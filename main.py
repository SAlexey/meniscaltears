# %%

from functools import partial
from typing import Dict
from util.eval_utils import (
    balanced_accuracy_score,
    confusion_matrix,
    pick,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from engine import evaluate, train
import json
import logging
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate, to_absolute_path, call
from torch import nn
from torch.utils.data import DataLoader
from data import transforms as T
from data.oai import CropDataset, MOAKSDataset
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from einops import rearrange
import sys
from util.cam import MenisciCAM


def _set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class MixCriterion(nn.Module):
    """
    Computes a set of losses depending on the  weights passetd to init
    As input expects a dictionary for both target and output
    Returns a dictionary containning losses

    Example:
    output = model(input) = {"labels": torch.Tensor[bs, *], ...}
    target = {"labels": torch.Tensor[bs, *], ...}
    criterion = MixCriterion(labels=1) <- will only see the labels in output and target
    loss_dict = criterion(output, target) <- pass  dictionaries straight to forward

    loss_dict = {"labels": torch.Tensor[1]}
    """

    def __init__(self, **weights):
        super().__init__()
        self.weight = weights

    @pick("labels")
    def loss_labels(self, out, tgt, **kwargs):
        loss = F.binary_cross_entropy_with_logits(out, tgt, **kwargs)
        return loss

    @pick("boxes")
    def loss_boxes(self, out, tgt, **kwargs):
        loss = F.l1_loss(out, tgt)
        return loss

    @pick("boxes")
    def loss_giou(self, out, tgt, **kwargs):
        out_xyxy = box_cxcywh_to_xyxy(out.flatten(0, 1))
        tgt_xyxy = box_cxcywh_to_xyxy(tgt.flatten(0, 1))
        giou = generalized_box_iou(out_xyxy, tgt_xyxy)
        loss = (1 - giou.diag()).mean()
        return loss

    def forward(
        self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor], **kwargs
    ):
        losses = {
            name: getattr(self, f"loss_{name}")(out, tgt, **kwargs)
            for name in self.weight
        }
        assert losses
        return losses


class Postprocess(nn.Module):
    def forward(self, output: Dict[str, torch.Tensor]):
        if "labels" in output:
            output["labels"] = rearrange(
                output["labels"],
                "bs (menisci labels) -> bs menisci labels",
                menisci=2,
                labels=3,
            )

        if "boxes" in output:
            output["boxes"] = rearrange(
                output["boxes"].sigmoid(),
                "bs (menisci boxes) -> bs menisci boxes",
                menisci=2,
                boxes=6,
            )

        return output


def _load_state(args, model, optimizer=None, scheduler=None, **kwargs):
    state_dict = {}

    device = torch.device(args.device)

    if args.checkpoint:
        checkpoint_path = to_absolute_path(args.checkpoint)
        state_dict = torch.load(checkpoint_path, map_location=device)
        logging.info(f"Loaded a checkpoint from {checkpoint_path}")

    if args.state_dict:
        state_dict_path = to_absolute_path(args.state_dict)
        state_dict["model"] = torch.load(state_dict_path, map_location=device)
        logging.info(f"Loaded model weights from {state_dict_path}")
        if args.checkpoint:
            logging.warning("Model weights in checkpoint have been overwritten!")

    if "model" in state_dict:
        container = model
        if not args.load_mlp:
            logging.info("Only loading backbone weights")
            # will ignore cls_out and box_out and only load backbone parameters
            # usefull when swapping mlp heads
            state_dict["model"] = {
                k.replace("backbone.", ""): v
                for k, v in state_dict["model"].items()
                if "out" not in k
            }
            container = container.backbone
        container.load_state_dict(state_dict["model"])

    if "optimizer" in state_dict and optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])

    if "scheduler" in state_dict and scheduler is not None:
        scheduler.load_state_dict(state_dict["scheduler"])

    best_val_loss = state_dict.get("best_val_loss", kwargs.get("best_val_loss", np.inf))
    start = state_dict.get("epoch", kwargs.get("epoch", 0))

    if start > 0:
        start += 1

    logging.info(
        f"State loaded successfully! Epoch {start}; Best Validation Loss {best_val_loss:.4f}"
    )
    return {"start": start, "best_val_loss": best_val_loss}


@hydra.main(config_path=".config/", config_name="config")
def main(args):
    _set_random_seed(50899)

    root = Path("/scratch/htc/ashestak")

    if not root.exists():
        root = Path("/scratch/visual/ashestak")

    if not root.exists():
        raise ValueError(f"Invalid root directory: {root}")

    data_dir = root / args.data_dir
    anns_dir = root / args.anns_dir

    assert data_dir.exists(), "Provided data directory doesn't exist!"
    assert anns_dir.exists(), "Provided annotations directory doesn't exist!"

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=(0.4945), std=(0.3782,))

    if args.crop:
        train_transforms = T.Compose([to_tensor, T.CropIMG(), normalize])
        dataset_train = CropDataset(
            data_dir,
            anns_dir / "train.json",
            transforms=train_transforms,
        )
        val_transforms = T.Compose([T.ToTensor(), T.CropIMG(random=False), normalize])
        dataset_val = CropDataset(
            data_dir,
            anns_dir / "val.json",
            transforms=val_transforms,
            size=dataset_train.img_size,
        )

    else:
        train_transforms = T.Compose(
            [to_tensor, T.RandomResizedBBoxSafeCrop(), normalize]
        )
        dataset_train = MOAKSDataset(
            data_dir,
            anns_dir / "train.json",
            binary=args.binary,
            multilabel=args.multilabel,
            transforms=train_transforms,
        )
        val_transforms = T.Compose([to_tensor, normalize])
        dataset_val = MOAKSDataset(
            data_dir,
            anns_dir / "val.json",
            binary=args.binary,
            multilabel=args.multilabel,
            transforms=val_transforms,
        )

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)

    logging.info(f"Running On Device: {device}")

    model = instantiate(args.model)
    criterion = MixCriterion(**args.weights)
    model.to(device)

    mlp_params = [
        *model.out_cls.parameters(),
    ]

    if hasattr(model, "out_box"):
        for param in model.out_box.parameters():
            mlp_params.append(param)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": mlp_params, "lr": args.lr_head},
    ]

    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )
    epochs = args.num_epochs

    state = _load_state(args, model, optimizer, scheduler)
    start = state["start"]
    best_val_loss = state["best_val_loss"]

    names = ("LAH", "LB", "LPH", "MAH", "MB", "MPH")

    METRICS = {
        "balanced_accuracy": partial(balanced_accuracy_score, names=names),
        "roc_curve": partial(roc_curve, names=names),
        "roc_auc_score": partial(roc_auc_score, names=names),
        "precision_recall_curve": partial(precision_recall_curve, names=names),
        "confusion_matrix": partial(confusion_matrix, names=names),
    }

    metrics = {key: METRICS[key] for key in args.metrics}

    postprocess = Postprocess()

    if args.eval:

        logging.info("Running evaluation on the test set")

        dataset = CropDataset(
            data_dir,
            anns_dir / "test.json",
            size=dataset_train.img_size,
            transforms=val_transforms,
        )

        loader = DataLoader(
            dataset, shuffle=False, batch_size=1, num_workers=args.num_workers
        )

        eval_results = evaluate(
            model, loader, postprocess=postprocess, progress=True, **metrics
        )

        logging.info(f"Obtain GradCAM: {args.cam}")

        if args.cam:
            cam = MenisciCAM(
                model,
                model.layer4,
                use_cuda=args.device == "cuda",
                postprocess=postprocess,
            )

        torch.save(eval_results, "test_results.pt")
        logging.info("Testing finished, exitting")
        sys.exit(0)

    # start training
    logging.info(f"Starting training epoch {start} ({time.strftime('%H:%M:%S')})")
    for epoch in range(start, epochs):

        pos_weight = dataloader_train.dataset.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)

        train_results = train(
            model,
            dataloader_train,
            optimizer,
            criterion,
            criterion_kwargs={"pos_weight": pos_weight},
            postprocess=postprocess,
            window=args.window,
            epoch=epoch,
        )

        epoch_time = train_results["total_time"]
        epoch_loss = train_results["total_loss"] / train_results["total_steps"]
        step_time = epoch_time / train_results["total_steps"]

        logging.info(
            f"epoch [{epoch:04d} / {epochs:04d}] | training loss [{epoch_loss:.4f}] | trainig time [{epoch_time:.2f} ({step_time:.3f})]"
        )

        pos_weight = dataloader_val.dataset.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)

        eval_results = evaluate(
            model,
            dataloader_val,
            criterion,
            criterion_kwargs={"pos_weight": pos_weight},
            postprocess=postprocess,
            **metrics,
        )

        epoch_time = eval_results["total_time"]
        epoch_loss = eval_results["total_loss"] / eval_results["total_steps"]
        step_time = epoch_time / eval_results["total_steps"]

        logging.info(
            f"epoch [{epoch:04d} / {epochs:04d}] | validation loss {epoch_loss:.4f} | inference time [{epoch_time:.2f} ({step_time:.3f})]"
        )

        for metric in ("balanced_accuracy", "roc_auc_score"):

            if metric in eval_results:

                logs = [metric]

                for name, value in eval_results[metric].items():

                    logs.append(f"{name} [{value}]")

                logging.info(" | ".join(logs))

        if "confusion_matrix" in eval_results:

            for name, value in eval_results["confusion_matrix"].items():
                tn, fp, fn, tp = value.flatten()
                logging.info(
                    f"confusion matrix for {name} TP [{tp}] | TN [{tn}] | FP [{fp}] | FN [{fn}]"
                )

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss

            torch.save(model.state_dict(), "best_model.pt")
            torch.save(eval_results, "best_model_eval_results.ps")

            with open("best_model.json", "w") as fh:
                json.dump({"epoch": epoch, "val_loss": best_val_loss}, fh)

        scheduler.step()

        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                "checkpoint.ckpt",
            )


if __name__ == "__main__":
    main()

# %%ple indices must be integers or slices, not str
