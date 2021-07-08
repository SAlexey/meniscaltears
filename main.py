# %%

from util.eval_utils import (
    balanced_accuracy_score,
    confusion_matrix,
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
from collections import defaultdict, deque
from math import ceil
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate, to_absolute_path, call
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import transforms as T
from data.oai import MOAKSDataset
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import SmoothedValue
from einops import rearrange
import sys


def _set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class MixCriterion(nn.Module):
    def __init__(self, **weights):
        self.weight = weights
        self.keys = {"giou": "boxes"}
        super().__init__()

    def loss_labels(self, out, tgt, **kwargs):
        loss = F.binary_cross_entropy_with_logits(out, tgt, **kwargs)
        return loss

    def loss_boxes(self, out, tgt, **kwargs):
        loss = F.l1_loss(out, tgt)
        return loss

    def loss_giou(self, out, tgt, **kwargs):
        out_xyxy = box_cxcywh_to_xyxy(out.flatten(0, 1))
        tgt_xyxy = box_cxcywh_to_xyxy(tgt.flatten(0, 1))
        giou = generalized_box_iou(out_xyxy, tgt_xyxy)
        loss = (1 - giou.diag()).mean()
        return loss

    def forward(self, out, tgt, return_interm_losses=True, **kwargs):
        losses = {}
        loss = 0
        for name, weight in self.weight.items():

            key = self.keys.get(name, name)
            value = getattr(self, f"loss_{name}")(out[key], tgt[key], **kwargs)
            losses[name] = value
            loss += value * weight

        assert losses
        if return_interm_losses:
            return loss, losses

        return loss


def _load_state(args, model, optimizer=None, scheduler=None, **kwargs):
    state_dict = {}

    device = torch.device(args.device)

    if args.checkpoint:
        state_dict = torch.load(to_absolute_path(args.checkpoint), map_location=device)

    if args.state_dict:
        state_dict["model"] = torch.load(
            to_absolute_path(args.state_dict), map_location=device
        )

    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])

    if "optimizer" in state_dict and optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])

    if "scheduler" in state_dict and scheduler is not None:
        scheduler.load_state_dict(state_dict["scheduler"])

    best_val_loss = state_dict.get("best_val_loss", kwargs.get("best_val_loss", np.inf))
    start = state_dict.get("epoch", kwargs.get("epoch", 0))

    if start > 0:
        start += 1

    return {"start": start, "best_val_loss": best_val_loss}


@hydra.main(config_path=".config/", config_name="config")
def main(args):
    _set_random_seed(50899)
    root = Path(os.environ["SCRATCH_ROOT"])

    data_dir = root / args.data_dir
    anns_dir = root / args.anns_dir

    assert data_dir.exists(), "Provided data directory doesn't exist!"
    assert anns_dir.exists(), "Provided annotations directory doesn't exist!"

    train_transforms = T.Compose(
        [T.ToTensor(), T.RandomResizedBBoxSafeCrop(), T.Normalize()]
    )

    val_transforms = T.Compose([T.ToTensor(), T.Normalize()])

    dataset_train = MOAKSDataset(
        data_dir,
        anns_dir / "train.json",
        multilabel=args.multilabel,
        transforms=train_transforms,
    )
    # limit number of training images
    if args.limit_train_items:
        dataset_train.anns = dataset_train.anns[: args.limit_train_items]

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataset_val = MOAKSDataset(
        data_dir,
        anns_dir / "val.json",
        multilabel=args.multilabel,
        transforms=val_transforms,
    )
    # limit number of val images

    if args.limit_val_items:
        dataset_val.anns = dataset_val.anns[: args.limit_val_items]

    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)

    model = instantiate(args.model)

    model.to(device)

    criterion = MixCriterion(**args.weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )
    epochs = args.num_epochs

    state = _load_state(args, model, optimizer, scheduler)
    start = state["start"]
    best_val_loss = state["best_val_loss"]

    metrics = {
        "balanced_accuracy": balanced_accuracy_score,
        "roc_curve": roc_curve,
        "roc_auc_score": roc_auc_score,
        "precision_recall_curve": precision_recall_curve,
        "confusion_matrix": confusion_matrix,
    }

    postprocess = lambda out: {
        "labels": rearrange(
            out["labels"],
            "bs tokens (obj cls) -> bs (tokens obj) cls",
            cls=args.model.cls_out // 2,
            obj=2,
            tokens=args.model.cls_tokens,
        ),
        "boxes": rearrange(
            out["boxes"].sigmoid(),
            "bs tokens (obj box) -> bs (tokens obj) box",
            box=args.model.box_out // 2,
            obj=2,
            tokens=args.model.cls_tokens,
        ),
    }

    if args.eval:

        logging.info("Running evaluation on the test set")

        dataset = MOAKSDataset(
            data_dir,
            anns_dir / "test.json",
            binary=args.binary,
            multilabel=args.multilabel,
            transforms=val_transforms,
        )

        loader = DataLoader(
            dataset, shuffle=False, batch_size=1, num_workers=args.num_workers
        )

        eval_results = evaluate(
            model, loader, postprocess=postprocess, progress=True, **metrics
        )

        torch.save(eval_results, "test_results.pt")
        logging.info("Testing finished, exitting")
        sys.exit(0)

    # start training
    logging.info(f"Startinng training epoch {start} ({time.strftime('%H:%M:%S')})")
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
                logging.info(f"confusion matrix for {name} TP [{tp}] | TN [{tn}] | FP [{fp}] | FN [{fn}]")

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

# %%
