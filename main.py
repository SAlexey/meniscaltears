#%%
from functools import partial
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict
import warnings

from matplotlib import pyplot as plt
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
import random
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import call, instantiate, to_absolute_path
from torch import nn, hub
from util.box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    denormalize_boxes,
    generalized_box_iou,
)
from einops import rearrange
import sys
from util.xai import SmoothGradientSaliency
from util.misc import SmoothedValue
from torch.utils.data import DataLoader


REGION = {0: "anterior_horn", 1: "body", 2: "posterior_horn"}
LAT_MED = {0: "lateral", 1: "medial"}


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

    @pick("boxes")
    def loss_inter(self, out, tgt, **kwargs):
        # iou term that separates output boxes from each other
        out_xyxy = box_cxcywh_to_xyxy(out.flatten(0, 1))
        loss = box_iou(out_xyxy[0::2, :], out_xyxy[1::2, :]).diag()
        loss = loss.mean()

        return loss

    def forward(
        self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor], **kwargs
    ):
        losses = {
            name: getattr(self, f"loss_{name}")(out, tgt, **kwargs)
            for name in self.weight
        }
        if "aux" in out:
            losses["aux"] = dict()
            for i, o in enumerate(out["aux"]):
                losses["aux"][i] = {
                    name: getattr(self, f"loss_{name}")(o, tgt, **kwargs)
                    for name in self.weight
                }

        assert losses
        return losses


# WARNING: NO SIGMOID IN POSTPROCESS FOR LABELS
# THEY WILL BE ACTIVATED IN EVALIATION [enpgine.py]
# AFTER BCEWithLogitsLoss HAS DONE ITS THING
class Postprocess(nn.Module):
    def forward(self, output: Dict[str, torch.Tensor]):
        if "pred_logits" in output:
            output["labels"] = output.pop("pred_logits")

        if "pred_boxes" in output:
            output["boxes"] = output.pop("pred_boxes")
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
        sd = torch.load(state_dict_path, map_location=device)

        remap_keys = {

            "backbone.0.body.conv1.weight": "backbone.0.body.stem.0.weight", 
            "backbone.0.body.bn1.weight": "backbone.0.body.stem.1.weight", 
            "backbone.0.body.bn1.bias": "backbone.0.body.stem.1.bias", 
            "backbone.0.body.bn1.running_mean": "backbone.0.body.stem.1.running_mean",
            "backbone.0.body.bn1.running_var": "backbone.0.body.stem.1.running_var",
            "backbone.0.body.bn1.num_batches_tracked": "backbone.0.body.stem.1.num_batches_tracked"
                }


        sd = {remap_keys.get(k, k): v for k,v in sd.items()}
        state_dict["model"] = sd
        logging.info(f"Loaded model weights from {state_dict_path}")
        if args.checkpoint:
            logging.warning("Model weights in checkpoint have been overwritten!")

    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])

    if "optimizer" in state_dict and optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])

    if "scheduler" in state_dict and scheduler is not None:
        scheduler.load_state_dict(state_dict["scheduler"])

    best_val_loss = state_dict.get("best_val_loss", kwargs.get("best_val_loss", np.inf))
    best_roc_auc = state_dict.get("best_roc_auc", kwargs.get("best_roc_auc", -np.inf))
    start = state_dict.get("epoch", kwargs.get("epoch", 0))

    if start > 0:
        start += 1

    return {
        "start": start,
        "best_val_loss": best_val_loss,
        "best_roc_auc": best_roc_auc,
    }


@hydra.main(config_path=".config/", config_name="config")
def main(args):
    _set_random_seed(50899)

    model = call(args.model)
    criterion = MixCriterion(**args.weights)

    device = torch.device(args.device)
    logging.info(f"Running On Device: {device}")

    model.to(device)
    criterion.to(device)

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if ("backbone" in n) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if ("backbone" not in n) and p.requires_grad
            ]
        },
    ]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )
    epochs = args.num_epochs

    losses = defaultdict(lambda: SmoothedValue(window=None))

    state = _load_state(args, model, optimizer, scheduler)
    start = state["start"]
    best_val_loss = state["best_val_loss"]
    best_roc_auc = -np.inf

    names = ("LAH", "LB", "LPH", "MAH", "MB", "MPH")

    METRICS = {
        "balanced_accuracy": partial(balanced_accuracy_score, names=names),
        "roc_curve": partial(roc_curve, names=names),
        "roc_auc_score": partial(roc_auc_score, names=names),
        "precision_recall_curve": partial(precision_recall_curve, names=names),
        "confusion_matrix": partial(confusion_matrix, names=names),
    }
    logging.info(f"Running: {model}")
    metrics = {key: METRICS[key] for key in args.metrics}

    postprocess = Postprocess()
    # WARNING: NO SIGMOID IN POSTPROCESS FOR LABELS
    # THEY WILL BE ACTIVATED IN EVALIATION [enpgine.py]
    # AFTER BCEWithLogitsLoss HAS DONE ITS THING

    # smooth saliency maps
    sg_sal = SmoothGradientSaliency(model, postprocess=postprocess, vanilla=True)

    data = call(args.data, train=not args.eval)

    if args.eval:
        loader = DataLoader(data, batch_size=args.batch_size)
        logging.info("Running evaluation on the test set")
        test_results = evaluate(
            model, loader, postprocess=postprocess, progress=True, **metrics
        )

        if "roc_auc_score" in test_results:
            logging.info(f"Test AUC: {test_results['roc_auc_score']}")

        if args.cam:
            logging.info(f"Obtaining GradCAM")
            for b_img, b_tgt in loader:
                b_targets = b_tgt["labels"]
                ids = b_tgt["image_id"]
                for i, (img, tgt, img_id) in enumerate(zip(b_img, b_targets, ids)):
                    img = img.unsqueeze(0)
                    for meniscus in range(2):
                        pos_labels = tgt[meniscus].nonzero()  # indeces where labels > 0
                        index = (0, meniscus, pos_labels)

                        name = f"{img_id.item()}_{LAT_MED[meniscus]}"

                        if args.crop:
                            target = None
                        else:
                            target = b_tgt["boxes"][i]

                        sg_sal(
                            img,
                            index,
                            tgt[meniscus],
                            save_as=name,
                            boxes=target,
                            num_passes=1,
                        )
        torch.save(test_results, "test_results.pt")
        logging.info("Testing finished, exitting")
        sys.exit(0)

    train_data, val_data = data

    loader_train = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    loader_val = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # start training
    logging.info(f"Epoch {start}; Best Validation Loss {best_val_loss:.4f}")
    logging.info(f"Starting training")
    for epoch in range(start, epochs):

        pos_weight = loader_train.dataset.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)

        train_results = train(
            model,
            loader_train,
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

        losses["train"] += epoch_loss

        logging.info(
            f"epoch [{epoch:04d} / {epochs:04d}] | training loss [{epoch_loss:.4f}] | trainig time [{epoch_time:.2f} ({step_time:.3f})]"
        )

        pos_weight = loader_val.dataset.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)

        eval_results = evaluate(
            model,
            loader_val,
            criterion,
            criterion_kwargs={"pos_weight": pos_weight},
            postprocess=postprocess,
            **metrics,
        )

        epoch_time = eval_results["total_time"]
        epoch_loss = eval_results["total_loss"] / eval_results["total_steps"]
        step_time = epoch_time / eval_results["total_steps"]

        losses["val"] += epoch_loss

        logging.info(
            f"epoch [{epoch:04d} / {epochs:04d}] | validation loss {epoch_loss:.4f} | inference time [{epoch_time:.2f} ({step_time:.3f})]"
        )

        for metric in ("balanced_accuracy", "roc_auc_score"):

            if metric in eval_results:

                logs = [f"{metric:>30}"]

                for name, value in eval_results[metric].items():

                    logs.append(f"{name} [{value:.4f}]")

                logging.info(" | ".join(logs))

        if (metric := "confusion_matrix") in eval_results:
            for name, value in eval_results[metric].items():
                log = [f"{metric:>26} {name:3}"]
                for label, each in zip(("tn", "fp", "fn", "tp"), value.flatten()):
                    log.append(f"{label.capitalize()} [{each:3d}]")
                logging.info(" | ".join(log))

        avg_auc = np.mean(
            list(eval_results.get("roc_auc_score", dict(default=-np.inf)).values())
        )

        if avg_auc > best_roc_auc:
            best_roc_auc = avg_auc

            torch.save(model.state_dict(), "best_roc_model.pt")
            torch.save(eval_results, "best_roc_model_eval.pt")

            with open("best_roc_model.json", "w") as fh:
                json.dump(
                    {
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "roc_auc": best_roc_auc,
                        "bs": args.batch_size,
                        "lr": args.lr,
                        "lr_b": args.lr_backbone,
                    },
                    fh,
                )

        if epoch_loss < best_val_loss:

            logging.info(f"Best Epoch Validation loss achieved!")
            best_val_loss = epoch_loss

            torch.save(model.state_dict(), "best_bce_model.pt")
            torch.save(eval_results, "best_bce_model_eval.pt")

            with open("best_bce_model.json", "w") as fh:
                json.dump(
                    {
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "roc_auc": avg_auc,
                        "batch": args.batch_size,
                    },
                    fh,
                )
            logging.info("Best Model Saved!")
        scheduler.step()

        if epoch % 5 == 0:

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_roc_auc": best_roc_auc,
                },
                "checkpoint.ckpt",
            )
            logging.info("Checkpoint Saved!")

        for name, loss in losses.items():
            plt.plot(loss.values, label=name)
        plt.legend()
        plt.savefig("epoch_loss.jpg")
        plt.close()

        # boxes = box_cxcywh_to_xyxy(eval_results["outputs"]["boxes"].flatten(0, 1))
        # ious = box_iou(boxes[0::2], boxes[1::2]).diag()
        # inter_iou_epoch.append(ious)

        # plt.violinplot(inter_iou_epoch)
        # plt.savefig("inter_iou.jpg")
        # plt.close()

    return best_val_loss


if __name__ == "__main__":

    p = Path("/scratch/")

    suffix = "ashestak/torch_hub"

    if (p / "visual").exists():
        hub.set_dir(p / "visual" / suffix)

    elif (p / "htc").exists():
        hub.set_dir(p / "htc" / suffix)

    else:
        warnings.warn("Unable to set torch hub directory! check yo /home/")

    main()

# %%
