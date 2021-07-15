#%%

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data.transforms import RandomResizedBBoxSafeCrop, Normalize, Compose

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
import random
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate, to_absolute_path
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from einops import rearrange
import sys
from data.oai import build, CropDataset, MOAKSDataset, MixDataset
from util.cam import MenisciCAM, to_gif, MenisciSaliency, GuidedBackprop


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

    best_val_loss = state_dict.get("best_val_loss", kwargs.get("best_val_loss", -np.inf))
    start = state_dict.get("epoch", kwargs.get("epoch", 0))

    if start > 0:
        start += 1

    return {"start": start, "best_val_loss": best_val_loss}


@hydra.main(config_path=".config/", config_name="config")
def main(args):
    _set_random_seed(50899)
    dataloader_train, dataloader_val, dataloader_test = build(args)
    
    if args.mix:

        dataset_train_mix = MixDataset(
                dataloader_train.dataset.root, 
                "/scratch/htc/ashestak/meniscaltears/data/train.json",
                "/scratch/htc/ashestak/meniscaltears/data/tse/train.json",
                train=True, transforms=Compose((RandomResizedBBoxSafeCrop(), Normalize()))
                )

        dataloader_train_mix = DataLoader(dataset_train_mix, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)

        #dataset_val_tse = MOAKSDataset(
        #        dataloader_val.dataset.root, 
        #        "/scratch/htc/ashestak/meniscaltears/data/tse/val.json",
        #        transform=dataloader_val.dataset.transform
        #        )
        #
        #dataloader_val_tse = DataLoader(dataset_val_tse, num_workers=args.num_workers, batch_size=args.batch_size)
        #dataset_test_tse = MOAKSDataset(
        #        dataloader_val.dataset.root, 
        #        "/scratch/htc/ashestak/meniscaltears/data/tse/test.json",
        #        transform=dataloader_val.dataset.transform
        #        )
        
        # dataloader_train_mix = DataLoader(dataset_test_tse, num_workers=args.num_workers, batch_size=args.batch_size)

        dataloader_train = dataloader_train_mix
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
    logging.info(f"Running: {model}")
    logging.info(f"Running model on {'TSE' if args.tse else 'DESS'} sequence")
    logging.info(
        f'Running model on dataset {"with" if isinstance(dataloader_train.dataset, CropDataset) else "without"} cropping\n'
    )

    metrics = {key: METRICS[key] for key in args.metrics}

    postprocess = Postprocess()

    if args.eval:

        logging.info("Running evaluation on the validation set")
        val_results = evaluate(
            model, dataloader_val, postprocess=postprocess, progress=True, **metrics
        )
        logging.info(f"Validation AUC: {val_results['roc_auc_score']}")

        logging.info("Running evaluation on the test set")
        test_results = evaluate(
            model, dataloader_test, postprocess=postprocess, progress=True, **metrics
        )
        logging.info(f"Test AUC: {test_results['roc_auc_score']}")

        if args.cam:
            logging.info(f"Obtaining GradCAM")

            cam = MenisciCAM(
                model,
                model.backbone.layer4,
                use_cuda=args.device == "cuda",
                postprocess=postprocess,
            )
            saliency = MenisciSaliency(
                model, use_cuda=args.device == "cuda", 
                postprocess=postprocess
            )
            g_back = GuidedBackprop(
                model,
                use_cuda=args.device == "cuda",
                postprocess=postprocess,
            )

            for bs_img, bs_ann in dataloader_test:
                for i in range(len(bs_img)):
                    img = bs_img[i].unsqueeze(0)
                    ann = dict()
                    for key in bs_ann.keys():
                        ann[key] = bs_ann[key][i]
                    for meniscus in args.meniscus:
                        if ann["labels"][meniscus].any():
                            men_labels = (
                                ann["labels"][meniscus].detach().cpu().numpy().flatten()
                            )
                            for idx in np.argwhere(men_labels > 0):
                                cam_img = (
                                    cam(img, meniscus, idx)
                                    .squeeze()
                                    .numpy()
                                )
                                sal_img = (
                                    saliency(img, meniscus, idx)
                                    .detach()
                                    .cpu()
                                    .squeeze()
                                    .numpy()
                                )
                                back_img = (
                                    g_back.forward(img, meniscus, idx)
                                    .detach()
                                    .cpu()
                                    .squeeze()
                                    .numpy()
                                )
                                np.save(
                                    f"{ann['image_id'].item()}_{meniscus}_cam", cam_img
                                )
                                to_gif(
                                    img,
                                    cam_img,
                                    f"{ann['image_id'].item()}_{LAT_MED[meniscus]}_{REGION[idx[0]]}_gradcam.gif",
                                    cam_type="grad"
                                )
                                to_gif(
                                    img,
                                    sal_img,
                                    f"{ann['image_id'].item()}_{LAT_MED[meniscus]}_{REGION[idx[0]]}_saliency.gif",
                                    cam_type="back",
                                )
                                to_gif(
                                    img,
                                    back_img,
                                    f"{ann['image_id'].item()}_{LAT_MED[meniscus]}_{REGION[idx[0]]}_guided.gif",
                                    cam_type="back"
                                )

        torch.save(test_results, "test_results.pt")
        logging.info("Testing finished, exitting")
        sys.exit(0)

    # start training
    logging.info(f"Epoch {start}; Best Validation Loss {best_val_loss:.4f}")
    logging.info(f"Starting training")
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
        #weighting = pos_weight.detach().cpu().numpy().flatten()
        #weighting /= weighting.sum()
        #print(weighting)

        epoch_eval = np.fromiter(eval_results["roc_auc_score"].values(), dtype=float).mean() #* weighting
        if epoch_eval > best_val_loss:
            best_val_loss = epoch_eval

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
    return best_val_loss


if __name__ == "__main__":
    main()

# %%ple indices must be integers or slices, not str
