# %%

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
from hydra.utils import instantiate, to_absolute_path
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import transforms as T
from data.oai import MOAKSDataset
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import SmoothedValue
from einops import rearrange

def _set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _get_device(name):
    device = torch.device("cpu")
    if "cuda" in str(name):
        if not torch.cuda.is_available():
            logging.warning(
                f"selected device={name}, but cuda is not available, returning device=cpu"
            )
            return device
        if not os.environ["USE_CUDA"]:
            logging.warning(
                f"selected device={name}, but it is prohibited by the $USE_CUDA env. variable, returning device=cpu"
            )
            return device
    device = torch.device(name)
    return device


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

    device = _get_device(args.device)

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
    # dataset_train.keys = dataset_train.keys[:1]
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
    # dataset_val.keys = dataset_val.keys[:1]
    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = _get_device(args.device)

    model = instantiate(args.model)

    model.to(device)

    criterion = MixCriterion(**args.weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )
    epochs = args.num_epochs
    window = args.window

    state = _load_state(args, model, optimizer, scheduler)
    start = state["start"]
    best_val_loss = state["best_val_loss"]

    metrics = defaultdict(lambda: SmoothedValue(window=args.window))

    train_steps = ceil(len(dataset_train) / dataloader_train.batch_size)
    val_steps = ceil(len(dataset_val) / dataloader_val.batch_size)

    postprocess = lambda out: {
            "labels": rearrange(out["labels"], "bs (obj l) -> bs obj l", obj=2), 
            "boxes": rearrange(out["boxes"].sigmoid(), "bs (obj box) -> bs obj box", obj=2)
    }
    logger = SummaryWriter()
    logging.info(f"Startinng training Epoch {start} ({time.strftime('%H:%M:%S')})")
    for epoch in range(start, epochs):

        total_loss = 0
        total_steps = 0
        total_time = 0

        pos_weight = dataloader_train.dataset.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)
        model.train()
        for step, (img, tgt) in enumerate(dataloader_train):
            img = img.to(device)
            tgt = {k: v.to(device).float() for k, v in tgt.items()}

            global_step = step + epoch * train_steps

            t0 = time.time()
            out = model(img)

            if postprocess is not None:
                out = postprocess(out)
            loss, loss_dict = criterion(out, tgt, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_time = time.time() - t0
            total_time += step_time

            loss_value = loss.detach().cpu().item()

            metrics["loss"] += loss_value

            for key, val in loss_dict.items():
                metrics[key] += val.detach().cpu().item()

            if step and (step % window == 0):

                lr = optimizer.param_groups[0].get("lr")

                metric_str = f"Epoch [{epoch:03d} / {epochs:03d}] || Step [{step:6d}] ||  Time [{step_time:0.3f} ({total_time / total_steps:.3f})] || lr [{lr:.5f}]"

                for name, metric in metrics.items():
                    avg_metric = metric.mean()
                    metric_str += f" || {name} [{metric.value:.4f} ({avg_metric:.4f})]"
                    logger.add_scalar(name, avg_metric, global_step=global_step)

                logging.info(metric_str)

            total_loss += loss_value
            total_steps += 1

        logger.add_scalar(
            "train_loss_epoch", total_loss / total_steps, global_step=epoch
        )

        with torch.no_grad():

            model.eval()

            total_loss = 0
            total_steps = 0
            total_time = 0

            pos_weight = dataloader_val.dataset.pos_weight
            if isinstance(pos_weight, torch.Tensor):
                pos_weight = pos_weight.to(device)

            for step, (img, tgt) in enumerate(dataloader_val):
                img = img.to(device)
                tgt = {k: v.to(device).float() for k, v in tgt.items()}

                global_step = step + epoch * val_steps
                t0 = time.time()

                out = model(img)

                if postprocess is not None:
                    out = postprocess(out)

                total_time += step_time
                loss = criterion(
                    out, tgt, return_interm_losses=False, pos_weight=pos_weight
                )

                loss_value = loss.detach().cpu().item()

                total_loss += loss_value
                total_steps += 1

            epoch_loss = total_loss / total_steps

            metric_str = f"Epoch [{epoch:03d} / {epochs:03d}] || Steps [{total_steps:5d}] || Time [{total_time:.2f} ({total_time / total_steps:.2f})] || Validation loss [{epoch_loss:.4f}]"

            logging.info(metric_str)
            logger.add_scalar("val_loss_epoch", epoch_loss, global_step=epoch)

            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss

                torch.save(model.state_dict(), "best_model.pt")

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

        if time.strftime("%H:%M") == "18:00":
            logging.info("Exitting due to time constraint")
            quit()


if __name__ == "__main__":
    # args = parse_args()
    main()

# %%
