#%%
from functools import partial
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict
import warnings

from matplotlib import pyplot as plt
from torch.nn.modules import activation
from util.eval_utils import (
    balanced_accuracy_score,
    confusion_matrix,
    pick,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from engine import evaluate, train
from argparse import Namespace
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
from einops.layers.torch import Rearrange
import sys
from util.xai import SmoothGradientSaliency
from util.misc import EarlyStopping, SmoothedValue
from util.cam import to_gif
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
    def __init__(self, cfg):
        super().__init__()
        _, _, class_set = cfg.data.setting.split("-")
        
        labels_axes = {
            "query": cfg.model.num_queries,
            "class": cfg.model.num_classes,
        }

        coords_axes = {
            "query": cfg.model.num_queries,
            "box": 6
        }
        moaks = class_set == "moaks"
        
        if cfg.model.transformer is None:
            labels = f"bs (query class) -> bs query class"
            coords = f"bs (query box) -> bs query box"
        else:
            if moaks:
                labels = f"bs query (moaks class) -> bs class query moaks"
            else:
                labels = f"bs query class -> bs query class"
            coords = f"bs query box -> bs query box"

        self.labels = nn.Sequential(
            Rearrange(labels, **labels_axes),
            nn.Softmax(dim=1) if moaks else nn.Identity()
        )

        self.boxes = nn.Sequential(
            Rearrange(coords, **coords_axes),
            nn.Sigmoid()
        )
    

    def forward(
        self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], **kwargs
    ):
        if "labels" in output:
            output["labels"] = self.labels(output["labels"])
        if "boxes" in output:
            output["boxes"] = self.boxes(output["boxes"])
        return output



#%%


def _load_state(args, model, optimizer=None, scheduler=None, **kwargs):
    state_dict = {}

    device = torch.device(args.device)

    if args.checkpoint:
        checkpoint_path = to_absolute_path(args.checkpoint)
        state_dict = torch.load(checkpoint_path, map_location=device)
        logging.info(f"Loaded a checkpoint from {checkpoint_path}")

    if args.state_dict:
        state_dict_path = to_absolute_path(args.state_dict)
        model_state_dict = torch.load(state_dict_path, map_location=device)

        if args.backbone_only:
            model_state_dict = {k.replace("backbone.", ""): v for k, v in model_state_dict.items() if "backbone" in k}
            model = model.backbone

        state_dict["model"] = model_state_dict
        logging.info(f"Loaded model weights from {state_dict_path}")

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

    device = torch.device(args.device)
    logging.info(f"Running On Device: {device}")

    if args.data.setting.endswith("moaks"):
        assert args.model.transformer is not None, "MOAKS works only for transformers!"

    model = call(args.model)
    model.to(device)

    
    NAMES = {
        "global": ("anywhere", ),
        "meniscus": ("lateral", "medial"),
        "region": ("LAH", "LB", "LPH", "MAH", "MB", "MPH")
    }

    names = NAMES[args.data.setting.split("-")[0]]

    METRICS = {
        "balanced_accuracy": partial(balanced_accuracy_score, names=names),
        "roc_curve": partial(roc_curve, names=names),
        "roc_auc_score": partial(roc_auc_score, names=names),
        "precision_recall_curve": partial(precision_recall_curve, names=names),
        "confusion_matrix": partial(confusion_matrix, names=names),
    }
    metrics = {key: METRICS[key] for key in args.metrics}
    
    logging.info(f"Running: {model}")

    postprocess = Postprocess(args)
    # WARNING: NO SIGMOID IN POSTPROCESS FOR LABELS
    # THEY WILL BE ACTIVATED IN EVALIATION [enpgine.py]
    # AFTER BCEWithLogitsLoss HAS DONE ITS THING

    # smooth saliency maps
    sg_sal = SmoothGradientSaliency(model, postprocess=postprocess, vanilla=True)
    data = call(args.data, train=not args.eval)

    if args.eval:
        _load_state(args, model)
        loader = DataLoader(data, batch_size=args.batch_size)
        logging.info("Running evaluation on the test set")
        test_results = evaluate(
            model, loader, postprocess=postprocess, progress=True, **metrics
        )

        if "roc_auc_score" in test_results:
            logging.info(f"Test AUC: {test_results['roc_auc_score']}")

        if args.cam:
            assert args.batch_size == 1, "Only batch size 1 is supported!"    
            logging.info(f"Obtaining GradCAM")

            
            for input, target in loader:
                conv_features, enc_attn_weights, dec_attn_weights = [], [], []

                hooks = [
                    model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)
                    ),
                    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])
                    ),
                    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])
                    ),
                ]

                outputs = model(input)
                print("o", outputs)
                print("t", target)

                for hook in hooks:
                    hook.remove()

                image_id = target['image_id'].squeeze().item()

                # don't need the list anymore
                conv_features = conv_features[-1]
                enc_attn_weights = enc_attn_weights[-1]
                dec_attn_weights = dec_attn_weights[-1]
                d, h, w = conv_features[0].tensors.shape[-3:]

                fig, ax = plt.subplots(figsize=(10, 10))

                dec_attn_weights = dec_attn_weights.view(1, 1, d,h,w).detach()

                # 5 points where decoder attention is max
                sattn = enc_attn_weights.reshape((d, h, w, d, h, w))


                attention = []
                attention.append(sattn[..., 4, 4, 4])
                attention.append(sattn[..., 8, 4, 4])

                attention = torch.stack(attention).mean(0).detach().view(1, 1, 10, 10, 10)
                print(attention.shape)


                dec_attn_weights = F.interpolate(dec_attn_weights, input.size()[-3:], mode="trilinear")
                enc_attn_weights = F.interpolate(attention, input.size()[-3:], mode="trilinear")
                
                ax.imshow(dec_attn_weights[0, 0, 40])
                ax.axis('off')
                ax.set_title(f'query id: {1}')

                to_gif(input.detach().squeeze(), dec_attn_weights.squeeze(),f"/scratch/visual/ashestak/meniscaltears/attention/dec_attention_{image_id}.gif")
                to_gif(input.detach().squeeze(), enc_attn_weights.squeeze(),f"/scratch/visual/ashestak/meniscaltears/attention/enc_attention_{image_id}.gif")

                
                # logging.info(f"Image id: {image_id}; labels={target['labels'].flatten()}")
                # index = (0, 0, ...)
                # name = f"{image_id}_global"
                # sg_sal(input, index, save_as=name, num_passes=20, target=target)

            


        torch.save(test_results, "test_results.pt")
        logging.info("Testing finished, exitting")
        sys.exit(0)

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

    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )
    epochs = args.num_epochs

    losses = defaultdict(lambda: SmoothedValue(window=None))

    state = _load_state(args, model, optimizer, scheduler)
    start = state["start"]
    best_val_loss = state["best_val_loss"]

    
    tracker = EarlyStopping(name="val_loss", patience=args.early_stop.patience, warmup=args.early_stop.warmup)

    criterion = MixCriterion(**args.weights)
    criterion.to(device)


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
        
        scheduler.step()

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

        early_stop = tracker(epoch_loss, model, epoch)

        if early_stop:
            tracker.checkpoint(model, epoch, optimizer, scheduler)
            logging.info("Early stopping criterion met, exitting")
            return tracker.best[0]


        if epoch % 5 == 0:
            tracker.checkpoint(model, epoch, optimizer, scheduler)
            logging.info("Checkpoint Saved!")

        for name, loss in losses.items():
            plt.plot(loss.values, label=name)

        plt.legend()
        plt.savefig("epoch_loss.jpg")
        plt.close()

    return tracker.best[0]


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
