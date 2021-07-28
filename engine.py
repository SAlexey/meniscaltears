from collections import defaultdict
from tempfile import TemporaryDirectory
import time
from typing import Dict, Iterable
import torch
from torch import nn
from tqdm import tqdm

from util.misc import SmoothedValue, _reduce, _to_device, _get_model_device
import logging

# utilities
def _reduce_loss_dict(loss_dict, weight_dict=dict()):
    return sum(loss * weight_dict.get(name, 1) for name, loss in loss_dict.items())


def evaluate(
    model,
    loader,
    criterion=None,
    criterion_kwargs=dict(),
    postprocess=None,
    postprocess_kwargs=dict(),
    progress=False,
    metrics_kwargs=dict(),
    **metrics,
):

    outputs = []
    targets = []
    losses = []

    eval_results = {}

    device = _get_model_device(model)

    total_loss = 0
    total_time = 0
    total_steps = 0

    model.eval()

    if progress:
        loader = tqdm(loader)

    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)

            t0 = time.time()
            output = model(input)

            total_time += time.time() - t0
            total_steps += 1

            if postprocess is not None:
                output = postprocess(output, **postprocess_kwargs)

            if criterion is not None:
                target = {k: v.to(device) for k, v in target.items()}

                loss_dict = criterion(output, target, **criterion_kwargs)
                loss = _reduce_loss_dict(loss_dict, criterion.weight)

                total_loss += loss.detach().cpu().item()

                losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})
            
            target = {k: v.detach().cpu() for k, v in target.items()}
            output = {k: v.detach().cpu() for k, v in output.items()} 
            
            targets.append(target)
            outputs.append(output)

    outputs = _reduce(outputs)
    targets = _reduce(targets)

    # WARNING: NO SIGMOID IN POSTPROCESS FOR LABELS
    # THEY WILL BE ACTIVATED HERE
    # AFTER BCEWithLogitsLoss HAS DONE ITS THING
    outputs["labels"] = outputs["labels"].sigmoid()

    outputs = {k: v.cpu() for k, v in outputs.items()}
    targets = {k: v.cpu() for k, v in targets.items()}

    eval_results["outputs"] = outputs
    eval_results["targets"] = targets
    eval_results["total_loss"] = total_loss
    eval_results["total_time"] = total_time
    eval_results["total_steps"] = total_steps

    if losses:
        eval_results["losses"] = _reduce(losses)

    for name, metric in metrics.items():
        eval_results[name] = metric(
            targets, outputs, **metrics_kwargs.get(name, dict())
        )

        if name == "roc_auc_score":
            tgt, out = {}, {}
            
            # reduce labels to menisci first 
            tgt["labels"] = targets["labels"].max(-1).values
            out["labels"] = outputs["labels"].max(-1).values

            # compute roc auc scores for menisci
            eval_results[f"menisci_{name}"] = metric(tgt, out, names=("lateral", "medial"))

            # reduce labels further to anywhere

            tgt["labels"] = tgt["labels"].max(-1).values.unsqueeze(1)
            out["labels"] = out["labels"].max(-1).values.unsqueeze(1)
            
            # compute roc auc for anywhere in the knee
            eval_results[f"anywhere_{name}"] = metric(tgt, out, names=("knee", ))

    return eval_results


def train(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    criterion_kwargs=dict(),
    postprocess=None,
    window: int = 1,
    epoch: int = 0,
    progress=False,
):

    model.train()

    total_loss = 0
    total_time = 0
    total_steps = 0
    
    meters = defaultdict(lambda: SmoothedValue(window=window))

    device = _get_model_device(model)

    if progress:
        loader = tqdm(loader)

    for step, (input, target) in enumerate(loader):

        input = input.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        t0 = time.time()

        output = model(input)

        if postprocess is not None:
            output = postprocess(output)

        loss_dict = criterion(output, target, **criterion_kwargs)
        loss = _reduce_loss_dict(loss_dict, criterion.weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_time = time.time() - t0

        loss = loss.detach().cpu().item()
        
        total_loss += loss
        meters["loss"] += loss
        for name, loss in loss_dict.items():
            meters[name] += loss.detach().cpu().item()

        if step and step % window == 0:

            logs = [
                f"epoch [{epoch:04d}]",
                f"step [{step:05d}]",
                f"time [{step_time:.3f}]",
            ]

            for i, group in enumerate(optimizer.param_groups):
                logs.append(f"lr_{i} [{group.get('lr'):.5f}]")

            for name, meter in meters.items():
                logs.append(f"{name} [{meter.value:.4f} ({meter.mean():.4f})]")

            logging.info(" | ".join(logs))

        total_time += step_time
        total_steps += 1

    return {
        "total_loss": total_loss,
        "total_steps": total_steps,
        "total_time": total_time,
    }
