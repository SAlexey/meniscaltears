from collections import defaultdict
from tempfile import TemporaryDirectory
import time
from typing import Dict, Iterable, Sequence
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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

    med_lat_pred = []
    med_lat_target = []

    with torch.no_grad():
        for input, target in loader:
            input = input.to(_get_model_device(model))

            t0 = time.time()
            output = model(input)

            total_time += time.time() - t0
            total_steps += 1

            if postprocess is not None:
                output = postprocess(output, **postprocess_kwargs)

            targets.append(target)
            outputs.append({k: v.detach().cpu() for k, v in output.items()})
            med_lat_pred.append(torch.max(output["labels"], dim=-1)[0])
            med_lat_target.append(torch.max(target["labels"], dim=-1)[0])

            if criterion is not None:
                target = {k: v.to(device) for k, v in target.items()}

                loss_dict = criterion(output, target, **criterion_kwargs)
                loss = _reduce_loss_dict(loss_dict, criterion.weight)

                total_loss += loss.detach().cpu().item()

                if loss_dict:
                    losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})

    med_lat_pred = torch.cat(med_lat_pred, dim=0).cpu()
    med_lat_target = torch.cat(med_lat_target, dim=0).cpu()
    eval_results["lat_auc"] = roc_auc_score(med_lat_target[:, 0], med_lat_pred[:, 0])
    eval_results["med_auc"] = roc_auc_score(med_lat_target[:, 1], med_lat_pred[:, 1])
    eval_results["anywhere_auc"] = roc_auc_score(
        torch.max(med_lat_target, dim=-1)[0], torch.max(med_lat_pred, dim=-1)[0]
    )

    outputs = _reduce(outputs)
    targets = _reduce(targets)

    outputs["labels"] = outputs["labels"].sigmoid()

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

    outputs = []
    targets = []

    meters = defaultdict(lambda: SmoothedValue(window=window))

    device = _get_model_device(model)

    if progress:
        loader = tqdm(loader)

    for step, (input, target) in enumerate(loader):

        input, target = _to_device(input, target, device=device)

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

        _, output = _to_device(torch.empty(0), output)
        _, target = _to_device(torch.empty(0), output)

        outputs.append(output)
        targets.append(target)

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

    outputs = _reduce(outputs)
    targets = _reduce(targets)

    return {
        "outputs": outputs,
        "targets": targets,
        "total_loss": total_loss,
        "total_steps": total_steps,
        "total_time": total_time,
    }
