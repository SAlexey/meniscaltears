#%%
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from hydra.utils import instantiate
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import transforms as T
from data.oai import MOAKSDatasetBinaryMultilabel
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.eval_utils import (
    evaluate_ious,
    evaluate_scores,
)
from util.misc import bbox_image, save_as_nifty

from util.plot_uitls import plot_confusion_matrix, plot_recall_iou


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


def _get_model(cfg, device="cpu", state_dict=None):
    if isinstance(device, str):
        device = _get_device(device)
    model = instantiate(cfg.model)
    if state_dict is not None:
        state_dict = torch.load(state_dict, map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


class Postprocess(nn.Module):
    def __init__(self, menisci=2, labels=3, boxes=6):
        super().__init__()
        self.boxes = nn.Sequential(
            nn.Sigmoid(),
            Rearrange(
                "bs (menisci boxes) -> bs menisci boxes", menisci=menisci, boxes=boxes
            ),
        )

        self.labels = nn.Sequential(
            nn.Sigmoid(),
            Rearrange(
                "bs (menisci labels) -> bs menisci labels",
                menisci=menisci,
                labels=labels,
            ),
        )

    def forward(self, model_out):

        if "boxes" in model_out:
            model_out["boxes"] = self.boxes(model_out["boxes"])

        if "labels" in model_out:
            model_out["labels"] = self.labels(model_out["labels"])

        return model_out


@hydra.main(config_path=".config/", config_name="test")
def main(args):

    root = Path(os.environ["SCRATCH_ROOT"])

    data_dir = root / args.data_dir
    anns_dir = root / args.anns_dir
    state_dict = root / args.weights

    assert anns_dir.exists(), "Provided path for annotations does not exist!"
    assert data_dir.exists(), "Provided path for data does not exist!"
    assert state_dict.exists(), "Provided path for state_dict does not exist!"

    if not args.output_dir:
        output_dir = state_dict.parents[1] / "testing"
    else:
        output_dir = root / args.output_dir

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    info = {}

    device = _get_device(args.device)
    model = _get_model(args, device, state_dict=state_dict)

    postprocess = Postprocess()

    info["name"] = args.model.backbone
    info["device"] = device.type
    info["nparams"] = sum(
        torch.prod(torch.as_tensor(p.size()))
        for p in model.parameters()
        if p.requires_grad
    )

    transforms = T.Compose((T.ToTensor(), T.Normalize()))
    data = MOAKSDatasetBinaryMultilabel(
        data_dir, anns_dir / "test.json", transforms=transforms
    )

    # data.keys = data.keys[:4]

    loader = DataLoader(data, batch_size=1, num_workers=3)
    image_ids = []
    scale_fct = []
    boxes_out = []
    boxes_tgt = []
    class_out = []
    class_tgt = []

    with torch.no_grad():

        inference_time = 0
        inference_step = 0

        for img, tgt in tqdm(loader):

            img = img.to(device)
            tgt = {k: v.to(device) for k, v in tgt.items()}

            t0 = time.time()

            out = postprocess(model(img))

            inference_time += time.time() - t0
            inference_step += 1

            out_box = out.get("boxes", [])
            out_cls = out.get("labels", [])

            tgt_box = tgt.get("boxes", [])
            tgt_cls = tgt.get("labels", [])
            tgt_ids = tgt["image_id"]

            shape = torch.as_tensor(img.shape[-3:]).to(device)

            shape = repeat(
                shape, "shape ->  repeat (obj shape)", obj=out_box.size(1), repeat=2
            )
            tgt_ids = repeat(
                tgt_ids, "ids -> (repeat obj ids)", obj=out_box.size(1), repeat=2
            )

            out_cls = rearrange(out_cls, "bs obj labels -> (bs obj) labels")
            tgt_cls = rearrange(tgt_cls, "bs obj labels -> (bs obj) labels")
            out_box = rearrange(out_box, "bs obj coords -> (bs obj) coords")
            tgt_box = rearrange(tgt_box, "bs obj coords -> (bs obj) coords")

            image_ids.extend([each for each in tgt_ids])
            class_out.extend([each for each in out_cls])
            class_tgt.extend([each for each in tgt_cls])
            boxes_out.extend([each for each in out_box])
            boxes_tgt.extend([each for each in tgt_box])
            scale_fct.extend([each for each in shape])

        info["mean_inference_time"] = inference_time / inference_step

    boxes_out = rearrange(boxes_out, "list coord -> list coord").cpu()
    boxes_tgt = rearrange(boxes_tgt, "list coord -> list coord").cpu()
    class_out = rearrange(class_out, "list class -> list class").cpu()
    class_tgt = rearrange(class_tgt, "list class -> list class").cpu()
    scale_fct = rearrange(scale_fct, "list fct -> list fct").cpu()
    image_ids = rearrange(image_ids, "ids -> ids").cpu()

    boxes_xyxy_out = box_cxcywh_to_xyxy(boxes_out)
    boxes_xyxy_tgt = box_cxcywh_to_xyxy(boxes_tgt)

    boxes_iou = box_iou(boxes_xyxy_out, boxes_xyxy_tgt).diag()

    boxes_xyxy_out = boxes_xyxy_out * scale_fct
    boxes_xyxy_tgt = boxes_xyxy_tgt * scale_fct

    if not (output_data := (output_dir / "data")).exists():
        output_data.mkdir(parents=True)

    if not (output_metrics := (output_dir / "metrics")).exists():
        output_metrics.mkdir(parents=True)

    # save everything

    torch.save(
        {
            **info,
            "image_ids": image_ids,
            "output_boxes": boxes_xyxy_out,
            "target_boxes": boxes_xyxy_tgt,
            "boxes_iou": boxes_iou,
            "output_scores": class_out,
            "target_labels": class_tgt,
        },
        output_data / "predictions.pt",
    )

    data = defaultdict(dict)

    ax = None

    for i, meniscus in enumerate(("medial", "lateral")):
        evals = evaluate_ious(boxes_iou[i::2])
        data["iou_eval"][meniscus] = evals
        rec = evals["recall_curve"]
        fig = plot_recall_iou(
            rec["recall"], rec["iou"], rec["auc"], ax=ax, name=meniscus
        )
        ax = fig.axes[0]

        # box_ev_out = boxes_out[i::2]
        # box_ev_tgt = boxes_tgt[i::2]
        # evals = evaluate_boxes(box_ev_out, box_ev_tgt)
        # data["box_eval"][meniscus] = evals

        # frac = box_ev_out / box_ev_tgt
        # f, axis = plt.subplots()
        # axis.violinplot(frac[:3].unbind(-1))

        # axis.set_xlim((0, 4))
        # axis.set_xticks((1, 2, 3))
        # axis.set_xticklabels(("Z Min", "Y Min", "X Min"))
        # axis.axhline(1, c="darkblue")
        # axis.text(3.5, 1.15, "Over-shoot")
        # axis.text(3.5, 0.85, "Under-shoot")

        # f.savefig(output_metrics / f"BoxFracMin_{meniscus}.png")
        # plt.close()

        # f, axis = plt.subplots()
        # axis.violinplot(frac[3:].unbind(-1))
        # axis.set_xlim((0, 4))
        # axis.set_xticks((1, 2, 3))
        # axis.set_xticklabels(("Z Max", "Y Max", "X Max"))
        # axis.axhline(1, c="darkblue")
        # axis.text(3.5, 1.15, "Under-shoot")
        # axis.text(3.5, 0.85, "Over-shoot")
        # f.savefig(output_metrics / f"BoxFracMax_{meniscus}.png")
        # plt.close()

    fig.savefig(output_metrics / "recall_iou_curve.png")
    plt.close()

    plt.violinplot([boxes_iou[0::2], boxes_iou[1::2]])
    ax = plt.gca()
    ax.set_xticks((1, 2))
    ax.set_xticklabels(("MED", "LAT"))
    plt.savefig(output_metrics / "IOU_violin.png")

    pr_fig, pr_axis = plt.subplots()
    roc_fig, roc_axis = plt.subplots()

    for tgt_cls, out_cls, part in zip(class_tgt.T, class_out.T, ("AH", "B", "PH")):
        evals = evaluate_scores(out_cls.float(), tgt_cls.int())

        data["cls_eval"][part] = evals

        pr = evals["pr_curve"]
        roc = evals["roc_curve"]

        metrics.PrecisionRecallDisplay(
            estimator_name=part,
            precision=pr["precision"],
            recall=pr["recall"],
            average_precision=pr["auc"],
        ).plot(ax=pr_axis)

        metrics.RocCurveDisplay(
            estimator_name=part, fpr=roc["fpr"], tpr=roc["tpr"], roc_auc=roc["auc"]
        ).plot(ax=roc_axis)

    roc_fig.savefig(output_metrics / "roc_curve.png")
    pr_fig.savefig(output_metrics / "pr_curve.png")
    torch.save(data, output_metrics / "metrics.pt")

    # sort iou value indices in ascending order
    iou_idx_asc = boxes_iou.argsort()

    # get 5 worst cases
    n = 10

    worst_ids = image_ids[iou_idx_asc[:n]]
    worst_out = boxes_out[iou_idx_asc[:n]]
    worst_tgt = boxes_tgt[iou_idx_asc[:n]]

    # group results by image_id

    for img_id in worst_ids.unique():

        idxs = (image_ids == img_id).nonzero()
        id_out = worst_out[idxs].flatten(0, 1)
        id_tgt = worst_tgt[idxs].flatten(0, 1)

        img_out = bbox_image(id_out)
        img_tgt = bbox_image(id_tgt)
        
        save_as_nifty(img_out, output_metrics / f"{img_id}_out")
        save_as_nifty(img_tgt, output_metrics / f"{img_id}_tgt")



if __name__ == "__main__":
    main()

# %%
