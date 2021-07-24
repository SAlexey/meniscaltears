from matplotlib.patches import Rectangle
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from util.box_ops import box_cxcywh_to_xyxy
import SimpleITK as sitk


def main():

    anns_path = "/scratch/visual/ashestak/meniscaltears/data/tse/test.json"
    anns_name = "TSE"
    test_path = (
        "/scratch/visual/ashestak/meniscaltears/selected_models/resnet50_tse/test_results.pt",
    )

    with open() as fh:
        anns = json.load(fh)

    test_results = torch.load(test_path)

    boxes = test_results["targets"]["boxes"]

    xy_areas = boxes[-2:].prod(-1)

    coords = boxes.unbind(-1)
    coords = [coord.flatten() for coord in coords]
    plt.violinplot(coords)
    plt.xlabel("Box Coordinate")
    plt.ylabel("Pixel")
    plt.xticks(list(range(1, 7)), ("cz", "cy", "cz", "d", "h", "w"))
    plt.savefig(f"outputs/BoxCoordsDistribution_{anns_name}.tiff")

    xy_areas = xy_areas.flatten().float()

    index = (xy_areas > xy_areas.quantile(0.98)).nonzero()
    for ann_idx, box_idx in zip(index // 2, index % 2):
        ann = anns[ann_idx]

        image = np.load(
            f"/scratch/htc/ashestak/oai/v00/data/inputs/{ann['image_id']}.npy"
        )
        d, h, w = image.shape
        midbox = boxes[ann_idx, box_idx, 0] * d

        out_box = test_results["outputs"]["boxes"][ann_idx, box_idx, [1, 2, 4, 5]]
        tgt_box = test_results["targets"]["boxes"][ann_idx, box_idx, [1, 2, 4, 5]]

        plt.imshow(image[midbox], "gray")
        plt.gca().add_patch(
            (
                plt.Rectangle(
                    (
                        (tgt_box[1] - 0.5 * tgt_box[3]) * w,
                        (tgt_box[0] - 0.5 * tgt_box[2]) * h,
                    ),
                    tgt_box[2] * w,
                    tgt_box[0] * h,
                    fill=False,
                    ec="blue",
                )
            )
        )

        plt.gca().add_patch(
            (
                plt.Rectangle(
                    (
                        (out_box[1] - 0.5 * out_box[3]) * w,
                        (out_box[0] - 0.5 * out_box[2]) * h,
                    ),
                    out_box[2] * w,
                    out_box[0] * h,
                    fill=False,
                    ec="orange",
                )
            )
        )

        plt.savefig(f"outputs/{ann['image_id']}_{box_idx}_{anns_name}.tiff")


if __name__ == "__main__":
    main()
