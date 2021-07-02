"""
Script to generate annotations

requires moaks_complete.csv to work

Args:
    
    train   (float) - fraction of data to use for training
    val     (float) - validation fraction
    test    (float) - testing fraction
    boxes   (bool)  - generate boxes

    output  (string) - where to put the annotations

Returns:
    annotations (saves files to output)

Notes: 
    - it is not guaranteed that every image has 2 boxes for menisci (some images have missing menisci)
        
        IMOPORTANT!!
    -   in the segmentation masks the order of the menisci labels is always 5 -> 6 
        however on the left side 5 corresponds to lateral and 6 to medial meniscus
        and on the right side this order is flipped. You need to take care of this 
        discrepancy. This script takes the boxes in order of the segmentation label  
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from numpy.random import seed
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from scipy import ndimage

# copy paste from  utils.box_ops
def _objects_to_boxes(objects):
    boxes = []
    for obj in objects:
        mins = []
        maxs = []
        if obj is not None:
            for ax in obj:
                mins.append(ax.start)
                maxs.append(ax.stop)
        boxes.append(mins + maxs)
    return boxes


def find_objects(mask, boxes=True, **kwargs):
    objects = ndimage.find_objects(mask, **kwargs)
    if not boxes:
        return objects
    return _objects_to_boxes(objects)


# end copy paste


def main(args):

    moaks = pd.read_csv("moaks_complete.csv")
    moaks = moaks.sample(frac=1.0, random_state=args.seed)  # randomize

    train = int(round(len(moaks) * args.train))
    val = train + int(round(len(moaks)) * args.val)

    if args.boxes:

        bxs = []

        segm_dir = Path("/vis/scratchN/bzftacka/OAI_DESS_Data_AllTPs/Merged/v00/OAI/")
        reader = sitk.ImageSeriesReader()

        for _, row in tqdm(moaks.iterrows(), desc="Generating Boxes", total=len(moaks)):

            # segmentation dicom file names
            segm_path = str(segm_dir / row["PATH"] / "Segmentation")
            file_names = reader.GetGDCMSeriesFileNames(segm_path)
            reader.SetFileNames(file_names)
            image = reader.Execute()

            mask = sitk.GetArrayFromImage(image)

            boxes = find_objects(mask)
            bxs.append(boxes[-2:])

    moaks["boxes"] = bxs

    train, val, test = np.split(moaks, [train, val], seed=args.seed)

    train.to_json(Path(args.output) / "train.json", orient="records")
    val.to_json(Path(args.output) / "val.json", orient="records")
    test.to_json(Path(args.output) / "test.json", orient="records")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--train", type=float, default=0.5)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.35)
    parser.add_argument("--boxes", action="store_true")

    parser.add_argument("--output", type=str, default="./")

    args = parser.parse_args()

    assert sum((args.train, args.val, args.test)) == 1.0, "Splits must add  up  to 1.0"
    assert Path(
        args.output
    ).exists(), "Provided path for annotation output does not exist!"

    main(args)
