import json
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt


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


def main():

    path = Path(
        "/nfs/datavisual/online/projects/PrevOp-Overload/thesis/VReddy/data/SAG_IW_TSE/"
    )
    medial = Path(
        "/nfs/datavisual/online/projects/PrevOp-Overload/thesis/VReddy/data/MM_Masks/"
    )
    lateral = Path(
        "/nfs/datavisual/online/projects/PrevOp-Overload/thesis/VReddy/data/LM_Masks/"
    )

    df = pd.read_csv(
        "/scratch/visual/ashestak/meniscaltears/data/kMRI_FNIH_SQ_MOAKS_BICL00.txt",
        sep="|",
    )

    rows = []
    boxes = []
    for each in path.rglob("*.mhd"):
        if not (row := df[df["ID"] == int(each.stem)]).empty:
            rows.append(row)

            if each.stem == "9001695":  # corrupt file
                continue
            # img = sitk.GetArrayFromImage(sitk.ReadImage(str(each)))
            lat = np.asarray(
                nibabel.load(lateral / f"{each.stem}.nii.gz").dataobj
            ).transpose(2, 1, 0)
            med = np.asarray(
                nibabel.load(medial / f"{each.stem}.nii.gz").dataobj
            ).transpose(2, 1, 0)

            lat_box = find_objects(lat)
            med_box = find_objects(med)

            boxes.append(lat_box + med_box)

    tse = pd.concat(rows)

    columns = [
        "ID",
        "SIDE",
        "V00MMTLA",
        "V00MMTLB",
        "V00MMTLP",
        "V00MMTMA",
        "V00MMTMB",
        "V00MMTMP",
    ]

    data = tse[columns]
    side = data["SIDE"].str.extract("([a-zA-Z]+)")
    side = side[0].str.lower()
    variables = []
    for each in columns[2:]:
        variable = data.loc[:, each].str.extract("(\d{1})")
        variables.append(pd.to_numeric(variable[0]))

    data = pd.concat([data["ID"], side] + variables, axis=1)
    data.columns = columns
    data.rename({"ID": "image_id", "SIDE": "side"}, axis=1, inplace=True)
    data["LAT"] = (data[columns[2:5]].mean(axis=1) > 1).astype(int)
    data["MED"] = (data[columns[5:]].mean(axis=1) > 1).astype(int)
    data = data[data["image_id"] != 9001695]
    data["boxes"] = boxes
    data.to_json("data/moaks_tse.json", orient="records")


if __name__ == "__main__":
    main()
