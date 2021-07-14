import pandas as pd
import numpy as np


def main():

    anns = pd.read_json(
        "/scratch/visual/ashestak/meniscaltears/data/moaks_tse.json"
    ).sample(frac=1.0)

    train = int(round(len(anns) * 0.50))
    val = train + int(round(len(anns) * 0.15))

    print(len(anns))
    print(train, val)

    train, val, test = np.split(anns, (train, val))

    print("train", len(train))
    print((train.filter(like="V00") > 1).astype(int).sum())
    train.to_json(
        "/scratch/visual/ashestak/meniscaltears/data/train_tse.json", orient="records"
    )

    print("val", len(val))
    print((val.filter(like="V00") > 1).astype(int).sum())
    val.to_json(
        "/scratch/visual/ashestak/meniscaltears/data/val_tse.json", orient="records"
    )
    print("test", len(test))
    print((test.filter(like="V00") > 1).astype(int).sum())
    test.to_json(
        "/scratch/visual/ashestak/meniscaltears/data/test_tse.json", orient="records"
    )


if __name__ == "__main__":
    main()
