import torch
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np


results = [
    "/scratch/visual/ashestak/meniscaltears/selected_models/resnet50_ablation/test_results.pt",
    "/scratch/visual/ashestak/meniscaltears/selected_models/Best_Dess_ResNet26/test_results.pt",
    "/scratch/visual/ashestak/meniscaltears/selected_models/resnet50_baseline/test_results.pt",
]


method_names = [
    "Full-scale",
    "BB-loss",
    "BB-loss",
]

plot_name = "ROC_curve_plot.tiff"


# create 2x1 subplots
fig, big_axes = plt.subplots(figsize=(22.0, 12.0), nrows=2, ncols=1, sharey=True)

# create sub-plots
for row, big_ax in enumerate(big_axes, start=1):
    if row == 1:
        big_ax.set_title("Medial Meniscus\n", fontsize=24, fontweight="bold")
    else:
        big_ax.set_title("Lateral Meniscus\n", fontsize=24, fontweight="bold")

    # Turn off axis lines and ticks of the big subplot
    # obs alpha is 0 in RGBA string!
    big_ax.tick_params(
        labelcolor=(1.0, 1.0, 1.0, 0.0),
        top="off",
        bottom="off",
        left="off",
        right="off",
    )
    # removes the white frame
    big_ax._frameon = False
    big_ax.set_yticklabels([])
    big_ax.set_xticklabels([])
    big_ax.set_yticks([])
    big_ax.set_xticks([])

plot_titles = [
    "Anterior Horn",
    "Body",
    "Posterior Horn",
    "Anterior Horn",
    "Body",
    "Posterior Horn",
]
all_keys = ["MAH", "MB", "MPH", "LAH", "LB", "LPH"]

# fill sub plots with ROC curves
for i in range(1, 7):
    # create subplot
    ax = fig.add_subplot(2, 3, i)

    # add subplot title
    ax.set_title(plot_titles[i - 1], fontsize=20)

    # current key
    my_key = all_keys[i - 1]

    for res, name in zip(results, method_names):

        res = torch.load(res)
        # ROC Curve of method 1
        fpr, tpr, thresholds = res["roc_curve"][my_key]
        d = RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=res["roc_auc_score"][my_key],
            estimator_name=name,
        ).plot(ax=ax)

    # set label font size
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)

    # set axes ticks and font size
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.80, 1.0], fontsize=20)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.80, 1.0], fontsize=20)

    # set legend font size
    plt.legend(fontsize=18)

# adjust horizontal space between the subplots
plt.subplots_adjust(wspace=1)

plt.tight_layout()
plt.savefig(plot_name)
# plt.show()
