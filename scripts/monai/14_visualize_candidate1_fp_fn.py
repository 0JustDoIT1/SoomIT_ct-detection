from pathlib import Path
import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LUNA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "LUNA16"
)

RAW_ROOT = (
    LUNA_ROOT
    / "raw"
)

BUNDLE_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "monai"
    / "bundles"
    / "lung_nodule_ct_detection"
)

FOLD0_PATH = (
    BUNDLE_ROOT
    / "LUNA16_datasplit"
    / "mhd_original"
    / "dataset_fold0.json"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "candidate1_fold0"
)

PREDICTION_CSV = (
    RESULT_ROOT
    / "predictions.csv"
)

OUTPUT_DIR = (
    RESULT_ROOT
    / "visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Colors
# ============================================================

GT_COLOR = "lime"
PRED_COLOR = "red"
FP_COLOR = "orange"


# ============================================================
# Utility
# ============================================================

def find_mhd(seriesuid):

    matches = list(
        RAW_ROOT.rglob(
            f"{seriesuid}.mhd"
        )
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"MHD 없음: {seriesuid}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"MHD 중복: {seriesuid}"
        )

    return matches[0]


def get_uid_from_image(image_value):

    name = Path(
        str(image_value)
    ).name

    if name.endswith(".mhd"):
        return name[:-4]

    return Path(name).stem


def world_to_voxel(
    image,
    x,
    y,
    z,
):

    return image.TransformPhysicalPointToContinuousIndex(
        (
            float(x),
            float(y),
            float(z),
        )
    )


def center_distance(
    center_a,
    center_b,
):

    return math.sqrt(
        (center_a[0] - center_b[0]) ** 2
        + (center_a[1] - center_b[1]) ** 2
        + (center_a[2] - center_b[2]) ** 2
    )


# ============================================================
# Load GT
# ============================================================

with open(
    FOLD0_PATH,
    "r",
    encoding="utf-8",
) as f:

    fold0 = json.load(f)


gt_by_uid = {}


for item in fold0["validation"]:

    uid = get_uid_from_image(
        item["image"]
    )

    gt_by_uid[uid] = [
        [
            float(v)
            for v in box
        ]
        for box in item.get(
            "box",
            []
        )
    ]


# ============================================================
# Load Predictions
# ============================================================

records = []


with open(
    PREDICTION_CSV,
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        records.append(
            {
                "seriesuid": row["seriesuid"],

                "box": [
                    float(row["coordX"]),
                    float(row["coordY"]),
                    float(row["coordZ"]),
                    float(row["width"]),
                    float(row["height"]),
                    float(row["depth"]),
                ],

                "score": float(
                    row["score"]
                ),

                "status": row["status"],
            }
        )


# ============================================================
# Select Highest-score FP
# ============================================================

fp_records = [
    record
    for record in records
    if record["status"] == "FP"
]


if not fp_records:
    raise RuntimeError(
        "FP prediction 없음"
    )


selected_fp = max(
    fp_records,
    key=lambda x: x["score"],
)


# ============================================================
# Find FN GT
#
# GT 가운데 어떤 prediction center도
# diameter/2 이내에 들어오지 않은 결절
# ============================================================

pred_by_uid = {}


for record in records:

    pred_by_uid.setdefault(
        record["seriesuid"],
        []
    ).append(
        record
    )


fn_candidates = []


for uid, gt_boxes in gt_by_uid.items():

    predictions = pred_by_uid.get(
        uid,
        []
    )

    for gt_idx, gt_box in enumerate(
        gt_boxes
    ):

        gx, gy, gz, gw, gh, gd = gt_box

        gt_center = (
            gx,
            gy,
            gz,
        )

        diameter = (
            gw + gh + gd
        ) / 3.0

        radius = (
            diameter
            / 2.0
        )

        matched = False

        nearest_distance = None
        nearest_score = None


        for pred in predictions:

            px, py, pz = pred["box"][:3]

            distance = center_distance(
                gt_center,
                (
                    px,
                    py,
                    pz,
                ),
            )

            if (
                nearest_distance is None
                or distance < nearest_distance
            ):

                nearest_distance = distance
                nearest_score = pred["score"]


            if distance < radius:

                matched = True
                break


        if not matched:

            fn_candidates.append(
                {
                    "seriesuid": uid,
                    "gt_index": gt_idx,
                    "gt_box": gt_box,
                    "nearest_distance": nearest_distance,
                    "nearest_score": nearest_score,
                }
            )


if not fn_candidates:

    raise RuntimeError(
        "FN GT 없음"
    )


# 가장 큰 결절 FN을 대표 사례로 선택
selected_fn = max(
    fn_candidates,
    key=lambda item: (
        item["gt_box"][3]
        + item["gt_box"][4]
        + item["gt_box"][5]
    ) / 3.0,
)


# ============================================================
# Visualization: FP
# ============================================================

def visualize_fp(record):

    seriesuid = record["seriesuid"]

    px, py, pz, pw, ph, pd = (
        record["box"]
    )

    score = record["score"]


    mhd_path = find_mhd(
        seriesuid
    )


    image = sitk.ReadImage(
        str(mhd_path)
    )

    volume = sitk.GetArrayFromImage(
        image
    ).astype(
        np.float32
    )

    spacing = image.GetSpacing()


    pred_idx = world_to_voxel(
        image,
        px,
        py,
        pz,
    )


    pred_cx, pred_cy, pred_cz_float = (
        pred_idx
    )


    slice_z = int(
        round(
            pred_cz_float
        )
    )


    pred_w_vox = (
        pw
        / spacing[0]
    )

    pred_h_vox = (
        ph
        / spacing[1]
    )


    pred_xmin = (
        pred_cx
        - pred_w_vox / 2
    )

    pred_ymin = (
        pred_cy
        - pred_h_vox / 2
    )


    WINDOW_MIN = -1000
    WINDOW_MAX = 400


    slice_image = np.clip(
        volume[slice_z],
        WINDOW_MIN,
        WINDOW_MAX,
    )


    fig, ax = plt.subplots(
        figsize=(9, 9)
    )


    ax.imshow(
        slice_image,
        cmap="gray",
        vmin=WINDOW_MIN,
        vmax=WINDOW_MAX,
    )


    pred_rect = Rectangle(
        (
            pred_xmin,
            pred_ymin,
        ),
        pred_w_vox,
        pred_h_vox,
        fill=False,
        edgecolor=FP_COLOR,
        linewidth=3,
        linestyle="--",
    )

    ax.add_patch(
        pred_rect
    )


    ax.scatter(
        [pred_cx],
        [pred_cy],
        marker="+",
        s=160,
        c=FP_COLOR,
        linewidths=3,
    )


    ax.text(
        pred_xmin,
        pred_ymin - 8,
        f"False Positive | Score={score:.4f}",
        color=FP_COLOR,
        fontsize=12,
        fontweight="bold",
    )


    ax.set_title(
        "Candidate 1 - High Confidence False Positive\n"
        f"Score={score:.4f} | Slice={slice_z}",
        fontsize=15,
    )


    ax.axis(
        "off"
    )


    legend = [
        Line2D(
            [0],
            [0],
            color=FP_COLOR,
            linewidth=3,
            linestyle="--",
            label="False Positive Prediction",
        )
    ]


    ax.legend(
        handles=legend,
        loc="upper left",
        framealpha=0.85,
    )


    output_path = (
        OUTPUT_DIR
        / f"{seriesuid}_fp_detection.png"
    )


    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


    return output_path


# ============================================================
# Visualization: FN
# ============================================================

def visualize_fn(record):

    seriesuid = record["seriesuid"]

    gt_box = record["gt_box"]

    gx, gy, gz, gw, gh, gd = gt_box


    # ========================================================
    # Load CT
    # ========================================================

    mhd_path = find_mhd(
        seriesuid
    )

    image = sitk.ReadImage(
        str(mhd_path)
    )

    volume = sitk.GetArrayFromImage(
        image
    ).astype(
        np.float32
    )

    spacing = image.GetSpacing()


    # ========================================================
    # GT World → Voxel
    # ========================================================

    gt_idx = world_to_voxel(
        image,
        gx,
        gy,
        gz,
    )

    gt_cx, gt_cy, gt_cz_float = gt_idx


    slice_z = int(
        round(
            gt_cz_float
        )
    )


    # ========================================================
    # GT size mm → voxel
    # ========================================================

    gt_w_vox = (
        gw
        / spacing[0]
    )

    gt_h_vox = (
        gh
        / spacing[1]
    )


    gt_xmin = (
        gt_cx
        - gt_w_vox / 2
    )

    gt_ymin = (
        gt_cy
        - gt_h_vox / 2
    )


    # ========================================================
    # Find Nearest Prediction
    # ========================================================

    predictions = pred_by_uid.get(
        seriesuid,
        []
    )


    nearest_pred = None
    nearest_distance = None


    for pred in predictions:

        px, py, pz = pred["box"][:3]

        distance = center_distance(
            (
                gx,
                gy,
                gz,
            ),
            (
                px,
                py,
                pz,
            ),
        )


        if (
            nearest_distance is None
            or distance < nearest_distance
        ):

            nearest_distance = distance
            nearest_pred = pred


    # ========================================================
    # Lung Window
    # ========================================================

    WINDOW_MIN = -1000
    WINDOW_MAX = 400


    slice_image = np.clip(
        volume[slice_z],
        WINDOW_MIN,
        WINDOW_MAX,
    )


    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )


    ax.imshow(
        slice_image,
        cmap="gray",
        vmin=WINDOW_MIN,
        vmax=WINDOW_MAX,
    )


    # ========================================================
    # GT Box
    # ========================================================

    gt_rect = Rectangle(
        (
            gt_xmin,
            gt_ymin,
        ),
        gt_w_vox,
        gt_h_vox,
        fill=False,
        edgecolor=GT_COLOR,
        linewidth=3,
        linestyle="-",
    )

    ax.add_patch(
        gt_rect
    )


    # GT center
    ax.scatter(
        [gt_cx],
        [gt_cy],
        marker="x",
        s=150,
        c=GT_COLOR,
        linewidths=3,
        zorder=10,
    )


    ax.text(
        gt_xmin,
        gt_ymin - 8,
        "Ground Truth",
        color=GT_COLOR,
        fontsize=12,
        fontweight="bold",
    )


    # ========================================================
    # Nearest Prediction
    # ========================================================

    nearest_score = None


    if nearest_pred is not None:

        px, py, pz, pw, ph, pd = nearest_pred[
            "box"
        ]

        nearest_score = nearest_pred[
            "score"
        ]


        pred_idx = world_to_voxel(
            image,
            px,
            py,
            pz,
        )


        pred_cx, pred_cy, pred_cz_float = (
            pred_idx
        )


        pred_w_vox = (
            pw
            / spacing[0]
        )

        pred_h_vox = (
            ph
            / spacing[1]
        )


        pred_xmin = (
            pred_cx
            - pred_w_vox / 2
        )

        pred_ymin = (
            pred_cy
            - pred_h_vox / 2
        )


        # ----------------------------------------------------
        # Prediction box
        # ----------------------------------------------------

        pred_rect = Rectangle(
            (
                pred_xmin,
                pred_ymin,
            ),
            pred_w_vox,
            pred_h_vox,
            fill=False,
            edgecolor=PRED_COLOR,
            linewidth=3,
            linestyle="--",
        )

        ax.add_patch(
            pred_rect
        )


        # Prediction center
        ax.scatter(
            [pred_cx],
            [pred_cy],
            marker="+",
            s=160,
            c=PRED_COLOR,
            linewidths=3,
            zorder=11,
        )


        ax.text(
            pred_xmin,
            pred_ymin + pred_h_vox + 14,
            f"Nearest Prediction {nearest_score:.4f}",
            color=PRED_COLOR,
            fontsize=12,
            fontweight="bold",
        )


    # ========================================================
    # Information
    # ========================================================

    diameter = (
        gw + gh + gd
    ) / 3.0


    radius = (
        diameter
        / 2.0
    )


    if nearest_distance is not None:

        title = (
            "Candidate 1 - False Negative\n"
            f"GT Diameter={diameter:.2f} mm | "
            f"GT Radius={radius:.2f} mm | "
            f"Nearest Dist={nearest_distance:.2f} mm | "
            f"Score={nearest_score:.4f}"
        )

    else:

        title = (
            "Candidate 1 - False Negative\n"
            f"GT Diameter={diameter:.2f} mm | "
            "No Prediction Candidate"
        )


    ax.set_title(
        title,
        fontsize=15,
    )


    ax.axis(
        "off"
    )


    # ========================================================
    # Legend
    # ========================================================

    legend_elements = [

        Line2D(
            [0],
            [0],
            color=GT_COLOR,
            linewidth=3,
            linestyle="-",
            label="Ground Truth",
        ),
    ]


    if nearest_pred is not None:

        legend_elements.append(

            Line2D(
                [0],
                [0],
                color=PRED_COLOR,
                linewidth=3,
                linestyle="--",
                label="Nearest Prediction",
            )
        )


    ax.legend(
        handles=legend_elements,
        loc="upper left",
        framealpha=0.85,
    )


    # ========================================================
    # Save
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / f"{seriesuid}_fn_detection.png"
    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close(fig)


    return output_path


# ============================================================
# Run
# ============================================================

fp_output = visualize_fp(
    selected_fp
)

fn_output = visualize_fn(
    selected_fn
)


# ============================================================
# Final
# ============================================================

print("=" * 80)
print("✅ Candidate 1 FP / FN visualization 완료")
print()

print("[False Positive]")
print(
    "SeriesUID:",
    selected_fp["seriesuid"],
)
print(
    "Score:",
    f'{selected_fp["score"]:.6f}',
)
print(
    "Saved:",
    fp_output,
)

print()

print("[False Negative]")
print(
    "SeriesUID:",
    selected_fn["seriesuid"],
)
print(
    "GT index:",
    selected_fn["gt_index"],
)

diameter = (
    selected_fn["gt_box"][3]
    + selected_fn["gt_box"][4]
    + selected_fn["gt_box"][5]
) / 3.0

print(
    "GT diameter:",
    f"{diameter:.3f} mm",
)

print(
    "Nearest prediction distance:",
    selected_fn["nearest_distance"],
)

print(
    "Nearest prediction score:",
    selected_fn["nearest_score"],
)

print(
    "Saved:",
    fn_output,
)

print("=" * 80)