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
# Visualization Colors
# ============================================================

GT_COLOR = "lime"
PRED_COLOR = "red"


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
            f"MHD 파일을 찾을 수 없습니다: {seriesuid}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"MHD 파일이 중복되었습니다: {seriesuid}"
        )

    return matches[0]


def get_uid_from_image(image_value):

    name = Path(
        str(image_value)
    ).name

    if name.endswith(".mhd"):
        return name[:-4]

    return Path(name).stem


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

    gt_by_uid[uid] = item.get(
        "box",
        []
    )


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

        record = {
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

        records.append(record)


# ============================================================
# Select Highest-score TP
# ============================================================

tp_records = [
    record
    for record in records
    if record["status"] == "TP"
]


if not tp_records:

    raise RuntimeError(
        "TP prediction이 없습니다."
    )


selected_pred = max(
    tp_records,
    key=lambda x: x["score"],
)


seriesuid = selected_pred["seriesuid"]

pred_box = selected_pred["box"]

pred_score = selected_pred["score"]


gt_boxes = gt_by_uid[
    seriesuid
]


# ============================================================
# Find Nearest GT
# ============================================================

px, py, pz, pw, ph, pd = pred_box


def distance_to_gt(box):

    gx, gy, gz = (
        float(box[0]),
        float(box[1]),
        float(box[2]),
    )

    return math.sqrt(
        (px - gx) ** 2
        + (py - gy) ** 2
        + (pz - gz) ** 2
    )


gt_box = min(
    gt_boxes,
    key=distance_to_gt,
)


gx, gy, gz, gw, gh, gd = map(
    float,
    gt_box,
)


center_distance_mm = math.sqrt(
    (px - gx) ** 2
    + (py - gy) ** 2
    + (pz - gz) ** 2
)


# ============================================================
# Load CT
# ============================================================

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


print("=" * 70)
print("Selected CT")
print("SeriesUID:", seriesuid)
print("MHD:", mhd_path)
print("Volume shape:", volume.shape)
print("Spacing:", spacing)
print("=" * 70)


# ============================================================
# World → Voxel
# ============================================================

gt_center_idx = (
    image.TransformPhysicalPointToContinuousIndex(
        (
            gx,
            gy,
            gz,
        )
    )
)


pred_center_idx = (
    image.TransformPhysicalPointToContinuousIndex(
        (
            px,
            py,
            pz,
        )
    )
)


gt_cx, gt_cy, gt_cz_float = (
    gt_center_idx
)


pred_cx, pred_cy, pred_cz_float = (
    pred_center_idx
)


# GT 중심 slice 사용
slice_z = int(
    round(
        gt_cz_float
    )
)


if not (
    0
    <= slice_z
    < volume.shape[0]
):

    raise RuntimeError(
        f"Slice index 범위 초과: {slice_z}"
    )


# ============================================================
# Size mm → voxel
# ============================================================

gt_w_vox = (
    gw
    / spacing[0]
)

gt_h_vox = (
    gh
    / spacing[1]
)


pred_w_vox = (
    pw
    / spacing[0]
)

pred_h_vox = (
    ph
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


pred_xmin = (
    pred_cx
    - pred_w_vox / 2
)

pred_ymin = (
    pred_cy
    - pred_h_vox / 2
)


# ============================================================
# Lung Window
# ============================================================

WINDOW_MIN = -1000
WINDOW_MAX = 400


slice_image = np.clip(
    volume[slice_z],
    WINDOW_MIN,
    WINDOW_MAX,
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 9)
)


ax.imshow(
    slice_image,
    cmap="gray",
    vmin=WINDOW_MIN,
    vmax=WINDOW_MAX,
)


# ============================================================
# GT Bounding Box
# Green solid line
# ============================================================

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


# ============================================================
# Prediction Bounding Box
# Red dashed line
# ============================================================

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


# ============================================================
# Centers
# ============================================================

# GT center
ax.scatter(
    [gt_cx],
    [gt_cy],
    marker="x",
    s=130,
    c=GT_COLOR,
    linewidths=3,
    zorder=10,
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


# ============================================================
# Labels beside boxes
# ============================================================

ax.text(
    gt_xmin,
    gt_ymin - 6,
    "GT",
    color=GT_COLOR,
    fontsize=12,
    fontweight="bold",
)


ax.text(
    pred_xmin,
    pred_ymin + pred_h_vox + 14,
    f"Prediction {pred_score:.4f}",
    color=PRED_COLOR,
    fontsize=12,
    fontweight="bold",
)


# ============================================================
# Title
# ============================================================

ax.set_title(
    "Candidate 1 - True Positive Detection\n"
    f"Score={pred_score:.4f} | "
    f"Center Distance={center_distance_mm:.2f} mm | "
    f"Slice={slice_z}",
    fontsize=15,
)


ax.axis(
    "off"
)


# ============================================================
# Legend
# ============================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        color=GT_COLOR,
        linewidth=3,
        linestyle="-",
        label="Ground Truth",
    ),

    Line2D(
        [0],
        [0],
        color=PRED_COLOR,
        linewidth=3,
        linestyle="--",
        label="Prediction",
    ),

    Line2D(
        [0],
        [0],
        marker="x",
        color=GT_COLOR,
        linestyle="None",
        markersize=9,
        markeredgewidth=2,
        label="GT Center",
    ),

    Line2D(
        [0],
        [0],
        marker="+",
        color=PRED_COLOR,
        linestyle="None",
        markersize=11,
        markeredgewidth=2,
        label="Prediction Center",
    ),
]


ax.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=10,
    framealpha=0.85,
)


# ============================================================
# Save
# ============================================================

OUTPUT_PATH = (
    OUTPUT_DIR
    / (
        f"{seriesuid}"
        "_tp_detection_colored.png"
    )
)


fig.tight_layout()


fig.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("✅ Candidate 1 TP detection visualization 완료")
print()
print("SeriesUID:")
print(seriesuid)
print()
print(f"Prediction score : {pred_score:.6f}")
print(f"Center distance  : {center_distance_mm:.3f} mm")
print(f"Slice            : {slice_z}")
print()
print("Color:")
print("  Green solid = Ground Truth")
print("  Red dashed  = Prediction")
print()
print("Saved:")
print(OUTPUT_PATH)
print("=" * 70)