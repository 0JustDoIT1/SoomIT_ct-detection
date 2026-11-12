#!/usr/bin/env python3

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from matplotlib.patches import Rectangle


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

CPMNET_ROOT = (
    PROJECT_ROOT
    / "external"
    / "CPMNetv2"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "luna16"
)

IMAGE_ROOT = (
    DATA_ROOT
    / "imagesTr"
)

TRAIN_CSV = (
    DATA_ROOT
    / "train.csv"
)

MINI_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "mini_overfit"
)

CHECKPOINT = (
    MINI_ROOT
    / "best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "visualization"
    / "mini_predictions"
)

RESULT_JSON = (
    MINI_ROOT
    / "mini_prediction_diagnostic.json"
)


# =============================================================================
# CPMNetv2 imports
# =============================================================================

sys.path.insert(
    0,
    str(CPMNET_ROOT),
)

from networks.ResNet_3D_CPM import (
    resnet18,
    Detection_Postprocess,
)

from dataload.split_combine import (
    SplitComb,
)

from utils.box_utils import (
    nms_3D,
)


# =============================================================================
# Config
# =============================================================================

NUM_CASES = 5

CROP_SIZE = [
    64,
    128,
    128,
]

OVERLAP_SIZE = [
    16,
    32,
    32,
]

SPACING = np.array(
    [
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)

HU_MIN = -1200.0
HU_MAX = 600.0

# mini-overfit에서는 높은 threshold부터 보되,
# 너무 높아 아무것도 안 나오는 경우를 피하려고 낮게 설정
SCORE_THRESHOLD = 0.20

POSTPROCESS_TOPK = 60
POSTPROCESS_NUM_TOPK = 20

NMS_THRESHOLD = 0.05
NMS_TOPK = 40

INFERENCE_BATCH_SIZE = 8

SLICE_OFFSETS = [
    -2,
    0,
    2,
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# Utility
# =============================================================================

def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def normalize_ct(image):
    image = image.astype(
        np.float32,
        copy=False,
    )

    image = np.clip(
        image,
        HU_MIN,
        HU_MAX,
    )

    image = (
        image - HU_MIN
    ) / (
        HU_MAX - HU_MIN
    )

    return image.astype(
        np.float32,
        copy=False,
    )


def strip_nii_suffix(name):
    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return Path(name).stem


# =============================================================================
# Load mini cases
# =============================================================================

def load_positive_cases():

    df = pd.read_csv(
        TRAIN_CSV
    )

    df = df[
        df["label"].astype(str)
        == "nodule"
    ].copy()

    grouped = {
        str(uid): group.copy()
        for uid, group
        in df.groupby("seriesuid")
    }

    ct_paths = sorted(
        IMAGE_ROOT.glob(
            "*.nii.gz"
        )
    )

    selected = []

    for ct_path in ct_paths:

        keys = [
            ct_path.name,
            strip_nii_suffix(
                ct_path.name
            ),
        ]

        matched = None

        for key in keys:
            if key in grouped:
                matched = grouped[key]
                break

        if matched is None:
            continue

        selected.append(
            {
                "path": ct_path,
                "annotations": matched,
            }
        )

        if len(selected) >= NUM_CASES:
            break

    if len(selected) != NUM_CASES:
        raise RuntimeError(
            f"Expected {NUM_CASES} positive cases, "
            f"found {len(selected)}"
        )

    return selected


# =============================================================================
# Model
# =============================================================================

def build_model():

    model = resnet18(
        n_channels=1,

        n_blocks=[
            2,
            3,
            3,
            3,
        ],

        n_filters=[
            64,
            96,
            128,
            160,
        ],

        stem_filters=32,

        norm_type="batchnorm",

        head_norm="batchnorm",

        act_type="ReLU",

        se=False,

        first_stride=(
            1,
            2,
            2,
        ),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "model_state_dict not found in checkpoint"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch"
        ),
    )

    print(
        "Checkpoint metrics:",
        checkpoint.get(
            "metrics"
        ),
    )

    return model


# =============================================================================
# Inference
# =============================================================================

def infer_volume(
    model,
    image_zyx,
):

    split_comber = SplitComb(
        crop_size=CROP_SIZE,
        overlap=OVERLAP_SIZE,
        pad_value=-1,
    )

    detection_postprocess = (
        Detection_Postprocess(
            topk=POSTPROCESS_TOPK,
            threshold=SCORE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            num_topk=POSTPROCESS_NUM_TOPK,
            crop_size=CROP_SIZE,
        )
    )

    normalized = normalize_ct(
        image_zyx
    )

    network_input = (
        normalized
        * 2.0
        - 1.0
    )

    split_images, nzhw = (
        split_comber.split(
            network_input
        )
    )

    data = torch.from_numpy(
        split_images
    ).float()

    output_list = []

    for start in range(
        0,
        data.shape[0],
        INFERENCE_BATCH_SIZE,
    ):

        end = min(
            start
            + INFERENCE_BATCH_SIZE,
            data.shape[0],
        )

        batch = data[
            start:end
        ].to(
            DEVICE
        )

        with torch.no_grad():
            raw = model(
                batch
            )

            processed = (
                detection_postprocess(
                    raw,
                    device=DEVICE,
                )
            )

        output_list.append(
            processed
            .detach()
            .cpu()
            .numpy()
        )

    if not output_list:
        return np.zeros(
            (0, 7),
            dtype=np.float32,
        )

    output = np.concatenate(
        output_list,
        axis=0,
    )

    output = split_comber.combine(
        output,
        nzhw=nzhw,
    )

    output = (
        torch.from_numpy(
            output
        )
        .view(
            -1,
            8,
        )
    )

    valid = (
        output[:, 0]
        != -1.0
    )

    output = output[
        valid
    ]

    if len(output) == 0:
        return np.zeros(
            (0, 7),
            dtype=np.float32,
        )

    # output:
    # [class_id, prob, z, y, x, d, h, w]
    #
    # nms_3D에는 class id 제거 후
    # [prob,z,y,x,d,h,w]
    keep = nms_3D(
        output[:, 1:],
        overlap=NMS_THRESHOLD,
        top_k=NMS_TOPK,
    )

    output = (
        output[
            keep
        ]
        .cpu()
        .numpy()
    )

    # class id 제외
    return output[:, 1:]


# =============================================================================
# Matching
# =============================================================================

def build_gt_zyx(
    annotations,
):

    result = []

    for _, row in (
        annotations.iterrows()
    ):

        result.append(
            {
                "center_zyx": np.array(
                    [
                        float(
                            row["coordZ"]
                        ),
                        float(
                            row["coordY"]
                        ),
                        float(
                            row["coordX"]
                        ),
                    ],
                    dtype=np.float32,
                ),

                "size_dhw": np.array(
                    [
                        float(
                            row["d"]
                        ),
                        float(
                            row["h"]
                        ),
                        float(
                            row["w"]
                        ),
                    ],
                    dtype=np.float32,
                ),
            }
        )

    return result


def match_predictions(
    gts,
    predictions,
):

    results = []

    for gt_index, gt in enumerate(
        gts
    ):

        gt_center = (
            gt["center_zyx"]
        )

        diameter = float(
            np.max(
                gt["size_dhw"]
            )
        )

        radius = (
            diameter / 2.0
        )

        if len(predictions) == 0:

            results.append(
                {
                    "gt_index":
                        gt_index,

                    "nearest_distance":
                        None,

                    "nearest_score":
                        None,

                    "radius":
                        radius,

                    "matched":
                        False,
                }
            )

            continue

        pred_centers = (
            predictions[
                :,
                1:4,
            ]
        )

        distances = (
            np.linalg.norm(
                pred_centers
                - gt_center[
                    None,
                    :
                ],
                axis=1,
            )
        )

        nearest_index = int(
            np.argmin(
                distances
            )
        )

        nearest_distance = float(
            distances[
                nearest_index
            ]
        )

        nearest_score = float(
            predictions[
                nearest_index,
                0,
            ]
        )

        results.append(
            {
                "gt_index":
                    gt_index,

                "nearest_distance":
                    nearest_distance,

                "nearest_score":
                    nearest_score,

                "radius":
                    radius,

                "matched":
                    bool(
                        nearest_distance
                        <= radius
                    ),
            }
        )

    return results


# =============================================================================
# Visualization
# =============================================================================

def box_intersects_slice(
    center_z,
    depth,
    z_index,
):

    return (
        center_z - depth / 2.0
        <= z_index
        <= center_z + depth / 2.0
    )


def draw_gt(
    ax,
    gts,
    z_index,
):

    for gt in gts:

        z, y, x = (
            gt["center_zyx"]
        )

        d, h, w = (
            gt["size_dhw"]
        )

        if not box_intersects_slice(
            z,
            d,
            z_index,
        ):
            continue

        rect = Rectangle(
            (
                x - w / 2.0,
                y - h / 2.0,
            ),
            w,
            h,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(
            rect
        )

        ax.plot(
            x,
            y,
            marker="+",
            markersize=10,
            markeredgewidth=2,
        )

        ax.text(
            x,
            y - h / 2.0 - 5,
            "GT",
            fontsize=8,
        )


def draw_predictions(
    ax,
    predictions,
    z_index,
):

    for prediction in predictions:

        score = float(
            prediction[0]
        )

        z = float(
            prediction[1]
        )

        y = float(
            prediction[2]
        )

        x = float(
            prediction[3]
        )

        d = float(
            prediction[4]
        )

        h = float(
            prediction[5]
        )

        w = float(
            prediction[6]
        )

        if not box_intersects_slice(
            z,
            d,
            z_index,
        ):
            continue

        rect = Rectangle(
            (
                x - w / 2.0,
                y - h / 2.0,
            ),
            w,
            h,
            fill=False,
            linewidth=1.5,
            linestyle="--",
        )

        ax.add_patch(
            rect
        )

        ax.plot(
            x,
            y,
            marker="x",
            markersize=8,
            markeredgewidth=1.5,
        )

        ax.text(
            x,
            y + h / 2.0 + 8,
            f"P {score:.2f}",
            fontsize=7,
        )


def visualize_case(
    case_number,
    ct_path,
    image,
    gts,
    predictions,
):

    # GT 첫 결절을 기준으로 QA
    center_z = int(
        round(
            gts[
                0
            ][
                "center_zyx"
            ][0]
        )
    )

    z_size = (
        image.shape[
            0
        ]
    )

    slices = [
        int(
            np.clip(
                center_z
                + offset,
                0,
                z_size - 1,
            )
        )
        for offset
        in SLICE_OFFSETS
    ]

    display = normalize_ct(
        image
    )

    fig, axes = plt.subplots(
        1,
        len(slices),
        figsize=(
            15,
            5,
        ),
    )

    if len(slices) == 1:
        axes = [
            axes
        ]

    for ax, z_index in zip(
        axes,
        slices,
    ):

        ax.imshow(
            display[
                z_index
            ],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        draw_gt(
            ax,
            gts,
            z_index,
        )

        draw_predictions(
            ax,
            predictions,
            z_index,
        )

        ax.set_title(
            f"Axial z={z_index}"
        )

        ax.axis(
            "off"
        )

    fig.suptitle(
        f"Mini overfit case {case_number:02d} | "
        f"GT={len(gts)} | "
        f"Pred={len(predictions)}",
        fontsize=11,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / (
            f"{case_number:02d}_"
            f"{strip_nii_suffix(ct_path.name)}"
            f"_gt_pred.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


# =============================================================================
# Main
# =============================================================================

def main():

    header(
        "CPMNetv2 Mini Overfit Prediction Visualization"
    )

    print(
        "Device     :",
        DEVICE,
    )

    print(
        "Checkpoint :",
        CHECKPOINT,
    )

    print(
        "Threshold  :",
        SCORE_THRESHOLD,
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            CHECKPOINT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = (
        load_positive_cases()
    )

    model = (
        build_model()
    )

    all_results = []

    total_gt = 0
    total_matched = 0

    header(
        "Inference"
    )

    for case_number, case in enumerate(
        cases,
        start=1,
    ):

        ct_path = (
            case[
                "path"
            ]
        )

        image_itk = (
            sitk.ReadImage(
                str(
                    ct_path
                )
            )
        )

        spacing = np.asarray(
            image_itk
            .GetSpacing()[
                ::-1
            ],
            dtype=np.float32,
        )

        if not np.allclose(
            spacing,
            SPACING,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"Unexpected spacing: {spacing}"
            )

        image = (
            sitk.GetArrayFromImage(
                image_itk
            )
            .astype(
                np.float32
            )
        )

        gts = build_gt_zyx(
            case[
                "annotations"
            ]
        )

        predictions = infer_volume(
            model,
            image,
        )

        matches = (
            match_predictions(
                gts,
                predictions,
            )
        )

        matched = sum(
            1
            for item
            in matches
            if item[
                "matched"
            ]
        )

        total_gt += len(
            gts
        )

        total_matched += (
            matched
        )

        output_path = (
            visualize_case(
                case_number,
                ct_path,
                image,
                gts,
                predictions,
            )
        )

        nearest_values = [
            item[
                "nearest_distance"
            ]
            for item
            in matches
            if item[
                "nearest_distance"
            ]
            is not None
        ]

        nearest_median = (
            float(
                np.median(
                    nearest_values
                )
            )
            if nearest_values
            else None
        )

        max_score = (
            float(
                predictions[
                    :,
                    0,
                ].max()
            )
            if len(predictions)
            else None
        )

        print(
            f"[{case_number:02d}] "
            f"GT={len(gts):2d} | "
            f"Pred={len(predictions):2d} | "
            f"Matched={matched:2d} | "
            f"nearest_med="
            f"{nearest_median if nearest_median is not None else 'NA'} | "
            f"max_score="
            f"{max_score if max_score is not None else 'NA'}"
        )

        all_results.append(
            {
                "image":
                    ct_path.name,

                "gt_count":
                    len(gts),

                "prediction_count":
                    int(
                        len(
                            predictions
                        )
                    ),

                "matched_gt":
                    matched,

                "nearest_distance_median":
                    nearest_median,

                "max_score":
                    max_score,

                "matches":
                    matches,

                "predictions":
                    predictions.tolist(),

                "visualization":
                    str(
                        output_path
                    ),
            }
        )

    header(
        "Summary"
    )

    recall = (
        total_matched
        / total_gt
        if total_gt > 0
        else 0.0
    )

    print(
        f"Cases       : {len(cases)}"
    )

    print(
        f"GT nodules  : {total_gt}"
    )

    print(
        f"Matched GT  : {total_matched}"
    )

    print(
        f"Mini Recall : {recall:.4f}"
    )

    print()
    print(
        "Saved visualizations:"
    )

    print(
        OUTPUT_ROOT
    )

    result = {
        "checkpoint":
            str(
                CHECKPOINT
            ),

        "score_threshold":
            SCORE_THRESHOLD,

        "cases":
            len(
                cases
            ),

        "total_gt":
            total_gt,

        "matched_gt":
            total_matched,

        "mini_recall":
            recall,

        "results":
            all_results,
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "Diagnostic JSON:"
    )

    print(
        RESULT_JSON
    )

    print()

    if recall >= 0.8:

        print(
            "PASS: Mini-overfit case의 대부분 GT를 "
            "실제로 검출합니다."
        )

    elif recall > 0:

        print(
            "PARTIAL: 학습 신호는 있으나 "
            "검출 위치/threshold 추가 확인이 필요합니다."
        )

    else:

        print(
            "FAIL: Loss는 감소했지만 "
            "mini case의 GT를 검출하지 못했습니다."
        )


if __name__ == "__main__":
    main()