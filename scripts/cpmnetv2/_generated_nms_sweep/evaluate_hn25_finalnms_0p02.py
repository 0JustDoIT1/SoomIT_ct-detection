#!/usr/bin/env python3
"""
CPMNetv2 Full Validation Inference + Final Evaluation

Uses:
- best_val_loss.pt
- full-volume SplitComb inference
- CPM / FROC
- center-match AP / mAP
- 3D IoU AP@0.10 / 0.30 / 0.50
- Precision / Recall / F1
- size-wise recall
- localization / score statistics
- CSV / JSON / curves / QA examples

Conventions
-----------
Input volume      : [Z, Y, X]
GT center         : [z, y, x]
GT size           : [d, h, w]
Prediction format : [score, z, y, x, d, h, w]
Spacing           : 1 x 1 x 1 mm
"""

import gc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


# =============================================================================
# PATHS
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

VAL_ROOT = (
    DATA_ROOT
    / "imagesVa"
)

VAL_CSV = (
    DATA_ROOT
    / "val.csv"
)

TRAIN_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_training_ratio025"
)

CHECKPOINT = (
    TRAIN_OUTPUT_ROOT
    / "best_val_loss.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_evaluation_ratio025_finalnms_0p02"
)

PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "predictions.csv"
)

GT_CSV = (
    OUTPUT_DIR
    / "ground_truth.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "metrics_summary.json"
)

FROC_CSV = (
    OUTPUT_DIR
    / "froc_curve.csv"
)

PR_CSV = (
    OUTPUT_DIR
    / "center_pr_curve.csv"
)

SIZE_RECALL_CSV = (
    OUTPUT_DIR
    / "size_recall.csv"
)

FROC_PNG = (
    OUTPUT_DIR
    / "froc_curve.png"
)

PR_PNG = (
    OUTPUT_DIR
    / "center_pr_curve.png"
)

QA_DIR = (
    OUTPUT_DIR
    / "qa_examples"
)


# =============================================================================
# CPMNET IMPORT
# =============================================================================

sys.path.insert(
    0,
    str(CPMNET_ROOT),
)

from networks.ResNet_3D_CPM import (  # noqa: E402
    resnet18,
    Detection_Postprocess,
)

from dataload.split_combine import (  # noqa: E402
    SplitComb,
)

from utils.box_utils import (  # noqa: E402
    nms_3D,
)


# =============================================================================
# CONFIG
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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

SPACING_ZYX = np.asarray(
    [
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)

HU_MIN = -1200.0
HU_MAX = 600.0

INFER_BATCH_SIZE = 4

# FROC 계산용으로 낮은 점수 후보도 남김
CROP_SCORE_THRESHOLD = 0.01

CROP_POST_TOPK = 120
CROP_POST_NUM_TOPK = 60
CROP_NMS_THRESHOLD = 0.05

FINAL_NMS_THRESHOLD = 0.02
FINAL_TOPK_PER_SCAN = 300

FIXED_SCORE_THRESHOLD = 0.50

FROC_FP_PER_SCAN = [
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
]

IOU_AP_THRESHOLDS = [
    0.10,
    0.30,
    0.50,
]

SAVE_QA_EXAMPLES = True
QA_EXAMPLES_PER_TYPE = 5


# =============================================================================
# UTILITY
# =============================================================================

def header(title):

    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def strip_nii_suffix(name):

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return Path(name).stem


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
        image
        - HU_MIN
    ) / (
        HU_MAX
        - HU_MIN
    )

    image = (
        image
        * 2.0
        - 1.0
    )

    return image.astype(
        np.float32,
        copy=False,
    )


def percentile_stats(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:

        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p25": None,
            "p75": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


# =============================================================================
# MODEL
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

    return model.to(
        DEVICE
    )


def load_checkpoint(
    model,
):

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint 없음:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        epoch = checkpoint.get(
            "epoch"
        )

        best_val_loss = checkpoint.get(
            "best_val_loss"
        )

    else:

        state_dict = checkpoint
        epoch = None
        best_val_loss = None

    if any(
        key.startswith(
            "module."
        )
        for key in state_dict.keys()
    ):

        state_dict = {
            key.removeprefix(
                "module."
            ):
                value

            for key, value
            in state_dict.items()
        }

    result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "[PASS] checkpoint loaded"
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Epoch:",
        epoch,
    )

    print(
        "Best val loss:",
        best_val_loss,
    )

    print(
        "load_state_dict:",
        result,
    )

    return checkpoint


# =============================================================================
# DATA
# =============================================================================

def load_val_annotations():

    df = pd.read_csv(
        VAL_CSV
    )

    if "label" in df.columns:

        df = df[
            df["label"]
            .astype(str)
            == "nodule"
        ].copy()

    required = [
        "seriesuid",
        "coordX",
        "coordY",
        "coordZ",
        "w",
        "h",
        "d",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"val.csv required columns 없음: {missing}"
        )

    df[
        "seriesuid"
    ] = (
        df[
            "seriesuid"
        ]
        .astype(str)
        .map(
            strip_nii_suffix
        )
    )

    return df


def list_val_cases():

    image_paths = sorted(
        VAL_ROOT.glob(
            "*.nii.gz"
        )
    )

    if not image_paths:

        raise RuntimeError(
            f"No validation NIfTI:\n{VAL_ROOT}"
        )

    return [
        {
            "case_id":
                strip_nii_suffix(
                    path.name
                ),

            "path":
                path,
        }

        for path
        in image_paths
    ]


def build_gt_table(
    cases,
    annotations,
):

    rows = []

    gt_id = 0

    grouped = {
        str(uid):
            group.copy()

        for uid, group
        in annotations.groupby(
            "seriesuid"
        )
    }

    for case in cases:

        case_id = case[
            "case_id"
        ]

        group = grouped.get(
            case_id
        )

        if group is None:
            continue

        for _, row in group.iterrows():

            z = float(
                row[
                    "coordZ"
                ]
            )

            y = float(
                row[
                    "coordY"
                ]
            )

            x = float(
                row[
                    "coordX"
                ]
            )

            d = float(
                row[
                    "d"
                ]
            )

            h = float(
                row[
                    "h"
                ]
            )

            w = float(
                row[
                    "w"
                ]
            )

            diameter_mm = float(
                max(
                    d * h * w,
                    0.0,
                )
                ** (
                    1.0 / 3.0
                )
            )

            if diameter_mm < 6.0:

                size_group = (
                    "<6 mm"
                )

            elif diameter_mm < 10.0:

                size_group = (
                    "6-10 mm"
                )

            else:

                size_group = (
                    ">=10 mm"
                )

            rows.append(
                {
                    "gt_id":
                        gt_id,

                    "case_id":
                        case_id,

                    "z":
                        z,

                    "y":
                        y,

                    "x":
                        x,

                    "d":
                        d,

                    "h":
                        h,

                    "w":
                        w,

                    "diameter_mm":
                        diameter_mm,

                    "size_group":
                        size_group,
                }
            )

            gt_id += 1

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FULL-VOLUME INFERENCE
# =============================================================================

def build_postprocessor():

    return Detection_Postprocess(
        topk=CROP_POST_TOPK,
        threshold=CROP_SCORE_THRESHOLD,
        nms_threshold=CROP_NMS_THRESHOLD,
        num_topk=CROP_POST_NUM_TOPK,
        crop_size=CROP_SIZE,
    )


def infer_one_case(
    model,
    postprocessor,
    split_comber,
    case,
):

    case_id = case[
        "case_id"
    ]

    image_path = case[
        "path"
    ]

    image_itk = sitk.ReadImage(
        str(
            image_path
        )
    )

    spacing_zyx = np.asarray(
        image_itk
        .GetSpacing()[
            ::-1
        ],
        dtype=np.float32,
    )

    if not np.allclose(
        spacing_zyx,
        SPACING_ZYX,
        atol=1e-4,
    ):

        raise RuntimeError(
            f"Spacing mismatch: {case_id} "
            f"{spacing_zyx.tolist()}"
        )

    image = (
        sitk
        .GetArrayFromImage(
            image_itk
        )
        .astype(
            np.float32
        )
    )

    image_shape = tuple(
        int(x)
        for x in image.shape
    )

    image = normalize_ct(
        image
    )

    split_images, nzhw = (
        split_comber.split(
            image
        )
    )

    data = torch.from_numpy(
        split_images
    ).float()

    output_list = []

    with torch.no_grad():

        for start in range(
            0,
            data.shape[0],
            INFER_BATCH_SIZE,
        ):

            end = min(
                start
                + INFER_BATCH_SIZE,
                data.shape[0],
            )

            batch = (
                data[
                    start:end
                ]
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            raw_output = model(
                batch
            )

            detections = postprocessor(
                raw_output,
                device=DEVICE,
            )

            output_list.append(
                detections
                .detach()
                .cpu()
                .numpy()
            )

    if not output_list:

        return [], image_shape

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
        .reshape(
            -1,
            8,
        )
        .float()
    )

    valid = (
        output[
            :,
            0
        ]
        != -1.0
    )

    output = output[
        valid
    ]

    if len(output) > 0:

        keep = nms_3D(
            output[
                :,
                1:
            ],
            overlap=FINAL_NMS_THRESHOLD,
            top_k=FINAL_TOPK_PER_SCAN,
        )

        output = output[
            keep
        ]

    output = (
        output
        .cpu()
        .numpy()
    )

    rows = []

    for detection in output:

        score = float(
            detection[1]
        )

        z = float(
            detection[2]
        )

        y = float(
            detection[3]
        )

        x = float(
            detection[4]
        )

        d = float(
            detection[5]
        )

        h = float(
            detection[6]
        )

        w = float(
            detection[7]
        )

        if not np.all(
            np.isfinite(
                [
                    score,
                    z,
                    y,
                    x,
                    d,
                    h,
                    w,
                ]
            )
        ):
            continue

        if (
            d <= 0
            or h <= 0
            or w <= 0
        ):
            continue

        rows.append(
            {
                "case_id":
                    case_id,

                "score":
                    score,

                "z":
                    z,

                "y":
                    y,

                "x":
                    x,

                "d":
                    d,

                "h":
                    h,

                "w":
                    w,
            }
        )

    rows = sorted(
        rows,
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    return rows, image_shape


def run_full_inference(
    model,
    cases,
):

    header(
        "Full-volume Validation Inference"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    postprocessor = (
        build_postprocessor()
    )

    split_comber = SplitComb(
        crop_size=CROP_SIZE,
        overlap=OVERLAP_SIZE,
        pad_value=-1,
    )

    model.eval()

    all_predictions = []

    scan_rows = []

    start_time = time.time()

    for index, case in enumerate(
        cases,
        start=1,
    ):

        case_start = time.time()

        predictions, image_shape = (
            infer_one_case(
                model=model,
                postprocessor=postprocessor,
                split_comber=split_comber,
                case=case,
            )
        )

        all_predictions.extend(
            predictions
        )

        scan_rows.append(
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "shape_z":
                    image_shape[0],

                "shape_y":
                    image_shape[1],

                "shape_x":
                    image_shape[2],

                "num_predictions":
                    len(
                        predictions
                    ),

                "seconds":
                    time.time()
                    - case_start,
            }
        )

        elapsed = (
            time.time()
            - start_time
        )

        avg = (
            elapsed
            / index
        )

        eta = (
            len(cases)
            - index
        ) * avg

        print(
            f"[{index:03d}/{len(cases)}] "
            f"{case['case_id']} | "
            f"pred={len(predictions):3d} | "
            f"shape={image_shape} | "
            f"{time.time()-case_start:.1f}s | "
            f"ETA={eta/60:.1f} min"
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_df = pd.DataFrame(
        all_predictions,
        columns=[
            "case_id",
            "score",
            "z",
            "y",
            "x",
            "d",
            "h",
            "w",
        ],
    )

    scan_df = pd.DataFrame(
        scan_rows
    )

    pred_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    scan_df.to_csv(
        OUTPUT_DIR
        / "scan_inference_summary.csv",
        index=False,
    )

    print()
    print(
        "[PASS] predictions saved:",
        PREDICTIONS_CSV,
    )

    print(
        "Total predictions:",
        len(
            pred_df
        ),
    )

    print(
        "Predictions/scan:",
        len(pred_df)
        / max(
            len(cases),
            1,
        ),
    )

    return pred_df


# =============================================================================
# GEOMETRY
# =============================================================================

def center_distance_mm(
    pred,
    gt,
):

    diff = np.asarray(
        [
            float(pred["z"])
            - float(gt["z"]),

            float(pred["y"])
            - float(gt["y"]),

            float(pred["x"])
            - float(gt["x"]),
        ],
        dtype=float,
    )

    return float(
        np.sqrt(
            np.sum(
                (
                    diff
                    * SPACING_ZYX
                )
                ** 2
            )
        )
    )


def center_match(
    pred,
    gt,
):

    dz = (
        float(pred["z"])
        - float(gt["z"])
    )

    dy = (
        float(pred["y"])
        - float(gt["y"])
    )

    dx = (
        float(pred["x"])
        - float(gt["x"])
    )

    rz = max(
        float(gt["d"])
        / 2.0,
        1e-6,
    )

    ry = max(
        float(gt["h"])
        / 2.0,
        1e-6,
    )

    rx = max(
        float(gt["w"])
        / 2.0,
        1e-6,
    )

    value = (
        (dz / rz) ** 2
        + (dy / ry) ** 2
        + (dx / rx) ** 2
    )

    return bool(
        value <= 1.0
    )


def box_corners(row):

    z = float(
        row["z"]
    )

    y = float(
        row["y"]
    )

    x = float(
        row["x"]
    )

    d = max(
        float(
            row["d"]
        ),
        0.0,
    )

    h = max(
        float(
            row["h"]
        ),
        0.0,
    )

    w = max(
        float(
            row["w"]
        ),
        0.0,
    )

    return np.asarray(
        [
            z - d / 2.0,
            y - h / 2.0,
            x - w / 2.0,
            z + d / 2.0,
            y + h / 2.0,
            x + w / 2.0,
        ],
        dtype=float,
    )


def iou_3d(
    pred,
    gt,
):

    a = box_corners(
        pred
    )

    b = box_corners(
        gt
    )

    inter_min = np.maximum(
        a[:3],
        b[:3],
    )

    inter_max = np.minimum(
        a[3:],
        b[3:],
    )

    inter_shape = np.maximum(
        inter_max
        - inter_min,
        0.0,
    )

    inter = float(
        np.prod(
            inter_shape
        )
    )

    vol_a = float(
        np.prod(
            np.maximum(
                a[3:]
                - a[:3],
                0.0,
            )
        )
    )

    vol_b = float(
        np.prod(
            np.maximum(
                b[3:]
                - b[:3],
                0.0,
            )
        )
    )

    union = (
        vol_a
        + vol_b
        - inter
    )

    if union <= 0:
        return 0.0

    return float(
        inter
        / union
    )


# =============================================================================
# MATCHING
# =============================================================================

def greedy_detection_labels(
    pred_df,
    gt_df,
    mode="center",
    iou_threshold=None,
):

    predictions = (
        pred_df
        .sort_values(
            "score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    gt_groups = {
        case_id:
            group.reset_index(
                drop=True
            )

        for case_id, group
        in gt_df.groupby(
            "case_id"
        )
    }

    matched = {
        case_id:
            set()

        for case_id
        in gt_groups.keys()
    }

    labels = []
    matched_gt_ids = []
    distances = []
    matched_ious = []

    for _, pred in predictions.iterrows():

        case_id = str(
            pred[
                "case_id"
            ]
        )

        gt_group = gt_groups.get(
            case_id
        )

        best_local_idx = None
        best_metric = None
        best_distance = None
        best_iou = None

        if gt_group is not None:

            already = matched.setdefault(
                case_id,
                set(),
            )

            for local_idx, gt in gt_group.iterrows():

                if local_idx in already:
                    continue

                if mode == "center":

                    if not center_match(
                        pred,
                        gt,
                    ):
                        continue

                    distance = (
                        center_distance_mm(
                            pred,
                            gt,
                        )
                    )

                    metric = (
                        -distance
                    )

                    current_iou = (
                        iou_3d(
                            pred,
                            gt,
                        )
                    )

                elif mode == "iou":

                    current_iou = (
                        iou_3d(
                            pred,
                            gt,
                        )
                    )

                    if (
                        current_iou
                        < float(
                            iou_threshold
                        )
                    ):
                        continue

                    distance = (
                        center_distance_mm(
                            pred,
                            gt,
                        )
                    )

                    metric = (
                        current_iou
                    )

                else:

                    raise ValueError(
                        mode
                    )

                if (
                    best_metric is None
                    or metric > best_metric
                ):

                    best_metric = metric
                    best_local_idx = (
                        local_idx
                    )

                    best_distance = (
                        distance
                    )

                    best_iou = (
                        current_iou
                    )

        if best_local_idx is not None:

            matched[
                case_id
            ].add(
                best_local_idx
            )

            gt_row = gt_group.iloc[
                best_local_idx
            ]

            labels.append(
                1
            )

            matched_gt_ids.append(
                int(
                    gt_row[
                        "gt_id"
                    ]
                )
            )

            distances.append(
                float(
                    best_distance
                )
            )

            matched_ious.append(
                float(
                    best_iou
                )
            )

        else:

            labels.append(
                0
            )

            matched_gt_ids.append(
                None
            )

            distances.append(
                None
            )

            matched_ious.append(
                None
            )

    result = predictions.copy()

    result[
        "is_tp"
    ] = labels

    result[
        "matched_gt_id"
    ] = matched_gt_ids

    result[
        "center_distance_mm"
    ] = distances

    result[
        "matched_iou"
    ] = matched_ious

    return result


# =============================================================================
# PR / AP
# =============================================================================

def precision_recall_from_ranked(
    labeled_predictions,
    total_gt,
):

    if len(
        labeled_predictions
    ) == 0:

        return (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
        )

    tp = (
        labeled_predictions[
            "is_tp"
        ]
        .to_numpy(
            dtype=float
        )
    )

    fp = (
        1.0
        - tp
    )

    cum_tp = np.cumsum(
        tp
    )

    cum_fp = np.cumsum(
        fp
    )

    precision = (
        cum_tp
        / np.maximum(
            cum_tp
            + cum_fp,
            1e-12,
        )
    )

    recall = (
        cum_tp
        / max(
            total_gt,
            1,
        )
    )

    scores = (
        labeled_predictions[
            "score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return (
        precision,
        recall,
        scores,
    )


def interpolated_ap(
    precision,
    recall,
):

    if (
        len(precision) == 0
        or len(recall) == 0
    ):
        return 0.0

    mrec = np.concatenate(
        [
            [0.0],
            recall,
            [1.0],
        ]
    )

    mpre = np.concatenate(
        [
            [0.0],
            precision,
            [0.0],
        ]
    )

    for i in range(
        len(mpre) - 2,
        -1,
        -1,
    ):

        mpre[i] = max(
            mpre[i],
            mpre[i + 1],
        )

    idx = np.where(
        mrec[1:]
        != mrec[:-1]
    )[0]

    return float(
        np.sum(
            (
                mrec[
                    idx + 1
                ]
                - mrec[
                    idx
                ]
            )
            * mpre[
                idx + 1
            ]
        )
    )


def metrics_at_threshold(
    labeled_predictions,
    total_gt,
    threshold,
):

    selected = (
        labeled_predictions[
            labeled_predictions[
                "score"
            ]
            >= threshold
        ]
    )

    tp = int(
        selected[
            "is_tp"
        ].sum()
    )

    fp = int(
        len(selected)
        - tp
    )

    fn = int(
        max(
            total_gt
            - tp,
            0,
        )
    )

    precision = (
        tp
        / (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp
        / total_gt
        if total_gt > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),
    }


def find_best_f1_threshold(
    labeled_predictions,
    total_gt,
):

    if len(
        labeled_predictions
    ) == 0:

        return metrics_at_threshold(
            labeled_predictions,
            total_gt,
            1.0,
        )

    thresholds = np.unique(
        labeled_predictions[
            "score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    best = None

    for threshold in thresholds:

        metrics = (
            metrics_at_threshold(
                labeled_predictions,
                total_gt,
                threshold,
            )
        )

        if (
            best is None
            or metrics["f1"]
            > best["f1"]
        ):

            best = metrics

    return best


# =============================================================================
# FROC / CPM
# =============================================================================

def build_froc_curve(
    labeled_predictions,
    total_gt,
    num_scans,
):

    if len(
        labeled_predictions
    ) == 0:

        return pd.DataFrame(
            [
                {
                    "threshold":
                        1.0,

                    "fp_per_scan":
                        0.0,

                    "sensitivity":
                        0.0,

                    "tp":
                        0,

                    "fp":
                        0,
                }
            ]
        )

    scores = (
        labeled_predictions[
            "score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    is_tp = (
        labeled_predictions[
            "is_tp"
        ]
        .to_numpy(
            dtype=int
        )
    )

    cum_tp = np.cumsum(
        is_tp
    )

    cum_fp = np.cumsum(
        1
        - is_tp
    )

    rows = [
        {
            "threshold":
                float(
                    np.nextafter(
                        scores.max(),
                        np.inf,
                    )
                ),

            "fp_per_scan":
                0.0,

            "sensitivity":
                0.0,

            "tp":
                0,

            "fp":
                0,
        }
    ]

    for i in range(
        len(scores)
    ):

        if (
            i + 1
            < len(scores)
            and scores[
                i + 1
            ]
            == scores[i]
        ):
            continue

        rows.append(
            {
                "threshold":
                    float(
                        scores[i]
                    ),

                "fp_per_scan":
                    float(
                        cum_fp[i]
                        / max(
                            num_scans,
                            1,
                        )
                    ),

                "sensitivity":
                    float(
                        cum_tp[i]
                        / max(
                            total_gt,
                            1,
                        )
                    ),

                "tp":
                    int(
                        cum_tp[i]
                    ),

                "fp":
                    int(
                        cum_fp[i]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def froc_operating_points(
    froc_df,
):

    rows = []

    for target in (
        FROC_FP_PER_SCAN
    ):

        eligible = (
            froc_df[
                froc_df[
                    "fp_per_scan"
                ]
                <= target
            ]
        )

        if len(
            eligible
        ) == 0:

            best = froc_df.iloc[
                0
            ]

        else:

            max_sens = (
                eligible[
                    "sensitivity"
                ].max()
            )

            candidates = (
                eligible[
                    eligible[
                        "sensitivity"
                    ]
                    == max_sens
                ]
            )

            best = (
                candidates
                .sort_values(
                    "threshold",
                    ascending=False,
                )
                .iloc[
                    0
                ]
            )

        rows.append(
            {
                "target_fp_per_scan":
                    float(
                        target
                    ),

                "sensitivity":
                    float(
                        best[
                            "sensitivity"
                        ]
                    ),

                "threshold":
                    float(
                        best[
                            "threshold"
                        ]
                    ),

                "actual_fp_per_scan":
                    float(
                        best[
                            "fp_per_scan"
                        ]
                    ),

                "tp":
                    int(
                        best[
                            "tp"
                        ]
                    ),

                "fp":
                    int(
                        best[
                            "fp"
                        ]
                    ),
            }
        )

    return rows


# =============================================================================
# SIZE RECALL
# =============================================================================

def matched_gt_ids_at_threshold(
    labeled_predictions,
    threshold,
):

    selected = (
        labeled_predictions[
            (
                labeled_predictions[
                    "score"
                ]
                >= threshold
            )
            & (
                labeled_predictions[
                    "is_tp"
                ]
                == 1
            )
        ]
    )

    return set(
        int(x)
        for x in selected[
            "matched_gt_id"
        ].dropna()
    )


def calculate_size_recall(
    gt_df,
    labeled_predictions,
    thresholds,
):

    group_order = [
        "<6 mm",
        "6-10 mm",
        ">=10 mm",
    ]

    rows = []

    for threshold_name, threshold in thresholds.items():

        matched_ids = (
            matched_gt_ids_at_threshold(
                labeled_predictions,
                threshold,
            )
        )

        for group_name in (
            group_order
        ):

            group = (
                gt_df[
                    gt_df[
                        "size_group"
                    ]
                    == group_name
                ]
            )

            total = len(
                group
            )

            detected = int(
                group[
                    "gt_id"
                ]
                .isin(
                    matched_ids
                )
                .sum()
            )

            recall = (
                detected
                / total
                if total > 0
                else 0.0
            )

            rows.append(
                {
                    "operating_point":
                        threshold_name,

                    "threshold":
                        float(
                            threshold
                        ),

                    "size_group":
                        group_name,

                    "gt_nodules":
                        int(
                            total
                        ),

                    "detected":
                        detected,

                    "recall":
                        float(
                            recall
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PLOTS
# =============================================================================

def save_froc_plot(
    froc_df,
    operating_points,
):

    fig = plt.figure(
        figsize=(
            8,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    ax.plot(
        froc_df[
            "fp_per_scan"
        ],
        froc_df[
            "sensitivity"
        ],
    )

    points = pd.DataFrame(
        operating_points
    )

    ax.scatter(
        points[
            "target_fp_per_scan"
        ],
        points[
            "sensitivity"
        ],
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        0.1,
        10.0,
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        "False Positives / Scan"
    )

    ax.set_ylabel(
        "Sensitivity"
    )

    ax.set_title(
        "CPMNetv2 Validation FROC"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        FROC_PNG,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_pr_plot(
    precision,
    recall,
):

    fig = plt.figure(
        figsize=(
            8,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    ax.plot(
        recall,
        precision,
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "CPMNetv2 Validation PR Curve"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        PR_PNG,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# FINAL EVALUATION
# =============================================================================

def evaluate(
    pred_df,
    gt_df,
    num_scans,
):

    header(
        "Center-match Evaluation"
    )

    total_gt = len(
        gt_df
    )

    center_labeled = (
        greedy_detection_labels(
            pred_df=pred_df,
            gt_df=gt_df,
            mode="center",
        )
    )

    precision, recall, scores = (
        precision_recall_from_ranked(
            center_labeled,
            total_gt,
        )
    )

    center_ap = (
        interpolated_ap(
            precision,
            recall,
        )
    )

    best_f1 = (
        find_best_f1_threshold(
            center_labeled,
            total_gt,
        )
    )

    fixed_metrics = (
        metrics_at_threshold(
            center_labeled,
            total_gt,
            FIXED_SCORE_THRESHOLD,
        )
    )

    print(
        f"GT nodules           : {total_gt}"
    )

    print(
        f"Predictions          : {len(pred_df)}"
    )

    print(
        f"Center AP            : {center_ap:.6f}"
    )

    print(
        f"Center mAP           : {center_ap:.6f}"
    )

    print()
    print(
        "Best-F1 operating point"
    )

    print(
        f"  threshold : {best_f1['threshold']:.6f}"
    )

    print(
        f"  TP / FP / FN : "
        f"{best_f1['tp']} / "
        f"{best_f1['fp']} / "
        f"{best_f1['fn']}"
    )

    print(
        f"  Precision : {best_f1['precision']:.6f}"
    )

    print(
        f"  Recall    : {best_f1['recall']:.6f}"
    )

    print(
        f"  F1        : {best_f1['f1']:.6f}"
    )

    print()
    print(
        f"Fixed threshold = {FIXED_SCORE_THRESHOLD}"
    )

    print(
        f"  TP / FP / FN : "
        f"{fixed_metrics['tp']} / "
        f"{fixed_metrics['fp']} / "
        f"{fixed_metrics['fn']}"
    )

    print(
        f"  Precision : {fixed_metrics['precision']:.6f}"
    )

    print(
        f"  Recall    : {fixed_metrics['recall']:.6f}"
    )

    print(
        f"  F1        : {fixed_metrics['f1']:.6f}"
    )

    # -------------------------------------------------------------------------
    # FROC
    # -------------------------------------------------------------------------

    froc_df = (
        build_froc_curve(
            center_labeled,
            total_gt,
            num_scans,
        )
    )

    operating_points = (
        froc_operating_points(
            froc_df
        )
    )

    cpm = float(
        np.mean(
            [
                row[
                    "sensitivity"
                ]
                for row in operating_points
            ]
        )
    )

    header(
        "FROC / CPM"
    )

    for row in operating_points:

        print(
            f"FP/scan {row['target_fp_per_scan']:>5.3f} | "
            f"Sensitivity {row['sensitivity']:.6f} | "
            f"threshold {row['threshold']:.6f} | "
            f"actual FP/scan {row['actual_fp_per_scan']:.4f}"
        )

    print()
    print(
        f"CPM : {cpm:.6f}"
    )

    one_fp_row = next(
        row
        for row in operating_points
        if math.isclose(
            row[
                "target_fp_per_scan"
            ],
            1.0,
        )
    )

    one_fp_threshold = float(
        one_fp_row[
            "threshold"
        ]
    )

    # -------------------------------------------------------------------------
    # IoU AP
    # -------------------------------------------------------------------------

    header(
        "3D IoU AP"
    )

    iou_ap = {}

    for threshold in (
        IOU_AP_THRESHOLDS
    ):

        labeled = (
            greedy_detection_labels(
                pred_df=pred_df,
                gt_df=gt_df,
                mode="iou",
                iou_threshold=threshold,
            )
        )

        p, r, _ = (
            precision_recall_from_ranked(
                labeled,
                total_gt,
            )
        )

        ap = interpolated_ap(
            p,
            r,
        )

        key = (
            f"AP@{threshold:.2f}"
        )

        iou_ap[
            key
        ] = float(
            ap
        )

        print(
            f"{key:<10}: {ap:.6f}"
        )

    # -------------------------------------------------------------------------
    # Size recall
    # -------------------------------------------------------------------------

    thresholds = {
        "best_f1":
            float(
                best_f1[
                    "threshold"
                ]
            ),

        "1_fp_per_scan":
            one_fp_threshold,

        "fixed_0.50":
            FIXED_SCORE_THRESHOLD,
    }

    size_recall_df = (
        calculate_size_recall(
            gt_df=gt_df,
            labeled_predictions=center_labeled,
            thresholds=thresholds,
        )
    )

    header(
        "Size-wise Recall"
    )

    for (
        op_name,
        group,
    ) in size_recall_df.groupby(
        "operating_point",
        sort=False,
    ):

        print()
        print(
            f"[{op_name}] "
            f"threshold={group.iloc[0]['threshold']:.6f}"
        )

        for _, row in group.iterrows():

            print(
                f"  {row['size_group']:<9} | "
                f"GT {int(row['gt_nodules']):3d} | "
                f"Detected {int(row['detected']):3d} | "
                f"Recall {row['recall']:.6f}"
            )

    # -------------------------------------------------------------------------
    # Localization / score
    # -------------------------------------------------------------------------

    selected_best = (
        center_labeled[
            center_labeled[
                "score"
            ]
            >= best_f1[
                "threshold"
            ]
        ]
    )

    tp_best = (
        selected_best[
            selected_best[
                "is_tp"
            ]
            == 1
        ]
    )

    fp_best = (
        selected_best[
            selected_best[
                "is_tp"
            ]
            == 0
        ]
    )

    localization_stats = (
        percentile_stats(
            tp_best[
                "center_distance_mm"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )
    )

    tp_score_stats = (
        percentile_stats(
            tp_best[
                "score"
            ]
            .to_numpy(
                dtype=float
            )
        )
    )

    fp_score_stats = (
        percentile_stats(
            fp_best[
                "score"
            ]
            .to_numpy(
                dtype=float
            )
        )
    )

    header(
        "Localization / Score Statistics @ Best F1"
    )

    print(
        "Localization error:",
        localization_stats,
    )

    print(
        "TP score:",
        tp_score_stats,
    )

    print(
        "FP score:",
        fp_score_stats,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    froc_df.to_csv(
        FROC_CSV,
        index=False,
    )

    pr_df = pd.DataFrame(
        {
            "score":
                scores,

            "precision":
                precision,

            "recall":
                recall,
        }
    )

    pr_df.to_csv(
        PR_CSV,
        index=False,
    )

    size_recall_df.to_csv(
        SIZE_RECALL_CSV,
        index=False,
    )

    center_labeled.to_csv(
        OUTPUT_DIR
        / "center_match_predictions.csv",
        index=False,
    )

    save_froc_plot(
        froc_df,
        operating_points,
    )

    save_pr_plot(
        precision,
        recall,
    )

    summary = {
        "checkpoint":
            str(
                CHECKPOINT
            ),

        "num_scans":
            int(
                num_scans
            ),

        "num_gt_nodules":
            int(
                total_gt
            ),

        "num_predictions_retained":
            int(
                len(
                    pred_df
                )
            ),

        "center_match": {
            "ap":
                float(
                    center_ap
                ),

            "map":
                float(
                    center_ap
                ),

            "best_f1":
                best_f1,

            "fixed_0_50":
                fixed_metrics,
        },

        "froc": {
            "operating_points":
                operating_points,

            "cpm":
                cpm,
        },

        "iou_ap":
            iou_ap,

        "size_recall":
            size_recall_df.to_dict(
                orient="records"
            ),

        "localization_error_mm_best_f1":
            localization_stats,

        "tp_score_best_f1":
            tp_score_stats,

        "fp_score_best_f1":
            fp_score_stats,
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "CPMNetv2 Full Validation Evaluation"
    )

    print(
        "Device               :",
        DEVICE,
    )

    if torch.cuda.is_available():

        print(
            "GPU                  :",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Checkpoint           :",
        CHECKPOINT,
    )

    print(
        "Validation images    :",
        VAL_ROOT,
    )

    print(
        "Validation CSV       :",
        VAL_CSV,
    )

    print(
        "Output               :",
        OUTPUT_DIR,
    )

    print()
    print(
        "Crop                 :",
        CROP_SIZE,
    )

    print(
        "Overlap              :",
        OVERLAP_SIZE,
    )

    print(
        "Spacing              :",
        SPACING_ZYX.tolist(),
    )

    print(
        "Inference batch      :",
        INFER_BATCH_SIZE,
    )

    print(
        "Candidate threshold  :",
        CROP_SCORE_THRESHOLD,
    )

    for path in [
        CPMNET_ROOT,
        VAL_ROOT,
        VAL_CSV,
        CHECKPOINT,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    header(
        "Dataset"
    )

    cases = list_val_cases()

    annotations = (
        load_val_annotations()
    )

    gt_df = (
        build_gt_table(
            cases,
            annotations,
        )
    )

    gt_df.to_csv(
        GT_CSV,
        index=False,
    )

    positive_case_count = int(
        gt_df[
            "case_id"
        ]
        .nunique()
    )

    print(
        "Validation scans      :",
        len(
            cases
        ),
    )

    print(
        "Positive scans        :",
        positive_case_count,
    )

    print(
        "No-GT scans           :",
        len(cases)
        - positive_case_count,
    )

    print(
        "GT nodules            :",
        len(
            gt_df
        ),
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    header(
        "Model"
    )

    model = build_model()

    load_checkpoint(
        model
    )

    model.eval()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    pred_df = (
        run_full_inference(
            model=model,
            cases=cases,
        )
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    summary = evaluate(
        pred_df=pred_df,
        gt_df=gt_df,
        num_scans=len(
            cases
        ),
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    header(
        "FINAL RESULT"
    )

    print(
        f"Center AP       : "
        f"{summary['center_match']['ap']:.6f}"
    )

    print(
        f"Best F1         : "
        f"{summary['center_match']['best_f1']['f1']:.6f}"
    )

    print(
        f"Best F1 Recall  : "
        f"{summary['center_match']['best_f1']['recall']:.6f}"
    )

    print(
        f"Best F1 Prec.   : "
        f"{summary['center_match']['best_f1']['precision']:.6f}"
    )

    print(
        f"CPM             : "
        f"{summary['froc']['cpm']:.6f}"
    )

    for key, value in (
        summary[
            "iou_ap"
        ].items()
    ):

        print(
            f"{key:<15}: "
            f"{value:.6f}"
        )

    print()
    print(
        "Saved:"
    )

    print(
        PREDICTIONS_CSV
    )

    print(
        GT_CSV
    )

    print(
        SUMMARY_JSON
    )

    print(
        FROC_CSV
    )

    print(
        PR_CSV
    )

    print(
        SIZE_RECALL_CSV
    )

    print(
        FROC_PNG
    )

    print(
        PR_PNG
    )

    print()
    print(
        "Evaluation completed."
    )


if __name__ == "__main__":

    main()