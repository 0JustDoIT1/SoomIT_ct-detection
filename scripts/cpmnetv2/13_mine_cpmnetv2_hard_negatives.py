#!/usr/bin/env python3

"""
13_mine_cpmnetv2_hard_negatives.py

CPMNetv2 Hard Negative Mining

목적
----
현재 CPMNetv2 Baseline best checkpoint를 이용하여
TRAIN SET 800 scans에 대해 full-volume inference를 수행한다.

그 중:

    높은 confidence prediction
    +
    실제 GT nodule과 충분히 떨어져 있음

조건을 만족하는 prediction을 Hard Negative로 저장한다.

중요
----
Validation set은 절대 사용하지 않는다.

Pipeline
--------
Train CT
    ↓
Full-volume SplitComb inference
    ↓
3D NMS
    ↓
score >= MIN_HARD_NEG_SCORE
    ↓
GT nodule safety region 검사
    ↓
GT와 충분히 떨어진 prediction만 FP 후보
    ↓
score 순으로 scan당 최대 MAX_HARD_NEG_PER_SCAN개 저장

Output
------
hard_negatives.csv
hard_negative_summary.json
hard_negative_scan_summary.csv

좌표 convention
----------------
Volume      : [Z, Y, X]
GT center   : [z, y, x]
Prediction  : [score, z, y, x, d, h, w]
Spacing     : [1, 1, 1] mm
"""

import gc
import json
import math
import sys
import time
from pathlib import Path

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

TRAIN_ROOT = (
    DATA_ROOT
    / "imagesTr"
)

TRAIN_CSV = (
    DATA_ROOT
    / "train.csv"
)

TRAIN_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "full_training"
)

CHECKPOINT = (
    TRAIN_OUTPUT_ROOT
    / "best_val_loss.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_mining"
)

HARD_NEGATIVE_CSV = (
    OUTPUT_DIR
    / "hard_negatives.csv"
)

SCAN_SUMMARY_CSV = (
    OUTPUT_DIR
    / "hard_negative_scan_summary.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "hard_negative_summary.json"
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


# =============================================================================
# INFERENCE CONFIG
# =============================================================================

INFER_BATCH_SIZE = 4

# 낮은 단계에서 충분한 후보 확보
CROP_SCORE_THRESHOLD = 0.01

CROP_POST_TOPK = 120
CROP_POST_NUM_TOPK = 60
CROP_NMS_THRESHOLD = 0.05

FINAL_NMS_THRESHOLD = 0.05
FINAL_TOPK_PER_SCAN = 300


# =============================================================================
# HARD NEGATIVE CONFIG
# =============================================================================

# 이 값 이상인 FP만 hard negative 후보로 사용
MIN_HARD_NEG_SCORE = 0.90

# 한 scan이 hard negative를 독점하지 않도록 제한
MAX_HARD_NEG_PER_SCAN = 10

# GT 중심 주변은 애매한 영역이므로 negative로 사용하지 않는다.
#
# safety_radius =
#     (GT diameter / 2)
#     * GT_SAFETY_MARGIN
#
GT_SAFETY_MARGIN = 1.5

# GT가 없는 scan에서는 모든 high-confidence prediction이
# hard-negative 후보가 될 수 있음
ALLOW_NEGATIVE_SCAN_HARD_NEGATIVES = True


# =============================================================================
# UTILITY
# =============================================================================

def header(title):

    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def strip_nii_suffix(name):

    if name.endswith(
        ".nii.gz"
    ):
        return name[:-7]

    if name.endswith(
        ".nii"
    ):
        return name[:-4]

    return Path(
        name
    ).stem


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

    # training과 동일
    image = (
        image
        * 2.0
        - 1.0
    )

    return image.astype(
        np.float32,
        copy=False,
    )


def center_distance_mm(
    p1,
    p2,
):

    diff = np.asarray(
        [
            float(
                p1[
                    "z"
                ]
            )
            - float(
                p2[
                    "z"
                ]
            ),

            float(
                p1[
                    "y"
                ]
            )
            - float(
                p2[
                    "y"
                ]
            ),

            float(
                p1[
                    "x"
                ]
            )
            - float(
                p2[
                    "x"
                ]
            ),
        ],
        dtype=np.float32,
    )

    diff = (
        diff
        * SPACING_ZYX
    )

    return float(
        np.sqrt(
            np.sum(
                diff ** 2
            )
        )
    )


# =============================================================================
# DATA
# =============================================================================

def load_train_annotations():

    df = pd.read_csv(
        TRAIN_CSV
    )

    if "label" in df.columns:

        df = df[
            df[
                "label"
            ]
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
            f"TRAIN_CSV missing columns: "
            f"{missing}"
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


def build_train_cases(
    annotations,
):

    image_paths = sorted(
        TRAIN_ROOT.glob(
            "*.nii.gz"
        )
    )

    if not image_paths:

        raise RuntimeError(
            f"No training NIfTI found:\n"
            f"{TRAIN_ROOT}"
        )

    grouped = {
        str(
            uid
        ):
            group.copy()

        for uid, group
        in annotations.groupby(
            "seriesuid"
        )
    }

    cases = []

    positive = 0
    negative = 0
    nodules = 0

    for image_path in image_paths:

        case_id = (
            strip_nii_suffix(
                image_path.name
            )
        )

        gt = grouped.get(
            case_id
        )

        if gt is None:

            gt = pd.DataFrame(
                columns=annotations.columns
            )

            negative += 1

        else:

            positive += 1

            nodules += len(
                gt
            )

        cases.append(
            {
                "case_id":
                    case_id,

                "path":
                    image_path,

                "gt":
                    gt,
            }
        )

    return (
        cases,
        positive,
        negative,
        nodules,
    )


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
            f"Checkpoint 없음:\n"
            f"{CHECKPOINT}"
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
        for key
        in state_dict.keys()
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
        "[PASS] Checkpoint loaded"
    )

    print(
        "Checkpoint      :",
        CHECKPOINT,
    )

    print(
        "Epoch           :",
        epoch,
    )

    print(
        "Best val loss   :",
        best_val_loss,
    )

    print(
        "State dict      :",
        result,
    )


# =============================================================================
# POSTPROCESS
# =============================================================================

def build_postprocessor():

    return Detection_Postprocess(
        topk=CROP_POST_TOPK,

        threshold=(
            CROP_SCORE_THRESHOLD
        ),

        nms_threshold=(
            CROP_NMS_THRESHOLD
        ),

        num_topk=(
            CROP_POST_NUM_TOPK
        ),

        crop_size=CROP_SIZE,
    )


# =============================================================================
# FULL VOLUME INFERENCE
# =============================================================================

def infer_one_case(
    model,
    postprocessor,
    split_comber,
    case,
):

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
            f"Spacing mismatch:\n"
            f"{image_path}\n"
            f"{spacing_zyx}"
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
        int(
            x
        )
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

    outputs = []

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

            outputs.append(
                detections
                .detach()
                .cpu()
                .numpy()
            )

    if not outputs:

        return (
            [],
            image_shape,
        )

    output = np.concatenate(
        outputs,
        axis=0,
    )

    output = split_comber.combine(
        output,
        nzhw=nzhw,
    )

    output = (
        torch
        .from_numpy(
            output
        )
        .reshape(
            -1,
            8,
        )
        .float()
    )

    # CPMNetv2 invalid rows
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

    if len(
        output
    ) > 0:

        keep = nms_3D(
            output[
                :,
                1:
            ],

            overlap=(
                FINAL_NMS_THRESHOLD
            ),

            top_k=(
                FINAL_TOPK_PER_SCAN
            ),
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
            detection[
                1
            ]
        )

        z = float(
            detection[
                2
            ]
        )

        y = float(
            detection[
                3
            ]
        )

        x = float(
            detection[
                4
            ]
        )

        d = float(
            detection[
                5
            ]
        )

        h = float(
            detection[
                6
            ]
        )

        w = float(
            detection[
                7
            ]
        )

        values = [
            score,
            z,
            y,
            x,
            d,
            h,
            w,
        ]

        if not np.all(
            np.isfinite(
                values
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

        key=lambda row: (
            row[
                "score"
            ]
        ),

        reverse=True,
    )

    return (
        rows,
        image_shape,
    )


# =============================================================================
# GT CONVERSION
# =============================================================================

def convert_gt(
    gt_df,
):

    rows = []

    for _, row in gt_df.iterrows():

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

        rows.append(
            {
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
            }
        )

    return rows


# =============================================================================
# HARD NEGATIVE CHECK
# =============================================================================

def evaluate_candidate_against_gt(
    prediction,
    gt_rows,
):

    """
    prediction이 GT safety region 밖인지 확인.

    safety radius:
        GT equivalent diameter / 2
        * GT_SAFETY_MARGIN

    반환:
        is_safe_negative
        nearest_gt_distance
        nearest_gt_radius
        distance_over_radius
    """

    if len(
        gt_rows
    ) == 0:

        return {
            "is_safe_negative":
                True,

            "nearest_gt_distance_mm":
                None,

            "nearest_gt_radius_mm":
                None,

            "distance_over_safety_radius":
                None,
        }

    nearest_distance = None
    nearest_radius = None
    nearest_ratio = None

    inside_any_safety_region = False

    for gt in gt_rows:

        distance = (
            center_distance_mm(
                prediction,
                gt,
            )
        )

        radius = (
            (
                float(
                    gt[
                        "diameter_mm"
                    ]
                )
                / 2.0
            )
            * GT_SAFETY_MARGIN
        )

        radius = max(
            radius,
            1e-6,
        )

        ratio = (
            distance
            / radius
        )

        if (
            nearest_distance
            is None
            or distance
            < nearest_distance
        ):

            nearest_distance = (
                distance
            )

            nearest_radius = (
                radius
            )

            nearest_ratio = (
                ratio
            )

        if distance < radius:

            inside_any_safety_region = (
                True
            )

    return {
        "is_safe_negative":
            not inside_any_safety_region,

        "nearest_gt_distance_mm":
            nearest_distance,

        "nearest_gt_radius_mm":
            nearest_radius,

        "distance_over_safety_radius":
            nearest_ratio,
    }


# =============================================================================
# MINE ONE CASE
# =============================================================================

def mine_case(
    case,
    predictions,
):

    gt_rows = convert_gt(
        case[
            "gt"
        ]
    )

    high_score_predictions = [
        prediction
        for prediction in predictions
        if (
            prediction[
                "score"
            ]
            >= MIN_HARD_NEG_SCORE
        )
    ]

    safe_negatives = []

    rejected_near_gt = 0

    for prediction in (
        high_score_predictions
    ):

        check = (
            evaluate_candidate_against_gt(
                prediction,
                gt_rows,
            )
        )

        if not check[
            "is_safe_negative"
        ]:

            rejected_near_gt += 1
            continue

        row = dict(
            prediction
        )

        row.update(
            check
        )

        safe_negatives.append(
            row
        )

    safe_negatives = sorted(
        safe_negatives,

        key=lambda row: (
            row[
                "score"
            ]
        ),

        reverse=True,
    )

    selected = (
        safe_negatives[
            :
            MAX_HARD_NEG_PER_SCAN
        ]
    )

    return {
        "num_gt":
            len(
                gt_rows
            ),

        "num_predictions":
            len(
                predictions
            ),

        "num_high_score":
            len(
                high_score_predictions
            ),

        "num_rejected_near_gt":
            rejected_near_gt,

        "num_safe_fp":
            len(
                safe_negatives
            ),

        "selected":
            selected,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "CPMNetv2 Hard Negative Mining"
    )

    print(
        "Device                  :",
        DEVICE,
    )

    if torch.cuda.is_available():

        print(
            "GPU                     :",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Checkpoint              :",
        CHECKPOINT,
    )

    print(
        "Train root              :",
        TRAIN_ROOT,
    )

    print(
        "Train CSV               :",
        TRAIN_CSV,
    )

    print()
    print(
        "Hard-negative threshold :",
        MIN_HARD_NEG_SCORE,
    )

    print(
        "Max hard-neg / scan     :",
        MAX_HARD_NEG_PER_SCAN,
    )

    print(
        "GT safety margin        :",
        GT_SAFETY_MARGIN,
    )

    print(
        "Final prediction top-k  :",
        FINAL_TOPK_PER_SCAN,
    )

    # =========================================================================
    # CHECK
    # =========================================================================

    for path in [
        CPMNET_ROOT,
        TRAIN_ROOT,
        TRAIN_CSV,
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

    # =========================================================================
    # DATA
    # =========================================================================

    header(
        "Training Dataset"
    )

    annotations = (
        load_train_annotations()
    )

    (
        cases,
        positive_cases,
        negative_cases,
        total_nodules,
    ) = build_train_cases(
        annotations
    )

    print(
        "Train scans       :",
        len(
            cases
        ),
    )

    print(
        "Positive scans    :",
        positive_cases,
    )

    print(
        "No-GT scans       :",
        negative_cases,
    )

    print(
        "GT nodules        :",
        total_nodules,
    )

    if len(
        cases
    ) != 800:

        print(
            f"[WARN] Expected 800 scans, "
            f"found {len(cases)}"
        )

    # =========================================================================
    # MODEL
    # =========================================================================

    header(
        "Model"
    )

    model = build_model()

    load_checkpoint(
        model
    )

    model.eval()

    postprocessor = (
        build_postprocessor()
    )

    split_comber = SplitComb(
        crop_size=CROP_SIZE,
        overlap=OVERLAP_SIZE,
        pad_value=-1,
    )

    # =========================================================================
    # MINING
    # =========================================================================

    header(
        "Mining"
    )

    selected_rows = []

    scan_rows = []

    total_predictions = 0
    total_high_score = 0
    total_rejected_near_gt = 0
    total_safe_fp = 0
    total_selected = 0

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

        mining = mine_case(
            case,
            predictions,
        )

        selected = mining[
            "selected"
        ]

        for rank, row in enumerate(
            selected,
            start=1,
        ):

            selected_rows.append(
                {
                    "case_id":
                        case[
                            "case_id"
                        ],

                    "rank":
                        rank,

                    "score":
                        row[
                            "score"
                        ],

                    "z":
                        row[
                            "z"
                        ],

                    "y":
                        row[
                            "y"
                        ],

                    "x":
                        row[
                            "x"
                        ],

                    "d":
                        row[
                            "d"
                        ],

                    "h":
                        row[
                            "h"
                        ],

                    "w":
                        row[
                            "w"
                        ],

                    "nearest_gt_distance_mm":
                        row[
                            "nearest_gt_distance_mm"
                        ],

                    "nearest_gt_safety_radius_mm":
                        row[
                            "nearest_gt_radius_mm"
                        ],

                    "distance_over_safety_radius":
                        row[
                            "distance_over_safety_radius"
                        ],

                    "gt_count":
                        mining[
                            "num_gt"
                        ],

                    "source":
                        "baseline_train_fp",
                }
            )

        total_predictions += (
            mining[
                "num_predictions"
            ]
        )

        total_high_score += (
            mining[
                "num_high_score"
            ]
        )

        total_rejected_near_gt += (
            mining[
                "num_rejected_near_gt"
            ]
        )

        total_safe_fp += (
            mining[
                "num_safe_fp"
            ]
        )

        total_selected += len(
            selected
        )

        scan_rows.append(
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "gt_count":
                    mining[
                        "num_gt"
                    ],

                "volume_z":
                    image_shape[
                        0
                    ],

                "volume_y":
                    image_shape[
                        1
                    ],

                "volume_x":
                    image_shape[
                        2
                    ],

                "predictions":
                    mining[
                        "num_predictions"
                    ],

                "high_score_predictions":
                    mining[
                        "num_high_score"
                    ],

                "rejected_near_gt":
                    mining[
                        "num_rejected_near_gt"
                    ],

                "safe_fp_candidates":
                    mining[
                        "num_safe_fp"
                    ],

                "selected_hard_negatives":
                    len(
                        selected
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

        seconds_per_case = (
            elapsed
            / index
        )

        remaining = (
            len(
                cases
            )
            - index
        )

        eta_seconds = (
            remaining
            * seconds_per_case
        )

        print(
            f"[{index:03d}/{len(cases)}] "
            f"{case['case_id']} | "
            f"GT={mining['num_gt']:2d} | "
            f"pred={mining['num_predictions']:3d} | "
            f">={MIN_HARD_NEG_SCORE:.2f}="
            f"{mining['num_high_score']:3d} | "
            f"safeFP={mining['num_safe_fp']:3d} | "
            f"selected={len(selected):2d} | "
            f"{time.time()-case_start:.1f}s | "
            f"ETA={eta_seconds/60:.1f} min"
        )

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # =========================================================================
    # SAVE
    # =========================================================================

    hard_negative_df = (
        pd.DataFrame(
            selected_rows
        )
    )

    scan_summary_df = (
        pd.DataFrame(
            scan_rows
        )
    )

    hard_negative_df.to_csv(
        HARD_NEGATIVE_CSV,
        index=False,
    )

    scan_summary_df.to_csv(
        SCAN_SUMMARY_CSV,
        index=False,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    total_time = (
        time.time()
        - start_time
    )

    if len(
        hard_negative_df
    ) > 0:

        score_stats = {
            "mean":
                float(
                    hard_negative_df[
                        "score"
                    ].mean()
                ),

            "median":
                float(
                    hard_negative_df[
                        "score"
                    ].median()
                ),

            "min":
                float(
                    hard_negative_df[
                        "score"
                    ].min()
                ),

            "max":
                float(
                    hard_negative_df[
                        "score"
                    ].max()
                ),
        }

        cases_with_hard_neg = int(
            hard_negative_df[
                "case_id"
            ].nunique()
        )

    else:

        score_stats = {
            "mean":
                None,

            "median":
                None,

            "min":
                None,

            "max":
                None,
        }

        cases_with_hard_neg = 0

    summary = {
        "checkpoint":
            str(
                CHECKPOINT
            ),

        "train_scans":
            len(
                cases
            ),

        "positive_scans":
            positive_cases,

        "negative_scans":
            negative_cases,

        "gt_nodules":
            total_nodules,

        "config": {
            "min_hard_negative_score":
                MIN_HARD_NEG_SCORE,

            "max_hard_negative_per_scan":
                MAX_HARD_NEG_PER_SCAN,

            "gt_safety_margin":
                GT_SAFETY_MARGIN,

            "crop_size":
                CROP_SIZE,

            "overlap":
                OVERLAP_SIZE,

            "spacing":
                SPACING_ZYX.tolist(),

            "final_topk_per_scan":
                FINAL_TOPK_PER_SCAN,
        },

        "mining": {
            "total_predictions":
                total_predictions,

            "high_score_predictions":
                total_high_score,

            "rejected_near_gt":
                total_rejected_near_gt,

            "safe_fp_candidates":
                total_safe_fp,

            "selected_hard_negatives":
                total_selected,

            "cases_with_hard_negatives":
                cases_with_hard_neg,

            "score_stats":
                score_stats,
        },

        "runtime_seconds":
            total_time,

        "runtime_hours":
            total_time
            / 3600.0,
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

    # =========================================================================
    # FINAL
    # =========================================================================

    header(
        "Hard Negative Mining Result"
    )

    print(
        "Train scans             :",
        len(
            cases
        ),
    )

    print(
        "GT nodules              :",
        total_nodules,
    )

    print(
        "Raw predictions         :",
        total_predictions,
    )

    print(
        f"Score >= {MIN_HARD_NEG_SCORE:.2f}         :",
        total_high_score,
    )

    print(
        "Rejected near GT        :",
        total_rejected_near_gt,
    )

    print(
        "Safe FP candidates      :",
        total_safe_fp,
    )

    print(
        "Selected hard negatives :",
        total_selected,
    )

    print(
        "Cases with hard negatives:",
        cases_with_hard_neg,
    )

    print()

    print(
        "Hard-negative score:"
    )

    print(
        "  Mean   :",
        score_stats[
            "mean"
        ],
    )

    print(
        "  Median :",
        score_stats[
            "median"
        ],
    )

    print(
        "  Min    :",
        score_stats[
            "min"
        ],
    )

    print(
        "  Max    :",
        score_stats[
            "max"
        ],
    )

    print()

    print(
        f"Total mining time       : "
        f"{total_time/3600.0:.2f} h"
    )

    print()

    print(
        "Saved:"
    )

    print(
        HARD_NEGATIVE_CSV
    )

    print(
        SCAN_SUMMARY_CSV
    )

    print(
        SUMMARY_JSON
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "14_train_cpmnetv2_hard_negative.py"
    )

    print(
        "Baseline sampling에 mined hard-negative "
        "center만 추가하여 재학습합니다."
    )


if __name__ == "__main__":

    main()