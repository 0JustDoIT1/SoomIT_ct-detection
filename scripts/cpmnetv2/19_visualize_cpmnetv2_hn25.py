#!/usr/bin/env python3

"""
19_visualize_cpmnetv2_hn25.py

CPMNetv2 + Hard Negative 25%
Final Validation TP / FP / FN Visualization

Final fixed detection configuration
-----------------------------------
Model:
    CPMNetv2 + HN25

CROP_NMS_THRESHOLD:
    0.05

FINAL_NMS_THRESHOLD:
    0.05

Best-F1 threshold:
    0.9612025618553162

Expected validation result:
    TP = 62
    FP = 24
    FN = 43

Matching rule:
    prediction center와 GT center 거리
        <= GT equivalent diameter / 2

    한 GT는 한 prediction에만 matching.

Visualization:
    TP 5 cases
    FP 5 cases
    FN 5 cases

Output:
    visualization/
        tp/
        fp/
        fn/
        tp_fp_fn_events.csv
        visualization_manifest.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.patches import Rectangle


# =============================================================================
# PATH
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

VAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "luna16"
    / "imagesVa"
)

EVAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_evaluation_ratio025"
)

PREDICTIONS_CSV = (
    EVAL_DIR
    / "predictions.csv"
)

GT_CSV = (
    EVAL_DIR
    / "ground_truth.csv"
)

OUTPUT_DIR = (
    EVAL_DIR
    / "visualization"
)

EVENTS_CSV = (
    OUTPUT_DIR
    / "tp_fp_fn_events.csv"
)

MANIFEST_CSV = (
    OUTPUT_DIR
    / "visualization_manifest.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

# IMPORTANT
# 출력에 표시된 0.961203은 반올림 값.
#
# 실제 Best-F1 prediction minimum score:
# 0.9612025618553162
#
# 이 정확값을 사용해야:
#
# TP / FP / FN
# 62 / 24 / 43
#
# 가 재현됨.
SCORE_THRESHOLD = 0.9612025618553162

EXPECTED_TP = 62
EXPECTED_FP = 24
EXPECTED_FN = 43

NUM_EXAMPLES_PER_TYPE = 5

HU_MIN = -1200.0
HU_MAX = 600.0


# =============================================================================
# Utility
# =============================================================================

def header(title):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def normalize_case_id(name):
    """
    IMPORTANT

    LUNA16 series UID는 점(.)이 많이 포함됨.

    예:
    1.3.6.1.4.1.14519.5.2.1....

    Path(name).stem 을 사용하면 마지막 UID 구간을
    파일 확장자로 오인하여 잘라버릴 수 있음.

    따라서 .nii.gz / .nii만 명시적으로 제거한다.
    """

    name = str(
        name
    ).strip()

    if name.endswith(
        ".nii.gz"
    ):

        return name[:-7]

    if name.endswith(
        ".nii"
    ):

        return name[:-4]

    return name


# =============================================================================
# Input validation
# =============================================================================

def validate_inputs():

    header(
        "Input Validation"
    )

    print(
        "Validation CT:"
    )

    print(
        VAL_ROOT
    )

    print()

    print(
        "Predictions:"
    )

    print(
        PREDICTIONS_CSV
    )

    print()

    print(
        "Ground truth:"
    )

    print(
        GT_CSV
    )

    if not VAL_ROOT.exists():

        raise FileNotFoundError(
            f"Validation CT directory 없음:\n"
            f"{VAL_ROOT}"
        )

    if not PREDICTIONS_CSV.exists():

        raise FileNotFoundError(
            f"predictions.csv 없음:\n"
            f"{PREDICTIONS_CSV}"
        )

    if not GT_CSV.exists():

        raise FileNotFoundError(
            f"ground_truth.csv 없음:\n"
            f"{GT_CSV}"
        )

    print()

    print(
        "[PASS] Input files exist"
    )


# =============================================================================
# Load CSV
# =============================================================================

def load_csv():

    predictions = pd.read_csv(
        PREDICTIONS_CSV
    )

    ground_truth = pd.read_csv(
        GT_CSV
    )

    return (
        predictions,
        ground_truth,
    )


# =============================================================================
# CSV QA
# =============================================================================

def validate_columns(
    predictions,
    ground_truth,
):

    header(
        "CSV Column Inspection"
    )

    print(
        "Prediction columns:"
    )

    print(
        list(
            predictions.columns
        )
    )

    print()

    print(
        "GT columns:"
    )

    print(
        list(
            ground_truth.columns
        )
    )

    required_prediction_columns = {
        "case_id",
        "score",
        "z",
        "y",
        "x",
        "d",
        "h",
        "w",
    }

    required_gt_columns = {
        "gt_id",
        "case_id",
        "z",
        "y",
        "x",
        "d",
        "h",
        "w",
        "diameter_mm",
    }

    missing_pred = (
        required_prediction_columns
        - set(
            predictions.columns
        )
    )

    missing_gt = (
        required_gt_columns
        - set(
            ground_truth.columns
        )
    )

    if missing_pred:

        raise RuntimeError(
            "Prediction CSV missing columns:\n"
            f"{sorted(missing_pred)}"
        )

    if missing_gt:

        raise RuntimeError(
            "GT CSV missing columns:\n"
            f"{sorted(missing_gt)}"
        )

    print()

    print(
        "[PASS] Required columns exist"
    )


# =============================================================================
# Normalize case IDs
# =============================================================================

def normalize_case_ids(
    predictions,
    ground_truth,
):

    predictions = (
        predictions.copy()
    )

    ground_truth = (
        ground_truth.copy()
    )

    predictions[
        "case_id"
    ] = (
        predictions[
            "case_id"
        ]
        .astype(str)
        .map(
            normalize_case_id
        )
    )

    ground_truth[
        "case_id"
    ] = (
        ground_truth[
            "case_id"
        ]
        .astype(str)
        .map(
            normalize_case_id
        )
    )

    return (
        predictions,
        ground_truth,
    )


# =============================================================================
# Build TP / FP / FN
# =============================================================================

def build_detection_events(
    predictions,
    ground_truth,
):

    header(
        "Build TP / FP / FN @ Best-F1 threshold"
    )

    print(
        "Score threshold:"
    )

    print(
        f"{SCORE_THRESHOLD:.16f}"
    )

    # -------------------------------------------------------------------------
    # Best-F1 threshold filter
    # -------------------------------------------------------------------------

    filtered_predictions = (
        predictions[
            predictions[
                "score"
            ]
            >= SCORE_THRESHOLD
        ]
        .copy()
    )

    print()

    print(
        "Predictions before threshold:",
        len(
            predictions
        ),
    )

    print(
        "Predictions after threshold :",
        len(
            filtered_predictions
        ),
    )

    # -------------------------------------------------------------------------
    # All cases
    # -------------------------------------------------------------------------

    case_ids = sorted(
        set(
            ground_truth[
                "case_id"
            ].tolist()
        )
        |
        set(
            filtered_predictions[
                "case_id"
            ].tolist()
        )
    )

    events = []

    # -------------------------------------------------------------------------
    # Case-by-case matching
    # -------------------------------------------------------------------------

    for case_id in case_ids:

        case_gt = (
            ground_truth[
                ground_truth[
                    "case_id"
                ]
                == case_id
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        # evaluator와 동일하게 score descending
        case_pred = (
            filtered_predictions[
                filtered_predictions[
                    "case_id"
                ]
                == case_id
            ]
            .copy()
            .sort_values(
                "score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        matched_gt_indices = set()

        # =====================================================================
        # Prediction -> GT
        # =====================================================================

        for _, pred_row in (
            case_pred.iterrows()
        ):

            pred_center = np.asarray(
                [
                    float(
                        pred_row[
                            "z"
                        ]
                    ),
                    float(
                        pred_row[
                            "y"
                        ]
                    ),
                    float(
                        pred_row[
                            "x"
                        ]
                    ),
                ],
                dtype=np.float64,
            )

            best_gt_index = None
            best_distance = float(
                "inf"
            )

            # -----------------------------------------------------------------
            # Find nearest valid unmatched GT
            # -----------------------------------------------------------------

            for gt_index, gt_row in (
                case_gt.iterrows()
            ):

                if (
                    gt_index
                    in matched_gt_indices
                ):

                    continue

                gt_center = np.asarray(
                    [
                        float(
                            gt_row[
                                "z"
                            ]
                        ),
                        float(
                            gt_row[
                                "y"
                            ]
                        ),
                        float(
                            gt_row[
                                "x"
                            ]
                        ),
                    ],
                    dtype=np.float64,
                )

                distance = float(
                    np.linalg.norm(
                        pred_center
                        - gt_center
                    )
                )

                radius_mm = (
                    float(
                        gt_row[
                            "diameter_mm"
                        ]
                    )
                    / 2.0
                )

                if (
                    distance
                    <= radius_mm
                    and
                    distance
                    < best_distance
                ):

                    best_gt_index = (
                        gt_index
                    )

                    best_distance = (
                        distance
                    )

            # -----------------------------------------------------------------
            # TP
            # -----------------------------------------------------------------

            if (
                best_gt_index
                is not None
            ):

                matched_gt_indices.add(
                    best_gt_index
                )

                gt_row = (
                    case_gt.iloc[
                        best_gt_index
                    ]
                )

                events.append(
                    {
                        "type":
                            "TP",

                        "case_id":
                            case_id,

                        "score":
                            float(
                                pred_row[
                                    "score"
                                ]
                            ),

                        "pred_z":
                            float(
                                pred_row[
                                    "z"
                                ]
                            ),

                        "pred_y":
                            float(
                                pred_row[
                                    "y"
                                ]
                            ),

                        "pred_x":
                            float(
                                pred_row[
                                    "x"
                                ]
                            ),

                        "pred_d":
                            float(
                                pred_row[
                                    "d"
                                ]
                            ),

                        "pred_h":
                            float(
                                pred_row[
                                    "h"
                                ]
                            ),

                        "pred_w":
                            float(
                                pred_row[
                                    "w"
                                ]
                            ),

                        "gt_id":
                            gt_row[
                                "gt_id"
                            ],

                        "gt_z":
                            float(
                                gt_row[
                                    "z"
                                ]
                            ),

                        "gt_y":
                            float(
                                gt_row[
                                    "y"
                                ]
                            ),

                        "gt_x":
                            float(
                                gt_row[
                                    "x"
                                ]
                            ),

                        "gt_d":
                            float(
                                gt_row[
                                    "d"
                                ]
                            ),

                        "gt_h":
                            float(
                                gt_row[
                                    "h"
                                ]
                            ),

                        "gt_w":
                            float(
                                gt_row[
                                    "w"
                                ]
                            ),

                        "gt_diameter_mm":
                            float(
                                gt_row[
                                    "diameter_mm"
                                ]
                            ),

                        "center_distance_mm":
                            best_distance,
                    }
                )

            # -----------------------------------------------------------------
            # FP
            # -----------------------------------------------------------------

            else:

                events.append(
                    {
                        "type":
                            "FP",

                        "case_id":
                            case_id,

                        "score":
                            float(
                                pred_row[
                                    "score"
                                ]
                            ),

                        "pred_z":
                            float(
                                pred_row[
                                    "z"
                                ]
                            ),

                        "pred_y":
                            float(
                                pred_row[
                                    "y"
                                ]
                            ),

                        "pred_x":
                            float(
                                pred_row[
                                    "x"
                                ]
                            ),

                        "pred_d":
                            float(
                                pred_row[
                                    "d"
                                ]
                            ),

                        "pred_h":
                            float(
                                pred_row[
                                    "h"
                                ]
                            ),

                        "pred_w":
                            float(
                                pred_row[
                                    "w"
                                ]
                            ),

                        "gt_id":
                            np.nan,

                        "gt_z":
                            np.nan,

                        "gt_y":
                            np.nan,

                        "gt_x":
                            np.nan,

                        "gt_d":
                            np.nan,

                        "gt_h":
                            np.nan,

                        "gt_w":
                            np.nan,

                        "gt_diameter_mm":
                            np.nan,

                        "center_distance_mm":
                            np.nan,
                    }
                )

        # =====================================================================
        # Unmatched GT -> FN
        # =====================================================================

        for gt_index, gt_row in (
            case_gt.iterrows()
        ):

            if (
                gt_index
                in matched_gt_indices
            ):

                continue

            events.append(
                {
                    "type":
                        "FN",

                    "case_id":
                        case_id,

                    "score":
                        np.nan,

                    "pred_z":
                        np.nan,

                    "pred_y":
                        np.nan,

                    "pred_x":
                        np.nan,

                    "pred_d":
                        np.nan,

                    "pred_h":
                        np.nan,

                    "pred_w":
                        np.nan,

                    "gt_id":
                        gt_row[
                            "gt_id"
                        ],

                    "gt_z":
                        float(
                            gt_row[
                                "z"
                            ]
                        ),

                    "gt_y":
                        float(
                            gt_row[
                                "y"
                            ]
                        ),

                    "gt_x":
                        float(
                            gt_row[
                                "x"
                            ]
                        ),

                    "gt_d":
                        float(
                            gt_row[
                                "d"
                            ]
                        ),

                    "gt_h":
                        float(
                            gt_row[
                                "h"
                            ]
                        ),

                    "gt_w":
                        float(
                            gt_row[
                                "w"
                            ]
                        ),

                    "gt_diameter_mm":
                        float(
                            gt_row[
                                "diameter_mm"
                            ]
                        ),

                    "center_distance_mm":
                        np.nan,
                }
            )

    event_df = pd.DataFrame(
        events
    )

    # -------------------------------------------------------------------------
    # Counts
    # -------------------------------------------------------------------------

    tp = int(
        (
            event_df[
                "type"
            ]
            == "TP"
        ).sum()
    )

    fp = int(
        (
            event_df[
                "type"
            ]
            == "FP"
        ).sum()
    )

    fn = int(
        (
            event_df[
                "type"
            ]
            == "FN"
        ).sum()
    )

    print()

    print(
        f"TP : {tp}"
    )

    print(
        f"FP : {fp}"
    )

    print(
        f"FN : {fn}"
    )

    print()

    print(
        "Expected @ Best-F1:"
    )

    print(
        f"TP / FP / FN = "
        f"{EXPECTED_TP} / "
        f"{EXPECTED_FP} / "
        f"{EXPECTED_FN}"
    )

    if (
        tp == EXPECTED_TP
        and
        fp == EXPECTED_FP
        and
        fn == EXPECTED_FN
    ):

        print()

        print(
            "[PASS] Matching reproduced"
        )

    else:

        raise RuntimeError(
            "\nEvaluator matching 결과와 다릅니다.\n"
            f"Current : {tp} / {fp} / {fn}\n"
            f"Expected: "
            f"{EXPECTED_TP} / "
            f"{EXPECTED_FP} / "
            f"{EXPECTED_FN}\n\n"
            "Visualization을 진행하지 않습니다."
        )

    return event_df


# =============================================================================
# Find CT
# =============================================================================

def find_ct(
    case_id
):

    case_id = normalize_case_id(
        case_id
    )

    direct_candidates = [
        VAL_ROOT
        / f"{case_id}.nii.gz",

        VAL_ROOT
        / f"{case_id}.nii",
    ]

    for path in direct_candidates:

        if path.exists():

            return path

    # -------------------------------------------------------------------------
    # Fallback recursive search
    # -------------------------------------------------------------------------

    matches = list(
        VAL_ROOT.rglob(
            f"{case_id}.nii.gz"
        )
    )

    if len(
        matches
    ) == 1:

        return matches[0]

    if len(
        matches
    ) > 1:

        raise RuntimeError(
            f"동일 UID CT가 여러 개 발견됨:\n"
            f"{case_id}\n"
            f"{matches}"
        )

    matches = list(
        VAL_ROOT.rglob(
            f"{case_id}.nii"
        )
    )

    if len(
        matches
    ) == 1:

        return matches[0]

    if len(
        matches
    ) > 1:

        raise RuntimeError(
            f"동일 UID CT가 여러 개 발견됨:\n"
            f"{case_id}\n"
            f"{matches}"
        )

    raise FileNotFoundError(
        f"CT 없음:\n"
        f"{case_id}\n\n"
        f"Search root:\n"
        f"{VAL_ROOT}"
    )


# =============================================================================
# Load CT
# =============================================================================

def load_ct(
    case_id
):

    ct_path = find_ct(
        case_id
    )

    image_itk = sitk.ReadImage(
        str(
            ct_path
        )
    )

    volume = (
        sitk
        .GetArrayFromImage(
            image_itk
        )
        .astype(
            np.float32
        )
    )

    spacing_xyz = np.asarray(
        image_itk.GetSpacing(),
        dtype=np.float64,
    )

    spacing_zyx = (
        spacing_xyz[
            ::-1
        ]
    )

    return (
        ct_path,
        volume,
        spacing_zyx,
    )


# =============================================================================
# Rectangle helper
# =============================================================================

def get_xy_rectangle(
    center_y,
    center_x,
    height,
    width,
):

    x0 = (
        center_x
        - width
        / 2.0
    )

    y0 = (
        center_y
        - height
        / 2.0
    )

    return (
        x0,
        y0,
        width,
        height,
    )


# =============================================================================
# Select examples
# =============================================================================

def select_examples(
    event_df
):

    selected = {}

    # =========================================================================
    # TP
    #
    # localization error 전체 범위를 골고루 보여주기 위해
    # distance 정렬 후 균등 sampling
    # =========================================================================

    tp = (
        event_df[
            event_df[
                "type"
            ]
            == "TP"
        ]
        .sort_values(
            "center_distance_mm",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    tp_count = min(
        NUM_EXAMPLES_PER_TYPE,
        len(
            tp
        ),
    )

    if tp_count > 0:

        tp_indices = np.linspace(
            0,
            len(
                tp
            )
            - 1,
            tp_count,
            dtype=int,
        )

        selected[
            "TP"
        ] = (
            tp.iloc[
                tp_indices
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    else:

        selected[
            "TP"
        ] = pd.DataFrame()

    # =========================================================================
    # FP
    #
    # 가장 confidence가 높은 FP가 중요.
    # =========================================================================

    selected[
        "FP"
    ] = (
        event_df[
            event_df[
                "type"
            ]
            == "FP"
        ]
        .sort_values(
            "score",
            ascending=False,
        )
        .head(
            NUM_EXAMPLES_PER_TYPE
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    # =========================================================================
    # FN
    #
    # 작은 결절 failure부터 확인
    # =========================================================================

    selected[
        "FN"
    ] = (
        event_df[
            event_df[
                "type"
            ]
            == "FN"
        ]
        .sort_values(
            "gt_diameter_mm",
            ascending=True,
        )
        .head(
            NUM_EXAMPLES_PER_TYPE
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    return selected


# =============================================================================
# Draw
# =============================================================================

def draw_example(
    row,
    rank,
):

    kind = str(
        row[
            "type"
        ]
    )

    case_id = str(
        row[
            "case_id"
        ]
    )

    ct_path, volume, spacing_zyx = (
        load_ct(
            case_id
        )
    )

    # =========================================================================
    # Choose visualization slice
    # =========================================================================

    if kind == "TP":

        # TP에서는 GT 중심 slice를 기준으로 시각화.
        #
        # GT box가 가장 정확하게 보이고,
        # prediction center도 같은 slice에서 비교 가능.
        slice_z = int(
            round(
                float(
                    row[
                        "gt_z"
                    ]
                )
            )
        )

    elif kind == "FP":

        slice_z = int(
            round(
                float(
                    row[
                        "pred_z"
                    ]
                )
            )
        )

    elif kind == "FN":

        slice_z = int(
            round(
                float(
                    row[
                        "gt_z"
                    ]
                )
            )
        )

    else:

        raise RuntimeError(
            f"Unknown event type: {kind}"
        )

    slice_z = max(
        0,
        min(
            slice_z,
            volume.shape[0]
            - 1,
        ),
    )

    image = (
        volume[
            slice_z
        ]
    )

    # =========================================================================
    # Display range
    # =========================================================================

    image_min = float(
        np.nanmin(
            image
        )
    )

    image_max = float(
        np.nanmax(
            image
        )
    )

    # HU volume이면 lung window 비슷한 범위 사용.
    if (
        image_min < -100
        or
        image_max > 10
    ):

        display_image = np.clip(
            image,
            HU_MIN,
            HU_MAX,
        )

        vmin = HU_MIN
        vmax = HU_MAX

    else:

        # 혹시 normalized volume인 경우에도
        # 검은 화면이 되지 않도록 실제 range 사용.
        display_image = image

        vmin = image_min
        vmax = image_max

    # =========================================================================
    # Figure
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(
            9,
            9,
        )
    )

    ax.imshow(
        display_image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    # =========================================================================
    # GT
    # =========================================================================

    if kind in (
        "TP",
        "FN",
    ):

        gt_y = float(
            row[
                "gt_y"
            ]
        )

        gt_x = float(
            row[
                "gt_x"
            ]
        )

        gt_h = float(
            row[
                "gt_h"
            ]
        )

        gt_w = float(
            row[
                "gt_w"
            ]
        )

        x0, y0, width, height = (
            get_xy_rectangle(
                center_y=gt_y,
                center_x=gt_x,
                height=gt_h,
                width=gt_w,
            )
        )

        gt_rectangle = Rectangle(
            (
                x0,
                y0,
            ),
            width,
            height,
            fill=False,
            edgecolor="lime",
            linewidth=2.5,
            linestyle="-",
        )

        ax.add_patch(
            gt_rectangle
        )

        ax.scatter(
            [
                gt_x
            ],
            [
                gt_y
            ],
            marker="+",
            s=120,
            linewidths=2,
            color="lime",
        )

        ax.text(
            x0,
            max(
                0,
                y0 - 7,
            ),
            "GT",
            color="lime",
            fontsize=11,
            fontweight="bold",
            bbox={
                "facecolor":
                    "black",

                "alpha":
                    0.65,

                "edgecolor":
                    "none",
            },
        )

    # =========================================================================
    # Prediction
    # =========================================================================

    if kind in (
        "TP",
        "FP",
    ):

        pred_y = float(
            row[
                "pred_y"
            ]
        )

        pred_x = float(
            row[
                "pred_x"
            ]
        )

        pred_h = float(
            row[
                "pred_h"
            ]
        )

        pred_w = float(
            row[
                "pred_w"
            ]
        )

        x0, y0, width, height = (
            get_xy_rectangle(
                center_y=pred_y,
                center_x=pred_x,
                height=pred_h,
                width=pred_w,
            )
        )

        pred_rectangle = Rectangle(
            (
                x0,
                y0,
            ),
            width,
            height,
            fill=False,
            edgecolor="red",
            linewidth=2.5,
            linestyle="--",
        )

        ax.add_patch(
            pred_rectangle
        )

        ax.scatter(
            [
                pred_x
            ],
            [
                pred_y
            ],
            marker="x",
            s=100,
            linewidths=2,
            color="red",
        )

        ax.text(
            x0,
            max(
                0,
                y0 - 7,
            ),
            (
                f"Pred "
                f"{float(row['score']):.4f}"
            ),
            color="red",
            fontsize=11,
            fontweight="bold",
            bbox={
                "facecolor":
                    "black",

                "alpha":
                    0.65,

                "edgecolor":
                    "none",
            },
        )

    # =========================================================================
    # Title
    # =========================================================================

    title = [
        f"{kind}",
        f"Case: {case_id}",
        f"Slice Z: {slice_z}",
    ]

    if kind == "TP":

        title.append(
            f"Score: "
            f"{float(row['score']):.6f}"
        )

        title.append(
            f"Center distance: "
            f"{float(row['center_distance_mm']):.3f} mm"
        )

        title.append(
            f"GT diameter: "
            f"{float(row['gt_diameter_mm']):.2f} mm"
        )

    elif kind == "FP":

        title.append(
            f"Score: "
            f"{float(row['score']):.6f}"
        )

    elif kind == "FN":

        title.append(
            f"GT diameter: "
            f"{float(row['gt_diameter_mm']):.2f} mm"
        )

    ax.set_title(
        "\n".join(
            title
        ),
        fontsize=11,
        pad=12,
    )

    # =========================================================================
    # Legend text
    # =========================================================================

    if kind == "TP":

        legend_text = (
            "GT = green solid\n"
            "Prediction = red dashed"
        )

    elif kind == "FP":

        legend_text = (
            "False Positive\n"
            "Prediction = red dashed"
        )

    else:

        legend_text = (
            "False Negative\n"
            "GT = green solid"
        )

    ax.text(
        0.02,
        0.02,
        legend_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "facecolor":
                "black",

            "alpha":
                0.6,

            "edgecolor":
                "none",
        },
        color="white",
    )

    ax.axis(
        "off"
    )

    fig.tight_layout()

    # =========================================================================
    # Save
    # =========================================================================

    kind_dir = (
        OUTPUT_DIR
        / kind.lower()
    )

    kind_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        kind_dir
        / (
            f"{rank:02d}_"
            f"{case_id}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        output_path,
        ct_path,
        slice_z,
        spacing_zyx,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    header(
        "19 - CPMNetv2 HN25 TP / FP / FN Visualization"
    )

    validate_inputs()

    print()

    print(
        "Final fixed configuration:"
    )

    print(
        "  Model      : CPMNetv2 + HN25"
    )

    print(
        "  Crop NMS   : 0.05"
    )

    print(
        "  Final NMS  : 0.05"
    )

    print(
        "  Best-F1 threshold:"
    )

    print(
        f"  {SCORE_THRESHOLD:.16f}"
    )

    # =========================================================================
    # Load
    # =========================================================================

    predictions, ground_truth = (
        load_csv()
    )

    validate_columns(
        predictions,
        ground_truth,
    )

    predictions, ground_truth = (
        normalize_case_ids(
            predictions,
            ground_truth,
        )
    )

    # =========================================================================
    # Matching
    # =========================================================================

    event_df = (
        build_detection_events(
            predictions,
            ground_truth,
        )
    )

    # 여기까지 PASS가 아니면 build_detection_events에서
    # RuntimeError로 중단됨.
    #
    # 즉 잘못된 matching으로 visualization이 생성되지 않음.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_df.to_csv(
        EVENTS_CSV,
        index=False,
    )

    print()

    print(
        "[PASS] Detection events saved:"
    )

    print(
        EVENTS_CSV
    )

    # =========================================================================
    # Select examples
    # =========================================================================

    selected = (
        select_examples(
            event_df
        )
    )

    header(
        "Selected Visualization Examples"
    )

    manifest_rows = []

    for kind in (
        "TP",
        "FP",
        "FN",
    ):

        examples = (
            selected[
                kind
            ]
        )

        print()

        print(
            f"{kind}: "
            f"{len(examples)} examples"
        )

        for rank, (
            _,
            row,
        ) in enumerate(
            examples.iterrows(),
            start=1,
        ):

            (
                image_path,
                ct_path,
                slice_z,
                spacing_zyx,
            ) = draw_example(
                row=row,
                rank=rank,
            )

            manifest_rows.append(
                {
                    "type":
                        kind,

                    "rank":
                        rank,

                    "case_id":
                        row[
                            "case_id"
                        ],

                    "slice_z":
                        slice_z,

                    "score":
                        (
                            float(
                                row[
                                    "score"
                                ]
                            )
                            if not pd.isna(
                                row[
                                    "score"
                                ]
                            )
                            else np.nan
                        ),

                    "gt_diameter_mm":
                        (
                            float(
                                row[
                                    "gt_diameter_mm"
                                ]
                            )
                            if not pd.isna(
                                row[
                                    "gt_diameter_mm"
                                ]
                            )
                            else np.nan
                        ),

                    "center_distance_mm":
                        (
                            float(
                                row[
                                    "center_distance_mm"
                                ]
                            )
                            if not pd.isna(
                                row[
                                    "center_distance_mm"
                                ]
                            )
                            else np.nan
                        ),

                    "spacing_z":
                        float(
                            spacing_zyx[
                                0
                            ]
                        ),

                    "spacing_y":
                        float(
                            spacing_zyx[
                                1
                            ]
                        ),

                    "spacing_x":
                        float(
                            spacing_zyx[
                                2
                            ]
                        ),

                    "ct_path":
                        str(
                            ct_path
                        ),

                    "image_path":
                        str(
                            image_path
                        ),
                }
            )

            print(
                f"  [{rank}] "
                f"{row['case_id']}"
            )

            print(
                f"      {image_path}"
            )

    # =========================================================================
    # Save manifest
    # =========================================================================

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
    )

    # =========================================================================
    # Final summary
    # =========================================================================

    header(
        "Visualization Finished"
    )

    tp_count = int(
        (
            event_df[
                "type"
            ]
            == "TP"
        ).sum()
    )

    fp_count = int(
        (
            event_df[
                "type"
            ]
            == "FP"
        ).sum()
    )

    fn_count = int(
        (
            event_df[
                "type"
            ]
            == "FN"
        ).sum()
    )

    print(
        "Matching QA:"
    )

    print(
        f"  TP = {tp_count}"
    )

    print(
        f"  FP = {fp_count}"
    )

    print(
        f"  FN = {fn_count}"
    )

    print()

    print(
        "Visualizations:"
    )

    print(
        f"  TP = "
        f"{len(selected['TP'])}"
    )

    print(
        f"  FP = "
        f"{len(selected['FP'])}"
    )

    print(
        f"  FN = "
        f"{len(selected['FN'])}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        EVENTS_CSV
    )

    print(
        MANIFEST_CSV
    )

    print()

    print(
        "TP:"
    )

    print(
        OUTPUT_DIR
        / "tp"
    )

    print()

    print(
        "FP:"
    )

    print(
        OUTPUT_DIR
        / "fp"
    )

    print()

    print(
        "FN:"
    )

    print(
        OUTPUT_DIR
        / "fn"
    )

    print()

    print(
        "[PASS] HN25 final validation visualization completed"
    )


if __name__ == "__main__":

    main()