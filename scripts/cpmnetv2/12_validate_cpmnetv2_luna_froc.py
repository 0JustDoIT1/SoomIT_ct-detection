#!/usr/bin/env python3

"""
CPMNetv2 LUNA-style FROC / CPM Cross Validation

Purpose
-------
11_evaluate_cpmnetv2_full.py에서 생성한 full-volume predictions.csv를
다시 inference하지 않고 재사용하여 LUNA16 공식 evaluator의 핵심 방식에
가깝게 FROC / CPM을 교차검증한다.

Main differences from script 11
-------------------------------
11:
    GT d/h/w 기반 ellipsoid center matching

12:
    LUNA-style spherical center matching
    distance(pred_center, GT_center) < diameter_mm / 2

Evaluation
----------
- TP / FP / FN
- FROC
- Sensitivity @
    0.125
    0.25
    0.5
    1
    2
    4
    8 FP/scan
- CPM
- Previous custom CPM comparison
- Matching QA

No GPU inference is performed.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

EVAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "full_evaluation"
)

PREDICTIONS_CSV = (
    EVAL_ROOT
    / "predictions.csv"
)

GROUND_TRUTH_CSV = (
    EVAL_ROOT
    / "ground_truth.csv"
)

PREVIOUS_SUMMARY_JSON = (
    EVAL_ROOT
    / "metrics_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "luna_froc_validation"
)

FROC_CURVE_CSV = (
    OUTPUT_DIR
    / "luna_style_froc_curve.csv"
)

OPERATING_POINTS_CSV = (
    OUTPUT_DIR
    / "luna_style_operating_points.csv"
)

MATCH_CSV = (
    OUTPUT_DIR
    / "luna_style_matches.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "luna_style_summary.json"
)

FROC_PNG = (
    OUTPUT_DIR
    / "luna_style_froc.png"
)


# =============================================================================
# CONFIG
# =============================================================================

NUM_VALIDATION_SCANS = 88

FROC_POINTS = np.asarray(
    [
        0.125,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ],
    dtype=np.float64,
)


# =============================================================================
# UTILITY
# =============================================================================

def header(title):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File 없음:\n{path}"
        )


def safe_float(value):

    return float(
        value
    )


# =============================================================================
# LOAD
# =============================================================================

def load_data():

    check_file(
        PREDICTIONS_CSV
    )

    check_file(
        GROUND_TRUTH_CSV
    )

    predictions = pd.read_csv(
        PREDICTIONS_CSV
    )

    gt = pd.read_csv(
        GROUND_TRUTH_CSV
    )

    required_pred = [
        "case_id",
        "score",
        "z",
        "y",
        "x",
        "d",
        "h",
        "w",
    ]

    required_gt = [
        "gt_id",
        "case_id",
        "z",
        "y",
        "x",
        "d",
        "h",
        "w",
        "diameter_mm",
        "size_group",
    ]

    missing_pred = [
        column
        for column in required_pred
        if column not in predictions.columns
    ]

    missing_gt = [
        column
        for column in required_gt
        if column not in gt.columns
    ]

    if missing_pred:

        raise RuntimeError(
            f"Prediction columns missing: "
            f"{missing_pred}"
        )

    if missing_gt:

        raise RuntimeError(
            f"GT columns missing: "
            f"{missing_gt}"
        )

    predictions[
        "case_id"
    ] = (
        predictions[
            "case_id"
        ]
        .astype(str)
    )

    gt[
        "case_id"
    ] = (
        gt[
            "case_id"
        ]
        .astype(str)
    )

    return (
        predictions,
        gt,
    )


# =============================================================================
# GT QA
# =============================================================================

def gt_geometry_qa(gt):

    header(
        "GT Geometry QA"
    )

    d = gt[
        "d"
    ].to_numpy(
        dtype=float
    )

    h = gt[
        "h"
    ].to_numpy(
        dtype=float
    )

    w = gt[
        "w"
    ].to_numpy(
        dtype=float
    )

    diameter = gt[
        "diameter_mm"
    ].to_numpy(
        dtype=float
    )

    geometric_mean = (
        np.maximum(
            d * h * w,
            0.0,
        )
        ** (
            1.0 / 3.0
        )
    )

    diameter_diff = np.abs(
        geometric_mean
        - diameter
    )

    axis_range = (
        np.max(
            np.stack(
                [
                    d,
                    h,
                    w,
                ],
                axis=1,
            ),
            axis=1,
        )
        - np.min(
            np.stack(
                [
                    d,
                    h,
                    w,
                ],
                axis=1,
            ),
            axis=1,
        )
    )

    print(
        "GT nodules              :",
        len(
            gt
        ),
    )

    print(
        "Diameter recompute diff :"
    )

    print(
        f"  max  = "
        f"{diameter_diff.max():.8f}"
    )

    print(
        f"  mean = "
        f"{diameter_diff.mean():.8f}"
    )

    print()

    print(
        "GT box axis difference:"
    )

    print(
        f"  mean = "
        f"{axis_range.mean():.4f} mm"
    )

    print(
        f"  max  = "
        f"{axis_range.max():.4f} mm"
    )

    near_spherical = int(
        np.sum(
            axis_range
            < 1e-3
        )
    )

    print(
        f"  d≈h≈w cases = "
        f"{near_spherical}/{len(gt)}"
    )

    if np.max(
        diameter_diff
    ) < 1e-6:

        print()
        print(
            "[PASS] diameter_mm calculation consistent"
        )

    else:

        print()
        print(
            "[WARN] diameter_mm mismatch"
        )


# =============================================================================
# DISTANCE
# =============================================================================

def center_distance(
    pred,
    gt,
):

    dz = (
        safe_float(
            pred[
                "z"
            ]
        )
        - safe_float(
            gt[
                "z"
            ]
        )
    )

    dy = (
        safe_float(
            pred[
                "y"
            ]
        )
        - safe_float(
            gt[
                "y"
            ]
        )
    )

    dx = (
        safe_float(
            pred[
                "x"
            ]
        )
        - safe_float(
            gt[
                "x"
            ]
        )
    )

    # spacing = 1 mm isotropic
    return math.sqrt(
        dz * dz
        + dy * dy
        + dx * dx
    )


# =============================================================================
# LUNA-STYLE MATCHING
# =============================================================================

def build_luna_samples(
    predictions,
    gt,
):

    """
    LUNA16 evaluator 핵심 아이디어:

    각 GT nodule:
        distance < diameter/2 인 candidate 탐색

    여러 candidate가 하나의 GT에 들어오면:
        가장 confidence가 높은 candidate만 TP로 사용

    나머지 unmatched candidate:
        FP

    검출되지 않은 GT:
        FN이며 score = -inf에 해당하는 positive sample로 처리
    """

    header(
        "LUNA-style Center Matching"
    )

    pred_groups = {
        case_id:
            group.copy()

        for case_id, group
        in predictions.groupby(
            "case_id"
        )
    }

    gt_groups = {
        case_id:
            group.copy()

        for case_id, group
        in gt.groupby(
            "case_id"
        )
    }

    froc_labels = []
    froc_scores = []
    froc_case_ids = []

    match_rows = []

    total_tp = 0
    total_fn = 0
    total_fp = 0

    duplicate_candidates_ignored = 0

    all_cases = sorted(
        set(
            predictions[
                "case_id"
            ].astype(str)
        )
        | set(
            gt[
                "case_id"
            ].astype(str)
        )
    )

    for case_id in all_cases:

        pred_case = (
            pred_groups.get(
                case_id,
                pd.DataFrame(
                    columns=predictions.columns
                ),
            )
            .sort_values(
                "score",
                ascending=False,
            )
            .copy()
        )

        gt_case = (
            gt_groups.get(
                case_id,
                pd.DataFrame(
                    columns=gt.columns
                ),
            )
            .copy()
        )

        # unmatched candidate index
        available_pred_indices = set(
            pred_case.index.tolist()
        )

        case_tp = 0
        case_fn = 0

        for _, gt_row in gt_case.iterrows():

            diameter = float(
                gt_row[
                    "diameter_mm"
                ]
            )

            radius = (
                diameter
                / 2.0
            )

            matches = []

            for pred_idx, pred_row in pred_case.iterrows():

                distance = (
                    center_distance(
                        pred_row,
                        gt_row,
                    )
                )

                # Original LUNA evaluator uses dist < radius^2
                if distance < radius:

                    matches.append(
                        (
                            pred_idx,
                            float(
                                pred_row[
                                    "score"
                                ]
                            ),
                            distance,
                        )
                    )

            if matches:

                matches = sorted(
                    matches,
                    key=lambda item: item[1],
                    reverse=True,
                )

                best_pred_idx = (
                    matches[
                        0
                    ][
                        0
                    ]
                )

                best_score = (
                    matches[
                        0
                    ][
                        1
                    ]
                )

                best_distance = (
                    matches[
                        0
                    ][
                        2
                    ]
                )

                # all predictions within the same GT are not counted as FP
                # in LUNA evaluator.
                for (
                    pred_idx,
                    _,
                    _,
                ) in matches:

                    if (
                        pred_idx
                        in available_pred_indices
                    ):

                        available_pred_indices.remove(
                            pred_idx
                        )

                duplicate_candidates_ignored += max(
                    len(
                        matches
                    )
                    - 1,
                    0,
                )

                froc_labels.append(
                    1
                )

                froc_scores.append(
                    best_score
                )

                froc_case_ids.append(
                    case_id
                )

                total_tp += 1
                case_tp += 1

                match_rows.append(
                    {
                        "case_id":
                            case_id,

                        "gt_id":
                            int(
                                gt_row[
                                    "gt_id"
                                ]
                            ),

                        "diameter_mm":
                            diameter,

                        "status":
                            "TP",

                        "score":
                            best_score,

                        "distance_mm":
                            best_distance,

                        "num_candidates_in_gt":
                            len(
                                matches
                            ),

                        "prediction_index":
                            int(
                                best_pred_idx
                            ),
                    }
                )

            else:

                # For FROC curve this is effectively a positive with
                # confidence below every real prediction.
                froc_labels.append(
                    1
                )

                froc_scores.append(
                    -1.0
                )

                froc_case_ids.append(
                    case_id
                )

                total_fn += 1
                case_fn += 1

                match_rows.append(
                    {
                        "case_id":
                            case_id,

                        "gt_id":
                            int(
                                gt_row[
                                    "gt_id"
                                ]
                            ),

                        "diameter_mm":
                            diameter,

                        "status":
                            "FN",

                        "score":
                            -1.0,

                        "distance_mm":
                            None,

                        "num_candidates_in_gt":
                            0,

                        "prediction_index":
                            None,
                    }
                )

        case_fp = len(
            available_pred_indices
        )

        total_fp += (
            case_fp
        )

        for pred_idx in (
            available_pred_indices
        ):

            pred_row = pred_case.loc[
                pred_idx
            ]

            froc_labels.append(
                0
            )

            froc_scores.append(
                float(
                    pred_row[
                        "score"
                    ]
                )
            )

            froc_case_ids.append(
                case_id
            )

        print(
            f"{case_id} | "
            f"GT={len(gt_case):2d} "
            f"TP={case_tp:2d} "
            f"FN={case_fn:2d} "
            f"FP candidates={case_fp:3d}"
        )

    print()
    print(
        "Raw matching summary"
    )

    print(
        "TP:",
        total_tp,
    )

    print(
        "FN:",
        total_fn,
    )

    print(
        "FP candidates:",
        total_fp,
    )

    print(
        "Ignored duplicate candidates:",
        duplicate_candidates_ignored,
    )

    return {
        "labels":
            np.asarray(
                froc_labels,
                dtype=np.int8,
            ),

        "scores":
            np.asarray(
                froc_scores,
                dtype=np.float64,
            ),

        "case_ids":
            np.asarray(
                froc_case_ids,
                dtype=object,
            ),

        "matches":
            pd.DataFrame(
                match_rows
            ),

        "tp":
            int(
                total_tp
            ),

        "fn":
            int(
                total_fn
            ),

        "fp":
            int(
                total_fp
            ),

        "duplicate_candidates_ignored":
            int(
                duplicate_candidates_ignored
            ),
    }


# =============================================================================
# FROC
# =============================================================================

def calculate_froc(
    labels,
    scores,
    num_scans,
):

    """
    Threshold를 confidence score 순으로 낮추면서:

        sensitivity = TP / total GT
        FP/scan     = FP / scans

    계산한다.
    """

    total_gt = int(
        np.sum(
            labels
            == 1
        )
    )

    # FN pseudo samples have score -1.
    # Real predictions should be >= 0.
    real_mask = (
        scores
        >= 0.0
    )

    real_labels = labels[
        real_mask
    ]

    real_scores = scores[
        real_mask
    ]

    order = np.argsort(
        -real_scores
    )

    real_scores = real_scores[
        order
    ]

    real_labels = real_labels[
        order
    ]

    rows = [
        {
            "threshold":
                float(
                    np.nextafter(
                        real_scores.max(),
                        np.inf,
                    )
                    if len(
                        real_scores
                    ) > 0
                    else 1.0
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

    tp = 0
    fp = 0

    index = 0

    while index < len(
        real_scores
    ):

        current_score = (
            real_scores[
                index
            ]
        )

        while (
            index
            < len(
                real_scores
            )
            and real_scores[
                index
            ]
            == current_score
        ):

            if (
                real_labels[
                    index
                ]
                == 1
            ):

                tp += 1

            else:

                fp += 1

            index += 1

        rows.append(
            {
                "threshold":
                    float(
                        current_score
                    ),

                "fp_per_scan":
                    float(
                        fp
                        / num_scans
                    ),

                "sensitivity":
                    float(
                        tp
                        / total_gt
                    ),

                "tp":
                    int(
                        tp
                    ),

                "fp":
                    int(
                        fp
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# OFFICIAL-STYLE INTERPOLATION
# =============================================================================

def interpolate_operating_points(
    froc_df,
):

    """
    Original LUNA evaluator는 FROC curve를 만들고
    target FP/scan 위치에서 sensitivity를 interpolation한다.

    np.interp 방식 사용.
    """

    fps = (
        froc_df[
            "fp_per_scan"
        ]
        .to_numpy(
            dtype=float
        )
    )

    sensitivity = (
        froc_df[
            "sensitivity"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # same FP/scan value가 여러 번 존재할 수 있으므로
    # 각 FP 값에서 최대 sensitivity 유지
    compact = (
        froc_df
        .groupby(
            "fp_per_scan",
            as_index=False,
        )[
            "sensitivity"
        ]
        .max()
        .sort_values(
            "fp_per_scan"
        )
    )

    fps_unique = (
        compact[
            "fp_per_scan"
        ]
        .to_numpy(
            dtype=float
        )
    )

    sens_unique = (
        compact[
            "sensitivity"
        ]
        .to_numpy(
            dtype=float
        )
    )

    sensitivities = np.interp(
        FROC_POINTS,
        fps_unique,
        sens_unique,
        left=(
            sens_unique[
                0
            ]
            if len(
                sens_unique
            )
            else 0.0
        ),
        right=(
            sens_unique[
                -1
            ]
            if len(
                sens_unique
            )
            else 0.0
        ),
    )

    rows = []

    for fp_target, sens in zip(
        FROC_POINTS,
        sensitivities,
    ):

        rows.append(
            {
                "fp_per_scan":
                    float(
                        fp_target
                    ),

                "sensitivity":
                    float(
                        sens
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PREVIOUS RESULT
# =============================================================================

def load_previous_cpm():

    if not PREVIOUS_SUMMARY_JSON.exists():

        return None

    with open(
        PREVIOUS_SUMMARY_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    try:

        return float(
            data[
                "froc"
            ][
                "cpm"
            ]
        )

    except Exception:

        return None


# =============================================================================
# PLOT
# =============================================================================

def save_froc_plot(
    froc_df,
    operating_df,
):

    positive_fps = (
        froc_df[
            froc_df[
                "fp_per_scan"
            ]
            > 0
        ]
    )

    fig = plt.figure(
        figsize=(
            8,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    if len(
        positive_fps
    ) > 0:

        ax.plot(
            positive_fps[
                "fp_per_scan"
            ],
            positive_fps[
                "sensitivity"
            ],
        )

    ax.scatter(
        operating_df[
            "fp_per_scan"
        ],
        operating_df[
            "sensitivity"
        ],
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        0.125,
        8.0,
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xticks(
        FROC_POINTS
    )

    ax.set_xticklabels(
        [
            "0.125",
            "0.25",
            "0.5",
            "1",
            "2",
            "4",
            "8",
        ]
    )

    ax.set_xlabel(
        "False Positives / Scan"
    )

    ax.set_ylabel(
        "Sensitivity"
    )

    ax.set_title(
        "CPMNetv2 LUNA-style FROC"
    )

    ax.grid(
        True,
        which="both",
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


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "CPMNetv2 LUNA-style FROC / CPM Cross Validation"
    )

    print(
        "Predictions:",
        PREDICTIONS_CSV,
    )

    print(
        "GT:",
        GROUND_TRUTH_CSV,
    )

    print(
        "Validation scans:",
        NUM_VALIDATION_SCANS,
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    predictions, gt = (
        load_data()
    )

    header(
        "Dataset"
    )

    print(
        "Predictions:",
        len(
            predictions
        ),
    )

    print(
        "GT nodules:",
        len(
            gt
        ),
    )

    print(
        "Prediction scans:",
        predictions[
            "case_id"
        ].nunique(),
    )

    print(
        "GT positive scans:",
        gt[
            "case_id"
        ].nunique(),
    )

    if len(
        gt
    ) != 105:

        print(
            "[WARN] Expected 105 GT nodules"
        )

    if predictions[
        "case_id"
    ].nunique() != NUM_VALIDATION_SCANS:

        print(
            "[WARN] Prediction scan count != 88"
        )

    # -------------------------------------------------------------------------
    # GT QA
    # -------------------------------------------------------------------------

    gt_geometry_qa(
        gt
    )

    # -------------------------------------------------------------------------
    # Match
    # -------------------------------------------------------------------------

    result = (
        build_luna_samples(
            predictions,
            gt,
        )
    )

    # -------------------------------------------------------------------------
    # FROC
    # -------------------------------------------------------------------------

    froc_df = (
        calculate_froc(
            labels=result[
                "labels"
            ],

            scores=result[
                "scores"
            ],

            num_scans=(
                NUM_VALIDATION_SCANS
            ),
        )
    )

    operating_df = (
        interpolate_operating_points(
            froc_df
        )
    )

    cpm = float(
        operating_df[
            "sensitivity"
        ].mean()
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    header(
        "LUNA-style FROC"
    )

    for _, row in operating_df.iterrows():

        print(
            f"FP/scan "
            f"{row['fp_per_scan']:>5.3f}"
            f" | "
            f"Sensitivity "
            f"{row['sensitivity']:.6f}"
        )

    print()
    print(
        f"CPM : {cpm:.6f}"
    )

    previous_cpm = (
        load_previous_cpm()
    )

    header(
        "Cross-check"
    )

    if previous_cpm is not None:

        difference = (
            cpm
            - previous_cpm
        )

        print(
            f"11 Custom CPM    : "
            f"{previous_cpm:.6f}"
        )

        print(
            f"12 LUNA-style CPM: "
            f"{cpm:.6f}"
        )

        print(
            f"Difference       : "
            f"{difference:+.6f}"
        )

        if abs(
            difference
        ) < 0.02:

            print()
            print(
                "[PASS] CPM values are closely consistent."
            )

        elif abs(
            difference
        ) < 0.05:

            print()
            print(
                "[CHECK] Small evaluator difference exists."
            )

        else:

            print()
            print(
                "[WARN] Large CPM difference."
            )

            print(
                "Matching definition / interpolation / "
                "candidate handling을 확인해야 합니다."
            )

    else:

        difference = None

        print(
            "Previous CPM not available."
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    froc_df.to_csv(
        FROC_CURVE_CSV,
        index=False,
    )

    operating_df.to_csv(
        OPERATING_POINTS_CSV,
        index=False,
    )

    result[
        "matches"
    ].to_csv(
        MATCH_CSV,
        index=False,
    )

    save_froc_plot(
        froc_df,
        operating_df,
    )

    summary = {
        "evaluation":
            "LUNA-style spherical center matching",

        "num_scans":
            NUM_VALIDATION_SCANS,

        "num_gt_nodules":
            int(
                len(
                    gt
                )
            ),

        "num_predictions":
            int(
                len(
                    predictions
                )
            ),

        "raw_matching": {
            "tp":
                result[
                    "tp"
                ],

            "fn":
                result[
                    "fn"
                ],

            "fp_candidates":
                result[
                    "fp"
                ],

            "duplicate_candidates_ignored":
                result[
                    "duplicate_candidates_ignored"
                ],
        },

        "operating_points":
            operating_df.to_dict(
                orient="records"
            ),

        "cpm":
            cpm,

        "previous_custom_cpm":
            previous_cpm,

        "difference":
            difference,
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

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    header(
        "FINAL"
    )

    print(
        f"LUNA-style CPM : "
        f"{cpm:.6f}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        FROC_CURVE_CSV
    )

    print(
        OPERATING_POINTS_CSV
    )

    print(
        MATCH_CSV
    )

    print(
        SUMMARY_JSON
    )

    print(
        FROC_PNG
    )

    print()

    print(
        "Cross validation completed."
    )


if __name__ == "__main__":

    main()