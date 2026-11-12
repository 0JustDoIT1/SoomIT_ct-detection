#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np


# =============================================================================
# PATH
# =============================================================================

PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "evaluation"
    / "candidate2_fold0_predictions.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "evaluation"
)

MAP_OUTPUT_PATH = (
    OUTPUT_DIR
    / "candidate2_fold0_map.json"
)

PR_OUTPUT_PATH = (
    OUTPUT_DIR
    / "candidate2_fold0_pr_curve.csv"
)


# =============================================================================
# CPM
# =============================================================================

CPM_FP_RATES = [
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
]


# =============================================================================
# UTIL
# =============================================================================

def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def load_predictions():
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_PATH}"
        )

    with open(
        PREDICTION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(
            "prediction JSON 최상위 구조가 list가 아닙니다."
        )

    return data


def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return float(
        np.linalg.norm(a - b)
    )


# =============================================================================
# GT / PRED FLATTEN
# =============================================================================

def prepare_data(cases):
    """
    case 단위 JSON을 평가하기 쉬운 구조로 정리.
    """

    gt_by_case = {}
    predictions = []

    total_gt = 0

    for case_index, case in enumerate(cases):

        image = case.get("image", f"case_{case_index}")

        seriesuid = Path(image).stem

        gt_list = []

        for gt_local_index, gt in enumerate(
            case.get("gt_nodules", [])
        ):
            center = gt.get("center_xyz_mm")
            diameter = float(
                gt.get("diameter_mm", 0)
            )

            if center is None:
                continue

            gt_item = {
                "seriesuid": seriesuid,
                "gt_index": gt_local_index,
                "center_xyz_mm": [
                    float(x)
                    for x in center
                ],
                "diameter_mm": diameter,
                "radius_mm": diameter / 2.0,
            }

            gt_list.append(gt_item)
            total_gt += 1

        gt_by_case[seriesuid] = gt_list

        for pred_local_index, pred in enumerate(
            case.get("predictions", [])
        ):
            center = pred.get(
                "center_xyz_mm"
            )

            score = pred.get("score")

            if (
                center is None
                or score is None
            ):
                continue

            predictions.append(
                {
                    "seriesuid":
                        seriesuid,

                    "prediction_index":
                        pred_local_index,

                    "center_xyz_mm":
                        [
                            float(x)
                            for x in center
                        ],

                    "score":
                        float(score),

                    "box_xyzxyz":
                        pred.get(
                            "box_xyzxyz"
                        ),
                }
            )

    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True,
    )

    return (
        gt_by_case,
        predictions,
        total_gt,
    )


# =============================================================================
# LUNA CENTER MATCHING
# =============================================================================

def evaluate_ranked_predictions(
    gt_by_case,
    predictions,
    total_gt,
):
    """
    score가 높은 prediction부터 순서대로 평가.

    LUNA-style matching:

        distance(pred center, GT center)
        <= GT diameter / 2

    동일 GT는 최초 1회만 TP.
    그 이후 중복 prediction은 FP.
    """

    matched_gt = {
        uid: set()
        for uid in gt_by_case
    }

    tp_flags = []
    fp_flags = []

    events = []

    for rank, pred in enumerate(
        predictions,
        start=1,
    ):

        uid = pred["seriesuid"]

        case_gt = gt_by_case.get(
            uid,
            [],
        )

        best_gt_index = None
        best_distance = None

        # --------------------------------------------------
        # 아직 매칭되지 않은 GT 중
        # matching radius 안에 있는 가장 가까운 GT 선택
        # --------------------------------------------------

        for gt_index, gt in enumerate(
            case_gt
        ):

            if gt_index in matched_gt[uid]:
                continue

            distance = euclidean_distance(
                pred["center_xyz_mm"],
                gt["center_xyz_mm"],
            )

            allowed_radius = (
                gt["radius_mm"]
            )

            if distance <= allowed_radius:

                if (
                    best_distance is None
                    or distance < best_distance
                ):
                    best_distance = distance
                    best_gt_index = gt_index

        if best_gt_index is not None:

            matched_gt[uid].add(
                best_gt_index
            )

            is_tp = 1
            is_fp = 0

            matched_gt_item = (
                case_gt[
                    best_gt_index
                ]
            )

            matched_diameter = (
                matched_gt_item[
                    "diameter_mm"
                ]
            )

        else:

            is_tp = 0
            is_fp = 1

            matched_diameter = None

        tp_flags.append(is_tp)
        fp_flags.append(is_fp)

        events.append(
            {
                "rank": rank,
                "seriesuid": uid,
                "prediction_index":
                    pred[
                        "prediction_index"
                    ],
                "score":
                    pred["score"],
                "tp":
                    bool(is_tp),
                "fp":
                    bool(is_fp),
                "matched_gt_index":
                    best_gt_index,
                "distance_mm":
                    best_distance,
                "matched_gt_diameter_mm":
                    matched_diameter,
            }
        )

    tp_flags = np.asarray(
        tp_flags,
        dtype=float,
    )

    fp_flags = np.asarray(
        fp_flags,
        dtype=float,
    )

    cumulative_tp = np.cumsum(
        tp_flags
    )

    cumulative_fp = np.cumsum(
        fp_flags
    )

    precision = (
        cumulative_tp
        / np.maximum(
            cumulative_tp
            + cumulative_fp,
            1,
        )
    )

    recall = (
        cumulative_tp
        / max(
            total_gt,
            1,
        )
    )

    return {
        "tp_flags": tp_flags,
        "fp_flags": fp_flags,
        "cumulative_tp":
            cumulative_tp,
        "cumulative_fp":
            cumulative_fp,
        "precision":
            precision,
        "recall":
            recall,
        "events":
            events,
        "matched_gt":
            matched_gt,
    }


# =============================================================================
# AP
# =============================================================================

def calculate_ap(
    precision,
    recall,
):
    """
    Precision envelope 기반 area under PR curve.

    VOC/COCO 계열에서 흔히 사용하는
    monotonic precision envelope 방식.
    """

    if len(precision) == 0:
        return 0.0

    mrec = np.concatenate(
        (
            [0.0],
            recall,
            [1.0],
        )
    )

    mpre = np.concatenate(
        (
            [0.0],
            precision,
            [0.0],
        )
    )

    # precision envelope
    for i in range(
        len(mpre) - 2,
        -1,
        -1,
    ):
        mpre[i] = max(
            mpre[i],
            mpre[i + 1],
        )

    change_indices = np.where(
        mrec[1:]
        != mrec[:-1]
    )[0]

    ap = np.sum(
        (
            mrec[
                change_indices + 1
            ]
            - mrec[
                change_indices
            ]
        )
        * mpre[
            change_indices + 1
        ]
    )

    return float(ap)


# =============================================================================
# THRESHOLD METRICS
# =============================================================================

def evaluate_at_threshold(
    events,
    total_gt,
    threshold,
):
    selected = [
        e for e in events
        if e["score"] >= threshold
    ]

    tp = sum(
        1
        for e in selected
        if e["tp"]
    )

    fp = sum(
        1
        for e in selected
        if e["fp"]
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / total_gt
        if total_gt > 0
        else 0.0
    )

    return {
        "threshold":
            float(threshold),
        "predictions":
            len(selected),
        "tp":
            int(tp),
        "fp":
            int(fp),
        "precision":
            float(precision),
        "recall":
            float(recall),
    }


# =============================================================================
# FROC
# =============================================================================

def build_froc(
    evaluation,
    total_gt,
    num_scans,
):
    cumulative_tp = (
        evaluation[
            "cumulative_tp"
        ]
    )

    cumulative_fp = (
        evaluation[
            "cumulative_fp"
        ]
    )

    if len(cumulative_tp) == 0:
        return (
            np.array([0.0]),
            np.array([0.0]),
        )

    fp_per_scan = (
        cumulative_fp
        / max(
            num_scans,
            1,
        )
    )

    sensitivity = (
        cumulative_tp
        / max(
            total_gt,
            1,
        )
    )

    return (
        fp_per_scan,
        sensitivity,
    )


def sensitivity_at_fp_rate(
    fp_per_scan,
    sensitivity,
    target_fp,
):
    """
    target FP/scan 이하에서 달성 가능한
    최대 sensitivity.

    보간보다 diagnostic 목적에 명확하게
    operating-point 방식으로 계산.
    """

    valid = (
        fp_per_scan
        <= target_fp
    )

    if not np.any(valid):
        return 0.0

    return float(
        np.max(
            sensitivity[valid]
        )
    )


# =============================================================================
# SIZE-BASED RECALL
# =============================================================================

def size_group(
    diameter,
):
    if diameter < 6.0:
        return "<6 mm"

    if diameter < 10.0:
        return "6-10 mm"

    return ">=10 mm"


def calculate_size_recall(
    gt_by_case,
    matched_gt,
):
    stats = {
        "<6 mm": {
            "gt": 0,
            "detected": 0,
        },
        "6-10 mm": {
            "gt": 0,
            "detected": 0,
        },
        ">=10 mm": {
            "gt": 0,
            "detected": 0,
        },
    }

    for uid, gt_list in (
        gt_by_case.items()
    ):

        matched_indices = (
            matched_gt.get(
                uid,
                set(),
            )
        )

        for gt_index, gt in enumerate(
            gt_list
        ):

            group = size_group(
                gt["diameter_mm"]
            )

            stats[group]["gt"] += 1

            if (
                gt_index
                in matched_indices
            ):
                stats[group][
                    "detected"
                ] += 1

    for group in stats:

        gt_count = (
            stats[group]["gt"]
        )

        detected = (
            stats[group][
                "detected"
            ]
        )

        recall = (
            detected / gt_count
            if gt_count > 0
            else 0.0
        )

        stats[group][
            "recall"
        ] = float(recall)

    return stats


# =============================================================================
# SAVE
# =============================================================================

def save_pr_curve(
    predictions,
    evaluation,
):
    with open(
        PR_OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "score",
                "tp",
                "fp",
                "cumulative_tp",
                "cumulative_fp",
                "precision",
                "recall",
            ]
        )

        for i, pred in enumerate(
            predictions
        ):

            writer.writerow(
                [
                    i + 1,
                    pred["score"],
                    int(
                        evaluation[
                            "tp_flags"
                        ][i]
                    ),
                    int(
                        evaluation[
                            "fp_flags"
                        ][i]
                    ),
                    int(
                        evaluation[
                            "cumulative_tp"
                        ][i]
                    ),
                    int(
                        evaluation[
                            "cumulative_fp"
                        ][i]
                    ),
                    float(
                        evaluation[
                            "precision"
                        ][i]
                    ),
                    float(
                        evaluation[
                            "recall"
                        ][i]
                    ),
                ]
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "Candidate 2 Detection Evaluation"
    )

    cases = load_predictions()

    (
        gt_by_case,
        predictions,
        total_gt,
    ) = prepare_data(cases)

    num_scans = len(cases)

    total_predictions = len(
        predictions
    )

    print(
        f"Scans       : {num_scans}"
    )

    print(
        f"GT nodules  : {total_gt}"
    )

    print(
        f"Predictions : {total_predictions}"
    )

    if predictions:

        scores = np.asarray(
            [
                p["score"]
                for p in predictions
            ],
            dtype=float,
        )

        print()
        print(
            f"Score min   : "
            f"{scores.min():.6f}"
        )

        print(
            f"Score max   : "
            f"{scores.max():.6f}"
        )

        print(
            f"Score mean  : "
            f"{scores.mean():.6f}"
        )

        print(
            f"Score median: "
            f"{np.median(scores):.6f}"
        )

    # -------------------------------------------------------------------------
    # Ranked evaluation
    # -------------------------------------------------------------------------

    evaluation = (
        evaluate_ranked_predictions(
            gt_by_case,
            predictions,
            total_gt,
        )
    )

    precision = (
        evaluation["precision"]
    )

    recall = (
        evaluation["recall"]
    )

    ap = calculate_ap(
        precision,
        recall,
    )

    # Single-class detection
    map_value = ap

    total_tp = int(
        evaluation[
            "tp_flags"
        ].sum()
    )

    total_fp = int(
        evaluation[
            "fp_flags"
        ].sum()
    )

    max_recall = (
        total_tp / total_gt
        if total_gt > 0
        else 0.0
    )

    print_header(
        "AP / Recall"
    )

    print(
        "Matching rule:"
    )

    print(
        "distance(pred center, GT center) "
        "<= GT diameter / 2"
    )

    print()
    print(
        f"True Positives : {total_tp}"
    )

    print(
        f"False Positives: {total_fp}"
    )

    print(
        f"Maximum Recall : "
        f"{max_recall:.6f}"
    )

    print(
        f"Center-match AP: "
        f"{ap:.6f}"
    )

    print(
        f"mAP            : "
        f"{map_value:.6f}"
    )

    print(
        "(single class이므로 mAP = AP)"
    )

    # -------------------------------------------------------------------------
    # Size recall
    # -------------------------------------------------------------------------

    size_stats = (
        calculate_size_recall(
            gt_by_case,
            evaluation[
                "matched_gt"
            ],
        )
    )

    print_header(
        "Size-based Recall"
    )

    for group in [
        "<6 mm",
        "6-10 mm",
        ">=10 mm",
    ]:

        s = size_stats[group]

        print(
            f"{group:<10} | "
            f"GT {s['gt']:4d} | "
            f"Detected {s['detected']:4d} | "
            f"Recall {s['recall']:.4f}"
        )

    # -------------------------------------------------------------------------
    # Threshold diagnostic
    # -------------------------------------------------------------------------

    threshold_candidates = [
        0.30,
        0.25,
        0.23,
        0.22,
        0.20,
        0.15,
        0.10,
        0.05,
        0.01,
        0.00,
    ]

    threshold_results = []

    print_header(
        "Score Threshold Diagnostic"
    )

    print(
        f"{'Thr':>6} "
        f"{'Pred':>8} "
        f"{'TP':>6} "
        f"{'FP':>8} "
        f"{'Precision':>12} "
        f"{'Recall':>10}"
    )

    print("-" * 60)

    for threshold in (
        threshold_candidates
    ):

        result = (
            evaluate_at_threshold(
                evaluation[
                    "events"
                ],
                total_gt,
                threshold,
            )
        )

        threshold_results.append(
            result
        )

        print(
            f"{threshold:6.2f} "
            f"{result['predictions']:8d} "
            f"{result['tp']:6d} "
            f"{result['fp']:8d} "
            f"{result['precision']:12.6f} "
            f"{result['recall']:10.6f}"
        )

    # -------------------------------------------------------------------------
    # FROC / CPM independently recomputed
    # -------------------------------------------------------------------------

    (
        fp_per_scan,
        sensitivity,
    ) = build_froc(
        evaluation,
        total_gt,
        num_scans,
    )

    sensitivities_at_cpm = []

    for fp_rate in CPM_FP_RATES:

        sens = (
            sensitivity_at_fp_rate(
                fp_per_scan,
                sensitivity,
                fp_rate,
            )
        )

        sensitivities_at_cpm.append(
            sens
        )

    cpm = float(
        np.mean(
            sensitivities_at_cpm
        )
    )

    print_header(
        "Independent FROC / CPM"
    )

    for fp_rate, sens in zip(
        CPM_FP_RATES,
        sensitivities_at_cpm,
    ):
        print(
            f"Sensitivity @ "
            f"{fp_rate:>5g} FP/scan "
            f": {sens:.6f}"
        )

    print()
    print(
        f"Sensitivity @ 1 FP/scan : "
        f"{sensitivities_at_cpm[3]:.6f}"
    )

    print(
        f"Sensitivity @ 4 FP/scan : "
        f"{sensitivities_at_cpm[5]:.6f}"
    )

    print(
        f"CPM                     : "
        f"{cpm:.6f}"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_pr_curve(
        predictions,
        evaluation,
    )

    output = {
        "evaluation_type":
            "LUNA16 center-distance matching",

        "matching_rule":
            "distance(pred, GT) <= GT diameter / 2",

        "num_scans":
            num_scans,

        "total_gt_nodules":
            total_gt,

        "total_predictions":
            total_predictions,

        "true_positives":
            total_tp,

        "false_positives":
            total_fp,

        "max_recall":
            float(max_recall),

        "center_match_ap":
            float(ap),

        "map":
            float(map_value),

        "map_note":
            (
                "Single-class center-match AP. "
                "This is not IoU-based COCO mAP."
            ),

        "size_based_recall":
            size_stats,

        "threshold_diagnostic":
            threshold_results,

        "cpm_fp_rates":
            CPM_FP_RATES,

        "sensitivities_at_cpm_rates":
            [
                float(x)
                for x in
                sensitivities_at_cpm
            ],

        "sensitivity_at_1_fp_per_scan":
            float(
                sensitivities_at_cpm[3]
            ),

        "sensitivity_at_4_fp_per_scan":
            float(
                sensitivities_at_cpm[5]
            ),

        "cpm":
            cpm,
    }

    with open(
        MAP_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_header(
        "Saved"
    )

    print(
        MAP_OUTPUT_PATH
    )

    print(
        PR_OUTPUT_PATH
    )

    print()
    print(
        "Candidate 2 evaluation completed."
    )


if __name__ == "__main__":
    main()