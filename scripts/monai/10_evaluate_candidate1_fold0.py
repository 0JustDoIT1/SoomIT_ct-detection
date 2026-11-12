from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys

import numpy as np


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

INFERENCE_CONFIG = (
    BUNDLE_ROOT
    / "configs"
    / "inference.json"
)

METADATA_CONFIG = (
    BUNDLE_ROOT
    / "configs"
    / "metadata.json"
)

MODEL_PATH = (
    BUNDLE_ROOT
    / "models"
    / "model.pt"
)

FOLD0_PATH = (
    BUNDLE_ROOT
    / "LUNA16_datasplit"
    / "mhd_original"
    / "dataset_fold0.json"
)


# ============================================================
# Output
# ============================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "candidate1_fold0"
)

INFERENCE_OUTPUT_DIR = (
    OUTPUT_ROOT
    / "inference"
)

EVAL_DATALIST_PATH = (
    OUTPUT_ROOT
    / "dataset_fold0_validation_absolute.json"
)

RUN_CONFIG_PATH = (
    OUTPUT_ROOT
    / "inference_candidate1_fold0.json"
)

PREDICTION_PATH = (
    INFERENCE_OUTPUT_DIR
    / "candidate1_fold0_predictions.json"
)

FROC_CSV_PATH = (
    OUTPUT_ROOT
    / "froc_curve.csv"
)

CPM_CSV_PATH = (
    OUTPUT_ROOT
    / "cpm.csv"
)

METRICS_JSON_PATH = (
    OUTPUT_ROOT
    / "metrics.json"
)

PREDICTION_CSV_PATH = (
    OUTPUT_ROOT
    / "predictions.csv"
)


# ============================================================
# LUNA16 CPM operating points
# ============================================================

CPM_FP_RATES = np.array(
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


# ============================================================
# Utility
# ============================================================

def get_uid_from_image(
    image_value,
):

    name = Path(
        str(image_value)
    ).name

    if name.endswith(
        ".mhd"
    ):
        return name[:-4]

    if name.endswith(
        ".nii.gz"
    ):
        return name[:-7]

    return Path(
        name
    ).stem


def find_mhd(
    seriesuid,
):

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
            f"중복 MHD 발견: {seriesuid}"
        )

    return matches[0]


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
# Prepare evaluation datalist
# ============================================================

def prepare_eval_datalist():

    with open(
        FOLD0_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        fold0 = json.load(
            f
        )


    validation = fold0.get(
        "validation",
        []
    )


    if len(validation) != 67:

        raise RuntimeError(
            f"fold0 validation 수 이상: {len(validation)}"
        )


    eval_validation = []


    for item in validation:

        uid = get_uid_from_image(
            item["image"]
        )

        mhd_path = find_mhd(
            uid
        )

        raw_path = (
            mhd_path.with_suffix(
                ".raw"
            )
        )


        if not raw_path.exists():

            raise FileNotFoundError(
                raw_path
            )


        eval_validation.append(
            {
                "image": str(
                    mhd_path.resolve()
                ),
                "box": item.get(
                    "box",
                    [],
                ),
                "label": item.get(
                    "label",
                    [],
                ),
            }
        )


    dataset = {
        "training": [],
        "validation": eval_validation,
    }


    with open(
        EVAL_DATALIST_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
        )


    return validation


# ============================================================
# Prepare MONAI inference config
# ============================================================

def prepare_inference_config():

    with open(
        INFERENCE_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(
            f
        )


    config[
        "bundle_root"
    ] = str(
        BUNDLE_ROOT
    )

    config[
        "whether_raw_luna16"
    ] = True

    config[
        "data_list_file_path"
    ] = str(
        EVAL_DATALIST_PATH
    )

    # image path를 absolute path로 넣었으므로
    # base_dir는 root로 둔다.
    config[
        "dataset_dir"
    ] = "/"

    config[
        "output_dir"
    ] = str(
        INFERENCE_OUTPUT_DIR
    )

    config[
        "output_filename"
    ] = (
        "candidate1_fold0_predictions.json"
    )

    config[
        "load_pretrain"
    ] = True


    with open(
        RUN_CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )


# ============================================================
# Run inference
# ============================================================

def run_inference():

    print()
    print(
        "[3] Candidate 1 full inference"
    )

    print()

    command = [
        sys.executable,
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        str(
            RUN_CONFIG_PATH
        ),
    ]


    # metadata 경고 방지
    if METADATA_CONFIG.exists():

        command.extend(
            [
                "--meta_file",
                str(
                    METADATA_CONFIG
                ),
            ]
        )


    print(
        "COMMAND:"
    )

    print(
        " ".join(
            command
        )
    )

    print()


    result = subprocess.run(
        command,
        cwd=str(
            BUNDLE_ROOT
        ),
    )


    if result.returncode != 0:

        raise RuntimeError(
            f"MONAI inference 실패: {result.returncode}"
        )


    if not PREDICTION_PATH.exists():

        raise FileNotFoundError(
            f"Prediction JSON 생성 실패: {PREDICTION_PATH}"
        )


# ============================================================
# Load GT
# ============================================================

def load_gt():

    with open(
        FOLD0_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        fold0 = json.load(
            f
        )


    gt_by_uid = {}


    for item in fold0[
        "validation"
    ]:

        uid = get_uid_from_image(
            item["image"]
        )

        boxes = item.get(
            "box",
            [],
        )


        gt_by_uid[
            uid
        ] = []


        for box in boxes:

            if len(
                box
            ) != 6:

                raise RuntimeError(
                    f"GT box 형식 이상: {uid}"
                )


            values = [
                float(
                    value
                )
                for value in box
            ]


            if not all(
                math.isfinite(
                    value
                )
                for value in values
            ):

                raise RuntimeError(
                    f"GT NaN/Inf: {uid}"
                )


            cx, cy, cz, w, h, d = values


            # LUNA16 annotation에서는
            # width / height / depth가 동일한 diameter 기반.
            diameter = (
                w + h + d
            ) / 3.0


            gt_by_uid[
                uid
            ].append(
                {
                    "center": (
                        cx,
                        cy,
                        cz,
                    ),
                    "diameter": diameter,
                    "box": values,
                }
            )


    return gt_by_uid


# ============================================================
# Load predictions
# ============================================================

def load_predictions():

    with open(
        PREDICTION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        predictions = json.load(
            f
        )


    if not isinstance(
        predictions,
        list,
    ):

        raise RuntimeError(
            "Prediction JSON은 list여야 합니다."
        )


    pred_by_uid = {}


    for case in predictions:

        uid = get_uid_from_image(
            case.get(
                "image",
                ""
            )
        )

        boxes = case.get(
            "box",
            []
        )

        scores = case.get(
            "label_scores",
            case.get(
                "score",
                [],
            ),
        )


        if len(
            boxes
        ) != len(
            scores
        ):

            raise RuntimeError(
                f"box/score 개수 불일치: {uid}"
            )


        case_predictions = []


        for box, score in zip(
            boxes,
            scores,
        ):

            if len(
                box
            ) != 6:

                raise RuntimeError(
                    f"Prediction box 형식 이상: {uid}"
                )


            box_values = [
                float(
                    value
                )
                for value in box
            ]

            score = float(
                score
            )


            if not all(
                math.isfinite(
                    value
                )
                for value in box_values
            ):

                raise RuntimeError(
                    f"Prediction box NaN/Inf: {uid}"
                )


            if not math.isfinite(
                score
            ):

                raise RuntimeError(
                    f"Prediction score NaN/Inf: {uid}"
                )


            case_predictions.append(
                {
                    "center": (
                        box_values[0],
                        box_values[1],
                        box_values[2],
                    ),
                    "box": box_values,
                    "score": score,
                }
            )


        pred_by_uid[
            uid
        ] = case_predictions


    return pred_by_uid


# ============================================================
# Match predictions to GT
#
# LUNA16 hit criterion:
#
# distance(pred center, GT center)
# < GT diameter / 2
#
# 동일 결절을 여러 prediction이 검출한 경우:
# 가장 높은 score 하나만 TP candidate로 사용하고
# 나머지 duplicate detection은 FROC 계산에서 ignore.
# ============================================================

def classify_candidates(
    gt_by_uid,
    pred_by_uid,
):

    candidate_records = []

    total_gt = sum(
        len(
            nodules
        )
        for nodules in gt_by_uid.values()
    )


    detected_gt_candidates = 0
    ignored_duplicates = 0
    unmatched_candidates = 0


    for uid, gt_nodules in gt_by_uid.items():

        predictions = pred_by_uid.get(
            uid,
            []
        )


        # ----------------------------------------------------
        # 각 prediction이 어떤 GT 내부에 들어오는지 계산
        # ----------------------------------------------------

        candidate_gt_match = {}


        for pred_idx, pred in enumerate(
            predictions
        ):

            matched = []


            for gt_idx, gt in enumerate(
                gt_nodules
            ):

                distance = center_distance(
                    pred[
                        "center"
                    ],
                    gt[
                        "center"
                    ],
                )

                radius = (
                    gt[
                        "diameter"
                    ]
                    / 2.0
                )


                if distance < radius:

                    matched.append(
                        (
                            gt_idx,
                            distance,
                        )
                    )


            if matched:

                # 하나의 candidate가 여러 GT에 걸칠 경우
                # 가장 가까운 GT에 배정.
                matched.sort(
                    key=lambda item: item[1]
                )

                candidate_gt_match[
                    pred_idx
                ] = matched[0][0]


        # ----------------------------------------------------
        # 각 GT마다 가장 높은 score prediction 선택
        # ----------------------------------------------------

        representative = {}


        for pred_idx, gt_idx in candidate_gt_match.items():

            score = predictions[
                pred_idx
            ][
                "score"
            ]


            if (
                gt_idx not in representative
                or score
                > predictions[
                    representative[
                        gt_idx
                    ]
                ][
                    "score"
                ]
            ):

                representative[
                    gt_idx
                ] = pred_idx


        representative_indices = set(
            representative.values()
        )


        detected_gt_candidates += len(
            representative_indices
        )


        # ----------------------------------------------------
        # Candidate classification
        # ----------------------------------------------------

        for pred_idx, pred in enumerate(
            predictions
        ):

            if pred_idx in representative_indices:

                status = "TP"

                gt_idx = candidate_gt_match[
                    pred_idx
                ]

                distance = center_distance(
                    pred[
                        "center"
                    ],
                    gt_nodules[
                        gt_idx
                    ][
                        "center"
                    ],
                )


            elif pred_idx in candidate_gt_match:

                # 동일 nodule을 여러 번 검출한 candidate
                status = "IGNORE_DUPLICATE"

                gt_idx = candidate_gt_match[
                    pred_idx
                ]

                distance = center_distance(
                    pred[
                        "center"
                    ],
                    gt_nodules[
                        gt_idx
                    ][
                        "center"
                    ],
                )

                ignored_duplicates += 1


            else:

                status = "FP"

                gt_idx = None
                distance = None

                unmatched_candidates += 1


            candidate_records.append(
                {
                    "seriesuid": uid,
                    "score": pred[
                        "score"
                    ],
                    "status": status,
                    "gt_index": gt_idx,
                    "center_distance": distance,
                    "box": pred[
                        "box"
                    ],
                }
            )


    return {
        "records": candidate_records,
        "total_gt": total_gt,
        "detected_gt_candidates": detected_gt_candidates,
        "ignored_duplicates": ignored_duplicates,
        "unmatched_candidates": unmatched_candidates,
    }


# ============================================================
# Build FROC
# ============================================================

def build_froc(
    candidate_records,
    total_gt,
    num_scans,
):

    relevant = [
        record
        for record in candidate_records
        if record[
            "status"
        ] in (
            "TP",
            "FP",
        )
    ]


    scores = sorted(
        {
            float(
                record[
                    "score"
                ]
            )
            for record in relevant
        },
        reverse=True,
    )


    points = [
        {
            "threshold": float(
                "inf"
            ),
            "fp_per_scan": 0.0,
            "sensitivity": 0.0,
            "tp": 0,
            "fp": 0,
        }
    ]


    for threshold in scores:

        tp = sum(
            1
            for record in relevant
            if (
                record[
                    "status"
                ] == "TP"
                and record[
                    "score"
                ] >= threshold
            )
        )

        fp = sum(
            1
            for record in relevant
            if (
                record[
                    "status"
                ] == "FP"
                and record[
                    "score"
                ] >= threshold
            )
        )


        sensitivity = (
            tp / total_gt
            if total_gt > 0
            else 0.0
        )

        fp_per_scan = (
            fp / num_scans
        )


        points.append(
            {
                "threshold": threshold,
                "fp_per_scan": fp_per_scan,
                "sensitivity": sensitivity,
                "tp": tp,
                "fp": fp,
            }
        )


    # --------------------------------------------------------
    # 동일 FP/scan 값에서는 최대 sensitivity만 유지
    # --------------------------------------------------------

    collapsed = {}


    for point in points:

        fp_rate = point[
            "fp_per_scan"
        ]

        if (
            fp_rate not in collapsed
            or point[
                "sensitivity"
            ]
            > collapsed[
                fp_rate
            ][
                "sensitivity"
            ]
        ):

            collapsed[
                fp_rate
            ] = point


    curve = [
        collapsed[
            key
        ]
        for key in sorted(
            collapsed.keys()
        )
    ]


    return curve


# ============================================================
# CPM interpolation
# ============================================================

def calculate_cpm(
    curve,
):

    x = np.array(
        [
            point[
                "fp_per_scan"
            ]
            for point in curve
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            point[
                "sensitivity"
            ]
            for point in curve
        ],
        dtype=np.float64,
    )


    sensitivities = np.interp(
        CPM_FP_RATES,
        x,
        y,
        left=0.0,
        right=y[-1],
    )


    cpm = float(
        np.mean(
            sensitivities
        )
    )


    return sensitivities, cpm


# ============================================================
# Save results
# ============================================================

def save_results(
    match_result,
    curve,
    cpm_sensitivities,
    cpm,
    num_scans,
):

    # --------------------------------------------------------
    # Prediction CSV
    # --------------------------------------------------------

    with open(
        PREDICTION_CSV_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "seriesuid",
                "coordX",
                "coordY",
                "coordZ",
                "width",
                "height",
                "depth",
                "score",
                "status",
                "gt_index",
                "center_distance_mm",
            ]
        )


        for record in match_result[
            "records"
        ]:

            box = record[
                "box"
            ]

            writer.writerow(
                [
                    record[
                        "seriesuid"
                    ],
                    *box,
                    record[
                        "score"
                    ],
                    record[
                        "status"
                    ],
                    record[
                        "gt_index"
                    ],
                    record[
                        "center_distance"
                    ],
                ]
            )


    # --------------------------------------------------------
    # FROC CSV
    # --------------------------------------------------------

    with open(
        FROC_CSV_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "threshold",
                "fp_per_scan",
                "sensitivity",
                "tp",
                "fp",
            ]
        )


        for point in curve:

            writer.writerow(
                [
                    point[
                        "threshold"
                    ],
                    point[
                        "fp_per_scan"
                    ],
                    point[
                        "sensitivity"
                    ],
                    point[
                        "tp"
                    ],
                    point[
                        "fp"
                    ],
                ]
            )


    # --------------------------------------------------------
    # CPM CSV
    # --------------------------------------------------------

    with open(
        CPM_CSV_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "fp_per_scan",
                "sensitivity",
            ]
        )


        for fp_rate, sensitivity in zip(
            CPM_FP_RATES,
            cpm_sensitivities,
        ):

            writer.writerow(
                [
                    float(
                        fp_rate
                    ),
                    float(
                        sensitivity
                    ),
                ]
            )


    # --------------------------------------------------------
    # Metrics JSON
    # --------------------------------------------------------

    metrics = {
        "candidate": "MONAI official pretrained RetinaNet",
        "num_scans": num_scans,
        "total_gt_nodules": match_result[
            "total_gt"
        ],
        "detected_gt_at_score_0_02_or_higher": match_result[
            "detected_gt_candidates"
        ],
        "unmatched_candidates": match_result[
            "unmatched_candidates"
        ],
        "ignored_duplicate_candidates": match_result[
            "ignored_duplicates"
        ],
        "cpm_fp_rates": [
            float(
                value
            )
            for value in CPM_FP_RATES
        ],
        "sensitivities": [
            float(
                value
            )
            for value in cpm_sensitivities
        ],
        "cpm": cpm,
        "note": (
            "Fold0 validation 67 scans evaluation. "
            "Uses LUNA16 center-distance hit criterion. "
            "This is the common project test set, not the "
            "official LUNA16 challenge leaderboard evaluation."
        ),
    }


    with open(
        METRICS_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "기존 prediction JSON이 있어도 "
            "inference를 다시 실행"
        ),
    )

    args = parser.parse_args()


    print(
        "=" * 80
    )

    print(
        "Candidate 1 - Fold0 Validation Evaluation"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    print()
    print(
        "[1] Required files"
    )

    required = [
        MODEL_PATH,
        INFERENCE_CONFIG,
        FOLD0_PATH,
    ]


    for path in required:

        print(
            path
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    print(
        "✅ Required files 확인"
    )


    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    INFERENCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Prepare datalist
    # --------------------------------------------------------

    print()
    print(
        "[2] Prepare fold0 validation"
    )

    validation = (
        prepare_eval_datalist()
    )

    print(
        "Validation CT:",
        len(
            validation
        ),
    )

    print(
        "GT nodules:",
        sum(
            len(
                item.get(
                    "box",
                    []
                )
            )
            for item in validation
        ),
    )

    print(
        "✅ 67 CT / 112 GT 준비"
    )


    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    prepare_inference_config()


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    if (
        args.rerun
        or not PREDICTION_PATH.exists()
    ):

        run_inference()

    else:

        print()
        print(
            "[3] Existing prediction 사용"
        )

        print(
            PREDICTION_PATH
        )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print(
        "[4] Load GT / Predictions"
    )

    gt_by_uid = load_gt()

    pred_by_uid = (
        load_predictions()
    )


    gt_uids = set(
        gt_by_uid
    )

    pred_uids = set(
        pred_by_uid
    )


    missing_predictions = (
        gt_uids
        - pred_uids
    )


    extra_predictions = (
        pred_uids
        - gt_uids
    )


    print(
        "GT cases   :",
        len(
            gt_uids
        ),
    )

    print(
        "Pred cases :",
        len(
            pred_uids
        ),
    )

    print(
        "Missing    :",
        len(
            missing_predictions
        ),
    )

    print(
        "Extra      :",
        len(
            extra_predictions
        ),
    )


    if missing_predictions:

        raise RuntimeError(
            "Inference가 생성되지 않은 validation CT가 있습니다."
        )


    if extra_predictions:

        raise RuntimeError(
            "fold0 validation 이외 prediction이 존재합니다."
        )


    # --------------------------------------------------------
    # Candidate matching
    # --------------------------------------------------------

    print()
    print(
        "[5] LUNA16 center-distance matching"
    )

    match_result = classify_candidates(
        gt_by_uid,
        pred_by_uid,
    )


    print(
        "GT nodules          :",
        match_result[
            "total_gt"
        ],
    )

    print(
        "Detected GT         :",
        match_result[
            "detected_gt_candidates"
        ],
    )

    print(
        "Unmatched candidate :",
        match_result[
            "unmatched_candidates"
        ],
    )

    print(
        "Ignored duplicates  :",
        match_result[
            "ignored_duplicates"
        ],
    )


    # --------------------------------------------------------
    # FROC
    # --------------------------------------------------------

    print()
    print(
        "[6] FROC / CPM"
    )

    num_scans = len(
        gt_by_uid
    )


    curve = build_froc(
        match_result[
            "records"
        ],
        match_result[
            "total_gt"
        ],
        num_scans,
    )


    cpm_sensitivities, cpm = (
        calculate_cpm(
            curve
        )
    )


    print()

    print(
        "FP/scan      Sensitivity"
    )

    print(
        "-" * 32
    )


    for fp_rate, sensitivity in zip(
        CPM_FP_RATES,
        cpm_sensitivities,
    ):

        print(
            f"{fp_rate:7.3f}      "
            f"{sensitivity:.6f}"
        )


    print()
    print(
        f"CPM: {cpm:.6f}"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        match_result,
        curve,
        cpm_sensitivities,
        cpm,
        num_scans,
    )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "✅ Candidate 1 Fold0 Evaluation 완료"
    )

    print()
    print(
        "Prediction:"
    )
    print(
        PREDICTION_PATH
    )

    print()
    print(
        "FROC:"
    )
    print(
        FROC_CSV_PATH
    )

    print()
    print(
        "CPM:"
    )
    print(
        CPM_CSV_PATH
    )

    print()
    print(
        "Metrics:"
    )
    print(
        METRICS_JSON_PATH
    )

    print()
    print(
        "Per-candidate results:"
    )
    print(
        PREDICTION_CSV_PATH
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()