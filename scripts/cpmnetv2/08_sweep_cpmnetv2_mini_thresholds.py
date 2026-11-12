#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


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

OUTPUT_JSON = (
    MINI_ROOT
    / "threshold_sweep.json"
)

OUTPUT_CSV = (
    MINI_ROOT
    / "threshold_sweep.csv"
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

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]

# -------------------------------------------------------------------------
# 기존 07번은 NMS_TOPK=40이어서 모든 case가 Pred=40으로 잘렸음.
# 이번에는 threshold 영향 자체를 보기 위해 충분히 크게 둔다.
# -------------------------------------------------------------------------

POSTPROCESS_TOPK = 100
POSTPROCESS_NUM_TOPK = 50

NMS_THRESHOLD = 0.05
NMS_TOPK = 200

INFERENCE_BATCH_SIZE = 8

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
    print("=" * 110)
    print(title)
    print("=" * 110)


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

    return image.astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# Mini cases
# =============================================================================

def load_positive_cases():

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            TRAIN_CSV
        )

    df = pd.read_csv(
        TRAIN_CSV
    )

    df = df[
        df["label"]
        .astype(str)
        == "nodule"
    ].copy()

    grouped = {
        str(uid): group.copy()
        for uid, group
        in df.groupby(
            "seriesuid"
        )
    }

    ct_paths = sorted(
        IMAGE_ROOT.glob(
            "*.nii.gz"
        )
    )

    cases = []

    for ct_path in ct_paths:

        possible_keys = [
            ct_path.name,
            strip_nii_suffix(
                ct_path.name
            ),
        ]

        annotations = None

        for key in possible_keys:

            if key in grouped:
                annotations = grouped[
                    key
                ]
                break

        if annotations is None:
            continue

        cases.append(
            {
                "path":
                    ct_path,

                "annotations":
                    annotations,
            }
        )

        if len(
            cases
        ) >= NUM_CASES:
            break

    if len(
        cases
    ) != NUM_CASES:

        raise RuntimeError(
            f"Expected {NUM_CASES} cases, "
            f"found {len(cases)}"
        )

    return cases


# =============================================================================
# GT
# =============================================================================

def build_gt_zyx(
    annotations,
):

    gts = []

    for _, row in (
        annotations.iterrows()
    ):

        center = np.asarray(
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
        )

        size = np.asarray(
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
        )

        diameter = float(
            np.max(
                size
            )
        )

        gts.append(
            {
                "center_zyx":
                    center,

                "size_dhw":
                    size,

                "diameter":
                    diameter,

                "radius":
                    diameter
                    / 2.0,
            }
        )

    return gts


# =============================================================================
# Model
# =============================================================================

def build_model():

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            CHECKPOINT
        )

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
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Epoch     :",
        checkpoint.get(
            "epoch"
        ),
    )

    print(
        "Metrics   :",
        checkpoint.get(
            "metrics"
        ),
    )

    return model


# =============================================================================
# Load CT
# =============================================================================

def load_ct(
    ct_path,
):

    image_itk = sitk.ReadImage(
        str(
            ct_path
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
        SPACING,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"Spacing mismatch:\n"
            f"{ct_path}\n"
            f"{spacing_zyx}"
        )

    image = (
        sitk.GetArrayFromImage(
            image_itk
        )
        .astype(
            np.float32
        )
    )

    return image


# =============================================================================
# Inference
# =============================================================================

def infer_volume(
    model,
    image_zyx,
    score_threshold,
):

    split_comber = SplitComb(
        crop_size=CROP_SIZE,
        overlap=OVERLAP_SIZE,
        pad_value=-1,
    )

    postprocess = (
        Detection_Postprocess(
            topk=POSTPROCESS_TOPK,

            threshold=(
                score_threshold
            ),

            nms_threshold=(
                NMS_THRESHOLD
            ),

            num_topk=(
                POSTPROCESS_NUM_TOPK
            ),

            crop_size=CROP_SIZE,
        )
    )

    image = normalize_ct(
        image_zyx
    )

    # CPMNetv2 input
    # [0,1] → [-1,1]
    image = (
        image
        * 2.0
        - 1.0
    )

    split_images, nzhw = (
        split_comber.split(
            image
        )
    )

    split_images = (
        torch.from_numpy(
            split_images
        )
        .float()
    )

    outputs = []

    for start in range(
        0,
        split_images.shape[0],
        INFERENCE_BATCH_SIZE,
    ):

        end = min(
            start
            + INFERENCE_BATCH_SIZE,
            split_images.shape[0],
        )

        batch = (
            split_images[
                start:end
            ]
            .to(
                DEVICE
            )
        )

        with torch.no_grad():

            raw_output = model(
                batch
            )

            processed = (
                postprocess(
                    raw_output,
                    device=DEVICE,
                )
            )

        outputs.append(
            processed
            .detach()
            .cpu()
            .numpy()
        )

    if not outputs:

        return np.zeros(
            (
                0,
                7,
            ),
            dtype=np.float32,
        )

    outputs = np.concatenate(
        outputs,
        axis=0,
    )

    outputs = (
        split_comber.combine(
            outputs,
            nzhw=nzhw,
        )
    )

    outputs = (
        torch.from_numpy(
            outputs
        )
        .view(
            -1,
            8,
        )
    )

    # CPMNetv2 invalid detection marker
    valid = (
        outputs[
            :,
            0
        ]
        != -1.0
    )

    outputs = outputs[
        valid
    ]

    if len(
        outputs
    ) == 0:

        return np.zeros(
            (
                0,
                7,
            ),
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # CPMNetv2 output
    #
    # [class, prob, z, y, x, d, h, w]
    #
    # nms_3D input:
    # [prob, z, y, x, d, h, w]
    # ---------------------------------------------------------

    nms_input = (
        outputs[
            :,
            1:
        ]
    )

    keep = nms_3D(
        nms_input,
        overlap=NMS_THRESHOLD,
        top_k=NMS_TOPK,
    )

    outputs = (
        outputs[
            keep
        ]
        .detach()
        .cpu()
        .numpy()
    )

    # class 제거
    predictions = (
        outputs[
            :,
            1:
        ]
    )

    if len(
        predictions
    ) > 0:

        order = np.argsort(
            -predictions[
                :,
                0
            ]
        )

        predictions = (
            predictions[
                order
            ]
        )

    return predictions.astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# One-to-one matching
# =============================================================================

def match_predictions(
    gts,
    predictions,
):

    """
    LUNA-style center matching.

    prediction:
        [score, z, y, x, d, h, w]

    조건:
        distance(pred center, GT center)
        <= GT radius

    prediction은 score 높은 순서로 처리하며
    하나의 GT는 한 번만 TP로 인정.
    """

    matched_gt = set()

    tp = 0
    fp = 0

    matched_distances = []

    events = []

    for pred_index, prediction in enumerate(
        predictions
    ):

        score = float(
            prediction[
                0
            ]
        )

        pred_center = np.asarray(
            prediction[
                1:4
            ],
            dtype=np.float32,
        )

        best_gt = None
        best_distance = None

        for gt_index, gt in enumerate(
            gts
        ):

            if gt_index in matched_gt:
                continue

            distance = float(
                np.linalg.norm(
                    pred_center
                    - gt[
                        "center_zyx"
                    ]
                )
            )

            if (
                distance
                <= gt["radius"]
            ):

                if (
                    best_distance
                    is None
                    or distance
                    < best_distance
                ):

                    best_gt = (
                        gt_index
                    )

                    best_distance = (
                        distance
                    )

        if best_gt is not None:

            matched_gt.add(
                best_gt
            )

            tp += 1

            matched_distances.append(
                best_distance
            )

            events.append(
                {
                    "prediction_index":
                        pred_index,

                    "score":
                        score,

                    "tp":
                        True,

                    "matched_gt":
                        best_gt,

                    "distance":
                        best_distance,
                }
            )

        else:

            fp += 1

            events.append(
                {
                    "prediction_index":
                        pred_index,

                    "score":
                        score,

                    "tp":
                        False,

                    "matched_gt":
                        None,

                    "distance":
                        None,
                }
            )

    fn = (
        len(gts)
        - len(
            matched_gt
        )
    )

    return {
        "tp":
            int(
                tp
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "matched_gt":
            len(
                matched_gt
            ),

        "matched_distances":
            matched_distances,

        "events":
            events,
    }


# =============================================================================
# Threshold evaluation
# =============================================================================

def evaluate_threshold(
    model,
    cases,
    threshold,
):

    total_gt = 0
    total_pred = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_distances = []

    case_results = []

    start_time = time.time()

    for case_index, case in enumerate(
        cases,
        start=1,
    ):

        image = load_ct(
            case[
                "path"
            ]
        )

        gts = build_gt_zyx(
            case[
                "annotations"
            ]
        )

        predictions = (
            infer_volume(
                model,
                image,
                threshold,
            )
        )

        match = (
            match_predictions(
                gts,
                predictions,
            )
        )

        total_gt += len(
            gts
        )

        total_pred += len(
            predictions
        )

        total_tp += (
            match[
                "tp"
            ]
        )

        total_fp += (
            match[
                "fp"
            ]
        )

        total_fn += (
            match[
                "fn"
            ]
        )

        all_distances.extend(
            match[
                "matched_distances"
            ]
        )

        max_score = (
            float(
                predictions[
                    :,
                    0
                ].max()
            )
            if len(
                predictions
            )
            else None
        )

        case_results.append(
            {
                "case":
                    case_index,

                "image":
                    case[
                        "path"
                    ].name,

                "gt":
                    len(
                        gts
                    ),

                "predictions":
                    int(
                        len(
                            predictions
                        )
                    ),

                "tp":
                    match[
                        "tp"
                    ],

                "fp":
                    match[
                        "fp"
                    ],

                "fn":
                    match[
                        "fn"
                    ],

                "max_score":
                    max_score,
            }
        )

    recall = (
        total_tp
        / total_gt
        if total_gt > 0
        else 0.0
    )

    precision = (
        total_tp
        / (
            total_tp
            + total_fp
        )
        if (
            total_tp
            + total_fp
        ) > 0
        else 0.0
    )

    fp_per_scan = (
        total_fp
        / len(
            cases
        )
        if cases
        else 0.0
    )

    mean_preds = (
        total_pred
        / len(
            cases
        )
        if cases
        else 0.0
    )

    median_distance = (
        float(
            np.median(
                all_distances
            )
        )
        if all_distances
        else None
    )

    elapsed = (
        time.time()
        - start_time
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "cases":
            len(
                cases
            ),

        "gt":
            int(
                total_gt
            ),

        "predictions":
            int(
                total_pred
            ),

        "mean_predictions_per_scan":
            float(
                mean_preds
            ),

        "tp":
            int(
                total_tp
            ),

        "fp":
            int(
                total_fp
            ),

        "fn":
            int(
                total_fn
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "fp_per_scan":
            float(
                fp_per_scan
            ),

        "median_matched_distance":
            median_distance,

        "seconds":
            float(
                elapsed
            ),

        "case_results":
            case_results,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    header(
        "CPMNetv2 Mini Threshold Sweep"
    )

    print(
        "Device      :",
        DEVICE,
    )

    print(
        "Cases       :",
        NUM_CASES,
    )

    print(
        "Thresholds  :",
        THRESHOLDS,
    )

    print(
        "NMS top-k   :",
        NMS_TOPK,
    )

    print(
        "NMS IoU     :",
        NMS_THRESHOLD,
    )

    if DEVICE.type != "cuda":

        raise RuntimeError(
            "CUDA required."
        )

    cases = (
        load_positive_cases()
    )

    print()
    print(
        "Selected cases:"
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            f"[{index}] "
            f"{case['path'].name} "
            f"| GT="
            f"{len(case['annotations'])}"
        )

    model = (
        build_model()
    )

    results = []

    header(
        "Threshold Sweep"
    )

    print(
        f"{'Thr':>6} "
        f"{'Pred':>6} "
        f"{'Pred/scan':>10} "
        f"{'TP':>5} "
        f"{'FP':>6} "
        f"{'FN':>5} "
        f"{'Precision':>11} "
        f"{'Recall':>9} "
        f"{'FP/scan':>9} "
        f"{'Dist':>8}"
    )

    print(
        "-" * 100
    )

    for threshold in (
        THRESHOLDS
    ):

        result = (
            evaluate_threshold(
                model,
                cases,
                threshold,
            )
        )

        results.append(
            result
        )

        distance_text = (
            f"{result['median_matched_distance']:.2f}"
            if result[
                "median_matched_distance"
            ]
            is not None
            else "NA"
        )

        print(
            f"{threshold:6.2f} "
            f"{result['predictions']:6d} "
            f"{result['mean_predictions_per_scan']:10.1f} "
            f"{result['tp']:5d} "
            f"{result['fp']:6d} "
            f"{result['fn']:5d} "
            f"{result['precision']:11.4f} "
            f"{result['recall']:9.4f} "
            f"{result['fp_per_scan']:9.2f} "
            f"{distance_text:>8}"
        )

    # -------------------------------------------------------------------------
    # Select useful diagnostic threshold
    # -------------------------------------------------------------------------

    max_recall = max(
        r[
            "recall"
        ]
        for r
        in results
    )

    same_recall = [
        r
        for r
        in results
        if np.isclose(
            r[
                "recall"
            ],
            max_recall,
        )
    ]

    # 최대 Recall을 유지하면서 FP가 가장 작은 threshold
    recommended = min(
        same_recall,
        key=lambda x: (
            x[
                "fp_per_scan"
            ],
            -x[
                "threshold"
            ],
        ),
    )

    header(
        "Diagnostic Recommendation"
    )

    print(
        f"Maximum mini Recall : "
        f"{max_recall:.4f}"
    )

    print(
        f"Recommended threshold: "
        f"{recommended['threshold']:.2f}"
    )

    print(
        f"TP / FP / FN         : "
        f"{recommended['tp']} / "
        f"{recommended['fp']} / "
        f"{recommended['fn']}"
    )

    print(
        f"Recall               : "
        f"{recommended['recall']:.4f}"
    )

    print(
        f"Precision            : "
        f"{recommended['precision']:.4f}"
    )

    print(
        f"FP/scan              : "
        f"{recommended['fp_per_scan']:.2f}"
    )

    print()
    print(
        "주의: 이 threshold는 5개 mini-overfit case에서만 "
        "선택한 QA용 값입니다."
    )

    print(
        "Full validation의 최종 threshold로 확정하면 안 됩니다."
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output = {
        "checkpoint":
            str(
                CHECKPOINT
            ),

        "thresholds":
            THRESHOLDS,

        "nms_threshold":
            NMS_THRESHOLD,

        "nms_topk":
            NMS_TOPK,

        "num_cases":
            NUM_CASES,

        "max_mini_recall":
            float(
                max_recall
            ),

        "diagnostic_recommended_threshold":
            float(
                recommended[
                    "threshold"
                ]
            ),

        "results":
            results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    csv_rows = []

    for r in results:

        csv_rows.append(
            {
                "threshold":
                    r[
                        "threshold"
                    ],

                "predictions":
                    r[
                        "predictions"
                    ],

                "predictions_per_scan":
                    r[
                        "mean_predictions_per_scan"
                    ],

                "tp":
                    r[
                        "tp"
                    ],

                "fp":
                    r[
                        "fp"
                    ],

                "fn":
                    r[
                        "fn"
                    ],

                "precision":
                    r[
                        "precision"
                    ],

                "recall":
                    r[
                        "recall"
                    ],

                "fp_per_scan":
                    r[
                        "fp_per_scan"
                    ],

                "median_matched_distance":
                    r[
                        "median_matched_distance"
                    ],
            }
        )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    header(
        "Saved"
    )

    print(
        OUTPUT_JSON
    )

    print(
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()