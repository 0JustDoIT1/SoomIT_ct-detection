from pathlib import Path
import importlib.util
import json
import time

import numpy as np
import SimpleITK as sitk
import torch


# ============================================================
# Candidate 2 Final Held-out Evaluation
#
# Test:
#   67 CT / 112 GT nodules
#
# Model:
#   Med3D pretrained ResNet50
#   + Custom Multi-scale FPN
#   + MONAI 3D RetinaNet
#
# IMPORTANT
# - 19번에서 검증한 architecture/preprocessing 재사용
# - 21번 학습에서 저장한 best_val_loss.pt 사용
# - test 67 CT는 학습 중 사용하지 않음
# - full-volume sliding-window inference
# - FP32 inference
# ============================================================


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PIPELINE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "monai"
    / "19_validate_candidate2_training_pipeline.py"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "training_setup"
    / "candidate2_fold0_split.json"
)

BEST_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "training"
    / "best_val_loss.pt"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "evaluation"
)

PREDICTION_JSON = (
    RESULT_ROOT
    / "candidate2_fold0_predictions.json"
)

EVENTS_JSON = (
    RESULT_ROOT
    / "candidate2_fold0_detection_events.json"
)

FROC_JSON = (
    RESULT_ROOT
    / "candidate2_fold0_froc.json"
)

METRICS_JSON = (
    RESULT_ROOT
    / "candidate2_fold0_metrics.json"
)


# ============================================================
# Expected experiment conditions
# ============================================================

EXPECTED_TEST_CASES = 67
EXPECTED_GT_NODULES = 112

EXPECTED_SPACING = (
    0.703125,
    0.703125,
    1.25,
)

EXPECTED_PATCH_SIZE = (
    256,
    256,
    96,
)


# ============================================================
# LUNA16 CPM operating points
# ============================================================

CPM_FP_RATES = np.asarray(
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
# Post-processing
#
# NOTE:
# Candidate 1과 strict comparison을 할 때는
# score/NMS/evaluator 조건을 반드시 동일하게 맞춰야 한다.
# ============================================================

SCORE_THRESH = 0.02
TOPK_CANDIDATES_PER_LEVEL = 1000
NMS_THRESH = 0.22
DETECTIONS_PER_IMAGE = 300


# ============================================================
# Sliding-window inference
# ============================================================

OVERLAP = 0.25
SW_BATCH_SIZE = 1


# ============================================================
# JSON helpers
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"파일 없음:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Load 19번 pipeline
# ============================================================

def load_pipeline_module():

    if not PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"19번 pipeline script 없음:\n"
            f"{PIPELINE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "candidate2_pipeline",
        PIPELINE_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "19_validate_candidate2_training_pipeline.py "
            "import 실패"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Count GT
# ============================================================

def count_boxes(items):

    return sum(
        len(
            item.get(
                "box",
                [],
            )
        )
        for item in items
    )


# ============================================================
# Whole-volume resampling
#
# SimpleITK order:
#   X, Y, Z
#
# Target:
#   0.703125 × 0.703125 × 1.25 mm
# ============================================================

def resample_whole_ct(
    image,
    target_spacing_xyz,
):

    original_spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float64,
    )

    original_size = np.asarray(
        image.GetSize(),
        dtype=np.int64,
    )

    target_spacing = np.asarray(
        target_spacing_xyz,
        dtype=np.float64,
    )

    new_size = np.round(
        original_size
        * original_spacing
        / target_spacing
    ).astype(
        np.int64
    )

    new_size = np.maximum(
        new_size,
        1,
    )

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(
        tuple(
            float(v)
            for v in target_spacing
        )
    )

    resampler.SetSize(
        [
            int(v)
            for v in new_size
        ]
    )

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetTransform(
        sitk.Transform()
    )

    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    resampler.SetDefaultPixelValue(
        -1024.0
    )

    return resampler.Execute(
        image
    )


# ============================================================
# Prediction bbox → physical center
#
# box:
#   [xmin, ymin, zmin, xmax, ymax, zmax]
# ============================================================

def prediction_boxes_to_physical_centers(
    image,
    boxes_xyzxyz,
):

    physical_centers = []

    for box in boxes_xyzxyz:

        box = np.asarray(
            box,
            dtype=np.float64,
        )

        center_index = (
            box[:3]
            + box[3:]
        ) / 2.0

        physical = (
            image
            .TransformContinuousIndexToPhysicalPoint(
                tuple(
                    float(v)
                    for v in center_index
                )
            )
        )

        physical_centers.append(
            [
                float(v)
                for v in physical
            ]
        )

    return physical_centers


# ============================================================
# GT conversion
#
# Current split format:
#   [cx, cy, cz, width, height, depth]
#
# Current matching criterion:
#   distance(pred center, GT center)
#   <= max(width,height,depth)/2
# ============================================================

def build_gt_nodules(
    boxes,
):

    gt = []

    for index, box in enumerate(
        boxes
    ):

        if len(
            box
        ) != 6:
            raise RuntimeError(
                f"GT box format 이상: {box}"
            )

        cx, cy, cz, w, h, d = [
            float(v)
            for v in box
        ]

        diameter = max(
            w,
            h,
            d,
        )

        values = np.asarray(
            [
                cx,
                cy,
                cz,
                diameter,
            ],
            dtype=np.float64,
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"GT NaN/Inf: {box}"
            )

        if diameter <= 0:
            raise RuntimeError(
                f"GT diameter <= 0: {box}"
            )

        gt.append(
            {
                "gt_index": index,

                "center_xyz_mm": [
                    cx,
                    cy,
                    cz,
                ],

                "diameter_mm": diameter,

                "radius_mm": (
                    diameter / 2.0
                ),
            }
        )

    return gt


# ============================================================
# Build detector + load best checkpoint
# ============================================================

def build_detector(
    pipeline,
    device,
):

    print()
    print(
        "[3] Build Candidate 2 detector"
    )

    detector = pipeline.build_detector(
        device
    )

    print(
        "✅ Architecture 생성"
    )

    if not BEST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Best checkpoint 없음:\n"
            f"{BEST_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint에 model_state_dict 없음"
        )

    detector.network.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    print(
        "Checkpoint:",
        BEST_CHECKPOINT,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch"
        ),
    )

    print(
        "Best val loss:",
        checkpoint.get(
            "best_val_loss"
        ),
    )

    print(
        "✅ best_val_loss.pt load"
    )

    detector.set_box_selector_parameters(
        score_thresh=(
            SCORE_THRESH
        ),

        topk_candidates_per_level=(
            TOPK_CANDIDATES_PER_LEVEL
        ),

        nms_thresh=(
            NMS_THRESH
        ),

        detections_per_img=(
            DETECTIONS_PER_IMAGE
        ),
    )

    detector.set_sliding_window_inferer(
        roi_size=(
            pipeline.PATCH_SIZE
        ),

        overlap=(
            OVERLAP
        ),

        sw_batch_size=(
            SW_BATCH_SIZE
        ),

        mode="constant",

        device="cpu",
    )

    detector.eval()

    print()
    print(
        "Score threshold:",
        SCORE_THRESH,
    )

    print(
        "NMS threshold:",
        NMS_THRESH,
    )

    print(
        "Detections/image:",
        DETECTIONS_PER_IMAGE,
    )

    print(
        "Sliding window:",
        pipeline.PATCH_SIZE,
    )

    print(
        "Overlap:",
        OVERLAP,
    )

    print(
        "Inference dtype: FP32"
    )

    print(
        "✅ Detector inference 설정 완료"
    )

    return (
        detector,
        checkpoint,
    )


# ============================================================
# One full scan inference
# ============================================================

def infer_one_scan(
    sample,
    pipeline,
    detector,
    device,
):

    mhd_path = (
        pipeline
        .resolve_mhd_path(
            sample[
                "image"
            ]
        )
    )

    source_image = sitk.ReadImage(
        str(
            mhd_path
        )
    )

    # --------------------------------------------------------
    # Candidate1-aligned target spacing
    #
    # NOT 1 mm isotropic.
    # --------------------------------------------------------

    image_resampled = resample_whole_ct(
        source_image,
        pipeline.TARGET_SPACING,
    )

    actual_spacing = np.asarray(
        image_resampled.GetSpacing(),
        dtype=np.float64,
    )

    if not np.allclose(
        actual_spacing,
        np.asarray(
            pipeline.TARGET_SPACING,
            dtype=np.float64,
        ),
        atol=1e-6,
    ):
        raise RuntimeError(
            "Resampled spacing 불일치:\n"
            f"{actual_spacing}\n"
            f"{pipeline.TARGET_SPACING}"
        )

    # SimpleITK array:
    #   ZYX
    array_zyx = (
        sitk.GetArrayFromImage(
            image_resampled
        )
        .astype(
            np.float32
        )
    )

    if not np.isfinite(
        array_zyx
    ).all():
        raise RuntimeError(
            "Resampled CT에 NaN/Inf 존재"
        )

    # Network:
    #   XYZ
    volume_xyz = np.transpose(
        array_zyx,
        (
            2,
            1,
            0,
        ),
    )

    # same normalization as 19/20/21
    volume_xyz = pipeline.normalize_ct(
        volume_xyz
    )

    volume_xyz = np.asarray(
        volume_xyz,
        dtype=np.float32,
    )

    if not np.isfinite(
        volume_xyz
    ).all():
        raise RuntimeError(
            "Normalized CT에 NaN/Inf 존재"
        )

    min_value = float(
        volume_xyz.min()
    )

    max_value = float(
        volume_xyz.max()
    )

    if (
        min_value < -1e-6
        or max_value > 1.000001
    ):
        raise RuntimeError(
            "Normalized CT range 이상: "
            f"{min_value} ~ {max_value}"
        )

    # [C, X, Y, Z]
    image_tensor = (
        torch
        .from_numpy(
            np.ascontiguousarray(
                volume_xyz
            )
        )
        .unsqueeze(
            0
        )
        .float()
    )

    # --------------------------------------------------------
    # FP32 full-volume inference
    # --------------------------------------------------------

    with torch.no_grad():

        image_gpu = image_tensor.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        outputs = detector(
            [
                image_gpu
            ],
            use_inferer=True,
        )

    if len(
        outputs
    ) != 1:
        raise RuntimeError(
            "Detector output count 이상"
        )

    output = outputs[
        0
    ]

    box_key = detector.target_box_key
    label_key = detector.target_label_key
    score_key = detector.pred_score_key

    pred_boxes_tensor = output[
        box_key
    ]

    pred_labels_tensor = output[
        label_key
    ]

    pred_scores_tensor = output[
        score_key
    ]

    if not torch.isfinite(
        pred_boxes_tensor
    ).all():
        raise RuntimeError(
            "Prediction bbox에 NaN/Inf 존재"
        )

    if not torch.isfinite(
        pred_scores_tensor
    ).all():
        raise RuntimeError(
            "Prediction score에 NaN/Inf 존재"
        )

    pred_boxes = (
        pred_boxes_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    pred_labels = (
        pred_labels_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )

    pred_scores = (
        pred_scores_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    if not (
        len(
            pred_boxes
        )
        == len(
            pred_labels
        )
        == len(
            pred_scores
        )
    ):
        raise RuntimeError(
            "Prediction box/label/score count mismatch"
        )

    if len(
        pred_scores
    ) > 0:

        order = np.argsort(
            -pred_scores
        )

        pred_boxes = pred_boxes[
            order
        ]

        pred_labels = pred_labels[
            order
        ]

        pred_scores = pred_scores[
            order
        ]

    physical_centers = (
        prediction_boxes_to_physical_centers(
            image_resampled,
            pred_boxes,
        )
    )

    gt_nodules = build_gt_nodules(
        sample.get(
            "box",
            [],
        )
    )

    predictions = []

    for index in range(
        len(
            pred_boxes
        )
    ):

        predictions.append(
            {
                "prediction_index": index,

                "box_xyzxyz": (
                    pred_boxes[
                        index
                    ].tolist()
                ),

                "center_xyz_mm": (
                    physical_centers[
                        index
                    ]
                ),

                "score": float(
                    pred_scores[
                        index
                    ]
                ),

                "label": int(
                    pred_labels[
                        index
                    ]
                ),
            }
        )

    result = {
        "image": (
            sample[
                "image"
            ]
        ),

        "mhd_path": str(
            mhd_path
        ),

        "original_size_xyz": [
            int(v)
            for v in source_image.GetSize()
        ],

        "original_spacing_xyz": [
            float(v)
            for v in source_image.GetSpacing()
        ],

        "resampled_size_xyz": [
            int(v)
            for v in image_resampled.GetSize()
        ],

        "resampled_spacing_xyz": [
            float(v)
            for v in image_resampled.GetSpacing()
        ],

        "gt_nodules": (
            gt_nodules
        ),

        "predictions": (
            predictions
        ),
    }

    del image_gpu
    del image_tensor
    del output
    del outputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Build global TP/FP events
#
# hit:
# distance(pred center, GT center)
# <= GT radius
#
# One GT matched once.
# Duplicate predictions become FP.
# ============================================================

def build_detection_events(
    scan_results,
):

    all_predictions = []

    total_gt = 0

    for scan_index, scan in enumerate(
        scan_results
    ):

        total_gt += len(
            scan[
                "gt_nodules"
            ]
        )

        for prediction in scan[
            "predictions"
        ]:

            all_predictions.append(
                {
                    "scan_index": scan_index,
                    "prediction": prediction,
                }
            )

    all_predictions.sort(
        key=lambda x: (
            x[
                "prediction"
            ][
                "score"
            ]
        ),
        reverse=True,
    )

    matched_gt = {
        index: set()
        for index in range(
            len(
                scan_results
            )
        )
    }

    events = []

    cumulative_tp = 0
    cumulative_fp = 0

    for item in all_predictions:

        scan_index = item[
            "scan_index"
        ]

        pred = item[
            "prediction"
        ]

        pred_center = np.asarray(
            pred[
                "center_xyz_mm"
            ],
            dtype=np.float64,
        )

        gt_nodules = (
            scan_results[
                scan_index
            ][
                "gt_nodules"
            ]
        )

        best_gt_index = None
        best_distance = float(
            "inf"
        )

        for gt in gt_nodules:

            gt_index = gt[
                "gt_index"
            ]

            if (
                gt_index
                in matched_gt[
                    scan_index
                ]
            ):
                continue

            gt_center = np.asarray(
                gt[
                    "center_xyz_mm"
                ],
                dtype=np.float64,
            )

            distance = float(
                np.linalg.norm(
                    pred_center
                    - gt_center
                )
            )

            if (
                distance
                <= gt[
                    "radius_mm"
                ]
                and distance
                < best_distance
            ):
                best_distance = distance
                best_gt_index = gt_index

        if best_gt_index is not None:

            matched_gt[
                scan_index
            ].add(
                best_gt_index
            )

            cumulative_tp += 1
            is_tp = True

        else:

            cumulative_fp += 1
            is_tp = False

        events.append(
            {
                "score": float(
                    pred[
                        "score"
                    ]
                ),

                "scan_index": scan_index,

                "scan_image": (
                    scan_results[
                        scan_index
                    ][
                        "image"
                    ]
                ),

                "is_tp": is_tp,

                "matched_gt_index": (
                    best_gt_index
                ),

                "match_distance_mm": (
                    None
                    if best_gt_index is None
                    else float(
                        best_distance
                    )
                ),

                "cumulative_tp": (
                    cumulative_tp
                ),

                "cumulative_fp": (
                    cumulative_fp
                ),
            }
        )

    return (
        events,
        total_gt,
    )


# ============================================================
# FROC / CPM
#
# For each CPM target FP/scan:
# use best sensitivity actually achieved with FP/scan <= target.
# ============================================================

def calculate_froc(
    events,
    num_scans,
    total_gt,
):

    if num_scans <= 0:
        raise RuntimeError(
            "num_scans <= 0"
        )

    if total_gt <= 0:
        raise RuntimeError(
            "total_gt <= 0"
        )

    fp_per_scan = [
        0.0
    ]

    sensitivity = [
        0.0
    ]

    thresholds = [
        float(
            "inf"
        )
    ]

    for event in events:

        fp_rate = (
            event[
                "cumulative_fp"
            ]
            / num_scans
        )

        sens = (
            event[
                "cumulative_tp"
            ]
            / total_gt
        )

        fp_per_scan.append(
            float(
                fp_rate
            )
        )

        sensitivity.append(
            float(
                sens
            )
        )

        thresholds.append(
            float(
                event[
                    "score"
                ]
            )
        )

    fp_per_scan = np.asarray(
        fp_per_scan,
        dtype=np.float64,
    )

    sensitivity = np.asarray(
        sensitivity,
        dtype=np.float64,
    )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    unique_fp = np.unique(
        fp_per_scan
    )

    envelope_sensitivity = []

    for rate in unique_fp:

        mask = np.isclose(
            fp_per_scan,
            rate,
        )

        envelope_sensitivity.append(
            float(
                np.max(
                    sensitivity[
                        mask
                    ]
                )
            )
        )

    envelope_sensitivity = np.asarray(
        envelope_sensitivity,
        dtype=np.float64,
    )

    envelope_sensitivity = (
        np.maximum.accumulate(
            envelope_sensitivity
        )
    )

    sensitivities_at_cpm = []
    achieved_fp_at_cpm = []

    for target_fp in CPM_FP_RATES:

        valid_indices = np.where(
            unique_fp
            <= (
                target_fp
                + 1e-12
            )
        )[0]

        if len(
            valid_indices
        ) == 0:

            sensitivities_at_cpm.append(
                0.0
            )

            achieved_fp_at_cpm.append(
                0.0
            )

            continue

        idx = int(
            valid_indices[
                -1
            ]
        )

        sensitivities_at_cpm.append(
            float(
                envelope_sensitivity[
                    idx
                ]
            )
        )

        achieved_fp_at_cpm.append(
            float(
                unique_fp[
                    idx
                ]
            )
        )

    sensitivities_at_cpm = np.asarray(
        sensitivities_at_cpm,
        dtype=np.float64,
    )

    achieved_fp_at_cpm = np.asarray(
        achieved_fp_at_cpm,
        dtype=np.float64,
    )

    cpm = float(
        np.mean(
            sensitivities_at_cpm
        )
    )

    return {
        "curve_fp_per_scan": (
            fp_per_scan.tolist()
        ),

        "curve_sensitivity": (
            sensitivity.tolist()
        ),

        "curve_threshold": (
            thresholds.tolist()
        ),

        "envelope_fp_per_scan": (
            unique_fp.tolist()
        ),

        "envelope_sensitivity": (
            envelope_sensitivity.tolist()
        ),

        "cpm_fp_rates": (
            CPM_FP_RATES.tolist()
        ),

        "achieved_fp_rates": (
            achieved_fp_at_cpm.tolist()
        ),

        "sensitivities_at_cpm_rates": (
            sensitivities_at_cpm.tolist()
        ),

        "cpm": cpm,
    }


# ============================================================
# Main
# ============================================================

def main():

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "Candidate 2 - Final Held-out Evaluation"
    )

    print(
        "=" * 80
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Torch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ========================================================
    # 1. Pipeline
    # ========================================================

    print()
    print(
        "[1] Load Candidate 2 pipeline"
    )

    pipeline = load_pipeline_module()

    print(
        "Target spacing:",
        pipeline.TARGET_SPACING,
    )

    print(
        "Patch size:",
        pipeline.PATCH_SIZE,
    )

    if not np.allclose(
        np.asarray(
            pipeline.TARGET_SPACING,
            dtype=np.float64,
        ),
        np.asarray(
            EXPECTED_SPACING,
            dtype=np.float64,
        ),
        atol=1e-6,
    ):
        raise RuntimeError(
            "TARGET_SPACING 불일치"
        )

    if tuple(
        pipeline.PATCH_SIZE
    ) != EXPECTED_PATCH_SIZE:
        raise RuntimeError(
            "PATCH_SIZE 불일치"
        )

    print(
        "✅ Pipeline config 확인"
    )

    # ========================================================
    # 2. Final held-out split
    # ========================================================

    print()
    print(
        "[2] Load final held-out split"
    )

    split = load_json(
        SPLIT_PATH
    )

    test_samples = split[
        "test"
    ]

    print(
        "Final test cases:",
        len(
            test_samples
        ),
    )

    if len(
        test_samples
    ) != EXPECTED_TEST_CASES:
        raise RuntimeError(
            "Final test count 이상: "
            f"{len(test_samples)}"
        )

    total_gt_from_json = (
        count_boxes(
            test_samples
        )
    )

    print(
        "GT nodules:",
        total_gt_from_json,
    )

    if (
        total_gt_from_json
        != EXPECTED_GT_NODULES
    ):
        raise RuntimeError(
            "Final test GT count 이상: "
            f"{total_gt_from_json}"
        )

    print(
        "✅ 67 CT / 112 GT 확인"
    )

    # ========================================================
    # 3. Detector + checkpoint
    # ========================================================

    detector, checkpoint = build_detector(
        pipeline,
        device,
    )

    # ========================================================
    # 4. Full-volume inference
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "Final 67 CT full-volume inference"
    )

    print(
        "=" * 80
    )

    scan_results = []

    total_start = time.time()

    for index, sample in enumerate(
        test_samples,
        start=1,
    ):

        case_start = time.time()

        print()
        print(
            "-" * 80
        )

        print(
            f"[{index}/{len(test_samples)}]"
        )

        print(
            sample[
                "image"
            ]
        )

        result = infer_one_scan(
            sample,
            pipeline,
            detector,
            device,
        )

        scan_results.append(
            result
        )

        elapsed = (
            time.time()
            - case_start
        )

        print(
            "GT nodules:",
            len(
                result[
                    "gt_nodules"
                ]
            ),
        )

        print(
            "Predictions:",
            len(
                result[
                    "predictions"
                ]
            ),
        )

        if result[
            "predictions"
        ]:

            print(
                "Top score:",
                f"{result['predictions'][0]['score']:.4f}",
            )

        print(
            "Time:",
            f"{elapsed:.1f} sec",
        )

        # progressive save
        save_json(
            PREDICTION_JSON,
            scan_results,
        )

    # ========================================================
    # 5. TP / FP events
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "Build TP / FP events"
    )

    print(
        "=" * 80
    )

    events, total_gt = (
        build_detection_events(
            scan_results
        )
    )

    if total_gt != EXPECTED_GT_NODULES:
        raise RuntimeError(
            "Evaluation GT total 이상: "
            f"{total_gt}"
        )

    save_json(
        EVENTS_JSON,
        events,
    )

    # ========================================================
    # 6. FROC / CPM
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "FROC / CPM"
    )

    print(
        "=" * 80
    )

    froc = calculate_froc(
        events,

        num_scans=(
            len(
                scan_results
            )
        ),

        total_gt=(
            total_gt
        ),
    )

    print()
    print(
        "Final Test CT:",
        len(
            scan_results
        ),
    )

    print(
        "GT Nodules:",
        total_gt,
    )

    print()
    print(
        "Sensitivity @ FP/scan"
    )

    for (
        target_fp,
        achieved_fp,
        sensitivity_value,
    ) in zip(
        froc[
            "cpm_fp_rates"
        ],
        froc[
            "achieved_fp_rates"
        ],
        froc[
            "sensitivities_at_cpm_rates"
        ],
    ):

        print(
            f"{target_fp:>6.3f} FP/scan : "
            f"{sensitivity_value:.4f} "
            f"(achieved {achieved_fp:.4f})"
        )

    print()
    print(
        "CPM:",
        f"{froc['cpm']:.4f}",
    )

    # ========================================================
    # 7. Save
    # ========================================================

    save_json(
        FROC_JSON,
        froc,
    )

    total_inference_hours = (
        (
            time.time()
            - total_start
        )
        / 3600.0
    )

    metrics = {
        "candidate": 2,

        "model": (
            "Med3D pretrained ResNet50 "
            "+ Custom Multi-scale FPN "
            "+ MONAI 3D RetinaNet"
        ),

        "checkpoint": str(
            BEST_CHECKPOINT
        ),

        "checkpoint_epoch": (
            checkpoint.get(
                "epoch"
            )
        ),

        "best_internal_val_loss": (
            checkpoint.get(
                "best_val_loss"
            )
        ),

        "num_test_scans": (
            len(
                scan_results
            )
        ),

        "num_gt_nodules": (
            total_gt
        ),

        "target_spacing_xyz": [
            float(v)
            for v in pipeline.TARGET_SPACING
        ],

        "patch_size_xyz": [
            int(v)
            for v in pipeline.PATCH_SIZE
        ],

        "score_threshold": (
            SCORE_THRESH
        ),

        "nms_threshold": (
            NMS_THRESH
        ),

        "detections_per_image": (
            DETECTIONS_PER_IMAGE
        ),

        "sliding_window_overlap": (
            OVERLAP
        ),

        "sw_batch_size": (
            SW_BATCH_SIZE
        ),

        "inference_dtype": (
            "float32"
        ),

        "cpm_fp_rates": (
            froc[
                "cpm_fp_rates"
            ]
        ),

        "achieved_fp_rates": (
            froc[
                "achieved_fp_rates"
            ]
        ),

        "sensitivity": (
            froc[
                "sensitivities_at_cpm_rates"
            ]
        ),

        "cpm": (
            froc[
                "cpm"
            ]
        ),

        # Historical project reference only.
        "candidate1_cpm_reference": (
            0.7130
        ),

        "candidate1_reference_note": (
            "Historical project reference. "
            "Verify Candidate 1 checkpoint/evaluator provenance "
            "before claiming strict apples-to-apples comparison."
        ),

        "total_inference_hours": (
            total_inference_hours
        ),
    }

    save_json(
        METRICS_JSON,
        metrics,
    )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "✅ Candidate 2 Final Evaluation Finished"
    )

    print()
    print(
        "Checkpoint epoch:"
    )

    print(
        checkpoint.get(
            "epoch"
        )
    )

    print()
    print(
        "Best internal val loss:"
    )

    print(
        checkpoint.get(
            "best_val_loss"
        )
    )

    print()
    print(
        "Candidate 2 CPM:"
    )

    print(
        f"{froc['cpm']:.4f}"
    )

    print()
    print(
        "Candidate 1 historical CPM reference:"
    )

    print(
        "0.7130"
    )

    difference = (
        froc[
            "cpm"
        ]
        - 0.7130
    )

    print()
    print(
        "Difference vs historical reference:"
    )

    print(
        f"{difference:+.4f}"
    )

    print()
    print(
        "Predictions:"
    )

    print(
        PREDICTION_JSON
    )

    print()
    print(
        "Detection events:"
    )

    print(
        EVENTS_JSON
    )

    print()
    print(
        "Metrics:"
    )

    print(
        METRICS_JSON
    )

    print()
    print(
        "FROC:"
    )

    print(
        FROC_JSON
    )

    print()
    print(
        "Total inference time:"
    )

    print(
        f"{total_inference_hours:.2f} h"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
