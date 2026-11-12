from pathlib import Path
import importlib.util
import json
import random

import numpy as np
import torch


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


# ============================================================
# Validation configuration
# ============================================================

SEED = 42

EXPECTED_TRAIN = 779
EXPECTED_INTERNAL_VAL = 42
EXPECTED_FINAL_TEST = 67

EXPECTED_TRAIN_POSITIVE = 507
EXPECTED_TRAIN_NEGATIVE = 272

EXPECTED_VAL_POSITIVE = 27
EXPECTED_VAL_NEGATIVE = 15

EXPECTED_FINAL_GT = 112


# ------------------------------------------------------------
# 여러 케이스에서 실제 transform을 반복 검증
# ------------------------------------------------------------

NUM_POSITIVE_CASES = 5
NUM_NEGATIVE_CASES = 5


# ============================================================
# Helpers
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


def load_pipeline_module():

    if not PIPELINE_SCRIPT.exists():

        raise FileNotFoundError(
            f"19번 script 없음:\n"
            f"{PIPELINE_SCRIPT}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "candidate2_pipeline",
            PIPELINE_SCRIPT,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "19번 module import 실패"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def count_pos_neg(items):

    positive = sum(
        1
        for item in items
        if len(
            item.get(
                "box",
                [],
            )
        ) > 0
    )

    negative = (
        len(items)
        - positive
    )

    return (
        positive,
        negative,
    )


def total_boxes(items):

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
# Box validation
# ============================================================

def validate_box_array(
    boxes,
    patch_size,
):

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    )

    if boxes.ndim != 2:

        raise RuntimeError(
            f"bbox ndim 이상: {boxes.ndim}"
        )

    if boxes.shape[1] != 6:

        raise RuntimeError(
            f"bbox shape 이상: {boxes.shape}"
        )

    if len(boxes) == 0:

        return

    if not np.isfinite(
        boxes
    ).all():

        raise RuntimeError(
            "bbox에 NaN / Inf 존재"
        )

    mins = boxes[
        :,
        :3
    ]

    maxs = boxes[
        :,
        3:
    ]

    patch_size = np.asarray(
        patch_size,
        dtype=np.float32,
    )

    if np.any(
        mins < 0
    ):

        raise RuntimeError(
            "bbox min < 0"
        )

    if np.any(
        maxs > patch_size
    ):

        raise RuntimeError(
            "bbox max > patch size"
        )

    if np.any(
        maxs <= mins
    ):

        raise RuntimeError(
            "invalid bbox width/height/depth"
        )


# ============================================================
# Image validation
# ============================================================

def validate_patch(
    patch,
    patch_size,
):

    if tuple(
        patch.shape
    ) != tuple(
        patch_size
    ):

        raise RuntimeError(
            "Patch shape 이상: "
            f"{patch.shape} "
            f"!= {patch_size}"
        )

    if patch.dtype != np.float32:

        raise RuntimeError(
            f"Patch dtype 이상: "
            f"{patch.dtype}"
        )

    if not np.isfinite(
        patch
    ).all():

        raise RuntimeError(
            "Patch에 NaN / Inf 존재"
        )

    minimum = float(
        patch.min()
    )

    maximum = float(
        patch.max()
    )

    if (
        minimum < -1e-6
        or maximum > 1.000001
    ):

        raise RuntimeError(
            "Normalization range 이상: "
            f"{minimum} ~ {maximum}"
        )


# ============================================================
# Positive real transform validation
# ============================================================

def validate_positive_case(
    sample,
    pipeline,
    index,
):

    print()
    print(
        "-" * 80
    )

    print(
        f"[Positive {index}]"
    )

    print(
        "Image:",
        sample[
            "image"
        ],
    )

    print(
        "Original GT:",
        len(
            sample.get(
                "box",
                [],
            )
        ),
    )

    (
        mhd_path,
        original_image,
        resampled_image,
        volume_xyz,
        boxes_xyzxyz,
        labels,
    ) = pipeline.load_real_sample(
        sample
    )

    print(
        "MHD:",
        mhd_path,
    )

    print(
        "Original size:",
        original_image.GetSize(),
    )

    print(
        "Original spacing:",
        original_image.GetSpacing(),
    )

    print(
        "Resampled size:",
        resampled_image.GetSize(),
    )

    print(
        "Resampled spacing:",
        resampled_image.GetSpacing(),
    )

    print(
        "Voxel GT:",
        len(
            boxes_xyzxyz
        ),
    )

    # --------------------------------------------------------
    # Resampled spacing
    # --------------------------------------------------------

    actual_spacing = np.asarray(
        resampled_image.GetSpacing(),
        dtype=np.float64,
    )

    expected_spacing = np.asarray(
        pipeline.TARGET_SPACING,
        dtype=np.float64,
    )

    if not np.allclose(
        actual_spacing,
        expected_spacing,
        atol=1e-6,
    ):

        raise RuntimeError(
            "Resampled spacing 불일치:\n"
            f"{actual_spacing}\n"
            f"{expected_spacing}"
        )

    # --------------------------------------------------------
    # Full volume
    # --------------------------------------------------------

    if not np.isfinite(
        volume_xyz
    ).all():

        raise RuntimeError(
            "Full CT에 NaN/Inf 존재"
        )

    full_min = float(
        volume_xyz.min()
    )

    full_max = float(
        volume_xyz.max()
    )

    if (
        full_min < -1e-6
        or full_max > 1.000001
    ):

        raise RuntimeError(
            "Full CT normalization 이상"
        )

    # --------------------------------------------------------
    # GT structure
    # --------------------------------------------------------

    if len(
        boxes_xyzxyz
    ) != len(
        labels
    ):

        raise RuntimeError(
            "bbox / label count mismatch"
        )

    if len(
        boxes_xyzxyz
    ) == 0:

        raise RuntimeError(
            "Positive case인데 voxel bbox 없음"
        )

    # --------------------------------------------------------
    # Positive crop
    # --------------------------------------------------------

    (
        patch,
        crop_boxes,
        crop_labels,
    ) = pipeline.make_positive_crop(
        volume_xyz,
        boxes_xyzxyz,
        labels,
    )

    validate_patch(
        patch,
        pipeline.PATCH_SIZE,
    )

    validate_box_array(
        crop_boxes,
        pipeline.PATCH_SIZE,
    )

    if len(
        crop_boxes
    ) == 0:

        raise RuntimeError(
            "Positive crop 내부 bbox 없음"
        )

    if len(
        crop_boxes
    ) != len(
        crop_labels
    ):

        raise RuntimeError(
            "Positive crop box/label mismatch"
        )

    print(
        "Patch:",
        patch.shape,
    )

    print(
        "Crop boxes:",
        len(
            crop_boxes
        ),
    )

    print(
        "Intensity:",
        f"{patch.min():.4f}",
        "~",
        f"{patch.max():.4f}",
    )

    print()
    print(
        "First cropped bbox:"
    )

    print(
        crop_boxes[
            0
        ].tolist()
    )

    print(
        "✅ Positive real transform PASS"
    )

    return {
        "mhd": str(
            mhd_path
        ),

        "original_gt": len(
            sample.get(
                "box",
                [],
            )
        ),

        "crop_gt": len(
            crop_boxes
        ),

        "patch_shape": list(
            patch.shape
        ),

        "spacing": [
            float(v)
            for v in (
                resampled_image
                .GetSpacing()
            )
        ],
    }


# ============================================================
# Negative real transform validation
# ============================================================

def validate_negative_case(
    sample,
    pipeline,
    index,
):

    print()
    print(
        "-" * 80
    )

    print(
        f"[Negative {index}]"
    )

    print(
        "Image:",
        sample[
            "image"
        ],
    )

    (
        mhd_path,
        original_image,
        resampled_image,
        volume_xyz,
        boxes_xyzxyz,
        labels,
    ) = pipeline.load_real_sample(
        sample
    )

    print(
        "MHD:",
        mhd_path,
    )

    print(
        "Original size:",
        original_image.GetSize(),
    )

    print(
        "Original spacing:",
        original_image.GetSpacing(),
    )

    print(
        "Resampled size:",
        resampled_image.GetSize(),
    )

    print(
        "Resampled spacing:",
        resampled_image.GetSpacing(),
    )

    print(
        "GT boxes:",
        len(
            boxes_xyzxyz
        ),
    )

    # --------------------------------------------------------
    # Negative must be empty
    # --------------------------------------------------------

    if len(
        boxes_xyzxyz
    ) != 0:

        raise RuntimeError(
            "Negative case에 bbox 존재"
        )

    if len(
        labels
    ) != 0:

        raise RuntimeError(
            "Negative case에 label 존재"
        )

    # --------------------------------------------------------
    # Spacing
    # --------------------------------------------------------

    actual_spacing = np.asarray(
        resampled_image.GetSpacing(),
        dtype=np.float64,
    )

    expected_spacing = np.asarray(
        pipeline.TARGET_SPACING,
        dtype=np.float64,
    )

    if not np.allclose(
        actual_spacing,
        expected_spacing,
        atol=1e-6,
    ):

        raise RuntimeError(
            "Negative CT spacing 불일치"
        )

    # --------------------------------------------------------
    # Negative crop
    # --------------------------------------------------------

    (
        patch,
        crop_boxes,
        crop_labels,
    ) = pipeline.make_negative_crop(
        volume_xyz
    )

    validate_patch(
        patch,
        pipeline.PATCH_SIZE,
    )

    validate_box_array(
        crop_boxes,
        pipeline.PATCH_SIZE,
    )

    if len(
        crop_boxes
    ) != 0:

        raise RuntimeError(
            "Negative crop bbox 존재"
        )

    if len(
        crop_labels
    ) != 0:

        raise RuntimeError(
            "Negative crop label 존재"
        )

    print(
        "Patch:",
        patch.shape,
    )

    print(
        "Crop boxes:",
        len(
            crop_boxes
        ),
    )

    print(
        "Intensity:",
        f"{patch.min():.4f}",
        "~",
        f"{patch.max():.4f}",
    )

    print(
        "✅ Negative real transform PASS"
    )

    return {
        "mhd": str(
            mhd_path
        ),

        "crop_gt": 0,

        "patch_shape": list(
            patch.shape
        ),

        "spacing": [
            float(v)
            for v in (
                resampled_image
                .GetSpacing()
            )
        ],
    }


# ============================================================
# Detector smoke test
#
# 19번과 같은 실제 positive / negative patch를 사용하되
# 여기서는 forward만 다시 검증
# ============================================================

def validate_detector_forward(
    pipeline,
    positive_sample,
    negative_sample,
    device,
):

    print()
    print(
        "=" * 80
    )

    print(
        "[6] Real transformed patch detector forward"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    (
        _,
        _,
        _,
        pos_volume,
        pos_boxes,
        pos_labels,
    ) = pipeline.load_real_sample(
        positive_sample
    )

    (
        pos_patch,
        pos_crop_boxes,
        pos_crop_labels,
    ) = pipeline.make_positive_crop(
        pos_volume,
        pos_boxes,
        pos_labels,
    )

    # --------------------------------------------------------
    # Negative
    # --------------------------------------------------------

    (
        _,
        _,
        _,
        neg_volume,
        _,
        _,
    ) = pipeline.load_real_sample(
        negative_sample
    )

    (
        neg_patch,
        neg_crop_boxes,
        neg_crop_labels,
    ) = pipeline.make_negative_crop(
        neg_volume
    )

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    detector = pipeline.build_detector(
        device
    )

    detector.train()

    # --------------------------------------------------------
    # Positive forward
    # --------------------------------------------------------

    images, targets = (
        pipeline.to_detector_batch(
            pos_patch,
            pos_crop_boxes,
            pos_crop_labels,
            device,
        )
    )

    with torch.no_grad():

        outputs = detector(
            images,
            targets,
        )

    pos_cls = outputs[
        detector.cls_key
    ]

    pos_box = outputs[
        detector.box_reg_key
    ]

    pos_total = (
        pos_cls
        + pos_box
    )

    if not torch.isfinite(
        pos_total
    ):

        raise RuntimeError(
            "Positive detector loss NaN/Inf"
        )

    print()
    print(
        "Positive:"
    )

    print(
        "  Classification:",
        float(
            pos_cls.detach().cpu()
        ),
    )

    print(
        "  Box regression:",
        float(
            pos_box.detach().cpu()
        ),
    )

    print(
        "  Total:",
        float(
            pos_total.detach().cpu()
        ),
    )

    # --------------------------------------------------------
    # Negative forward
    # --------------------------------------------------------

    images, targets = (
        pipeline.to_detector_batch(
            neg_patch,
            neg_crop_boxes,
            neg_crop_labels,
            device,
        )
    )

    with torch.no_grad():

        outputs = detector(
            images,
            targets,
        )

    neg_cls = outputs[
        detector.cls_key
    ]

    neg_box = outputs[
        detector.box_reg_key
    ]

    neg_total = (
        neg_cls
        + neg_box
    )

    if not torch.isfinite(
        neg_total
    ):

        raise RuntimeError(
            "Negative detector loss NaN/Inf"
        )

    print()
    print(
        "Negative:"
    )

    print(
        "  Classification:",
        float(
            neg_cls.detach().cpu()
        ),
    )

    print(
        "  Box regression:",
        float(
            neg_box.detach().cpu()
        ),
    )

    print(
        "  Total:",
        float(
            neg_total.detach().cpu()
        ),
    )

    # Negative에는 GT가 없으므로 regression loss = 0 기대
    if abs(
        float(
            neg_box.detach().cpu()
        )
    ) > 1e-8:

        raise RuntimeError(
            "Negative bbox regression loss가 "
            "0이 아님"
        )

    print()
    print(
        "✅ Real transformed detector forward PASS"
    )

    return detector


# ============================================================
# Main
# ============================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

        torch.cuda.reset_peak_memory_stats()


    print(
        "=" * 80
    )

    print(
        "Candidate 2 - "
        "Candidate 1 Aligned Real Transform Validation"
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
    # 1. Load shared pipeline
    # ========================================================

    print()
    print(
        "[1] Load 19번 shared pipeline"
    )


    pipeline = load_pipeline_module()


    print(
        "Spacing:",
        pipeline.TARGET_SPACING,
    )

    print(
        "Patch:",
        pipeline.PATCH_SIZE,
    )


    expected_spacing = (
        0.703125,
        0.703125,
        1.25,
    )

    expected_patch = (
        256,
        256,
        96,
    )


    if tuple(
        pipeline.TARGET_SPACING
    ) != expected_spacing:

        raise RuntimeError(
            "19번 TARGET_SPACING이 "
            "Candidate 1 기준과 다름"
        )


    if tuple(
        pipeline.PATCH_SIZE
    ) != expected_patch:

        raise RuntimeError(
            "19번 PATCH_SIZE가 "
            "Candidate 1 기준과 다름"
        )


    print(
        "✅ Shared preprocessing config PASS"
    )


    # ========================================================
    # 2. Load split
    # ========================================================

    print()
    print(
        "[2] Load aligned Candidate 2 split"
    )


    split = load_json(
        SPLIT_PATH
    )


    training = split[
        "training"
    ]

    validation = split[
        "validation"
    ]

    final_test = split[
        "test"
    ]


    print(
        "Train:",
        len(
            training
        ),
    )

    print(
        "Internal Val:",
        len(
            validation
        ),
    )

    print(
        "Final Test:",
        len(
            final_test
        ),
    )


    if len(
        training
    ) != EXPECTED_TRAIN:

        raise RuntimeError(
            "Train count 이상"
        )


    if len(
        validation
    ) != EXPECTED_INTERNAL_VAL:

        raise RuntimeError(
            "Internal Val count 이상"
        )


    if len(
        final_test
    ) != EXPECTED_FINAL_TEST:

        raise RuntimeError(
            "Final Test count 이상"
        )


    print(
        "✅ Split count PASS"
    )


    # ========================================================
    # 3. Distribution check
    # ========================================================

    print()
    print(
        "[3] Positive / Negative distribution"
    )


    train_pos, train_neg = (
        count_pos_neg(
            training
        )
    )

    val_pos, val_neg = (
        count_pos_neg(
            validation
        )
    )

    final_pos, final_neg = (
        count_pos_neg(
            final_test
        )
    )


    print(
        "Train Positive:",
        train_pos,
    )

    print(
        "Train Negative:",
        train_neg,
    )

    print(
        "Val Positive:",
        val_pos,
    )

    print(
        "Val Negative:",
        val_neg,
    )

    print(
        "Test Positive:",
        final_pos,
    )

    print(
        "Test Negative:",
        final_neg,
    )


    if train_pos != EXPECTED_TRAIN_POSITIVE:

        raise RuntimeError(
            "Train positive count 이상"
        )


    if train_neg != EXPECTED_TRAIN_NEGATIVE:

        raise RuntimeError(
            "Train negative count 이상"
        )


    if val_pos != EXPECTED_VAL_POSITIVE:

        raise RuntimeError(
            "Val positive count 이상"
        )


    if val_neg != EXPECTED_VAL_NEGATIVE:

        raise RuntimeError(
            "Val negative count 이상"
        )


    final_gt = total_boxes(
        final_test
    )


    print(
        "Final GT nodules:",
        final_gt,
    )


    if final_gt != EXPECTED_FINAL_GT:

        raise RuntimeError(
            "Final GT count 이상"
        )


    print(
        "✅ Dataset distribution PASS"
    )


    # ========================================================
    # 4. Select actual cases
    # ========================================================

    positive_cases = [
        item
        for item in training
        if len(
            item.get(
                "box",
                [],
            )
        ) > 0
    ]


    negative_cases = [
        item
        for item in training
        if len(
            item.get(
                "box",
                [],
            )
        ) == 0
    ]


    random.shuffle(
        positive_cases
    )

    random.shuffle(
        negative_cases
    )


    positive_cases = (
        positive_cases[
            :NUM_POSITIVE_CASES
        ]
    )

    negative_cases = (
        negative_cases[
            :NUM_NEGATIVE_CASES
        ]
    )


    # ========================================================
    # 5. Positive transforms
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "[4] Positive real transform validation"
    )

    print(
        "=" * 80
    )


    positive_results = []


    for index, sample in enumerate(
        positive_cases,
        start=1,
    ):

        result = validate_positive_case(
            sample,
            pipeline,
            index,
        )

        positive_results.append(
            result
        )


    print()
    print(
        f"✅ Positive {len(positive_results)} cases PASS"
    )


    # ========================================================
    # 6. Negative transforms
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "[5] Negative real transform validation"
    )

    print(
        "=" * 80
    )


    negative_results = []


    for index, sample in enumerate(
        negative_cases,
        start=1,
    ):

        result = validate_negative_case(
            sample,
            pipeline,
            index,
        )

        negative_results.append(
            result
        )


    print()
    print(
        f"✅ Negative {len(negative_results)} cases PASS"
    )


    # ========================================================
    # 7. Detector real forward
    # ========================================================

    detector = validate_detector_forward(
        pipeline,
        positive_cases[
            0
        ],
        negative_cases[
            0
        ],
        device,
    )


    # ========================================================
    # 8. GPU memory
    # ========================================================

    if torch.cuda.is_available():

        print()
        print(
            "[7] GPU memory"
        )

        print(
            "Max allocated:",
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
        )

        print(
            "Max reserved:",
            f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GB",
        )


    # ========================================================
    # Final
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "✅ Candidate 2 "
        "Real Transform Validation PASS"
    )

    print()
    print(
        "Validated:"
    )

    print(
        "Aligned split 779 / 42 / 67         ✅"
    )

    print(
        "Train Pos / Neg = 507 / 272          ✅"
    )

    print(
        "Val Pos / Neg = 27 / 15              ✅"
    )

    print(
        "Final Test = 67 / GT 112             ✅"
    )

    print(
        "Spacing 0.703125×0.703125×1.25       ✅"
    )

    print(
        "Patch 256×256×96                     ✅"
    )

    print(
        "HU -1024~300 → 0~1                   ✅"
    )

    print(
        "Physical CCCWHD → voxel XYZXYZ       ✅"
    )

    print(
        f"Positive real CT × {NUM_POSITIVE_CASES}                 ✅"
    )

    print(
        f"Negative real CT × {NUM_NEGATIVE_CASES}                 ✅"
    )

    print(
        "Positive bbox bounds                 ✅"
    )

    print(
        "Negative empty bbox                  ✅"
    )

    print(
        "Med3D + FPN + RetinaNet forward      ✅"
    )

    print(
        "Negative regression loss = 0         ✅"
    )

    print()
    print(
        "Next:"
    )

    print(
        "21_train_candidate2.py"
    )

    print(
        "→ Candidate 1 aligned "
        "positive/negative training"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()