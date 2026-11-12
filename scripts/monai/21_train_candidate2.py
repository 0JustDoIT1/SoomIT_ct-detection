from pathlib import Path
import importlib.util
import hashlib
import json
import math
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "training"
)

BEST_CHECKPOINT = (
    OUTPUT_DIR
    / "best_val_loss.pt"
)

LAST_CHECKPOINT = (
    OUTPUT_DIR
    / "last.pt"
)

HISTORY_JSON = (
    OUTPUT_DIR
    / "history.json"
)

TRAINING_CONFIG_JSON = (
    OUTPUT_DIR
    / "training_config.json"
)


# ============================================================
# Fixed experiment configuration
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


# ============================================================
# Training
# ============================================================

MAX_EPOCHS = 80

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

GRAD_ACCUM_STEPS = 2
GRAD_CLIP_NORM = 5.0

EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

LR_FACTOR = 0.5
LR_PATIENCE = 4
MIN_LR = 1e-6


# ============================================================
# Augmentation
#
# Core preprocessing / crop conditions are shared with 19.
# This augmentation is applied only after the aligned crop.
# ============================================================

FLIP_PROB = 0.5

INTENSITY_AUG_PROB = 0.3

INTENSITY_SCALE_RANGE = (
    0.95,
    1.05,
)

INTENSITY_SHIFT_RANGE = (
    -0.03,
    0.03,
)


# ============================================================
# Logging
# ============================================================

PRINT_EVERY = 25


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


def save_json(
    path,
    obj,
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
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


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
            "19번 pipeline module import 실패"
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


def get_case_id(sample):

    return Path(
        str(
            sample.get(
                "image",
                ""
            )
        )
    ).name


def stable_seed_from_sample(
    sample,
    base_seed=SEED,
):

    text = (
        str(
            base_seed
        )
        + "|"
        + get_case_id(
            sample
        )
    )

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


# ============================================================
# Validation crop
#
# The same internal-validation crop is reproduced every epoch.
# This prevents random validation crop variation from deciding
# early stopping / LR scheduling.
# ============================================================

def make_deterministic_crop(
    sample,
    pipeline,
    volume_xyz,
    boxes,
    labels,
):

    random_state = (
        random.getstate()
    )

    np_state = (
        np.random.get_state()
    )

    seed = stable_seed_from_sample(
        sample
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
        % (
            2**32
            - 1
        )
    )

    try:

        if len(
            boxes
        ) > 0:

            patch, crop_boxes, crop_labels = (
                pipeline.make_positive_crop(
                    volume_xyz,
                    boxes,
                    labels,
                )
            )

        else:

            patch, crop_boxes, crop_labels = (
                pipeline.make_negative_crop(
                    volume_xyz
                )
            )

    finally:

        random.setstate(
            random_state
        )

        np.random.set_state(
            np_state
        )

    return (
        patch,
        crop_boxes,
        crop_labels,
    )


# ============================================================
# Augmentation
#
# patch coordinate convention:
# XYZ
#
# bbox:
# [xmin, ymin, zmin, xmax, ymax, zmax]
# ============================================================

def flip_patch_and_boxes(
    patch,
    boxes,
    axis,
):

    patch = np.flip(
        patch,
        axis=axis,
    ).copy()

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    ).copy()

    if len(
        boxes
    ) == 0:

        return (
            patch,
            boxes,
        )

    size = float(
        patch.shape[
            axis
        ]
    )

    old_min = (
        boxes[
            :,
            axis
        ].copy()
    )

    old_max = (
        boxes[
            :,
            axis + 3
        ].copy()
    )

    boxes[
        :,
        axis
    ] = (
        size
        - old_max
    )

    boxes[
        :,
        axis + 3
    ] = (
        size
        - old_min
    )

    return (
        patch,
        boxes,
    )


def augment_patch(
    patch,
    boxes,
):

    patch = np.asarray(
        patch,
        dtype=np.float32,
    ).copy()

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    ).copy()

    # --------------------------------------------------------
    # Random flip X / Y / Z
    # --------------------------------------------------------

    for axis in range(
        3
    ):

        if random.random() < FLIP_PROB:

            patch, boxes = (
                flip_patch_and_boxes(
                    patch,
                    boxes,
                    axis,
                )
            )

    # --------------------------------------------------------
    # Mild intensity scale + shift
    # --------------------------------------------------------

    if (
        random.random()
        < INTENSITY_AUG_PROB
    ):

        scale = random.uniform(
            *INTENSITY_SCALE_RANGE
        )

        shift = random.uniform(
            *INTENSITY_SHIFT_RANGE
        )

        patch = (
            patch
            * scale
            + shift
        )

        patch = np.clip(
            patch,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    return (
        np.ascontiguousarray(
            patch
        ),
        np.ascontiguousarray(
            boxes
        ),
    )


# ============================================================
# Sample preparation
# ============================================================

def prepare_training_sample(
    sample,
    pipeline,
):

    (
        _,
        _,
        _,
        volume_xyz,
        boxes,
        labels,
    ) = pipeline.load_real_sample(
        sample
    )

    if len(
        boxes
    ) > 0:

        (
            patch,
            crop_boxes,
            crop_labels,
        ) = pipeline.make_positive_crop(
            volume_xyz,
            boxes,
            labels,
        )

    else:

        (
            patch,
            crop_boxes,
            crop_labels,
        ) = pipeline.make_negative_crop(
            volume_xyz
        )

    patch, crop_boxes = (
        augment_patch(
            patch,
            crop_boxes,
        )
    )

    crop_labels = np.asarray(
        crop_labels,
        dtype=np.int64,
    )

    return (
        patch,
        crop_boxes,
        crop_labels,
    )


def prepare_validation_sample(
    sample,
    pipeline,
):

    (
        _,
        _,
        _,
        volume_xyz,
        boxes,
        labels,
    ) = pipeline.load_real_sample(
        sample
    )

    return make_deterministic_crop(
        sample,
        pipeline,
        volume_xyz,
        boxes,
        labels,
    )


# ============================================================
# Loss
# ============================================================

def compute_loss(
    detector,
    images,
    targets,
):

    outputs = detector(
        images,
        targets,
    )

    cls_loss = outputs[
        detector.cls_key
    ]

    box_loss = outputs[
        detector.box_reg_key
    ]

    total_loss = (
        cls_loss
        + box_loss
    )

    if not torch.isfinite(
        total_loss
    ):

        raise RuntimeError(
            "Loss NaN / Inf 발생"
        )

    return (
        total_loss,
        cls_loss,
        box_loss,
    )


# ============================================================
# BN mode helper
#
# Detector must stay in training mode to return losses.
# During validation, BatchNorm running statistics should not
# be updated.
# ============================================================

def set_batchnorm_eval(
    module,
):

    if isinstance(
        module,
        nn.modules.batchnorm._BatchNorm,
    ):

        module.eval()


# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(
    detector,
    optimizer,
    training,
    pipeline,
    device,
    epoch,
):

    detector.train()

    shuffled = list(
        training
    )

    random.shuffle(
        shuffled
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    sum_total = 0.0
    sum_cls = 0.0
    sum_box = 0.0

    positive_seen = 0
    negative_seen = 0

    start_time = time.time()

    num_items = len(
        shuffled
    )

    remainder = (
        num_items
        % GRAD_ACCUM_STEPS
    )

    for step, sample in enumerate(
        shuffled,
        start=1,
    ):

        (
            patch,
            crop_boxes,
            crop_labels,
        ) = prepare_training_sample(
            sample,
            pipeline,
        )

        if len(
            crop_boxes
        ) > 0:

            positive_seen += 1

        else:

            negative_seen += 1

        images, targets = (
            pipeline.to_detector_batch(
                patch,
                crop_boxes,
                crop_labels,
                device,
            )
        )

        (
            total_loss,
            cls_loss,
            box_loss,
        ) = compute_loss(
            detector,
            images,
            targets,
        )

        # ----------------------------------------------------
        # Correct divisor for the final incomplete
        # accumulation group.
        # ----------------------------------------------------

        if (
            remainder != 0
            and step
            > (
                num_items
                - remainder
            )
        ):

            accumulation_divisor = (
                remainder
            )

        else:

            accumulation_divisor = (
                GRAD_ACCUM_STEPS
            )

        (
            total_loss
            / accumulation_divisor
        ).backward()

        should_step = (
            step
            % GRAD_ACCUM_STEPS
            == 0
            or step
            == num_items
        )

        if should_step:

            torch.nn.utils.clip_grad_norm_(
                detector.network.parameters(),
                GRAD_CLIP_NORM,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        raw_total = float(
            total_loss.detach().cpu()
        )

        raw_cls = float(
            cls_loss.detach().cpu()
        )

        raw_box = float(
            box_loss.detach().cpu()
        )

        sum_total += raw_total
        sum_cls += raw_cls
        sum_box += raw_box

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == num_items
        ):

            elapsed = (
                time.time()
                - start_time
            )

            avg_time = (
                elapsed
                / step
            )

            remaining = (
                avg_time
                * (
                    num_items
                    - step
                )
            )

            print(
                f"  Train "
                f"{step:4d}/{num_items} | "
                f"loss {raw_total:.4f} | "
                f"cls {raw_cls:.4f} | "
                f"box {raw_box:.4f} | "
                f"ETA {remaining / 60:.1f} min"
            )

    duration = (
        time.time()
        - start_time
    )

    count = max(
        num_items,
        1,
    )

    return {
        "loss": (
            sum_total
            / count
        ),

        "cls_loss": (
            sum_cls
            / count
        ),

        "box_loss": (
            sum_box
            / count
        ),

        "positive_seen": (
            positive_seen
        ),

        "negative_seen": (
            negative_seen
        ),

        "seconds": (
            duration
        ),
    }


# ============================================================
# Validation
# ============================================================

def validate_one_epoch(
    detector,
    validation,
    pipeline,
    device,
):

    # --------------------------------------------------------
    # RetinaNetDetector needs training mode to calculate
    # target losses.
    #
    # BN layers are explicitly returned to eval mode.
    # --------------------------------------------------------

    detector.train()

    detector.apply(
        set_batchnorm_eval
    )

    sum_total = 0.0
    sum_cls = 0.0
    sum_box = 0.0

    positive_seen = 0
    negative_seen = 0

    start_time = time.time()

    with torch.no_grad():

        for step, sample in enumerate(
            validation,
            start=1,
        ):

            (
                patch,
                crop_boxes,
                crop_labels,
            ) = prepare_validation_sample(
                sample,
                pipeline,
            )

            if len(
                crop_boxes
            ) > 0:

                positive_seen += 1

            else:

                negative_seen += 1

            images, targets = (
                pipeline.to_detector_batch(
                    patch,
                    crop_boxes,
                    crop_labels,
                    device,
                )
            )

            (
                total_loss,
                cls_loss,
                box_loss,
            ) = compute_loss(
                detector,
                images,
                targets,
            )

            sum_total += float(
                total_loss.detach().cpu()
            )

            sum_cls += float(
                cls_loss.detach().cpu()
            )

            sum_box += float(
                box_loss.detach().cpu()
            )

    duration = (
        time.time()
        - start_time
    )

    count = max(
        len(
            validation
        ),
        1,
    )

    return {
        "loss": (
            sum_total
            / count
        ),

        "cls_loss": (
            sum_cls
            / count
        ),

        "box_loss": (
            sum_box
            / count
        ),

        "positive_seen": (
            positive_seen
        ),

        "negative_seen": (
            negative_seen
        ),

        "seconds": (
            duration
        ),
    }


# ============================================================
# Checkpoint
# ============================================================

def make_checkpoint(
    epoch,
    detector,
    optimizer,
    scheduler,
    best_val_loss,
    epochs_without_improvement,
    history,
    config,
):

    return {
        "epoch": epoch,

        "model_state_dict": (
            detector.network.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
        ),

        "best_val_loss": (
            best_val_loss
        ),

        "epochs_without_improvement": (
            epochs_without_improvement
        ),

        "history": (
            history
        ),

        "config": (
            config
        ),

        "saved_at": (
            datetime.now().isoformat()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "Candidate 2 - "
        "Candidate 1 Aligned Med3D Fine-Tuning"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # Device
    # ========================================================

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
    # Shared pipeline
    # ========================================================

    print()
    print(
        "[1] Load validated Candidate 2 pipeline"
    )

    pipeline = (
        load_pipeline_module()
    )

    print(
        "Spacing:",
        pipeline.TARGET_SPACING,
    )

    print(
        "Patch:",
        pipeline.PATCH_SIZE,
    )

    if tuple(
        pipeline.TARGET_SPACING
    ) != (
        0.703125,
        0.703125,
        1.25,
    ):

        raise RuntimeError(
            "TARGET_SPACING 불일치"
        )

    if tuple(
        pipeline.PATCH_SIZE
    ) != (
        256,
        256,
        96,
    ):

        raise RuntimeError(
            "PATCH_SIZE 불일치"
        )

    print(
        "✅ Pipeline config PASS"
    )


    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "[2] Load aligned split"
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

    final_gt = count_boxes(
        final_test
    )

    print()
    print(
        "TRAIN"
    )

    print(
        " Total:",
        len(
            training
        ),
    )

    print(
        " Positive:",
        train_pos,
    )

    print(
        " Negative:",
        train_neg,
    )

    print()
    print(
        "INTERNAL VAL"
    )

    print(
        " Total:",
        len(
            validation
        ),
    )

    print(
        " Positive:",
        val_pos,
    )

    print(
        " Negative:",
        val_neg,
    )

    print()
    print(
        "FINAL TEST"
    )

    print(
        " CT:",
        len(
            final_test
        ),
    )

    print(
        " GT:",
        final_gt,
    )

    print(
        " Training 사용: NO"
    )

    if (
        len(
            training
        )
        != EXPECTED_TRAIN
        or train_pos
        != EXPECTED_TRAIN_POSITIVE
        or train_neg
        != EXPECTED_TRAIN_NEGATIVE
    ):

        raise RuntimeError(
            "Training split 불일치"
        )

    if (
        len(
            validation
        )
        != EXPECTED_INTERNAL_VAL
        or val_pos
        != EXPECTED_VAL_POSITIVE
        or val_neg
        != EXPECTED_VAL_NEGATIVE
    ):

        raise RuntimeError(
            "Validation split 불일치"
        )

    if (
        len(
            final_test
        )
        != EXPECTED_FINAL_TEST
        or final_gt
        != EXPECTED_FINAL_GT
    ):

        raise RuntimeError(
            "Final test 불일치"
        )

    print()
    print(
        "✅ Dataset PASS"
    )


    # ========================================================
    # Build model
    # ========================================================

    print()
    print(
        "[3] Build Med3D + FPN + RetinaNet"
    )

    detector = (
        pipeline.build_detector(
            device
        )
    )

    print(
        "✅ Detector 생성"
    )


    # ========================================================
    # Optimizer / Scheduler
    # ========================================================

    optimizer = torch.optim.AdamW(
        detector.network.parameters(),

        lr=(
            LEARNING_RATE
        ),

        weight_decay=(
            WEIGHT_DECAY
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,

            mode="min",

            factor=(
                LR_FACTOR
            ),

            patience=(
                LR_PATIENCE
            ),

            min_lr=(
                MIN_LR
            ),
        )
    )


    # ========================================================
    # Save config
    # ========================================================

    config = {
        "candidate": 2,

        "model": (
            "Med3D pretrained ResNet50 "
            "+ Custom Multi-scale FPN "
            "+ MONAI 3D RetinaNet"
        ),

        "fine_tuning": True,

        "pretrained_weight": str(
            pipeline.MED3D_WEIGHT
        ),

        "split": {
            "train": len(
                training
            ),

            "train_positive": (
                train_pos
            ),

            "train_negative": (
                train_neg
            ),

            "internal_validation": len(
                validation
            ),

            "val_positive": (
                val_pos
            ),

            "val_negative": (
                val_neg
            ),

            "final_test": len(
                final_test
            ),

            "final_test_gt": (
                final_gt
            ),

            "final_test_used_during_training": False,
        },

        "preprocessing": {
            "spacing_xyz": list(
                pipeline.TARGET_SPACING
            ),

            "patch_size_xyz": list(
                pipeline.PATCH_SIZE
            ),

            "hu_min": -1024.0,

            "hu_max": 300.0,

            "normalization": (
                "0_to_1"
            ),
        },

        "sampling": {
            "one_random_crop_per_ct_per_epoch": True,

            "positive": (
                "GT-centered crop + jitter"
            ),

            "negative": (
                "random background crop"
            ),
        },

        "augmentation": {
            "flip_probability_per_axis": (
                FLIP_PROB
            ),

            "intensity_probability": (
                INTENSITY_AUG_PROB
            ),

            "intensity_scale_range": list(
                INTENSITY_SCALE_RANGE
            ),

            "intensity_shift_range": list(
                INTENSITY_SHIFT_RANGE
            ),
        },

        "training": {
            "max_epochs": (
                MAX_EPOCHS
            ),

            "optimizer": (
                "AdamW"
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "weight_decay": (
                WEIGHT_DECAY
            ),

            "batch_size": 1,

            "gradient_accumulation": (
                GRAD_ACCUM_STEPS
            ),

            "effective_batch_size": (
                GRAD_ACCUM_STEPS
            ),

            "gradient_clip_norm": (
                GRAD_CLIP_NORM
            ),

            "scheduler": (
                "ReduceLROnPlateau"
            ),

            "scheduler_factor": (
                LR_FACTOR
            ),

            "scheduler_patience": (
                LR_PATIENCE
            ),

            "min_lr": (
                MIN_LR
            ),

            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),

            "early_stopping_min_delta": (
                EARLY_STOPPING_MIN_DELTA
            ),

            "mixed_precision": False,
        },

        "validation": {
            "deterministic_crop": True,

            "reason": (
                "stable val loss for "
                "scheduler and early stopping"
            ),
        },

        "seed": (
            SEED
        ),
    }

    save_json(
        TRAINING_CONFIG_JSON,
        config,
    )

    print()
    print(
        "Training config:"
    )

    print(
        TRAINING_CONFIG_JSON
    )


    # ========================================================
    # Training loop
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "[4] Fine-tuning 시작"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Max epochs:",
        MAX_EPOCHS,
    )

    print(
        "LR:",
        LEARNING_RATE,
    )

    print(
        "Gradient accumulation:",
        GRAD_ACCUM_STEPS,
    )

    print(
        "Effective batch:",
        GRAD_ACCUM_STEPS,
    )

    print(
        "Early stopping patience:",
        EARLY_STOPPING_PATIENCE,
    )

    print(
        "Mixed precision: OFF (FP32)"
    )

    print()


    best_val_loss = math.inf

    epochs_without_improvement = 0

    history = []

    training_start = time.time()


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        print()
        print(
            "#" * 80
        )

        print(
            f"# Epoch "
            f"{epoch}/{MAX_EPOCHS}"
        )

        print(
            "#" * 80
        )

        current_lr = (
            optimizer
            .param_groups[
                0
            ][
                "lr"
            ]
        )

        print()
        print(
            "LR:",
            current_lr,
        )


        # ====================================================
        # Train
        # ====================================================

        train_metrics = train_one_epoch(
            detector,
            optimizer,
            training,
            pipeline,
            device,
            epoch,
        )


        # ====================================================
        # Validation
        # ====================================================

        val_metrics = validate_one_epoch(
            detector,
            validation,
            pipeline,
            device,
        )


        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step(
            val_metrics[
                "loss"
            ]
        )

        next_lr = (
            optimizer
            .param_groups[
                0
            ][
                "lr"
            ]
        )


        # ====================================================
        # Improvement
        # ====================================================

        improved = (
            val_metrics[
                "loss"
            ]
            <
            (
                best_val_loss
                - EARLY_STOPPING_MIN_DELTA
            )
        )

        if improved:

            best_val_loss = (
                val_metrics[
                    "loss"
                ]
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        # ====================================================
        # Epoch record
        # ====================================================

        record = {
            "epoch": (
                epoch
            ),

            "lr_before_scheduler": (
                current_lr
            ),

            "lr_after_scheduler": (
                next_lr
            ),

            "train": (
                train_metrics
            ),

            "validation": (
                val_metrics
            ),

            "best_val_loss": (
                best_val_loss
            ),

            "improved": (
                improved
            ),

            "epochs_without_improvement": (
                epochs_without_improvement
            ),
        }

        history.append(
            record
        )

        save_json(
            HISTORY_JSON,
            history,
        )


        # ====================================================
        # Save last checkpoint
        # ====================================================

        last_state = make_checkpoint(
            epoch,
            detector,
            optimizer,
            scheduler,
            best_val_loss,
            epochs_without_improvement,
            history,
            config,
        )

        torch.save(
            last_state,
            LAST_CHECKPOINT,
        )


        # ====================================================
        # Save best checkpoint
        # ====================================================

        if improved:

            torch.save(
                last_state,
                BEST_CHECKPOINT,
            )


        # ====================================================
        # Output
        # ====================================================

        print()
        print(
            "-" * 80
        )

        print(
            f"Epoch {epoch} summary"
        )

        print(
            "-" * 80
        )

        print(
            "Train loss:",
            f"{train_metrics['loss']:.6f}",
        )

        print(
            "  cls:",
            f"{train_metrics['cls_loss']:.6f}",
        )

        print(
            "  box:",
            f"{train_metrics['box_loss']:.6f}",
        )

        print(
            "  pos / neg:",
            f"{train_metrics['positive_seen']} "
            f"/ {train_metrics['negative_seen']}",
        )

        print(
            "  time:",
            f"{train_metrics['seconds'] / 60:.1f} min",
        )

        print()

        print(
            "Val loss:",
            f"{val_metrics['loss']:.6f}",
        )

        print(
            "  cls:",
            f"{val_metrics['cls_loss']:.6f}",
        )

        print(
            "  box:",
            f"{val_metrics['box_loss']:.6f}",
        )

        print(
            "  pos / neg:",
            f"{val_metrics['positive_seen']} "
            f"/ {val_metrics['negative_seen']}",
        )

        print(
            "  time:",
            f"{val_metrics['seconds'] / 60:.1f} min",
        )

        print()

        print(
            "Best val loss:",
            f"{best_val_loss:.6f}",
        )

        print(
            "Improved:",
            improved,
        )

        print(
            "No improvement:",
            f"{epochs_without_improvement}"
            f"/{EARLY_STOPPING_PATIENCE}",
        )

        print(
            "LR:",
            f"{current_lr:.8f}"
            " → "
            f"{next_lr:.8f}",
        )

        print()

        print(
            "Best checkpoint:",
            BEST_CHECKPOINT,
        )

        print(
            "Last checkpoint:",
            LAST_CHECKPOINT,
        )


        # ====================================================
        # Early stopping
        # ====================================================

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "=" * 80
            )

            print(
                "Early stopping"
            )

            print(
                f"{EARLY_STOPPING_PATIENCE} epochs "
                "동안 validation loss 개선 없음"
            )

            print(
                "=" * 80
            )

            break


    # ========================================================
    # Finished
    # ========================================================

    total_seconds = (
        time.time()
        - training_start
    )

    final_epoch = (
        history[
            -1
        ][
            "epoch"
        ]
        if history
        else 0
    )

    print()
    print(
        "=" * 80
    )

    print(
        "✅ Candidate 2 Fine-Tuning 완료"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Final epoch:",
        final_epoch,
    )

    print(
        "Best val loss:",
        best_val_loss,
    )

    print(
        "Total training time:",
        f"{total_seconds / 3600:.2f} h",
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        "Last checkpoint:"
    )

    print(
        LAST_CHECKPOINT
    )

    print()
    print(
        "History:"
    )

    print(
        HISTORY_JSON
    )

    print()
    print(
        "Final held-out test:"
    )

    print(
        f"{EXPECTED_FINAL_TEST} CT / "
        f"{EXPECTED_FINAL_GT} GT"
    )

    print(
        "학습 중 사용하지 않음 ✅"
    )

    print()
    print(
        "Next:"
    )

    print(
        "22_evaluate_candidate2_fold0.py"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
