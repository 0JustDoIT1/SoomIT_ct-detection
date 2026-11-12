#!/usr/bin/env python3

import gc
import json
import math
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


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

TRAIN_ROOT = (
    DATA_ROOT
    / "imagesTr"
)

VAL_ROOT = (
    DATA_ROOT
    / "imagesVa"
)

TRAIN_CSV = (
    DATA_ROOT
    / "train.csv"
)

VAL_CSV = (
    DATA_ROOT
    / "val.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "full_training"
)

BEST_CHECKPOINT = (
    OUTPUT_ROOT
    / "best_val_loss.pt"
)

LAST_CHECKPOINT = (
    OUTPUT_ROOT
    / "last.pt"
)

HISTORY_JSON = (
    OUTPUT_ROOT
    / "history.json"
)

HISTORY_CSV = (
    OUTPUT_ROOT
    / "history.csv"
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
    Detection_loss,
)

from dataload.crop import (
    InstanceCrop,
)

from dataload.dataset import (
    collate_fn_dict,
)

import transform


# =============================================================================
# Training config
# =============================================================================

SEED = 42

MAX_EPOCHS = 100

BATCH_SIZE = 2
SAMPLE_NUM = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

VAL_INTERVAL = 5

MIN_EPOCHS = 30

EARLY_STOP_PATIENCE_EPOCHS = 20

EARLY_STOP_PATIENCE_CHECKS = (
    EARLY_STOP_PATIENCE_EPOCHS
    // VAL_INTERVAL
)

CROP_SIZE = [
    64,
    128,
    128,
]

SPACING = [
    1.0,
    1.0,
    1.0,
]

HU_MIN = -1200.0
HU_MAX = 600.0


# =============================================================================
# Sampling
# =============================================================================

TRAIN_TP_RATIO = 0.75
TRAIN_POS_RATIO = 0.90

# Validation은 항상 같은 random sequence로 다시 생성해서
# val loss 비교가 지나치게 흔들리지 않도록 한다.
VAL_TP_RATIO = 0.75
VAL_POS_RATIO = 0.90

VAL_SEED = 12345


# =============================================================================
# DataLoader
# =============================================================================

NUM_WORKERS = 2
PIN_MEMORY = True


# =============================================================================
# Resume
# =============================================================================

# 처음 학습:
RESUME_FROM = None

# 중간부터 재개하려면:
# RESUME_FROM = LAST_CHECKPOINT


# =============================================================================
# Device
# =============================================================================

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


def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


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


def format_time(seconds):

    if not math.isfinite(
        seconds
    ):

        return "NA"

    return str(
        timedelta(
            seconds=int(
                seconds
            )
        )
    )


def bytes_to_gb(value):

    return (
        value
        / (
            1024 ** 3
        )
    )


# =============================================================================
# Annotation loader
# =============================================================================

def load_annotation_groups(
    csv_path,
):

    df = pd.read_csv(
        csv_path
    )

    if "label" in df.columns:

        df = df[
            df["label"]
            .astype(str)
            == "nodule"
        ].copy()

    groups = {
        str(uid): group.copy()
        for uid, group
        in df.groupby(
            "seriesuid"
        )
    }

    return groups


# =============================================================================
# Build case list
# =============================================================================

def build_cases(
    image_root,
    csv_path,
):

    annotation_groups = (
        load_annotation_groups(
            csv_path
        )
    )

    image_paths = sorted(
        image_root.glob(
            "*.nii.gz"
        )
    )

    if not image_paths:

        raise RuntimeError(
            f"No NIfTI found: {image_root}"
        )

    cases = []

    positive_cases = 0
    negative_cases = 0
    total_nodules = 0

    for image_path in image_paths:

        possible_keys = [
            image_path.name,
            strip_nii_suffix(
                image_path.name
            ),
        ]

        annotations = None

        for key in possible_keys:

            if key in annotation_groups:

                annotations = (
                    annotation_groups[
                        key
                    ]
                )

                break

        if annotations is None:

            annotations = pd.DataFrame(
                columns=[
                    "seriesuid",
                    "coordX",
                    "coordY",
                    "coordZ",
                    "w",
                    "h",
                    "d",
                    "label",
                ]
            )

            negative_cases += 1

        else:

            positive_cases += 1

            total_nodules += len(
                annotations
            )

        cases.append(
            {
                "path":
                    image_path,

                "annotations":
                    annotations,
            }
        )

    return (
        cases,
        positive_cases,
        negative_cases,
        total_nodules,
    )


# =============================================================================
# Dataset
# =============================================================================

class CPMFullDataset(
    Dataset
):

    def __init__(
        self,
        cases,
        sample_num,
        tp_ratio,
        pos_ratio,
    ):

        self.cases = cases

        self.crop_fn = InstanceCrop(
            crop_size=CROP_SIZE,

            tp_ratio=tp_ratio,

            spacing=SPACING,

            rand_trans=None,
            rand_rot=None,
            rand_space=None,

            sample_num=sample_num,

            blank_side=0,

            instance_crop=True,
        )

        self.transform_post = (
            torchvision
            .transforms
            .Compose(
                [
                    transform.Pad(
                        output_size=CROP_SIZE
                    ),

                    transform.RandomCrop(
                        output_size=CROP_SIZE,

                        pos_ratio=pos_ratio,
                    ),

                    transform.CoordToAnnot(
                        blank_side=0
                    ),
                ]
            )
        )

    def __len__(
        self,
    ):

        return len(
            self.cases
        )

    def __getitem__(
        self,
        idx,
    ):

        case = (
            self.cases[
                idx
            ]
        )

        image_path = (
            case[
                "path"
            ]
        )

        annotations = (
            case[
                "annotations"
            ]
        )

        # ---------------------------------------------------------------------
        # CT
        # ---------------------------------------------------------------------

        image_itk = (
            sitk.ReadImage(
                str(
                    image_path
                )
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
            np.asarray(
                SPACING,
                dtype=np.float32,
            ),
            atol=1e-6,
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

        image = normalize_ct(
            image
        )

        # ---------------------------------------------------------------------
        # GT
        #
        # CSV:
        # coordX, coordY, coordZ
        #
        # CPMNetv2:
        # Z, Y, X
        # ---------------------------------------------------------------------

        if len(
            annotations
        ) > 0:

            all_loc = (
                annotations[
                    [
                        "coordX",
                        "coordY",
                        "coordZ",
                    ]
                ]
                .to_numpy(
                    dtype=np.float32
                )[
                    :,
                    ::-1
                ]
            )

            # w,h,d -> d,h,w
            all_rad = (
                annotations[
                    [
                        "w",
                        "h",
                        "d",
                    ]
                ]
                .to_numpy(
                    dtype=np.float32
                )[
                    :,
                    ::-1
                ]
            )

            all_cls = np.zeros(
                len(
                    annotations
                ),
                dtype=np.int8,
            )

        else:

            all_loc = np.empty(
                (
                    0,
                    3,
                ),
                dtype=np.float32,
            )

            all_rad = np.empty(
                (
                    0,
                    3,
                ),
                dtype=np.float32,
            )

            all_cls = np.empty(
                (
                    0,
                ),
                dtype=np.int8,
            )

        data = {
            "image":
                image,

            "all_loc":
                all_loc,

            "all_rad":
                all_rad,

            "all_cls":
                all_cls,

            "file_name":
                str(
                    image_path
                ),
        }

        # ---------------------------------------------------------------------
        # InstanceCrop
        # ---------------------------------------------------------------------

        samples = self.crop_fn(
            data,
            spacing_zyx,
        )

        if len(
            samples
        ) == 0:

            raise RuntimeError(
                f"No crop generated:\n"
                f"{image_path}"
            )

        output_samples = []

        for sample in samples:

            sample = (
                self.transform_post(
                    sample
                )
            )

            # Normalized [0,1]
            # -> CPMNetv2 [-1,1]
            sample[
                "image"
            ] = (
                sample[
                    "image"
                ]
                * 2.0
                - 1.0
            )

            output_samples.append(
                sample
            )

        return output_samples


# =============================================================================
# Model
# =============================================================================

def build_model():

    detection_loss = (
        Detection_loss(
            crop_size=CROP_SIZE,

            topk=5,

            spacing=SPACING,
        )
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

        detection_loss=(
            detection_loss
        ),

        device=DEVICE,
    )

    return model.to(
        DEVICE
    )


# =============================================================================
# Loss
# =============================================================================

def compute_loss(
    model,
    images,
    labels,
):

    (
        cls_loss,
        shape_loss,
        offset_loss,
        iou_loss,
    ) = model(
        [
            images,
            labels,
        ]
    )

    cls_loss = (
        cls_loss.mean()
    )

    shape_loss = (
        shape_loss.mean()
    )

    offset_loss = (
        offset_loss.mean()
    )

    iou_loss = (
        iou_loss.mean()
    )

    total_loss = (
        4.0
        * cls_loss

        + 0.1
        * shape_loss

        + offset_loss

        + iou_loss
    )

    return (
        total_loss,
        cls_loss,
        shape_loss,
        offset_loss,
        iou_loss,
    )


# =============================================================================
# Train epoch
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch,
):

    model.train()

    sums = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "shape_loss": 0.0,
        "offset_loss": 0.0,
        "iou_loss": 0.0,
    }

    steps = 0

    epoch_start = time.time()

    for step, batch in enumerate(
        loader,
        start=1,
    ):

        images = (
            batch[
                "image"
            ]
            .float()
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        labels = (
            batch[
                "annot"
            ]
            .float()
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            total_loss,
            cls_loss,
            shape_loss,
            offset_loss,
            iou_loss,
        ) = compute_loss(
            model,
            images,
            labels,
        )

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                f"NaN/Inf loss "
                f"epoch={epoch} "
                f"step={step}"
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        sums[
            "loss"
        ] += float(
            total_loss
            .detach()
            .cpu()
        )

        sums[
            "cls_loss"
        ] += float(
            cls_loss
            .detach()
            .cpu()
        )

        sums[
            "shape_loss"
        ] += float(
            shape_loss
            .detach()
            .cpu()
        )

        sums[
            "offset_loss"
        ] += float(
            offset_loss
            .detach()
            .cpu()
        )

        sums[
            "iou_loss"
        ] += float(
            iou_loss
            .detach()
            .cpu()
        )

        steps += 1

        # ---------------------------------------------------------------------
        # Progress every 100 iterations
        # ---------------------------------------------------------------------

        if (
            step % 100 == 0
            or step == len(
                loader
            )
        ):

            elapsed = (
                time.time()
                - epoch_start
            )

            sec_per_iter = (
                elapsed
                / step
            )

            remaining = (
                len(
                    loader
                )
                - step
            ) * sec_per_iter

            print(
                f"  Iter "
                f"{step:4d}/"
                f"{len(loader):4d}"
                f" | "
                f"loss="
                f"{sums['loss']/steps:.4f}"
                f" | "
                f"{sec_per_iter:.2f}s/it"
                f" | "
                f"ETA "
                f"{format_time(remaining)}"
            )

    metrics = {
        key:
            value
            / max(
                steps,
                1,
            )
        for key, value
        in sums.items()
    }

    metrics[
        "seconds"
    ] = (
        time.time()
        - epoch_start
    )

    metrics[
        "iterations"
    ] = steps

    return metrics


# =============================================================================
# Validation
# =============================================================================

def validate(
    model,
    loader,
):

    # CPMNetv2는 train mode에서
    # model([images, labels]) -> loss를 반환한다.
    #
    # eval mode에서는 inference 경로로 들어가기 때문에
    # validation loss 계산 시 top-level model은 train mode를 유지한다.
    model.train()

    # Validation 중 BatchNorm running statistics가
    # 변경되지 않도록 BatchNorm만 eval mode로 고정한다.
    for module in model.modules():

        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()

        if isinstance(
            module,
            (
                torch.nn.Dropout,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            ),
        ):
            module.eval()

    sums = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "shape_loss": 0.0,
        "offset_loss": 0.0,
        "iou_loss": 0.0,
    }

    steps = 0

    start_time = (
        time.time()
    )

    with torch.no_grad():

        for batch in loader:

            images = (
                batch[
                    "image"
                ]
                .float()
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            labels = (
                batch[
                    "annot"
                ]
                .float()
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            (
                total_loss,
                cls_loss,
                shape_loss,
                offset_loss,
                iou_loss,
            ) = compute_loss(
                model,
                images,
                labels,
            )

            if not torch.isfinite(
                total_loss
            ):

                raise RuntimeError(
                    "NaN/Inf validation loss"
                )

            sums[
                "loss"
            ] += float(
                total_loss
                .detach()
                .cpu()
            )

            sums[
                "cls_loss"
            ] += float(
                cls_loss
                .detach()
                .cpu()
            )

            sums[
                "shape_loss"
            ] += float(
                shape_loss
                .detach()
                .cpu()
            )

            sums[
                "offset_loss"
            ] += float(
                offset_loss
                .detach()
                .cpu()
            )

            sums[
                "iou_loss"
            ] += float(
                iou_loss
                .detach()
                .cpu()
            )

            steps += 1

    metrics = {
        key:
            value
            / max(
                steps,
                1,
            )
        for key, value
        in sums.items()
    }

    metrics[
        "seconds"
    ] = (
        time.time()
        - start_time
    )

    metrics[
        "iterations"
    ] = steps

    return metrics


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    bad_val_checks,
    history,
):

    checkpoint = {
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_val_loss":
            best_val_loss,

        "bad_val_checks":
            bad_val_checks,

        "history":
            history,

        "config":
            {
                "max_epochs":
                    MAX_EPOCHS,

                "batch_size":
                    BATCH_SIZE,

                "sample_num":
                    SAMPLE_NUM,

                "effective_crop_batch":
                    BATCH_SIZE
                    * SAMPLE_NUM,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "val_interval":
                    VAL_INTERVAL,

                "min_epochs":
                    MIN_EPOCHS,

                "early_stop_patience_epochs":
                    EARLY_STOP_PATIENCE_EPOCHS,

                "early_stop_patience_checks":
                    EARLY_STOP_PATIENCE_CHECKS,

                "crop_size":
                    CROP_SIZE,

                "spacing":
                    SPACING,

                "train_tp_ratio":
                    TRAIN_TP_RATIO,

                "train_pos_ratio":
                    TRAIN_POS_RATIO,
            },
    }

    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# History
# =============================================================================

def save_history(
    history,
):

    with open(
        HISTORY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_CSV,
        index=False,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    set_seed(
        SEED
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    header(
        "CPMNetv2 Full Training"
    )

    if DEVICE.type != "cuda":

        raise RuntimeError(
            "CUDA GPU required."
        )

    gpu = (
        torch.cuda
        .get_device_properties(
            DEVICE
        )
    )

    print(
        "GPU                  :",
        gpu.name,
    )

    print(
        "VRAM                 :",
        f"{bytes_to_gb(gpu.total_memory):.2f} GB",
    )

    print(
        "PyTorch              :",
        torch.__version__,
    )

    print(
        "PyTorch CUDA         :",
        torch.version.cuda,
    )

    print()
    print(
        "Max epochs           :",
        MAX_EPOCHS,
    )

    print(
        "Batch size           :",
        BATCH_SIZE,
    )

    print(
        "Sample num           :",
        SAMPLE_NUM,
    )

    print(
        "Effective crop batch :",
        BATCH_SIZE
        * SAMPLE_NUM,
    )

    print(
        "Validation interval  :",
        VAL_INTERVAL,
    )

    print(
        "Minimum epochs       :",
        MIN_EPOCHS,
    )

    print(
        "Early stop patience  :",
        f"{EARLY_STOP_PATIENCE_EPOCHS} epochs "
        f"({EARLY_STOP_PATIENCE_CHECKS} val checks)",
    )

    print(
        "Learning rate        :",
        LEARNING_RATE,
    )

    # =========================================================================
    # Data
    # =========================================================================

    header(
        "Dataset"
    )

    (
        train_cases,
        train_positive,
        train_negative,
        train_nodules,
    ) = build_cases(
        TRAIN_ROOT,
        TRAIN_CSV,
    )

    (
        val_cases,
        val_positive,
        val_negative,
        val_nodules,
    ) = build_cases(
        VAL_ROOT,
        VAL_CSV,
    )

    print(
        f"Train scans       : "
        f"{len(train_cases)}"
    )

    print(
        f"Train positive    : "
        f"{train_positive}"
    )

    print(
        f"Train no-GT       : "
        f"{train_negative}"
    )

    print(
        f"Train GT nodules  : "
        f"{train_nodules}"
    )

    print()

    print(
        f"Val scans         : "
        f"{len(val_cases)}"
    )

    print(
        f"Val positive      : "
        f"{val_positive}"
    )

    print(
        f"Val no-GT         : "
        f"{val_negative}"
    )

    print(
        f"Val GT nodules    : "
        f"{val_nodules}"
    )

    # =========================================================================
    # Dataset / Loader
    # =========================================================================

    train_dataset = CPMFullDataset(
        cases=train_cases,

        sample_num=SAMPLE_NUM,

        tp_ratio=TRAIN_TP_RATIO,

        pos_ratio=TRAIN_POS_RATIO,
    )

    val_dataset = CPMFullDataset(
        cases=val_cases,

        sample_num=SAMPLE_NUM,

        tp_ratio=VAL_TP_RATIO,

        pos_ratio=VAL_POS_RATIO,
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=collate_fn_dict,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        drop_last=False,

        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        collate_fn=collate_fn_dict,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        drop_last=False,

        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    print()

    print(
        "Train iterations/epoch:",
        len(
            train_loader
        ),
    )

    print(
        "Approx crops/epoch    :",
        len(
            train_loader
        )
        * BATCH_SIZE
        * SAMPLE_NUM,
    )

    # =========================================================================
    # Model
    # =========================================================================

    header(
        "Model"
    )

    model = build_model()

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY,
        )
    )

    start_epoch = 1

    best_val_loss = float(
        "inf"
    )

    bad_val_checks = 0

    history = []

    # =========================================================================
    # Resume
    # =========================================================================

    if RESUME_FROM is not None:

        resume_path = Path(
            RESUME_FROM
        )

        if not resume_path.exists():

            raise FileNotFoundError(
                resume_path
            )

        header(
            "Resume"
        )

        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                float(
                    "inf"
                ),
            )
        )

        bad_val_checks = int(
            checkpoint.get(
                "bad_val_checks",
                0,
            )
        )

        history = checkpoint.get(
            "history",
            [],
        )

        print(
            "Resume checkpoint:",
            resume_path,
        )

        print(
            "Start epoch      :",
            start_epoch,
        )

        print(
            "Best val loss    :",
            best_val_loss,
        )

    # =========================================================================
    # Training
    # =========================================================================

    header(
        "Training Start"
    )

    total_training_start = (
        time.time()
    )

    early_stopped = False

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):

        epoch_wall_start = (
            time.time()
        )

        print()
        print(
            f"[Epoch {epoch:03d}/{MAX_EPOCHS}]"
        )

        # ---------------------------------------------------------------------
        # Train
        # ---------------------------------------------------------------------

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                epoch,
            )
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_cls_loss":
                train_metrics[
                    "cls_loss"
                ],

            "train_shape_loss":
                train_metrics[
                    "shape_loss"
                ],

            "train_offset_loss":
                train_metrics[
                    "offset_loss"
                ],

            "train_iou_loss":
                train_metrics[
                    "iou_loss"
                ],

            "train_seconds":
                train_metrics[
                    "seconds"
                ],

            "val_loss":
                None,

            "val_cls_loss":
                None,

            "val_shape_loss":
                None,

            "val_offset_loss":
                None,

            "val_iou_loss":
                None,

            "val_seconds":
                None,

            "best_val_loss":
                (
                    None
                    if not math.isfinite(
                        best_val_loss
                    )
                    else best_val_loss
                ),
        }

        print(
            f"  Train | "
            f"loss={train_metrics['loss']:.6f} | "
            f"cls={train_metrics['cls_loss']:.6f} | "
            f"shape={train_metrics['shape_loss']:.6f} | "
            f"offset={train_metrics['offset_loss']:.6f} | "
            f"iou={train_metrics['iou_loss']:.6f} | "
            f"time={format_time(train_metrics['seconds'])}"
        )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        should_validate = (
            epoch % VAL_INTERVAL == 0
            or epoch == MAX_EPOCHS
        )

        if should_validate:

            val_metrics = validate(
                model,
                val_loader,
            )

            row[
                "val_loss"
            ] = val_metrics[
                "loss"
            ]

            row[
                "val_cls_loss"
            ] = val_metrics[
                "cls_loss"
            ]

            row[
                "val_shape_loss"
            ] = val_metrics[
                "shape_loss"
            ]

            row[
                "val_offset_loss"
            ] = val_metrics[
                "offset_loss"
            ]

            row[
                "val_iou_loss"
            ] = val_metrics[
                "iou_loss"
            ]

            row[
                "val_seconds"
            ] = val_metrics[
                "seconds"
            ]

            print(
                f"  Val   | "
                f"loss={val_metrics['loss']:.6f} | "
                f"cls={val_metrics['cls_loss']:.6f} | "
                f"shape={val_metrics['shape_loss']:.6f} | "
                f"offset={val_metrics['offset_loss']:.6f} | "
                f"iou={val_metrics['iou_loss']:.6f} | "
                f"time={format_time(val_metrics['seconds'])}"
            )

            # -----------------------------------------------------------------
            # Best validation loss
            # -----------------------------------------------------------------

            if (
                val_metrics[
                    "loss"
                ]
                < best_val_loss
            ):

                best_val_loss = (
                    val_metrics[
                        "loss"
                    ]
                )

                bad_val_checks = 0

                row[
                    "best_val_loss"
                ] = best_val_loss

                print(
                    f"  NEW BEST VAL LOSS: "
                    f"{best_val_loss:.6f}"
                )

                save_checkpoint(
                    BEST_CHECKPOINT,
                    model,
                    optimizer,
                    epoch,
                    best_val_loss,
                    bad_val_checks,
                    history
                    + [
                        row
                    ],
                )

            else:

                bad_val_checks += 1

                row[
                    "best_val_loss"
                ] = best_val_loss

                print(
                    f"  No improvement: "
                    f"{bad_val_checks}/"
                    f"{EARLY_STOP_PATIENCE_CHECKS}"
                )

        # ---------------------------------------------------------------------
        # Save history
        # ---------------------------------------------------------------------

        history.append(
            row
        )

        save_history(
            history
        )

        # ---------------------------------------------------------------------
        # Last checkpoint
        # ---------------------------------------------------------------------

        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            epoch,
            best_val_loss,
            bad_val_checks,
            history,
        )

        # ---------------------------------------------------------------------
        # ETA
        # ---------------------------------------------------------------------

        epoch_wall_time = (
            time.time()
            - epoch_wall_start
        )

        completed_epochs = (
            epoch
            - start_epoch
            + 1
        )

        elapsed_total = (
            time.time()
            - total_training_start
        )

        avg_epoch_time = (
            elapsed_total
            / completed_epochs
        )

        remaining_epochs = (
            MAX_EPOCHS
            - epoch
        )

        estimated_remaining = (
            avg_epoch_time
            * remaining_epochs
        )

        print(
            f"  Epoch wall time : "
            f"{format_time(epoch_wall_time)}"
        )

        print(
            f"  Total elapsed   : "
            f"{format_time(elapsed_total)}"
        )

        print(
            f"  Max-run ETA     : "
            f"{format_time(estimated_remaining)}"
        )

        print(
            f"  Best val loss   : "
            f"{best_val_loss:.6f}"
        )

        # ---------------------------------------------------------------------
        # GPU
        # ---------------------------------------------------------------------

        if torch.cuda.is_available():

            peak_allocated = (
                torch.cuda
                .max_memory_allocated()
            )

            print(
                f"  GPU peak alloc  : "
                f"{bytes_to_gb(peak_allocated):.2f} GB"
            )

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if (
            epoch >= MIN_EPOCHS
            and should_validate
            and bad_val_checks
            >= EARLY_STOP_PATIENCE_CHECKS
        ):

            print()
            print(
                f"EARLY STOP:"
            )

            print(
                f"Validation loss가 "
                f"{EARLY_STOP_PATIENCE_EPOCHS} epoch 동안 "
                f"개선되지 않았습니다."
            )

            early_stopped = True

            break

    # =========================================================================
    # Finished
    # =========================================================================

    header(
        "Training Finished"
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

    total_seconds = (
        time.time()
        - total_training_start
    )

    print(
        "Final epoch       :",
        final_epoch,
    )

    print(
        "Early stopped     :",
        early_stopped,
    )

    print(
        "Best val loss     :",
        f"{best_val_loss:.6f}",
    )

    print(
        "Total train time  :",
        format_time(
            total_seconds
        ),
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
        "History JSON:"
    )

    print(
        HISTORY_JSON
    )

    print()
    print(
        "History CSV:"
    )

    print(
        HISTORY_CSV
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "best_val_loss.pt를 이용하여 "
        "전체 validation inference 후 "
        "CPM / FROC / mAP / AP / Precision / Recall / "
        "F1 / size recall / localization metrics를 계산합니다."
    )

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()