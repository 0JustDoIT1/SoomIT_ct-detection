#!/usr/bin/env python3

import json
import random
import sys
import time
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

TRAIN_CSV = (
    DATA_ROOT
    / "train.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "mini_overfit"
)

BEST_CHECKPOINT = (
    OUTPUT_ROOT
    / "best.pt"
)

LAST_CHECKPOINT = (
    OUTPUT_ROOT
    / "last.pt"
)

HISTORY_PATH = (
    OUTPUT_ROOT
    / "history.json"
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
# Config
# =============================================================================

SEED = 42

NUM_CASES = 5

NUM_EPOCHS = 20

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

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

TP_RATIO = 1.0

POS_RATIO = 1.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# Utility
# =============================================================================

def header(
    title,
):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def set_seed(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


def normalize_ct(
    image,
):
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
# Find positive CTs
# =============================================================================

def select_positive_cases():

    df = pd.read_csv(
        TRAIN_CSV
    )

    df = df[
        df["label"].astype(str)
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
        TRAIN_ROOT.glob(
            "*.nii.gz"
        )
    )

    selected = []

    for ct_path in ct_paths:

        possible_keys = [
            ct_path.name,
            ct_path.name.replace(
                ".nii.gz",
                "",
            ),
        ]

        matched_key = None

        for key in possible_keys:

            if key in grouped:
                matched_key = key
                break

        if matched_key is None:
            continue

        selected.append(
            {
                "path": ct_path,
                "key": matched_key,
                "annotations":
                    grouped[
                        matched_key
                    ],
            }
        )

        if len(
            selected
        ) >= NUM_CASES:
            break

    if len(
        selected
    ) < NUM_CASES:

        raise RuntimeError(
            f"Positive CT가 부족합니다: "
            f"{len(selected)}/{NUM_CASES}"
        )

    return selected


# =============================================================================
# Dataset
# =============================================================================

class MiniOverfitDataset(
    Dataset
):

    def __init__(
        self,
        cases,
        crop_fn,
        transform_post,
    ):

        self.cases = cases

        self.crop_fn = crop_fn

        self.transform_post = (
            transform_post
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

        case = self.cases[
            idx
        ]

        ct_path = case[
            "path"
        ]

        annotations = case[
            "annotations"
        ]

        image_itk = (
            sitk.ReadImage(
                str(
                    ct_path
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

        expected = np.asarray(
            SPACING,
            dtype=np.float32,
        )

        if not np.allclose(
            spacing_zyx,
            expected,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Spacing mismatch: "
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

        # ---------------------------------------------------------
        # CSV XYZ → CPM ZYX
        # ---------------------------------------------------------

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
            )
            [:, ::-1]
        )

        # w,h,d → d,h,w
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
            )
            [:, ::-1]
        )

        all_cls = np.zeros(
            len(
                annotations
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
                    ct_path
                ),
        }

        samples = self.crop_fn(
            data,
            spacing_zyx,
        )

        if len(
            samples
        ) == 0:

            raise RuntimeError(
                f"No crop generated: "
                f"{ct_path}"
            )

        output_samples = []

        for sample in samples:

            sample = (
                self.transform_post(
                    sample
                )
            )

            sample["image"] = (
                sample["image"]
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
# Train
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
):

    model.train()

    total_sum = 0.0
    cls_sum = 0.0
    shape_sum = 0.0
    offset_sum = 0.0
    iou_sum = 0.0

    steps = 0

    for batch in loader:

        images = (
            batch[
                "image"
            ]
            .float()
            .to(
                DEVICE
            )
        )

        labels = (
            batch[
                "annot"
            ]
            .float()
            .to(
                DEVICE
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

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

        # Original CPMNetv2 weighting
        total_loss = (
            4.0
            * cls_loss
            + 0.1
            * shape_loss
            + 1.0
            * offset_loss
            + 1.0
            * iou_loss
        )

        if not torch.isfinite(
            total_loss
        ):
            raise RuntimeError(
                "NaN/Inf loss"
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        total_sum += float(
            total_loss
            .detach()
            .cpu()
        )

        cls_sum += float(
            cls_loss
            .detach()
            .cpu()
        )

        shape_sum += float(
            shape_loss
            .detach()
            .cpu()
        )

        offset_sum += float(
            offset_loss
            .detach()
            .cpu()
        )

        iou_sum += float(
            iou_loss
            .detach()
            .cpu()
        )

        steps += 1

    steps = max(
        steps,
        1,
    )

    return {
        "loss":
            total_sum
            / steps,

        "cls_loss":
            cls_sum
            / steps,

        "shape_loss":
            shape_sum
            / steps,

        "offset_loss":
            offset_sum
            / steps,

        "iou_loss":
            iou_sum
            / steps,
    }


# =============================================================================
# Save
# =============================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
):

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                metrics,

            "crop_size":
                CROP_SIZE,

            "spacing":
                SPACING,

            "num_cases":
                NUM_CASES,

            "num_epochs":
                NUM_EPOCHS,
        },
        path,
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
        "CPMNetv2 Mini Overfit Test"
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
        "Epochs      :",
        NUM_EPOCHS,
    )

    print(
        "Crop        :",
        CROP_SIZE,
    )

    print(
        "Spacing     :",
        SPACING,
    )

    print(
        "TP ratio    :",
        TP_RATIO,
    )

    print(
        "POS ratio   :",
        POS_RATIO,
    )

    if DEVICE.type != "cuda":

        raise RuntimeError(
            "CUDA required."
        )

    # -------------------------------------------------------------------------
    # Select cases
    # -------------------------------------------------------------------------

    cases = (
        select_positive_cases()
    )

    header(
        "Selected Positive Cases"
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            f"[{index}] "
            f"{case['path'].name} "
            f"| GT={len(case['annotations'])}"
        )

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    transform_post = (
        torchvision
        .transforms
        .Compose(
            [
                transform.Pad(
                    output_size=(
                        CROP_SIZE
                    )
                ),

                transform.RandomCrop(
                    output_size=(
                        CROP_SIZE
                    ),
                    pos_ratio=(
                        POS_RATIO
                    ),
                ),

                transform.CoordToAnnot(
                    blank_side=0
                ),
            ]
        )
    )

    crop_fn = InstanceCrop(
        crop_size=CROP_SIZE,

        tp_ratio=TP_RATIO,

        spacing=SPACING,

        rand_trans=None,

        rand_rot=None,

        rand_space=None,

        sample_num=1,

        blank_side=0,

        instance_crop=True,
    )

    dataset = (
        MiniOverfitDataset(
            cases=cases,
            crop_fn=crop_fn,
            transform_post=(
                transform_post
            ),
        )
    )

    loader = DataLoader(
        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=(
            collate_fn_dict
        ),

        num_workers=0,

        pin_memory=False,

        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = (
        build_model()
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),

            lr=(
                LEARNING_RATE
            ),

            weight_decay=(
                WEIGHT_DECAY
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    best_loss = float(
        "inf"
    )

    history = []

    header(
        "Training"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        start = time.time()

        metrics = (
            train_one_epoch(
                model,
                loader,
                optimizer,
            )
        )

        duration = (
            time.time()
            - start
        )

        row = {
            "epoch":
                epoch,

            **metrics,

            "seconds":
                duration,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"loss {metrics['loss']:.6f} | "
            f"cls {metrics['cls_loss']:.6f} | "
            f"shape {metrics['shape_loss']:.6f} | "
            f"offset {metrics['offset_loss']:.6f} | "
            f"iou {metrics['iou_loss']:.6f} | "
            f"{duration:.1f}s"
        )

        if (
            metrics[
                "loss"
            ]
            < best_loss
        ):

            best_loss = (
                metrics[
                    "loss"
                ]
            )

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                epoch,
                metrics,
            )

        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            epoch,
            metrics,
        )

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    header(
        "Mini Overfit Result"
    )

    first = history[
        0
    ][
        "loss"
    ]

    last = history[
        -1
    ][
        "loss"
    ]

    reduction = (
        first
        - last
    )

    reduction_percent = (
        reduction
        / first
        * 100.0
        if first > 0
        else 0.0
    )

    print(
        f"Initial loss : "
        f"{first:.6f}"
    )

    print(
        f"Final loss   : "
        f"{last:.6f}"
    )

    print(
        f"Reduction    : "
        f"{reduction:.6f}"
    )

    print(
        f"Reduction %  : "
        f"{reduction_percent:.2f}%"
    )

    print(
        f"Best loss    : "
        f"{best_loss:.6f}"
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
        "History:"
    )

    print(
        HISTORY_PATH
    )

    print()

    if (
        last
        < first * 0.5
    ):

        print(
            "PASS: Mini dataset에서 "
            "강한 학습 신호가 확인되었습니다."
        )

    elif last < first:

        print(
            "PARTIAL: Loss는 감소했지만 "
            "충분한 overfit 여부는 추가 확인이 필요합니다."
        )

    else:

        print(
            "FAIL: Mini dataset에서도 "
            "loss 감소가 확인되지 않습니다."
        )


if __name__ == "__main__":
    main()