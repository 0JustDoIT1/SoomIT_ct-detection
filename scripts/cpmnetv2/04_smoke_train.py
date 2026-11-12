#!/usr/bin/env python3

import os
import sys
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

PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

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

TRAIN_ROOT = DATA_ROOT / "imagesTr"
TRAIN_CSV = DATA_ROOT / "train.csv"

SMOKE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "smoke"
)

SMOKE_CHECKPOINT = (
    SMOKE_ROOT
    / "smoke_checkpoint.pt"
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

from dataload.crop import InstanceCrop

from dataload.dataset import (
    collate_fn_dict,
)

import transform


# =============================================================================
# Configuration
# =============================================================================

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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# Utility
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_luna_ct(image):
    """
    LUNA16 / lung CT normalization.

    HU:
        [-1200, 600]
             ↓
        clip
             ↓
        [0, 1]

    Final [-1,1] conversion is performed
    after crop/transform, matching CPMNetv2.
    """

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
        image - HU_MIN
    ) / (
        HU_MAX - HU_MIN
    )

    return image.astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# LUNA16 Dataset
# =============================================================================

class LUNA16CPMDataset(Dataset):

    def __init__(
        self,
        root,
        csv_file,
        crop_fn,
        transform_post=None,
    ):
        self.root = Path(root)

        self.crop_fn = crop_fn

        self.transform_post = (
            transform_post
        )

        # ---------------------------------------------------------------------
        # CT list
        # ---------------------------------------------------------------------

        self.ct_list = sorted(
            self.root.glob(
                "*.nii.gz"
            )
        )

        if len(self.ct_list) == 0:
            raise RuntimeError(
                f"No CT found in {self.root}"
            )

        # ---------------------------------------------------------------------
        # CSV
        # ---------------------------------------------------------------------

        self.csv_df = pd.read_csv(
            csv_file
        )

        required = [
            "seriesuid",
            "coordX",
            "coordY",
            "coordZ",
            "w",
            "h",
            "d",
            "label",
        ]

        missing = [
            c
            for c in required
            if c not in self.csv_df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing CSV columns: {missing}"
            )

        self.csv_groups = {
            str(uid): group.copy()
            for uid, group
            in self.csv_df.groupby(
                "seriesuid"
            )
        }

    def __len__(self):
        return len(
            self.ct_list
        )

    def __getitem__(
        self,
        idx,
    ):
        ct_path = (
            self.ct_list[idx]
        )

        # ---------------------------------------------------------------------
        # Read CT
        # ---------------------------------------------------------------------

        image_itk = sitk.ReadImage(
            str(ct_path)
        )

        image_spacing = np.asarray(
            image_itk.GetSpacing()[::-1],
            dtype=np.float32,
        )

        # expected [z,y,x]
        if not np.allclose(
            image_spacing,
            np.array(
                SPACING,
                dtype=np.float32,
            ),
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Unexpected spacing "
                f"{image_spacing} "
                f"for {ct_path.name}"
            )

        image = (
            sitk.GetArrayFromImage(
                image_itk
            )
            .astype(
                np.float32
            )
        )

        # ---------------------------------------------------------------------
        # CT-specific normalization
        # ---------------------------------------------------------------------

        image = normalize_luna_ct(
            image
        )

        # ---------------------------------------------------------------------
        # Annotation
        # ---------------------------------------------------------------------

        group = self.csv_groups.get(
            ct_path.name
        )

        if group is None:
            # Negative CT
            all_loc = np.zeros(
                (0, 3),
                dtype=np.float32,
            )

            all_rad = np.zeros(
                (0, 3),
                dtype=np.float32,
            )

            all_cls = np.zeros(
                (0,),
                dtype=np.int8,
            )

        else:
            # CSV:
            #
            # coordX coordY coordZ
            # ↓
            # dataset convention:
            # z y x

            all_loc = (
                group[
                    [
                        "coordX",
                        "coordY",
                        "coordZ",
                    ]
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            all_loc = (
                all_loc[:, ::-1]
            )

            # CSV:
            #
            # w h d (millimeters)
            # ↓
            # CPM:
            # d h w

            all_rad = (
                group[
                    [
                        "w",
                        "h",
                        "d",
                    ]
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            all_rad = (
                all_rad[:, ::-1]
            )

            # -------------------------------------------------------------
            # IMPORTANT:
            # all annotations in LUNA candidate
            # are pulmonary nodules.
            #
            # CPMNetv2:
            # lesion class = 0
            # -------------------------------------------------------------

            labels = (
                group["label"]
                .astype(str)
                .to_numpy()
            )

            valid = (
                labels == "nodule"
            )

            if not valid.all():
                invalid_labels = (
                    np.unique(
                        labels[~valid]
                    )
                )

                raise RuntimeError(
                    f"Unexpected labels: "
                    f"{invalid_labels}"
                )

            all_cls = np.zeros(
                len(group),
                dtype=np.int8,
            )

        # ---------------------------------------------------------------------
        # CPM crop input
        # ---------------------------------------------------------------------

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
                str(ct_path),
        }

        samples = self.crop_fn(
            data,
            image_spacing,
        )

        output_samples = []

        for sample in samples:

            if (
                self.transform_post
                is not None
            ):
                sample = (
                    self.transform_post(
                        sample
                    )
                )

            # CPMNetv2 network input:
            # [0,1] -> [-1,1]

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
# Main
# =============================================================================

def main():

    print_header(
        "CPMNetv2 LUNA16 Smoke Training"
    )

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Data root   : {DATA_ROOT}"
    )

    print(
        f"Train images: {TRAIN_ROOT}"
    )

    print(
        f"Train CSV   : {TRAIN_CSV}"
    )

    print(
        f"Crop size   : {CROP_SIZE}"
    )

    print(
        f"Spacing     : {SPACING}"
    )

    print(
        f"HU window   : "
        f"[{HU_MIN}, {HU_MAX}]"
    )

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA is required for "
            "this smoke test."
        )

    SMOKE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Transform
    #
    # Keep the smoke test conservative.
    # No random rotations or spacing augmentation yet.
    # -------------------------------------------------------------------------

    transform_list = [
        transform.Pad(
            output_size=CROP_SIZE
        ),

        transform.RandomCrop(
            output_size=CROP_SIZE,
            pos_ratio=0.9,
        ),

        transform.CoordToAnnot(
            blank_side=0
        ),
    ]

    transform_post = (
        torchvision.transforms.Compose(
            transform_list
        )
    )

    # -------------------------------------------------------------------------
    # Crop
    # -------------------------------------------------------------------------

    crop_fn = InstanceCrop(
        crop_size=CROP_SIZE,

        tp_ratio=0.75,

        spacing=SPACING,

        rand_trans=None,
        rand_rot=None,
        rand_space=None,

        sample_num=1,

        blank_side=0,

        instance_crop=True,
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    dataset = LUNA16CPMDataset(
        root=TRAIN_ROOT,
        csv_file=TRAIN_CSV,
        crop_fn=crop_fn,
        transform_post=transform_post,
    )

    print()
    print(
        f"Dataset CT count: "
        f"{len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn_dict,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # Detection loss
    # -------------------------------------------------------------------------

    detection_loss = (
        Detection_loss(
            crop_size=CROP_SIZE,
            topk=5,
            spacing=SPACING,
        )
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

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

    model = model.to(
        DEVICE
    )

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
    )

    # -------------------------------------------------------------------------
    # One smoke batch
    # -------------------------------------------------------------------------

    model.train()

    print_header(
        "Loading Smoke Batch"
    )

    batch = next(
        iter(loader)
    )

    images = (
        batch["image"]
        .float()
        .to(DEVICE)
    )

    labels = (
        batch["annot"]
        .float()
        .to(DEVICE)
    )

    print(
        f"Image shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Label shape : "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Image min   : "
        f"{images.min().item():.6f}"
    )

    print(
        f"Image max   : "
        f"{images.max().item():.6f}"
    )

    print(
        f"Image mean  : "
        f"{images.mean().item():.6f}"
    )

    print()
    print(
        "Annotations:"
    )

    print(
        labels.detach()
        .cpu()
        .numpy()
    )

    # -------------------------------------------------------------------------
    # Input QA
    # -------------------------------------------------------------------------

    if not torch.isfinite(
        images
    ).all():
        raise RuntimeError(
            "Image contains NaN/Inf."
        )

    if not torch.isfinite(
        labels
    ).all():
        raise RuntimeError(
            "Labels contain NaN/Inf."
        )

    if (
        images.min().item()
        < -1.001
        or
        images.max().item()
        > 1.001
    ):
        raise RuntimeError(
            "Image outside expected "
            "[-1,1] range."
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    print_header(
        "Forward Pass"
    )

    optimizer.zero_grad()

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

    cls_loss = cls_loss.mean()
    shape_loss = shape_loss.mean()
    offset_loss = offset_loss.mean()
    iou_loss = iou_loss.mean()

    print(
        f"cls loss    : "
        f"{cls_loss.item():.6f}"
    )

    print(
        f"shape loss  : "
        f"{shape_loss.item():.6f}"
    )

    print(
        f"offset loss : "
        f"{offset_loss.item():.6f}"
    )

    print(
        f"iou loss    : "
        f"{iou_loss.item():.6f}"
    )

    losses = [
        cls_loss,
        shape_loss,
        offset_loss,
        iou_loss,
    ]

    for name, loss in zip(
        [
            "cls",
            "shape",
            "offset",
            "iou",
        ],
        losses,
    ):
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{name} loss is NaN/Inf."
            )

    # Same weighting as original
    # train_CPMv2_MRA.py

    total_loss = (
        4.0 * cls_loss
        + 0.1 * shape_loss
        + 1.0 * offset_loss
        + 1.0 * iou_loss
    )

    print()
    print(
        f"Total loss  : "
        f"{total_loss.item():.6f}"
    )

    # -------------------------------------------------------------------------
    # Backward
    # -------------------------------------------------------------------------

    print_header(
        "Backward Pass"
    )

    total_loss.backward()

    finite_gradients = True
    gradient_parameters = 0

    for parameter in (
        model.parameters()
    ):
        if parameter.grad is None:
            continue

        gradient_parameters += 1

        if not torch.isfinite(
            parameter.grad
        ).all():
            finite_gradients = False
            break

    print(
        f"Parameters with gradients : "
        f"{gradient_parameters}"
    )

    print(
        f"Finite gradients          : "
        f"{finite_gradients}"
    )

    if not finite_gradients:
        raise RuntimeError(
            "Gradient contains NaN/Inf."
        )

    optimizer.step()

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                float(
                    total_loss.item()
                ),

            "crop_size":
                CROP_SIZE,

            "spacing":
                SPACING,

            "hu_window":
                [
                    HU_MIN,
                    HU_MAX,
                ],
        },
        SMOKE_CHECKPOINT,
    )

    print_header(
        "Smoke Test Result"
    )

    print(
        "[PASS] Dataset load"
    )

    print(
        "[PASS] LUNA16 CT normalization"
    )

    print(
        "[PASS] Positive nodule class"
    )

    print(
        "[PASS] InstanceCrop"
    )

    print(
        "[PASS] CoordToAnnot"
    )

    print(
        "[PASS] Model forward"
    )

    print(
        "[PASS] Detection losses finite"
    )

    print(
        "[PASS] Backward"
    )

    print(
        "[PASS] Optimizer step"
    )

    print(
        "[PASS] Checkpoint save"
    )

    print()
    print(
        f"Checkpoint:"
    )

    print(
        SMOKE_CHECKPOINT
    )

    print()
    print(
        "CPMNetv2 LUNA16 "
        "SMOKE TRAINING PASSED."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )