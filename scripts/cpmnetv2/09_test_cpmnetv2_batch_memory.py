#!/usr/bin/env python3

import gc
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

PROJECT_ROOT = Path("/mnt/c/AI/ct-project")
CPMNET_ROOT = PROJECT_ROOT / "external" / "CPMNetv2"

DATA_ROOT = PROJECT_ROOT / "results" / "cpmnetv2" / "luna16"
TRAIN_ROOT = DATA_ROOT / "imagesTr"
TRAIN_CSV = DATA_ROOT / "train.csv"

sys.path.insert(0, str(CPMNET_ROOT))


# =============================================================================
# CPMNetv2
# =============================================================================

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
# Config
# =============================================================================

SEED = 42

# batch_size, sample_num
TEST_CONFIGS = [
    (1, 2),
    (2, 2),
    (3, 2),
]

TEST_ITERATIONS = 3

# 테스트용 positive CT 개수
NUM_CASES = 12

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

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# Utilities
# =============================================================================

def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        image - HU_MIN
    ) / (
        HU_MAX - HU_MIN
    )

    return image


def bytes_to_gb(value):
    return value / (1024 ** 3)


def cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Positive cases
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
        in df.groupby("seriesuid")
    }

    selected = []

    for ct_path in sorted(
        TRAIN_ROOT.glob("*.nii.gz")
    ):

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
                "path":
                    ct_path,

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

    if not selected:

        print()
        print("DEBUG")
        print(
            "Example CT:",
            next(
                iter(
                    sorted(
                        TRAIN_ROOT.glob(
                            "*.nii.gz"
                        )
                    )
                ),
                None,
            ),
        )

        print(
            "Example CSV seriesuid:",
            next(
                iter(
                    grouped.keys()
                ),
                None,
            ),
        )

        raise RuntimeError(
            "Positive CT를 찾지 못했습니다."
        )

    print(
        f"Matched positive CTs: "
        f"{len(selected)}"
    )

    return selected


# =============================================================================
# Dataset
# =============================================================================

class BatchTestDataset(Dataset):

    def __init__(
        self,
        cases,
        sample_num,
    ):

        self.cases = cases

        self.crop_fn = InstanceCrop(
            crop_size=CROP_SIZE,
            tp_ratio=0.75,
            spacing=SPACING,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
            sample_num=sample_num,
            blank_side=0,
            instance_crop=True,
        )

        self.transform_post = (
            torchvision.transforms.Compose(
                [
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
            )
        )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):

        case = self.cases[idx]

        image_itk = sitk.ReadImage(
            str(case["path"])
        )

        spacing_zyx = np.asarray(
            image_itk.GetSpacing()[::-1],
            dtype=np.float32,
        )

        image = sitk.GetArrayFromImage(
            image_itk
        ).astype(np.float32)

        image = normalize_ct(
            image
        )

        anns = case[
            "annotations"
        ]

        # XYZ -> ZYX
        all_loc = (
            anns[
                [
                    "coordX",
                    "coordY",
                    "coordZ",
                ]
            ]
            .to_numpy(
                dtype=np.float32
            )[:, ::-1]
        )

        # w,h,d -> d,h,w
        all_rad = (
            anns[
                [
                    "w",
                    "h",
                    "d",
                ]
            ]
            .to_numpy(
                dtype=np.float32
            )[:, ::-1]
        )

        all_cls = np.zeros(
            len(anns),
            dtype=np.int8,
        )

        data = {
            "image": image,
            "all_loc": all_loc,
            "all_rad": all_rad,
            "all_cls": all_cls,
            "file_name": str(
                case["path"]
            ),
        }

        samples = self.crop_fn(
            data,
            spacing_zyx,
        )

        output_samples = []

        for sample in samples:

            sample = self.transform_post(
                sample
            )

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
# Model
# =============================================================================

def build_model():

    detection_loss = Detection_loss(
        crop_size=CROP_SIZE,
        topk=5,
        spacing=SPACING,
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

        detection_loss=detection_loss,

        device=DEVICE,
    )

    return model.to(
        DEVICE
    )


# =============================================================================
# Single config benchmark
# =============================================================================

def test_config(
    cases,
    batch_size,
    sample_num,
):

    cleanup()

    header(
        f"TEST batch_size={batch_size}, "
        f"sample_num={sample_num}"
    )

    effective_batch = (
        batch_size
        * sample_num
    )

    print(
        f"Effective crop batch : "
        f"{effective_batch}"
    )

    dataset = BatchTestDataset(
        cases=cases,
        sample_num=sample_num,
    )

    loader = DataLoader(
        dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

        collate_fn=collate_fn_dict,

        pin_memory=False,

        drop_last=True,
    )

    model = build_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    torch.cuda.reset_peak_memory_stats()

    iteration_times = []

    last_loss = None
    actual_batch_shape = None

    try:

        model.train()

        for step, batch in enumerate(
            loader,
            start=1,
        ):

            if step > TEST_ITERATIONS:
                break

            torch.cuda.synchronize()

            start_time = time.time()

            images = (
                batch["image"]
                .float()
                .to(
                    DEVICE,
                    non_blocking=False,
                )
            )

            labels = (
                batch["annot"]
                .float()
                .to(
                    DEVICE,
                    non_blocking=False,
                )
            )

            actual_batch_shape = tuple(
                images.shape
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

            cls_loss = cls_loss.mean()
            shape_loss = shape_loss.mean()
            offset_loss = offset_loss.mean()
            iou_loss = iou_loss.mean()

            total_loss = (
                4.0
                * cls_loss

                + 0.1
                * shape_loss

                + offset_loss

                + iou_loss
            )

            total_loss.backward()

            optimizer.step()

            torch.cuda.synchronize()

            elapsed = (
                time.time()
                - start_time
            )

            iteration_times.append(
                elapsed
            )

            last_loss = float(
                total_loss.detach().cpu()
            )

            allocated = bytes_to_gb(
                torch.cuda.memory_allocated()
            )

            reserved = bytes_to_gb(
                torch.cuda.memory_reserved()
            )

            peak = bytes_to_gb(
                torch.cuda.max_memory_allocated()
            )

            print(
                f"Step {step}/{TEST_ITERATIONS} | "
                f"input={tuple(images.shape)} | "
                f"loss={last_loss:.4f} | "
                f"time={elapsed:.2f}s | "
                f"allocated={allocated:.2f}GB | "
                f"reserved={reserved:.2f}GB | "
                f"peak={peak:.2f}GB"
            )

        peak_allocated = bytes_to_gb(
            torch.cuda.max_memory_allocated()
        )

        peak_reserved = bytes_to_gb(
            torch.cuda.max_memory_reserved()
        )

        total_memory = bytes_to_gb(
            torch.cuda.get_device_properties(
                DEVICE
            ).total_memory
        )

        mean_time = float(
            np.mean(
                iteration_times
            )
        )

        utilization = (
            peak_allocated
            / total_memory
            * 100.0
        )

        result = {
            "batch_size":
                batch_size,

            "sample_num":
                sample_num,

            "effective_batch":
                effective_batch,

            "shape":
                actual_batch_shape,

            "mean_time":
                mean_time,

            "peak_allocated":
                peak_allocated,

            "peak_reserved":
                peak_reserved,

            "total_memory":
                total_memory,

            "vram_percent":
                utilization,

            "status":
                "PASS",
        }

        print()
        print(
            f"PASS"
        )

        print(
            f"Mean sec/iteration : "
            f"{mean_time:.2f}"
        )

        print(
            f"Peak allocated     : "
            f"{peak_allocated:.2f} GB"
        )

        print(
            f"Peak reserved      : "
            f"{peak_reserved:.2f} GB"
        )

        print(
            f"GPU total          : "
            f"{total_memory:.2f} GB"
        )

        print(
            f"Peak usage         : "
            f"{utilization:.1f}%"
        )

        del optimizer
        del model
        del loader
        del dataset

        cleanup()

        return result

    except torch.cuda.OutOfMemoryError:

        print()
        print(
            "CUDA OOM"
        )

        del optimizer
        del model
        del loader
        del dataset

        cleanup()

        return {
            "batch_size":
                batch_size,

            "sample_num":
                sample_num,

            "effective_batch":
                effective_batch,

            "status":
                "OOM",
        }


# =============================================================================
# Main
# =============================================================================

def main():

    set_seed(
        SEED
    )

    header(
        "CPMNetv2 Batch / Sample_num GPU Benchmark"
    )

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA GPU가 필요합니다."
        )

    properties = (
        torch.cuda.get_device_properties(
            DEVICE
        )
    )

    print(
        "GPU            :",
        properties.name,
    )

    print(
        "Total VRAM     :",
        f"{bytes_to_gb(properties.total_memory):.2f} GB",
    )

    print(
        "PyTorch        :",
        torch.__version__,
    )

    print(
        "PyTorch CUDA   :",
        torch.version.cuda,
    )

    print(
        "Test iterations:",
        TEST_ITERATIONS,
    )

    cases = select_positive_cases()

    print(
        "Positive cases :",
        len(cases),
    )

    results = []

    for (
        batch_size,
        sample_num,
    ) in TEST_CONFIGS:

        result = test_config(
            cases=cases,
            batch_size=batch_size,
            sample_num=sample_num,
        )

        results.append(
            result
        )

    header(
        "FINAL COMPARISON"
    )

    print(
        f"{'Batch':>6} "
        f"{'Sample':>7} "
        f"{'Eff.Batch':>10} "
        f"{'sec/iter':>10} "
        f"{'Peak GB':>10} "
        f"{'VRAM %':>9} "
        f"{'Status':>8}"
    )

    print(
        "-" * 70
    )

    for result in results:

        if result[
            "status"
        ] == "PASS":

            print(
                f"{result['batch_size']:6d} "
                f"{result['sample_num']:7d} "
                f"{result['effective_batch']:10d} "
                f"{result['mean_time']:10.2f} "
                f"{result['peak_allocated']:10.2f} "
                f"{result['vram_percent']:9.1f} "
                f"{'PASS':>8}"
            )

        else:

            print(
                f"{result['batch_size']:6d} "
                f"{result['sample_num']:7d} "
                f"{result['effective_batch']:10d} "
                f"{'-':>10} "
                f"{'-':>10} "
                f"{'-':>9} "
                f"{'OOM':>8}"
            )

    print()
    print(
        "판단 기준:"
    )

    print(
        "  Peak VRAM <= 약 80~85% "
        "→ 본학습에 비교적 안전"
    )

    print(
        "  Peak VRAM 85~95% "
        "→ 가능하지만 여유가 적음"
    )

    print(
        "  Peak VRAM >95% / OOM "
        "→ 사용하지 않는 것을 권장"
    )


if __name__ == "__main__":
    main()