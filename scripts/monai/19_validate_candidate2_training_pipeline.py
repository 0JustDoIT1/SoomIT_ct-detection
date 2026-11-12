from pathlib import Path
from collections import OrderedDict
import json
import random
import sys

import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
)

from monai.apps.detection.networks.retinanet_detector import (
    RetinaNetDetector,
)

from monai.apps.detection.utils.anchor_utils import (
    AnchorGeneratorWithAnchorShape,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LUNA_RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "LUNA16"
    / "raw"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "training_setup"
    / "candidate2_fold0_split.json"
)

MEDICALNET_ROOT = (
    PROJECT_ROOT
    / "external"
    / "MedicalNet"
)

MED3D_WEIGHT = (
    PROJECT_ROOT
    / "weights"
    / "med3d"
    / "resnet_50_23dataset.pth"
)


# ============================================================
# Candidate 1 aligned configuration
# ============================================================

SEED = 42

EXPECTED_TRAIN = 779
EXPECTED_INTERNAL_VAL = 42
EXPECTED_FINAL_TEST = 67

TARGET_SPACING = (
    0.703125,
    0.703125,
    1.25,
)

PATCH_SIZE = (
    256,
    256,
    96,
)

NUM_CLASSES = 1

FPN_OUT_CHANNELS = 256

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


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


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# MHD resolver
# ============================================================

def resolve_mhd_path(image_value):

    filename = Path(
        str(image_value)
    ).name

    matches = sorted(
        LUNA_RAW_ROOT.glob(
            f"subset*/{filename}"
        )
    )

    if len(matches) == 0:

        raise FileNotFoundError(
            f"MHD 없음:\n{filename}"
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"MHD 중복:\n{filename}"
        )

    return matches[0].resolve()


# ============================================================
# CT normalization
#
# Candidate 1과 동일:
#
# -1024 ~ 300
# ↓
# 0 ~ 1
# ============================================================

def normalize_ct(volume):

    volume = np.clip(
        volume,
        -1024.0,
        300.0,
    )

    volume = (
        volume
        + 1024.0
    ) / 1324.0

    return volume.astype(
        np.float32
    )


# ============================================================
# Full CT resampling
# ============================================================

def resample_ct(
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
# Physical CCCWHD → voxel XYZXYZ
# ============================================================

def physical_boxes_to_voxel_xyzxyz(
    image,
    boxes,
):

    converted = []

    for box in boxes:

        cx, cy, cz, w, h, d = [
            float(v)
            for v in box
        ]

        mins = np.asarray(
            [
                cx - w / 2.0,
                cy - h / 2.0,
                cz - d / 2.0,
            ],
            dtype=np.float64,
        )

        maxs = np.asarray(
            [
                cx + w / 2.0,
                cy + h / 2.0,
                cz + d / 2.0,
            ],
            dtype=np.float64,
        )

        corners = []

        for x in [
            mins[0],
            maxs[0],
        ]:

            for y in [
                mins[1],
                maxs[1],
            ]:

                for z in [
                    mins[2],
                    maxs[2],
                ]:

                    idx = (
                        image
                        .TransformPhysicalPointToContinuousIndex(
                            (
                                float(x),
                                float(y),
                                float(z),
                            )
                        )
                    )

                    corners.append(
                        idx
                    )

        corners = np.asarray(
            corners,
            dtype=np.float64,
        )

        box_min = corners.min(
            axis=0
        )

        box_max = corners.max(
            axis=0
        )

        converted.append(
            np.concatenate(
                [
                    box_min,
                    box_max,
                ]
            )
        )

    if len(converted) == 0:

        return np.zeros(
            (
                0,
                6,
            ),
            dtype=np.float32,
        )

    return np.asarray(
        converted,
        dtype=np.float32,
    )


# ============================================================
# Positive crop
#
# GT nodule 중심 기반
#
# 256 × 256 × 96
# ============================================================

def make_positive_crop(
    volume_xyz,
    boxes_xyzxyz,
    labels,
):

    if len(boxes_xyzxyz) == 0:

        raise RuntimeError(
            "Positive sample인데 bbox 없음"
        )

    target_index = random.randrange(
        len(boxes_xyzxyz)
    )

    target_box = boxes_xyzxyz[
        target_index
    ]

    center = (
        target_box[:3]
        + target_box[3:]
    ) / 2.0

    patch_size = np.asarray(
        PATCH_SIZE,
        dtype=np.int64,
    )

    volume_size = np.asarray(
        volume_xyz.shape,
        dtype=np.int64,
    )


    # --------------------------------------------------------
    # 작은 random jitter
    # --------------------------------------------------------

    jitter = np.asarray(
        [
            random.uniform(
                -16,
                16,
            ),
            random.uniform(
                -16,
                16,
            ),
            random.uniform(
                -8,
                8,
            ),
        ],
        dtype=np.float64,
    )

    center = center + jitter


    start = np.floor(
        center
        - patch_size / 2
    ).astype(
        np.int64
    )

    max_start = (
        volume_size
        - patch_size
    )

    max_start = np.maximum(
        max_start,
        0,
    )

    start = np.maximum(
        start,
        0,
    )

    start = np.minimum(
        start,
        max_start,
    )

    end = (
        start
        + patch_size
    )


    patch = volume_xyz[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]


    if tuple(
        patch.shape
    ) != PATCH_SIZE:

        patch = pad_to_patch_size(
            patch
        )


    cropped_boxes = (
        boxes_xyzxyz.copy()
    )

    cropped_boxes[
        :,
        :3
    ] -= start

    cropped_boxes[
        :,
        3:
    ] -= start


    patch_size_float = np.asarray(
        PATCH_SIZE,
        dtype=np.float32,
    )


    clipped = cropped_boxes.copy()

    clipped[
        :,
        :3
    ] = np.maximum(
        clipped[
            :,
            :3
        ],
        0.0,
    )

    clipped[
        :,
        3:
    ] = np.minimum(
        clipped[
            :,
            3:
        ],
        patch_size_float,
    )


    valid = np.all(
        clipped[
            :,
            3:
        ]
        >
        clipped[
            :,
            :3
        ],
        axis=1,
    )


    clipped = clipped[
        valid
    ]

    labels = labels[
        valid
    ]


    if len(clipped) == 0:

        raise RuntimeError(
            "Positive crop 내부 GT 없음"
        )


    return (
        patch,
        clipped.astype(
            np.float32
        ),
        labels.astype(
            np.int64
        ),
    )


# ============================================================
# Negative crop
#
# GT가 없는 CT에서 random crop
# ============================================================

def make_negative_crop(
    volume_xyz,
):

    volume_size = np.asarray(
        volume_xyz.shape,
        dtype=np.int64,
    )

    patch_size = np.asarray(
        PATCH_SIZE,
        dtype=np.int64,
    )


    max_start = np.maximum(
        volume_size
        - patch_size,
        0,
    )


    start = np.asarray(
        [
            random.randint(
                0,
                int(max_start[0]),
            )
            if max_start[0] > 0
            else 0,

            random.randint(
                0,
                int(max_start[1]),
            )
            if max_start[1] > 0
            else 0,

            random.randint(
                0,
                int(max_start[2]),
            )
            if max_start[2] > 0
            else 0,
        ],
        dtype=np.int64,
    )


    end = (
        start
        + patch_size
    )


    patch = volume_xyz[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]


    if tuple(
        patch.shape
    ) != PATCH_SIZE:

        patch = pad_to_patch_size(
            patch
        )


    boxes = np.zeros(
        (
            0,
            6,
        ),
        dtype=np.float32,
    )

    labels = np.zeros(
        (
            0,
        ),
        dtype=np.int64,
    )


    return (
        patch,
        boxes,
        labels,
    )


# ============================================================
# Pad
# ============================================================

def pad_to_patch_size(
    volume,
):

    target = np.asarray(
        PATCH_SIZE,
        dtype=np.int64,
    )

    current = np.asarray(
        volume.shape,
        dtype=np.int64,
    )

    pad_total = np.maximum(
        target
        - current,
        0,
    )

    pad_before = (
        pad_total
        // 2
    )

    pad_after = (
        pad_total
        - pad_before
    )

    padded = np.pad(
        volume,
        (
            (
                int(pad_before[0]),
                int(pad_after[0]),
            ),
            (
                int(pad_before[1]),
                int(pad_after[1]),
            ),
            (
                int(pad_before[2]),
                int(pad_after[2]),
            ),
        ),
        mode="constant",
        constant_values=0.0,
    )

    return padded.astype(
        np.float32
    )


# ============================================================
# Load real CT sample
# ============================================================

def load_real_sample(
    sample,
):

    mhd_path = resolve_mhd_path(
        sample[
            "image"
        ]
    )

    image = sitk.ReadImage(
        str(
            mhd_path
        )
    )

    resampled = resample_ct(
        image,
        TARGET_SPACING,
    )

    array_zyx = sitk.GetArrayFromImage(
        resampled
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # SimpleITK ZYX
    # → network XYZ
    # --------------------------------------------------------

    volume_xyz = np.transpose(
        array_zyx,
        (
            2,
            1,
            0,
        ),
    )


    volume_xyz = normalize_ct(
        volume_xyz
    )


    boxes = physical_boxes_to_voxel_xyzxyz(
        resampled,
        sample.get(
            "box",
            [],
        ),
    )


    labels = np.asarray(
        sample.get(
            "label",
            [],
        ),
        dtype=np.int64,
    )


    return (
        mhd_path,
        image,
        resampled,
        volume_xyz,
        boxes,
        labels,
    )


# ============================================================
# MedicalNet
# ============================================================

def import_medicalnet():

    if str(
        MEDICALNET_ROOT
    ) not in sys.path:

        sys.path.insert(
            0,
            str(
                MEDICALNET_ROOT
            ),
        )

    from models import resnet

    return resnet


def build_med3d():

    resnet = import_medicalnet()

    return resnet.resnet50(
        sample_input_D=96,
        sample_input_H=96,
        sample_input_W=96,
        num_seg_classes=2,
        shortcut_type="B",
        no_cuda=False,
    )


def load_med3d_weights(
    model,
):

    checkpoint = torch.load(
        MED3D_WEIGHT,
        map_location="cpu",
        weights_only=False,
    )


    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )


    model_state = (
        model.state_dict()
    )


    compatible = {}


    for key, value in (
        state_dict.items()
    ):

        clean_key = key

        if clean_key.startswith(
            "module."
        ):

            clean_key = clean_key[
                len(
                    "module."
                ):
            ]


        if (
            clean_key in model_state
            and model_state[
                clean_key
            ].shape
            == value.shape
        ):

            compatible[
                clean_key
            ] = value


    model.load_state_dict(
        compatible,
        strict=False,
    )


    ratio = (
        len(
            compatible
        )
        / len(
            model_state
        )
    )


    print(
        "Med3D load ratio:",
        f"{ratio:.3f}",
    )


    if ratio < 0.90:

        raise RuntimeError(
            "Med3D pretrained weight "
            "load ratio 너무 낮음"
        )


    return model


# ============================================================
# Backbone wrapper
# ============================================================

class MedicalNetFeatureBackbone(
    nn.Module
):

    def __init__(
        self,
        model,
    ):

        super().__init__()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4


    def forward(
        self,
        x,
    ):

        x = self.conv1(
            x
        )

        x = self.bn1(
            x
        )

        x = self.relu(
            x
        )

        x = self.maxpool(
            x
        )


        c1 = self.layer1(
            x
        )

        c2 = self.layer2(
            c1
        )

        c3 = self.layer3(
            c2
        )

        c4 = self.layer4(
            c3
        )


        return (
            c1,
            c2,
            c3,
            c4,
        )


# ============================================================
# Multi-scale FPN
# ============================================================

class Med3DMultiScaleFPN(
    nn.Module
):

    def __init__(
        self,
        backbone,
        out_channels=256,
    ):

        super().__init__()

        self.backbone = backbone
        self.out_channels = out_channels


        self.lateral_c1 = nn.Conv3d(
            256,
            out_channels,
            kernel_size=1,
        )

        self.lateral_c2 = nn.Conv3d(
            512,
            out_channels,
            kernel_size=1,
        )

        self.lateral_c4 = nn.Conv3d(
            2048,
            out_channels,
            kernel_size=1,
        )


        self.smooth_p2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.smooth_p3 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )


        self.p4_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.p5_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )


    def forward(
        self,
        x,
    ):

        (
            c1,
            c2,
            c3,
            c4,
        ) = self.backbone(
            x
        )


        p3 = (
            self.lateral_c2(
                c2
            )
            +
            self.lateral_c4(
                c4
            )
        )


        p3 = self.smooth_p3(
            p3
        )


        p3_up = F.interpolate(
            p3,
            size=c1.shape[
                -3:
            ],
            mode="trilinear",
            align_corners=False,
        )


        p2 = (
            self.lateral_c1(
                c1
            )
            +
            p3_up
        )


        p2 = self.smooth_p2(
            p2
        )


        p4 = self.p4_conv(
            F.relu(
                p3
            )
        )


        p5 = self.p5_conv(
            F.relu(
                p4
            )
        )


        return OrderedDict(
            [
                (
                    "p2",
                    p2,
                ),
                (
                    "p3",
                    p3,
                ),
                (
                    "p4",
                    p4,
                ),
                (
                    "p5",
                    p5,
                ),
            ]
        )


# ============================================================
# Detector
# ============================================================

def build_detector(
    device,
):

    med3d = build_med3d()

    med3d = load_med3d_weights(
        med3d
    )


    backbone = (
        MedicalNetFeatureBackbone(
            med3d
        )
    )


    feature_extractor = (
        Med3DMultiScaleFPN(
            backbone,
            out_channels=(
                FPN_OUT_CHANNELS
            ),
        )
    )


    base_anchor_shapes = [
        [
            6.0,
            8.0,
            4.0,
        ],
        [
            8.0,
            6.0,
            5.0,
        ],
        [
            10.0,
            10.0,
            6.0,
        ],
    ]


    anchor_generator = (
        AnchorGeneratorWithAnchorShape(
            feature_map_scales=[
                1,
                2,
                4,
                8,
            ],

            base_anchor_shapes=(
                base_anchor_shapes
            ),
        )
    )


    num_anchors = (
        anchor_generator
        .num_anchors_per_location()[
            0
        ]
    )


    network = RetinaNet(
        spatial_dims=3,

        num_classes=(
            NUM_CLASSES
        ),

        num_anchors=(
            num_anchors
        ),

        feature_extractor=(
            feature_extractor
        ),

        size_divisible=4,
    )


    detector = RetinaNetDetector(
        network=network,

        anchor_generator=(
            anchor_generator
        ),

        debug=False,
    )


    detector.set_atss_matcher(
        num_candidates=4,
        center_in_gt=False,
    )


    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=0.25,
        pool_size=20,
        min_neg=16,
    )


    detector.set_target_keys(
        box_key="box",
        label_key="label",
    )


    return detector.to(
        device
    )


# ============================================================
# Convert crop → tensors
# ============================================================

def to_detector_batch(
    patch,
    boxes,
    labels,
    device,
):

    image = torch.from_numpy(
        np.ascontiguousarray(
            patch
        )
    ).unsqueeze(
        0
    ).float()


    box_tensor = torch.from_numpy(
        np.ascontiguousarray(
            boxes
        )
    ).float()


    label_tensor = torch.from_numpy(
        np.ascontiguousarray(
            labels
        )
    ).long()


    images = [
        image.to(
            device
        )
    ]


    targets = [
        {
            "box": (
                box_tensor.to(
                    device
                )
            ),

            "label": (
                label_tensor.to(
                    device
                )
            ),
        }
    ]


    return (
        images,
        targets,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )


    print(
        "=" * 80
    )

    print(
        "Candidate 2 - Candidate 1 Aligned "
        "Training Pipeline Validation"
    )

    print(
        "=" * 80
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


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )


    # ========================================================
    # 1. Load split
    # ========================================================

    print()
    print(
        "[1] Load Candidate 2 split"
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


    if (
        len(training)
        != EXPECTED_TRAIN
        or len(validation)
        != EXPECTED_INTERNAL_VAL
        or len(final_test)
        != EXPECTED_FINAL_TEST
    ):

        raise RuntimeError(
            "Split count 이상\n"
            f"Train: {len(training)} "
            f"!= {EXPECTED_TRAIN}\n"
            f"Internal Val: {len(validation)} "
            f"!= {EXPECTED_INTERNAL_VAL}\n"
            f"Final Test: {len(final_test)} "
            f"!= {EXPECTED_FINAL_TEST}"
        )


    print(
        "✅ Split PASS"
    )


    # ========================================================
    # 2. Positive / negative distribution
    # ========================================================

    print()
    print(
        "[2] Positive / Negative distribution"
    )


    def count_pos_neg(
        items,
    ):

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

    test_pos, test_neg = (
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
        test_pos,
    )

    print(
        "Test Negative:",
        test_neg,
    )


    if train_pos == 0:

        raise RuntimeError(
            "Training positive 없음"
        )


    if train_neg == 0:

        raise RuntimeError(
            "Training negative 없음"
        )


    print(
        "✅ Positive / Negative 모두 존재"
    )


    # ========================================================
    # 3. Candidate 1 aligned preprocessing
    # ========================================================

    print()
    print(
        "[3] Candidate 1 aligned settings"
    )


    print(
        "Spacing:",
        TARGET_SPACING,
    )

    print(
        "Patch:",
        PATCH_SIZE,
    )

    print(
        "Normalization:",
        "-1024~300 → 0~1",
    )


    print(
        "✅ Candidate 1 preprocessing 기준 적용"
    )


    # ========================================================
    # 4. Find one positive / one negative CT
    # ========================================================

    positive_sample = next(
        item
        for item in training
        if len(
            item.get(
                "box",
                [],
            )
        ) > 0
    )


    negative_sample = next(
        item
        for item in training
        if len(
            item.get(
                "box",
                [],
            )
        ) == 0
    )


    # ========================================================
    # 5. Positive CT
    # ========================================================

    print()
    print(
        "[4] Positive CT preprocessing"
    )


    (
        pos_path,
        pos_original,
        pos_resampled,
        pos_volume,
        pos_boxes,
        pos_labels,
    ) = load_real_sample(
        positive_sample
    )


    print(
        "MHD:",
        pos_path,
    )

    print(
        "Original size:",
        pos_original.GetSize(),
    )

    print(
        "Original spacing:",
        pos_original.GetSpacing(),
    )

    print(
        "Resampled size:",
        pos_resampled.GetSize(),
    )

    print(
        "Resampled spacing:",
        pos_resampled.GetSpacing(),
    )

    print(
        "GT boxes:",
        len(
            pos_boxes
        ),
    )


    (
        pos_patch,
        pos_crop_boxes,
        pos_crop_labels,
    ) = make_positive_crop(
        pos_volume,
        pos_boxes,
        pos_labels,
    )


    print(
        "Positive patch:",
        pos_patch.shape,
    )

    print(
        "Positive crop boxes:",
        len(
            pos_crop_boxes
        ),
    )

    print(
        "Intensity min/max:",
        float(
            pos_patch.min()
        ),
        float(
            pos_patch.max()
        ),
    )


    if (
        pos_patch.shape
        != PATCH_SIZE
    ):

        raise RuntimeError(
            "Positive patch shape 이상"
        )


    print(
        "✅ Positive crop PASS"
    )


    # ========================================================
    # 6. Negative CT
    # ========================================================

    print()
    print(
        "[5] Negative CT preprocessing"
    )


    (
        neg_path,
        neg_original,
        neg_resampled,
        neg_volume,
        neg_boxes,
        neg_labels,
    ) = load_real_sample(
        negative_sample
    )


    print(
        "MHD:",
        neg_path,
    )

    print(
        "GT boxes:",
        len(
            neg_boxes
        ),
    )


    (
        neg_patch,
        neg_crop_boxes,
        neg_crop_labels,
    ) = make_negative_crop(
        neg_volume
    )


    print(
        "Negative patch:",
        neg_patch.shape,
    )

    print(
        "Negative boxes:",
        len(
            neg_crop_boxes
        ),
    )


    if len(
        neg_crop_boxes
    ) != 0:

        raise RuntimeError(
            "Negative crop bbox가 존재함"
        )


    print(
        "✅ Negative crop PASS"
    )


    # ========================================================
    # 7. Build detector
    # ========================================================

    print()
    print(
        "[6] Build Candidate 2 detector"
    )


    detector = build_detector(
        device
    )


    print(
        "✅ Detector 생성"
    )


    # ========================================================
    # 8. Optimizer
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


    detector.train()


    # ========================================================
    # 9. Positive real forward
    # ========================================================

    print()
    print(
        "[7] Positive real-data forward / backward"
    )


    images, targets = (
        to_detector_batch(
            pos_patch,
            pos_crop_boxes,
            pos_crop_labels,
            device,
        )
    )


    optimizer.zero_grad(
        set_to_none=True
    )


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


    print(
        "Classification loss:",
        float(
            cls_loss.detach().cpu()
        ),
    )

    print(
        "Box regression loss:",
        float(
            box_loss.detach().cpu()
        ),
    )

    print(
        "Total loss:",
        float(
            total_loss.detach().cpu()
        ),
    )


    if not torch.isfinite(
        total_loss
    ):

        raise RuntimeError(
            "Positive loss NaN/Inf"
        )


    total_loss.backward()


    nonzero_grad = 0


    for parameter in (
        detector.network.parameters()
    ):

        if (
            parameter.grad is not None
            and torch.any(
                parameter.grad != 0
            )
        ):

            nonzero_grad += 1


    print(
        "Non-zero gradient params:",
        nonzero_grad,
    )


    if nonzero_grad == 0:

        raise RuntimeError(
            "Positive backward gradient 없음"
        )


    optimizer.step()


    print(
        "✅ Positive forward / backward PASS"
    )


    # ========================================================
    # 10. Negative real forward
    # ========================================================

    print()
    print(
        "[8] Negative real-data forward / backward"
    )


    images, targets = (
        to_detector_batch(
            neg_patch,
            neg_crop_boxes,
            neg_crop_labels,
            device,
        )
    )


    optimizer.zero_grad(
        set_to_none=True
    )


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


    print(
        "Classification loss:",
        float(
            cls_loss.detach().cpu()
        ),
    )

    print(
        "Box regression loss:",
        float(
            box_loss.detach().cpu()
        ),
    )

    print(
        "Total loss:",
        float(
            total_loss.detach().cpu()
        ),
    )


    if not torch.isfinite(
        total_loss
    ):

        raise RuntimeError(
            "Negative loss NaN/Inf"
        )


    total_loss.backward()


    optimizer.step()


    print(
        "✅ Negative forward / backward PASS"
    )


    # ========================================================
    # GPU memory
    # ========================================================

    if torch.cuda.is_available():

        print()
        print(
            "[9] GPU memory"
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
        "✅ Candidate 2 Candidate-1-Aligned "
        "Training Pipeline Validation PASS"
    )

    print()
    print(
        "Validated:"
    )

    print(
        "Official fold0 split                ✅"
    )

    print(
        "Positive CT                         ✅"
    )

    print(
        "Negative CT                         ✅"
    )

    print(
        "Spacing 0.703125×0.703125×1.25      ✅"
    )

    print(
        "Patch 256×256×96                    ✅"
    )

    print(
        "HU normalization                    ✅"
    )

    print(
        "Physical bbox → voxel bbox          ✅"
    )

    print(
        "Positive crop                       ✅"
    )

    print(
        "Negative crop                       ✅"
    )

    print(
        "Med3D pretrained backbone           ✅"
    )

    print(
        "Custom Multi-scale FPN              ✅"
    )

    print(
        "MONAI RetinaNetDetector             ✅"
    )

    print(
        "Positive loss / backward            ✅"
    )

    print(
        "Negative loss / backward            ✅"
    )

    print()
    print(
        "Next:"
    )

    print(
        "20_validate_candidate2_real_transform.py"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()