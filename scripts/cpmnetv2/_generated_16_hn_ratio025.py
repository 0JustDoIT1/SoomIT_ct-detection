#!/usr/bin/env python3

"""
14_train_cpmnetv2_hard_negative.py

CPMNetv2 Hard Negative Sampling Candidate

Baseline:
    scripts/cpmnetv2/10_train_cpmnetv2_full.py

변경점:
    TRAIN negative sampling only

Baseline sampling:
    Positive            75%
    Ordinary Negative   25%

Candidate:
    Positive            75%
    Negative            25%
        ├─ Ordinary     50% of negative
        └─ Hard Neg     50% of negative

전체 probability:
    Positive            75.0%
    Ordinary Negative   12.5%
    Hard Negative       12.5%

Hard negatives:
    results/cpmnetv2/hard_negative_mining/hard_negatives.csv

Important:
    Validation에는 Hard Negative 정보를 사용하지 않는다.
    Candidate는 처음부터 새로 학습한다.
    Baseline checkpoint에서 resume하지 않는다.

UID 처리:
    LUNA16 SeriesInstanceUID는 "."이 포함된 문자열이므로
    Path.stem을 사용하면 안 된다.

    제거하는 것은 오직:
        .nii.gz
        .nii

    뿐이다.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# =============================================================================
# PATH
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

BASELINE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
    / "10_train_cpmnetv2_full.py"
)

HARD_NEGATIVE_CSV = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_mining"
    / "hard_negatives.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_training_ratio025"
)


# =============================================================================
# HARD NEGATIVE CONFIG
# =============================================================================

# Negative sampling 안에서 Hard Negative가 차지하는 비율
#
# 전체:
#
# positive:
#     0.75
#
# negative:
#     0.25
#
# negative 내부:
#     ordinary = 0.50
#     hard     = 0.50
#
# 따라서 전체 probability:
#
# positive = 0.750
# ordinary = 0.125
# hard     = 0.125

HARD_NEGATIVE_RATIO = 0.25


# =============================================================================
# UID NORMALIZATION
# =============================================================================

def normalize_case_id(name):

    """
    LUNA16 SeriesInstanceUID를 안전하게 정규화한다.

    IMPORTANT:
        UID에 "."이 많으므로 Path.stem 사용 금지.

    Example:

        1.3.6....100332161840....nii.gz
            ->
        1.3.6....100332161840....

        1.3.6....100332161840....
            ->
        그대로 유지
    """

    name = str(
        name
    ).strip()

    if name.endswith(
        ".nii.gz"
    ):

        return name[
            :-7
        ]

    if name.endswith(
        ".nii"
    ):

        return name[
            :-4
        ]

    return name


# =============================================================================
# LOAD BASELINE SCRIPT
# =============================================================================

def load_baseline_module():

    if not BASELINE_SCRIPT.exists():

        raise FileNotFoundError(
            f"Baseline training script 없음:\n"
            f"{BASELINE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "cpmnetv2_baseline_training",
        BASELINE_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "10_train_cpmnetv2_full.py import 실패"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


baseline = load_baseline_module()


# =============================================================================
# CPMNetv2 CROP UTILITIES
# =============================================================================

from dataload.crop import (
    InstanceCrop,
    reorient,
    rand_rot_coord,
)


# =============================================================================
# LOAD HARD NEGATIVES
# =============================================================================

def load_hard_negative_map():

    if not HARD_NEGATIVE_CSV.exists():

        raise FileNotFoundError(
            f"Hard-negative CSV 없음:\n"
            f"{HARD_NEGATIVE_CSV}\n\n"
            f"13_mine_cpmnetv2_hard_negatives.py를 "
            f"먼저 실행해야 합니다."
        )

    df = baseline.pd.read_csv(
        HARD_NEGATIVE_CSV
    )

    required = [
        "case_id",
        "score",
        "z",
        "y",
        "x",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Hard-negative CSV columns missing: "
            f"{missing}"
        )

    hard_map = {}

    for case_id, group in df.groupby(
        "case_id"
    ):

        # -------------------------------------------------------------
        # IMPORTANT FIX
        #
        # 기존:
        #
        # baseline.strip_nii_suffix()
        #
        # 내부 Path.stem 때문에 UID가
        #
        # 1.3.6....6001.100332....
        #
        # ->
        #
        # 1.3.6....6001
        #
        # 로 잘려 390개가 1개 key로 뭉쳤음.
        #
        # 이제 .nii.gz / .nii만 제거한다.
        # -------------------------------------------------------------

        key = normalize_case_id(
            case_id
        )

        hard_map[
            key
        ] = (
            group[
                [
                    "z",
                    "y",
                    "x",
                ]
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

    return (
        df,
        hard_map,
    )


HARD_NEGATIVE_DF, HARD_NEGATIVE_MAP = (
    load_hard_negative_map()
)


# =============================================================================
# HARD NEGATIVE INSTANCE CROP
# =============================================================================

class HardNegativeInstanceCrop(
    InstanceCrop
):

    """
    CPMNetv2 original InstanceCrop 기반.

    Original:
        Positive crop
        Ordinary negative crop

    Candidate:
        Positive crop
        Ordinary negative crop
        Hard-negative-centered crop

    Hard Negative 위치 자체는 lesion GT로 추가하지 않는다.

    실제 annotation은 기존 GT만 유지한다.
    """

    def __init__(
        self,
        *args,
        hard_negative_map=None,
        hard_negative_ratio=0.50,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.hard_negative_map = (
            hard_negative_map
            if hard_negative_map is not None
            else {}
        )

        self.hard_negative_ratio = float(
            hard_negative_ratio
        )

        if not (
            0.0
            <= self.hard_negative_ratio
            <= 1.0
        ):

            raise ValueError(
                "hard_negative_ratio must be "
                "between 0 and 1."
            )

    # =========================================================================
    # CASE ID
    # =========================================================================

    def _case_id_from_sample(
        self,
        sample,
    ):

        file_name = str(
            sample.get(
                "file_name",
                "",
            )
        )

        file_name = Path(
            file_name
        ).name

        # -------------------------------------------------------------
        # IMPORTANT FIX
        #
        # Path.stem 사용 금지.
        #
        # NIfTI suffix만 제거한다.
        # -------------------------------------------------------------

        return normalize_case_id(
            file_name
        )

    # =========================================================================
    # CROP
    # =========================================================================

    def __call__(
        self,
        sample,
        image_spacing,
    ):

        image = (
            sample[
                "image"
            ]
            .astype(
                "float32"
            )
        )

        all_loc = (
            sample[
                "all_loc"
            ]
        )

        all_rad = (
            sample[
                "all_rad"
            ]
        )

        all_cls = (
            sample[
                "all_cls"
            ]
        )

        # ---------------------------------------------------------------------
        # GT instance locations
        # ---------------------------------------------------------------------

        instance_loc = all_loc[
            np.sum(
                [
                    all_cls == cls
                    for cls
                    in self.sample_cls
                ],
                axis=0,
                dtype="bool",
            )
        ]

        # ---------------------------------------------------------------------
        # Hard-negative locations
        # ---------------------------------------------------------------------

        case_id = (
            self._case_id_from_sample(
                sample
            )
        )

        hard_locations = (
            self.hard_negative_map.get(
                case_id
            )
        )

        if hard_locations is None:

            hard_locations = (
                np.empty(
                    (
                        0,
                        3,
                    ),
                    dtype=np.float32,
                )
            )

        else:

            hard_locations = np.asarray(
                hard_locations,
                dtype=np.float32,
            ).reshape(
                -1,
                3,
            )

        # ---------------------------------------------------------------------
        # Original image / shadow
        # ---------------------------------------------------------------------

        image_itk = (
            sitk.GetImageFromArray(
                image
            )
        )

        shadow = np.zeros(
            image.shape,
            dtype=np.float32,
        )

        shadow_itk = (
            sitk.GetImageFromArray(
                shadow
            )
        )

        shape = image.shape

        # ---------------------------------------------------------------------
        # Original InstanceCrop spacing
        # ---------------------------------------------------------------------

        re_spacing = (
            np.array(
                self.spacing
            )
            / np.array(
                self.base_spacing
            )
        )

        crop_size = (
            np.array(
                self.crop_size
            )
            * re_spacing
        )

        overlap = (
            np.array(
                self.overlap
            )
            * re_spacing
        )

        z_stride = (
            crop_size[0]
            - overlap[0]
        )

        y_stride = (
            crop_size[1]
            - overlap[1]
        )

        x_stride = (
            crop_size[2]
            - overlap[2]
        )

        # ---------------------------------------------------------------------
        # Sliding crop centers
        # ---------------------------------------------------------------------

        z_range = (
            np.arange(
                0,
                shape[0]
                - overlap[0],
                z_stride,
            )
            + crop_size[0]
            / 2
        )

        y_range = (
            np.arange(
                0,
                shape[1]
                - overlap[1],
                y_stride,
            )
            + crop_size[1]
            / 2
        )

        x_range = (
            np.arange(
                0,
                shape[2]
                - overlap[2],
                x_stride,
            )
            + crop_size[2]
            / 2
        )

        z_range = np.clip(
            z_range,
            a_min=None,
            a_max=(
                shape[0]
                - crop_size[0]
                / 2
            ),
        )

        y_range = np.clip(
            y_range,
            a_min=None,
            a_max=(
                shape[1]
                - crop_size[1]
                / 2
            ),
        )

        x_range = np.clip(
            x_range,
            a_min=None,
            a_max=(
                shape[2]
                - crop_size[2]
                / 2
            ),
        )

        crop_centers = []

        # source:
        #
        # 0 = sliding
        # 1 = GT-centered
        # 2 = hard-negative-centered

        center_sources = []

        # ---------------------------------------------------------------------
        # Sliding centers
        # ---------------------------------------------------------------------

        for z in z_range:

            for y in y_range:

                for x in x_range:

                    crop_centers.append(
                        np.asarray(
                            [
                                z,
                                y,
                                x,
                            ],
                            dtype=np.float32,
                        )
                    )

                    center_sources.append(
                        0
                    )

        # ---------------------------------------------------------------------
        # GT-centered candidates
        # ---------------------------------------------------------------------

        if self.instance_crop:

            if self.rand_trans is not None:

                instance_crop = (
                    instance_loc
                    + np.random.randint(
                        low=-self.rand_trans,
                        high=self.rand_trans,
                        size=3,
                    )
                )

            else:

                instance_crop = (
                    instance_loc
                )

            for center in instance_crop:

                crop_centers.append(
                    np.asarray(
                        center,
                        dtype=np.float32,
                    )
                )

                center_sources.append(
                    1
                )

        # ---------------------------------------------------------------------
        # Hard-negative candidates
        # ---------------------------------------------------------------------

        for center in hard_locations:

            crop_centers.append(
                np.asarray(
                    center,
                    dtype=np.float32,
                )
            )

            center_sources.append(
                2
            )

        crop_centers = np.asarray(
            crop_centers,
            dtype=np.float32,
        )

        center_sources = np.asarray(
            center_sources,
            dtype=np.int8,
        )

        # ---------------------------------------------------------------------
        # Crop metadata
        # ---------------------------------------------------------------------

        tp_num = []

        CT_crops = []

        image_spacing_crops = []

        all_loc_crops = []
        all_rad_crops = []
        all_cls_crops = []

        matrix_crops = []
        space_crops = []

        # ---------------------------------------------------------------------
        # Build candidate crop metadata
        # ---------------------------------------------------------------------

        for i in range(
            len(
                crop_centers
            )
        ):

            C = (
                crop_centers[
                    i
                ].copy()
            )

            if self.rand_trans is not None:

                C = (
                    C
                    + np.random.randint(
                        low=-self.rand_trans,
                        high=self.rand_trans,
                        size=3,
                    )
                    * re_spacing
                )

            O = (
                C
                - np.array(
                    crop_size
                )
                / 2
            )

            Z = (
                O
                + np.array(
                    [
                        crop_size[0]
                        - 1,
                        0,
                        0,
                    ]
                )
            )

            Y = (
                O
                + np.array(
                    [
                        0,
                        crop_size[1]
                        - 1,
                        0,
                    ]
                )
            )

            X = (
                O
                + np.array(
                    [
                        0,
                        0,
                        crop_size[2]
                        - 1,
                    ]
                )
            )

            matrix = np.array(
                [
                    O,
                    X,
                    Y,
                    Z,
                ]
            )

            # -------------------------------------------------------------
            # Rotation - baseline unchanged
            # -------------------------------------------------------------

            if self.rand_rot is not None:

                matrix = rand_rot_coord(
                    matrix,

                    [
                        -self.rand_rot[0],
                        self.rand_rot[0],
                    ],

                    [
                        -self.rand_rot[1],
                        self.rand_rot[1],
                    ],

                    [
                        -self.rand_rot[2],
                        self.rand_rot[2],
                    ],

                    rot_center=C,
                    p=0.8,
                )

            # -------------------------------------------------------------
            # Spacing augmentation - baseline unchanged
            # -------------------------------------------------------------

            if (
                self.rand_space
                is not None
                and baseline.random.random()
                < 0.8
            ):

                space = np.random.uniform(
                    self.rand_space[0],
                    self.rand_space[1],
                    size=3,
                ) * re_spacing

            else:

                space = (
                    re_spacing
                )

            # ITK axis
            matrix = matrix[
                :,
                ::-1
            ]

            image_itk_crop = reorient(
                shadow_itk,
                matrix,
                spacing=list(
                    space
                ),
                interp1=(
                    sitk.sitkNearestNeighbor
                ),
            )

            # -------------------------------------------------------------
            # GT coordinate transform
            # -------------------------------------------------------------

            all_loc_crop = [
                image_itk_crop
                .TransformPhysicalPointToContinuousIndex(
                    c.tolist()[
                        ::-1
                    ]
                )[
                    ::-1
                ]

                for c
                in all_loc
            ]

            all_loc_crop = np.array(
                all_loc_crop
            )

            if all_loc_crop.size == 0:

                all_loc_crop = np.empty(
                    (
                        0,
                        3,
                    ),
                    dtype=np.float32,
                )

            else:

                all_loc_crop = (
                    all_loc_crop.reshape(
                        -1,
                        3,
                    )
                )

            # -------------------------------------------------------------
            # Check GT inside crop
            # -------------------------------------------------------------

            in_idx = []

            for j in range(
                all_loc_crop.shape[0]
            ):

                inside = (
                    (
                        all_loc_crop[j]
                        <= np.array(
                            image_itk_crop
                            .GetSize()[
                                ::-1
                            ]
                        )
                    ).all()
                    and
                    (
                        all_loc_crop[j]
                        >= np.zeros(
                            3
                        )
                    ).all()
                )

                in_idx.append(
                    inside
                )

            in_idx = np.asarray(
                in_idx,
                dtype=bool,
            )

            if in_idx.size > 0:

                loc_crop = (
                    all_loc_crop[
                        in_idx
                    ]
                )

                rad_crop = (
                    all_rad[
                        in_idx
                    ]
                )

                cls_crop = (
                    all_cls[
                        in_idx
                    ]
                )

            else:

                loc_crop = np.empty(
                    (
                        0,
                        3,
                    ),
                    dtype=np.float32,
                )

                rad_crop = np.empty(
                    (
                        0,
                        3,
                    ),
                    dtype=np.float32,
                )

                cls_crop = np.empty(
                    (
                        0,
                    ),
                    dtype=np.int8,
                )

            # -------------------------------------------------------------
            # TP count
            # -------------------------------------------------------------

            if all_cls.shape[0] == 0:

                current_tp = 0

            else:

                current_tp = np.sum(
                    [
                        all_cls[
                            in_idx
                        ]
                        == cls

                        for cls
                        in self.sample_cls
                    ],
                    axis=0,
                    dtype="bool",
                ).sum()

            tp_num.append(
                current_tp
            )

            matrix_crops.append(
                matrix
            )

            space_crops.append(
                space
            )

            all_loc_crops.append(
                loc_crop
            )

            all_rad_crops.append(
                rad_crop
            )

            all_cls_crops.append(
                cls_crop
            )

        tp_num = np.asarray(
            tp_num
        )

        # =====================================================================
        # SAMPLING
        # =====================================================================

        hard_idx = (
            center_sources
            == 2
        )

        # GT-containing non-hard-negative candidate
        positive_idx = (
            (tp_num > 0)
            & (~hard_idx)
        )

        # normal negative candidate
        ordinary_negative_idx = (
            (tp_num == 0)
            & (~hard_idx)
        )

        p = np.zeros(
            len(
                crop_centers
            ),
            dtype=np.float64,
        )

        num_positive = int(
            positive_idx.sum()
        )

        num_ordinary = int(
            ordinary_negative_idx.sum()
        )

        num_hard = int(
            hard_idx.sum()
        )

        # ---------------------------------------------------------------------
        # Positive / Negative probability
        # ---------------------------------------------------------------------

        if num_positive > 0:

            positive_mass = float(
                self.tp_ratio
            )

            negative_mass = (
                1.0
                - positive_mass
            )

        else:

            # no-GT CT
            positive_mass = 0.0

            negative_mass = 1.0

        # ---------------------------------------------------------------------
        # Negative = ordinary + hard
        # ---------------------------------------------------------------------

        if (
            num_hard > 0
            and num_ordinary > 0
        ):

            hard_mass = (
                negative_mass
                * self.hard_negative_ratio
            )

            ordinary_mass = (
                negative_mass
                - hard_mass
            )

        elif num_hard > 0:

            hard_mass = (
                negative_mass
            )

            ordinary_mass = 0.0

        elif num_ordinary > 0:

            hard_mass = 0.0

            ordinary_mass = (
                negative_mass
            )

        else:

            hard_mass = 0.0
            ordinary_mass = 0.0

            if num_positive > 0:

                positive_mass = 1.0

        # ---------------------------------------------------------------------
        # Candidate probabilities
        # ---------------------------------------------------------------------

        if num_positive > 0:

            p[
                positive_idx
            ] = (
                positive_mass
                / num_positive
            )

        if num_ordinary > 0:

            p[
                ordinary_negative_idx
            ] = (
                ordinary_mass
                / num_ordinary
            )

        if num_hard > 0:

            p[
                hard_idx
            ] = (
                hard_mass
                / num_hard
            )

        probability_sum = (
            p.sum()
        )

        if (
            not np.isfinite(
                probability_sum
            )
            or probability_sum
            <= 0
        ):

            raise RuntimeError(
                f"Invalid crop sampling probability\n"
                f"case={case_id}\n"
                f"positive={num_positive}\n"
                f"ordinary={num_ordinary}\n"
                f"hard={num_hard}\n"
                f"sum={probability_sum}"
            )

        p = (
            p
            / probability_sum
        )

        # ---------------------------------------------------------------------
        # Sampling
        #
        # np.random.choice default replacement=True
        # baseline InstanceCrop와 동일
        # ---------------------------------------------------------------------

        index = np.random.choice(
            np.arange(
                len(
                    crop_centers
                )
            ),
            size=self.sample_num,
            p=p,
        )

        # ---------------------------------------------------------------------
        # Selected annotations
        # ---------------------------------------------------------------------

        selected_loc = [
            all_loc_crops[i]
            for i
            in index
        ]

        selected_rad = [
            all_rad_crops[i]
            for i
            in index
        ]

        selected_cls = [
            all_cls_crops[i]
            for i
            in index
        ]

        # ---------------------------------------------------------------------
        # Actual image crop
        # ---------------------------------------------------------------------

        for i in index:

            matrix = (
                matrix_crops[
                    i
                ]
            )

            space = (
                space_crops[
                    i
                ]
            )

            image_itk_crop = reorient(
                image_itk,
                matrix,
                spacing=list(
                    space
                ),
                interp1=(
                    sitk.sitkLinear
                ),
            )

            image_crop = (
                sitk.GetArrayFromImage(
                    image_itk_crop
                )
            )

            CT_crops.append(
                np.expand_dims(
                    image_crop,
                    axis=0,
                )
            )

            image_spacing_crops.append(
                space
            )

        # ---------------------------------------------------------------------
        # Same output structure as InstanceCrop
        # ---------------------------------------------------------------------

        samples = []

        for i in range(
            len(
                CT_crops
            )
        ):

            ctr = (
                selected_loc[
                    i
                ]
            )

            rad = (
                selected_rad[
                    i
                ]
            )

            cls = (
                selected_cls[
                    i
                ]
            )

            scale_spacing = (
                image_spacing_crops[
                    i
                ]
            )

            real_space = (
                image_spacing
                * scale_spacing
            )

            if len(
                rad
            ) > 0:

                rad = (
                    rad
                    / real_space
                )

            output = {
                "image":
                    CT_crops[
                        i
                    ],

                "ctr":
                    ctr,

                "rad":
                    rad,

                "cls":
                    cls,
            }

            samples.append(
                output
            )

        return samples


# =============================================================================
# DATASET
# =============================================================================

class HardNegativeCPMFullDataset(
    baseline.CPMFullDataset
):

    """
    Train:
        Hard Negative ON

    Validation:
        Hard Negative OFF

    나머지는 baseline CPMFullDataset 그대로.
    """

    def __init__(
        self,
        cases,
        sample_num,
        tp_ratio,
        pos_ratio,
    ):

        super().__init__(
            cases=cases,
            sample_num=sample_num,
            tp_ratio=tp_ratio,
            pos_ratio=pos_ratio,
        )

        # ---------------------------------------------------------------------
        # Train / Validation 구분
        # ---------------------------------------------------------------------

        parents = {
            Path(
                case[
                    "path"
                ]
            ).parent.resolve()

            for case
            in cases
        }

        train_root = (
            baseline.TRAIN_ROOT
            .resolve()
        )

        is_training_dataset = (
            len(
                parents
            ) == 1
            and train_root
            in parents
        )

        self.is_hard_negative_training = (
            is_training_dataset
        )

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        if is_training_dataset:

            self.crop_fn = (
                HardNegativeInstanceCrop(
                    crop_size=(
                        baseline.CROP_SIZE
                    ),

                    tp_ratio=tp_ratio,

                    spacing=(
                        baseline.SPACING
                    ),

                    rand_trans=None,
                    rand_rot=None,
                    rand_space=None,

                    sample_num=sample_num,

                    blank_side=0,

                    instance_crop=True,

                    hard_negative_map=(
                        HARD_NEGATIVE_MAP
                    ),

                    hard_negative_ratio=(
                        HARD_NEGATIVE_RATIO
                    ),
                )
            )

            print()
            print(
                "[HardNegative Dataset]"
            )

            print(
                "  Dataset type       : TRAIN"
            )

            print(
                "  Hard-negative crop : ON"
            )

            print(
                "  Cases with HN      :",
                len(
                    HARD_NEGATIVE_MAP
                ),
            )

            print(
                "  Hard negatives     :",
                len(
                    HARD_NEGATIVE_DF
                ),
            )

            print(
                "  TP ratio           :",
                tp_ratio,
            )

            print(
                "  Negative ratio     :",
                1.0
                - tp_ratio,
            )

            print(
                "  HN within negative :",
                HARD_NEGATIVE_RATIO,
            )

            print(
                "  Expected total mass:"
            )

            print(
                f"    positive         = "
                f"{tp_ratio:.3f}"
            )

            print(
                f"    ordinary neg     = "
                f"{(1.0-tp_ratio)*(1.0-HARD_NEGATIVE_RATIO):.3f}"
            )

            print(
                f"    hard neg         = "
                f"{(1.0-tp_ratio)*HARD_NEGATIVE_RATIO:.3f}"
            )

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        else:

            print()
            print(
                "[HardNegative Dataset]"
            )

            print(
                "  Dataset type       : VALIDATION"
            )

            print(
                "  Hard-negative crop : OFF"
            )

            print(
                "  Original baseline InstanceCrop used."
            )


# =============================================================================
# CANDIDATE OUTPUT
# =============================================================================

def configure_candidate():

    baseline.OUTPUT_ROOT = (
        OUTPUT_ROOT
    )

    baseline.BEST_CHECKPOINT = (
        OUTPUT_ROOT
        / "best_val_loss.pt"
    )

    baseline.LAST_CHECKPOINT = (
        OUTPUT_ROOT
        / "last.pt"
    )

    baseline.HISTORY_JSON = (
        OUTPUT_ROOT
        / "history.json"
    )

    baseline.HISTORY_CSV = (
        OUTPUT_ROOT
        / "history.csv"
    )

    # 처음부터 재학습
    baseline.RESUME_FROM = None

    # Dataset만 교체
    baseline.CPMFullDataset = (
        HardNegativeCPMFullDataset
    )


# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check():

    baseline.header(
        "Hard Negative Candidate Configuration"
    )

    print(
        "Baseline script:"
    )

    print(
        BASELINE_SCRIPT
    )

    print()

    print(
        "Hard-negative CSV:"
    )

    print(
        HARD_NEGATIVE_CSV
    )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_ROOT
    )

    print()

    # -------------------------------------------------------------------------
    # CSV stats
    # -------------------------------------------------------------------------

    csv_case_count = int(
        HARD_NEGATIVE_DF[
            "case_id"
        ].nunique()
    )

    mapped_case_count = len(
        HARD_NEGATIVE_MAP
    )

    print(
        "Mined hard negatives :",
        len(
            HARD_NEGATIVE_DF
        ),
    )

    print(
        "CSV unique cases     :",
        csv_case_count,
    )

    print(
        "Mapped cases         :",
        mapped_case_count,
    )

    print(
        "HN score mean        :",
        float(
            HARD_NEGATIVE_DF[
                "score"
            ].mean()
        ),
    )

    print(
        "HN score median      :",
        float(
            HARD_NEGATIVE_DF[
                "score"
            ].median()
        ),
    )

    # -------------------------------------------------------------------------
    # Critical protection
    # -------------------------------------------------------------------------

    if (
        csv_case_count
        != mapped_case_count
    ):

        raise RuntimeError(
            "\nHard-negative case mapping FAILED.\n"
            f"CSV unique case count = {csv_case_count}\n"
            f"Mapped case count     = {mapped_case_count}\n\n"
            "UID normalization 문제를 확인하십시오."
        )

    print()

    print(
        "[PASS] Hard-negative UID mapping"
    )

    print(
        f"       {mapped_case_count} cases preserved."
    )

    print()

    # -------------------------------------------------------------------------
    # Check train filename matching
    # -------------------------------------------------------------------------

    train_case_ids = {
        normalize_case_id(
            path.name
        )

        for path
        in baseline.TRAIN_ROOT.glob(
            "*.nii.gz"
        )
    }

    mapped_ids = set(
        HARD_NEGATIVE_MAP.keys()
    )

    matched_ids = (
        mapped_ids
        & train_case_ids
    )

    missing_ids = (
        mapped_ids
        - train_case_ids
    )

    print(
        "HN cases in map      :",
        len(
            mapped_ids
        ),
    )

    print(
        "HN cases matched CT  :",
        len(
            matched_ids
        ),
    )

    print(
        "HN cases missing CT  :",
        len(
            missing_ids
        ),
    )

    if missing_ids:

        print()

        print(
            "Missing example:"
        )

        for uid in sorted(
            missing_ids
        )[:10]:

            print(
                " ",
                uid,
            )

        raise RuntimeError(
            "Hard-negative UID와 train NIfTI "
            "파일명이 일치하지 않습니다."
        )

    print()

    print(
        "[PASS] All Hard Negative cases matched "
        "training NIfTI."
    )

    print()

    # -------------------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------------------

    print(
        "Candidate change:"
    )

    print(
        "  HARD NEGATIVE SAMPLING ONLY"
    )

    print()

    print(
        "UNCHANGED:"
    )

    print(
        "  Model architecture"
    )

    print(
        "  Detection loss"
    )

    print(
        "  Loss weights"
    )

    print(
        "  Learning rate"
    )

    print(
        "  Weight decay"
    )

    print(
        "  Batch size"
    )

    print(
        "  Sample num"
    )

    print(
        "  Crop size"
    )

    print(
        "  Spacing"
    )

    print(
        "  Validation split"
    )

    print(
        "  Validation sampling"
    )

    print(
        "  Early stopping"
    )

    print()

    print(
        "Baseline -> Candidate:"
    )

    print(
        f"  Positive probability       : "
        f"{baseline.TRAIN_TP_RATIO:.3f}"
    )

    negative_mass = (
        1.0
        - baseline.TRAIN_TP_RATIO
    )

    print(
        f"  Ordinary negative          : "
        f"{negative_mass * (1-HARD_NEGATIVE_RATIO):.3f}"
    )

    print(
        f"  Mined hard negative        : "
        f"{negative_mass * HARD_NEGATIVE_RATIO:.3f}"
    )

    print()

    print(
        "Training starts FROM SCRATCH."
    )

    print(
        "Baseline best checkpoint is NOT used "
        "as initialization."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    configure_candidate()

    sanity_check()

    print()

    answer = input(
        "Hard Negative Candidate 본학습을 "
        "시작할까요? [y/N]: "
    ).strip().lower()

    if answer not in (
        "y",
        "yes",
    ):

        print(
            "Training cancelled."
        )

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 모든 training logic은 기존 10번 그대로 사용
    baseline.main()


if __name__ == "__main__":

    main()