#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from matplotlib.patches import Rectangle


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "visualization"
    / "input_gt"
)


# =============================================================================
# Configuration
# =============================================================================

MAX_CASES = 10

# 중심 slice와 앞/뒤 slice
SLICE_OFFSETS = [
    -2,
    0,
    2,
]

HU_MIN = -1200.0
HU_MAX = 600.0

EXPECTED_SPACING = np.array(
    [
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)


# =============================================================================
# Utility
# =============================================================================

def print_header(
    title,
):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def strip_nii_suffix(
    filename,
):
    if filename.endswith(
        ".nii.gz"
    ):
        return filename[:-7]

    if filename.endswith(
        ".nii"
    ):
        return filename[:-4]

    return Path(
        filename
    ).stem


def normalize_for_display(
    image,
):
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

    return image


# =============================================================================
# CSV
# =============================================================================

def load_annotations():

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"CSV 없음:\n{TRAIN_CSV}"
        )

    df = pd.read_csv(
        TRAIN_CSV
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
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"CSV 필수 컬럼 없음: {missing}"
        )

    # nodule만 사용
    labels = (
        df["label"]
        .astype(str)
    )

    invalid = (
        labels
        != "nodule"
    )

    if invalid.any():
        print(
            "[WARNING] nodule 이외 label 발견:"
        )

        print(
            df.loc[
                invalid,
                "label",
            ].value_counts()
        )

    df = df[
        labels == "nodule"
    ].copy()

    return df


# =============================================================================
# Image
# =============================================================================

def load_ct(
    ct_path,
):

    image_itk = sitk.ReadImage(
        str(
            ct_path
        )
    )

    # SimpleITK GetSpacing:
    # [x, y, z]
    spacing_xyz = np.asarray(
        image_itk.GetSpacing(),
        dtype=np.float32,
    )

    if not np.allclose(
        spacing_xyz,
        EXPECTED_SPACING,
        atol=1e-6,
    ):
        raise RuntimeError(
            "\nSpacing 불일치\n"
            f"File    : {ct_path}\n"
            f"Spacing : {spacing_xyz.tolist()}\n"
            f"Expected: {EXPECTED_SPACING.tolist()}"
        )

    # numpy:
    # [z, y, x]
    array_zyx = (
        sitk.GetArrayFromImage(
            image_itk
        )
        .astype(
            np.float32
        )
    )

    if not np.isfinite(
        array_zyx
    ).all():
        raise RuntimeError(
            f"NaN/Inf CT: {ct_path}"
        )

    return (
        image_itk,
        array_zyx,
    )


# =============================================================================
# Box drawing
# =============================================================================

def annotation_intersects_slice(
    annotation,
    z_index,
):

    cz = float(
        annotation["coordZ"]
    )

    depth = float(
        annotation["d"]
    )

    z_min = (
        cz
        - depth / 2.0
    )

    z_max = (
        cz
        + depth / 2.0
    )

    return (
        z_min
        <= z_index
        <= z_max
    )


def draw_gt_boxes(
    ax,
    annotations,
    z_index,
):

    box_count = 0

    for _, ann in (
        annotations.iterrows()
    ):

        if not (
            annotation_intersects_slice(
                ann,
                z_index,
            )
        ):
            continue

        cx = float(
            ann["coordX"]
        )

        cy = float(
            ann["coordY"]
        )

        width = float(
            ann["w"]
        )

        height = float(
            ann["h"]
        )

        x_min = (
            cx
            - width / 2.0
        )

        y_min = (
            cy
            - height / 2.0
        )

        rectangle = Rectangle(
            (
                x_min,
                y_min,
            ),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(
            rectangle
        )

        # center
        ax.plot(
            cx,
            cy,
            marker="+",
            markersize=10,
            markeredgewidth=2,
        )

        diameter = max(
            width,
            height,
            float(
                ann["d"]
            ),
        )

        ax.text(
            x_min,
            max(
                0,
                y_min - 4,
            ),
            f"{diameter:.1f} mm",
            fontsize=8,
            bbox={
                "facecolor": "black",
                "alpha": 0.5,
                "pad": 2,
            },
        )

        box_count += 1

    return box_count


# =============================================================================
# Visualize one case
# =============================================================================

def visualize_case(
    ct_path,
    annotations,
    case_number,
):

    (
        _,
        array_zyx,
    ) = load_ct(
        ct_path
    )

    display = (
        normalize_for_display(
            array_zyx
        )
    )

    # 첫 번째 GT의 중심 z를 기준으로
    # 3개 slice를 표시.
    primary_gt = (
        annotations.iloc[
            0
        ]
    )

    center_z = int(
        round(
            float(
                primary_gt[
                    "coordZ"
                ]
            )
        )
    )

    z_size = (
        array_zyx.shape[
            0
        ]
    )

    slices = []

    for offset in (
        SLICE_OFFSETS
    ):

        z = int(
            np.clip(
                center_z
                + offset,
                0,
                z_size - 1,
            )
        )

        slices.append(
            z
        )

    fig, axes = plt.subplots(
        1,
        len(
            slices
        ),
        figsize=(
            15,
            5,
        ),
    )

    if len(
        slices
    ) == 1:
        axes = [
            axes
        ]

    total_boxes = 0

    for ax, z_index in zip(
        axes,
        slices,
    ):

        ax.imshow(
            display[
                z_index
            ],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        boxes_on_slice = (
            draw_gt_boxes(
                ax,
                annotations,
                z_index,
            )
        )

        total_boxes += (
            boxes_on_slice
        )

        ax.set_title(
            f"Axial z={z_index}\n"
            f"GT boxes={boxes_on_slice}"
        )

        ax.axis(
            "off"
        )

    seriesuid = str(
        annotations.iloc[
            0
        ][
            "seriesuid"
        ]
    )

    fig.suptitle(
        f"Case {case_number:02d} | "
        f"{strip_nii_suffix(ct_path.name)} | "
        f"GT nodules={len(annotations)}",
        fontsize=11,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / (
            f"{case_number:02d}_"
            f"{strip_nii_suffix(ct_path.name)}"
            f"_gt.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return {
        "seriesuid":
            seriesuid,

        "ct":
            ct_path.name,

        "gt_count":
            len(
                annotations
            ),

        "center_z":
            center_z,

        "visualized_slices":
            slices,

        "boxes_drawn":
            total_boxes,

        "output":
            output_path,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    print_header(
        "CPMNetv2 Input + GT Visualization QA"
    )

    print(
        f"Image root : {IMAGE_ROOT}"
    )

    print(
        f"CSV        : {TRAIN_CSV}"
    )

    print(
        f"Output     : {OUTPUT_ROOT}"
    )

    print(
        f"Max cases  : {MAX_CASES}"
    )

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"imagesTr 없음:\n{IMAGE_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_annotations()

    print()
    print(
        f"Total annotations: "
        f"{len(df)}"
    )

    grouped = {
        str(
            uid
        ): group.copy()
        for uid, group
        in df.groupby(
            "seriesuid"
        )
    }

    # Smoke dataset 구현과 동일하게
    # CT filename으로 CSV group을 찾는다.
    ct_paths = sorted(
        IMAGE_ROOT.glob(
            "*.nii.gz"
        )
    )

    positive_cases = []

    for ct_path in ct_paths:

        keys_to_try = [
            ct_path.name,
            strip_nii_suffix(
                ct_path.name
            ),
        ]

        group = None
        matched_key = None

        for key in keys_to_try:

            if key in grouped:

                group = grouped[
                    key
                ]

                matched_key = key

                break

        if group is None:
            continue

        positive_cases.append(
            (
                ct_path,
                group,
                matched_key,
            )
        )

    print(
        f"Positive CTs found: "
        f"{len(positive_cases)}"
    )

    if len(
        positive_cases
    ) == 0:

        print()
        print(
            "CSV seriesuid 예시:"
        )

        print(
            df[
                "seriesuid"
            ]
            .astype(str)
            .head()
            .tolist()
        )

        print(
            "CT filename 예시:"
        )

        print(
            [
                p.name
                for p in ct_paths[
                    :5
                ]
            ]
        )

        raise RuntimeError(
            "CT와 CSV seriesuid가 매칭되지 않습니다."
        )

    selected = (
        positive_cases[
            :MAX_CASES
        ]
    )

    results = []

    print_header(
        "Visualization"
    )

    for case_number, (
        ct_path,
        annotations,
        matched_key,
    ) in enumerate(
        selected,
        start=1,
    ):

        result = visualize_case(
            ct_path,
            annotations,
            case_number,
        )

        results.append(
            result
        )

        print(
            f"[{case_number:02d}] "
            f"GT={result['gt_count']:2d} | "
            f"z={result['center_z']:4d} | "
            f"slices={result['visualized_slices']} | "
            f"{ct_path.name}"
        )

    print_header(
        "QA Summary"
    )

    print(
        f"Visualized cases : "
        f"{len(results)}"
    )

    print(
        f"Visualized GTs   : "
        f"{sum(r['gt_count'] for r in results)}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "직접 PNG를 확인하여:"
    )

    print(
        "1. GT box가 실제 폐결절 위에 위치하는지"
    )

    print(
        "2. GT center가 결절 중심과 일치하는지"
    )

    print(
        "3. bbox 크기가 실제 결절 크기와 유사한지"
    )

    print(
        "4. z±2 slice에서도 bbox가 자연스럽게 이어지는지"
    )

    print()
    print(
        "CPMNetv2 Input + GT visualization completed."
    )


if __name__ == "__main__":
    main()