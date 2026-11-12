#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

PRED_PATH = (
    PROJECT_ROOT
    / "results/candidate2/evaluation/candidate2_fold0_predictions.json"
)


def resample_whole_ct(image, target_spacing_xyz):

    old_spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float64,
    )

    old_size = np.asarray(
        image.GetSize(),
        dtype=np.int64,
    )

    new_spacing = np.asarray(
        target_spacing_xyz,
        dtype=np.float64,
    )

    new_size = np.round(
        old_size * old_spacing / new_spacing
    ).astype(np.int64)

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(
        tuple(float(x) for x in new_spacing)
    )

    resampler.SetSize(
        [int(x) for x in new_size]
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

    return resampler.Execute(image)


with open(PRED_PATH, "r") as f:
    cases = json.load(f)


print("=" * 90)
print("GT physical -> resampled index sanity check")
print("=" * 90)


for case in cases[:5]:

    mhd = Path(case["mhd_path"])

    source = sitk.ReadImage(str(mhd))

    spacing = case[
        "resampled_spacing_xyz"
    ]

    image = resample_whole_ct(
        source,
        spacing,
    )

    size = np.asarray(
        image.GetSize(),
        dtype=float,
    )

    print()
    print("SCAN:", case["image"])
    print("resampled size:", size.tolist())

    for gt in case["gt_nodules"]:

        physical = tuple(
            float(x)
            for x in gt["center_xyz_mm"]
        )

        index = (
            image
            .TransformPhysicalPointToContinuousIndex(
                physical
            )
        )

        index = np.asarray(
            index,
            dtype=float,
        )

        inside = np.all(
            (index >= 0)
            & (index < size)
        )

        print()
        print(
            "GT physical:",
            np.round(
                physical,
                2,
            ).tolist()
        )

        print(
            "GT index   :",
            np.round(
                index,
                2,
            ).tolist()
        )

        print(
            "inside CT  :",
            bool(inside)
        )

        if case["predictions"]:

            pred_centers = []

            for p in case["predictions"]:

                box = np.asarray(
                    p["box_xyzxyz"],
                    dtype=float,
                )

                center = (
                    box[:3]
                    + box[3:]
                ) / 2

                pred_centers.append(center)

            pred_centers = np.asarray(
                pred_centers
            )

            distances = np.linalg.norm(
                pred_centers - index,
                axis=1,
            )

            j = int(
                np.argmin(distances)
            )

            print(
                "nearest pred index:",
                np.round(
                    pred_centers[j],
                    2,
                ).tolist()
            )

            print(
                "index distance     :",
                round(
                    float(distances[j]),
                    2,
                )
            )

            print(
                "nearest pred score :",
                case["predictions"][j]["score"]
            )

    print("-" * 90)
