#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# =============================================================================
# Default paths
# =============================================================================

PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

LUNA_ROOT = PROJECT_ROOT / "data" / "LUNA16"
RAW_ROOT = LUNA_ROOT / "raw"
ANNOTATION_CSV = LUNA_ROOT / "annotations.csv"

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "luna16"
)


# =============================================================================
# Configuration
# =============================================================================

TARGET_SPACING_XYZ = (1.0, 1.0, 1.0)

# Initial candidate split:
# subset0 ~ subset8 = training
# subset9         = validation
DEFAULT_VAL_SUBSET = 9


# =============================================================================
# Helpers
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def find_subset_from_path(path):
    for parent in path.parents:
        name = parent.name.lower()

        if name.startswith("subset"):
            try:
                return int(name.replace("subset", ""))
            except ValueError:
                pass

    raise RuntimeError(
        f"Could not determine subset number from: {path}"
    )


# =============================================================================
# Resampling
# =============================================================================

def calculate_new_size(
    image,
    target_spacing_xyz=TARGET_SPACING_XYZ,
):
    original_size = np.asarray(
        image.GetSize(),
        dtype=np.float64,
    )

    original_spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float64,
    )

    target_spacing = np.asarray(
        target_spacing_xyz,
        dtype=np.float64,
    )

    new_size = np.round(
        original_size
        * original_spacing
        / target_spacing
    ).astype(np.int64)

    new_size = np.maximum(
        new_size,
        1,
    )

    return tuple(
        int(v)
        for v in new_size
    )


def resample_ct(
    image,
    target_spacing_xyz=TARGET_SPACING_XYZ,
):
    new_size = calculate_new_size(
        image,
        target_spacing_xyz,
    )

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(new_size)

    resampler.SetOutputSpacing(
        tuple(float(v) for v in target_spacing_xyz)
    )

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetTransform(
        sitk.Transform(
            3,
            sitk.sitkIdentity,
        )
    )

    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    # Air outside the original FOV
    resampler.SetDefaultPixelValue(-1024)

    resampler.SetOutputPixelType(
        sitk.sitkFloat32
    )

    return resampler.Execute(image)


# =============================================================================
# Annotation conversion
# =============================================================================

def world_to_resampled_voxel(
    image,
    coord_x,
    coord_y,
    coord_z,
):
    world_xyz = (
        float(coord_x),
        float(coord_y),
        float(coord_z),
    )

    voxel_xyz = (
        image.TransformPhysicalPointToContinuousIndex(
            world_xyz
        )
    )

    return np.asarray(
        voxel_xyz,
        dtype=np.float64,
    )


def is_inside_image(
    voxel_xyz,
    image,
):
    size_xyz = np.asarray(
        image.GetSize(),
        dtype=np.float64,
    )

    return bool(
        np.all(voxel_xyz >= 0)
        and np.all(voxel_xyz < size_xyz)
    )


# =============================================================================
# Data discovery
# =============================================================================

def discover_cases():
    mhd_files = sorted(
        RAW_ROOT.glob(
            "subset*/*.mhd"
        )
    )

    cases = []

    for mhd_path in mhd_files:
        subset = find_subset_from_path(
            mhd_path
        )

        seriesuid = mhd_path.stem

        cases.append(
            {
                "seriesuid": seriesuid,
                "subset": subset,
                "path": mhd_path,
            }
        )

    return cases


# =============================================================================
# Main preprocessing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LUNA16 for CPMNetv2 "
            "using 1 mm isotropic CTs."
        )
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
    )

    parser.add_argument(
        "--val-subset",
        type=int,
        default=DEFAULT_VAL_SUBSET,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of CTs to process. "
            "Useful for QA."
        ),
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)

    images_tr = output_root / "imagesTr"
    images_va = output_root / "imagesVa"

    ensure_dir(output_root)
    ensure_dir(images_tr)
    ensure_dir(images_va)

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    print_header(
        "CPMNetv2 LUNA16 Preprocessing"
    )

    print(
        f"LUNA root       : {LUNA_ROOT}"
    )

    print(
        f"Raw root        : {RAW_ROOT}"
    )

    print(
        f"Annotation CSV  : {ANNOTATION_CSV}"
    )

    print(
        f"Output root     : {output_root}"
    )

    print(
        f"Target spacing  : "
        f"{TARGET_SPACING_XYZ} [X,Y,Z] mm"
    )

    print(
        f"Validation fold : subset{args.val_subset}"
    )

    # -------------------------------------------------------------------------
    # Validate source
    # -------------------------------------------------------------------------

    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            RAW_ROOT
        )

    if not ANNOTATION_CSV.exists():
        raise FileNotFoundError(
            ANNOTATION_CSV
        )

    # -------------------------------------------------------------------------
    # Load annotation
    # -------------------------------------------------------------------------

    annotations = pd.read_csv(
        ANNOTATION_CSV
    )

    required_columns = {
        "seriesuid",
        "coordX",
        "coordY",
        "coordZ",
        "diameter_mm",
    }

    missing = (
        required_columns
        - set(annotations.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing annotation columns: {missing}"
        )

    annotation_groups = {
        str(uid): group
        for uid, group
        in annotations.groupby(
            "seriesuid"
        )
    }

    # -------------------------------------------------------------------------
    # Cases
    # -------------------------------------------------------------------------

    cases = discover_cases()

    if not cases:
        raise RuntimeError(
            "No LUNA16 .mhd cases found."
        )

    if args.limit is not None:
        cases = cases[:args.limit]

    print()
    print(
        f"Cases to process : {len(cases)}"
    )

    # -------------------------------------------------------------------------
    # Output rows
    # -------------------------------------------------------------------------

    train_rows = []
    val_rows = []

    manifest_rows = []

    failed_cases = []

    coordinate_failures = []

    # -------------------------------------------------------------------------
    # Process
    # -------------------------------------------------------------------------

    for index, case in enumerate(
        cases,
        start=1,
    ):
        seriesuid = case["seriesuid"]
        subset = case["subset"]
        source_path = case["path"]

        split = (
            "val"
            if subset == args.val_subset
            else "train"
        )

        output_dir = (
            images_va
            if split == "val"
            else images_tr
        )

        output_filename = (
            f"{seriesuid}.nii.gz"
        )

        output_path = (
            output_dir
            / output_filename
        )

        print()
        print(
            f"[{index}/{len(cases)}] "
            f"subset{subset} "
            f"{seriesuid}"
        )

        try:
            # -----------------------------------------------------------------
            # Read source
            # -----------------------------------------------------------------

            image = sitk.ReadImage(
                str(source_path)
            )

            original_size_xyz = (
                image.GetSize()
            )

            original_spacing_xyz = (
                image.GetSpacing()
            )

            original_origin_xyz = (
                image.GetOrigin()
            )

            # -----------------------------------------------------------------
            # Resample
            # -----------------------------------------------------------------

            if (
                output_path.exists()
                and not args.overwrite
            ):
                print(
                    "  Existing output found "
                    "- loading without overwrite."
                )

                resampled = sitk.ReadImage(
                    str(output_path)
                )

            else:
                resampled = resample_ct(
                    image,
                    TARGET_SPACING_XYZ,
                )

                sitk.WriteImage(
                    resampled,
                    str(output_path),
                    True,
                )

            new_size_xyz = (
                resampled.GetSize()
            )

            new_spacing_xyz = (
                resampled.GetSpacing()
            )

            # -----------------------------------------------------------------
            # Validate spacing
            # -----------------------------------------------------------------

            spacing_ok = np.allclose(
                np.asarray(
                    new_spacing_xyz
                ),
                np.asarray(
                    TARGET_SPACING_XYZ
                ),
                atol=1e-6,
            )

            if not spacing_ok:
                raise RuntimeError(
                    "Unexpected resampled spacing: "
                    f"{new_spacing_xyz}"
                )

            # -----------------------------------------------------------------
            # Annotation
            # -----------------------------------------------------------------

            case_annotations = (
                annotation_groups.get(
                    seriesuid
                )
            )

            annotation_count = 0

            if case_annotations is not None:

                for _, row in (
                    case_annotations.iterrows()
                ):
                    voxel_xyz = (
                        world_to_resampled_voxel(
                            resampled,
                            row["coordX"],
                            row["coordY"],
                            row["coordZ"],
                        )
                    )

                    inside = is_inside_image(
                        voxel_xyz,
                        resampled,
                    )

                    if not inside:
                        coordinate_failures.append(
                            {
                                "seriesuid": seriesuid,
                                "subset": subset,
                                "coordX_world": row["coordX"],
                                "coordY_world": row["coordY"],
                                "coordZ_world": row["coordZ"],
                                "voxelX": voxel_xyz[0],
                                "voxelY": voxel_xyz[1],
                                "voxelZ": voxel_xyz[2],
                            }
                        )

                        print(
                            "  [WARNING] Annotation "
                            "outside resampled CT"
                        )

                        continue

                    diameter_mm = float(
                        row["diameter_mm"]
                    )

                    # LUNA16 provides a spherical
                    # nodule diameter.
                    #
                    # CPMNetv2 expects:
                    # x y z w h d
                    #
                    # Therefore use the same
                    # diameter for all 3 dimensions.
                    output_row = {
                        "seriesuid":
                            output_filename,

                        "coordX":
                            float(voxel_xyz[0]),

                        "coordY":
                            float(voxel_xyz[1]),

                        "coordZ":
                            float(voxel_xyz[2]),

                        "w":
                            diameter_mm,

                        "h":
                            diameter_mm,

                        "d":
                            diameter_mm,

                        "label":
                            "nodule",
                    }

                    if split == "val":
                        val_rows.append(
                            output_row
                        )
                    else:
                        train_rows.append(
                            output_row
                        )

                    annotation_count += 1

            # -----------------------------------------------------------------
            # Manifest
            # -----------------------------------------------------------------

            manifest_rows.append(
                {
                    "seriesuid":
                        seriesuid,

                    "filename":
                        output_filename,

                    "subset":
                        subset,

                    "split":
                        split,

                    "positive":
                        annotation_count > 0,

                    "annotation_count":
                        annotation_count,

                    "source_path":
                        str(source_path),

                    "output_path":
                        str(output_path),

                    "original_size_x":
                        original_size_xyz[0],

                    "original_size_y":
                        original_size_xyz[1],

                    "original_size_z":
                        original_size_xyz[2],

                    "original_spacing_x":
                        original_spacing_xyz[0],

                    "original_spacing_y":
                        original_spacing_xyz[1],

                    "original_spacing_z":
                        original_spacing_xyz[2],

                    "original_origin_x":
                        original_origin_xyz[0],

                    "original_origin_y":
                        original_origin_xyz[1],

                    "original_origin_z":
                        original_origin_xyz[2],

                    "new_size_x":
                        new_size_xyz[0],

                    "new_size_y":
                        new_size_xyz[1],

                    "new_size_z":
                        new_size_xyz[2],

                    "new_spacing_x":
                        new_spacing_xyz[0],

                    "new_spacing_y":
                        new_spacing_xyz[1],

                    "new_spacing_z":
                        new_spacing_xyz[2],
                }
            )

            print(
                f"  split       : {split}"
            )

            print(
                f"  original    : "
                f"{original_size_xyz} "
                f"@ {original_spacing_xyz}"
            )

            print(
                f"  resampled   : "
                f"{new_size_xyz} "
                f"@ {new_spacing_xyz}"
            )

            print(
                f"  annotations : "
                f"{annotation_count}"
            )

        except Exception as exc:
            print(
                f"  [FAIL] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed_cases.append(
                {
                    "seriesuid":
                        seriesuid,

                    "subset":
                        subset,

                    "source_path":
                        str(source_path),

                    "error":
                        f"{type(exc).__name__}: {exc}",
                }
            )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------

    train_columns = [
        "seriesuid",
        "coordX",
        "coordY",
        "coordZ",
        "w",
        "h",
        "d",
        "label",
    ]

    train_df = pd.DataFrame(
        train_rows,
        columns=train_columns,
    )

    val_df = pd.DataFrame(
        val_rows,
        columns=train_columns,
    )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    train_csv_path = (
        output_root / "train.csv"
    )

    val_csv_path = (
        output_root / "val.csv"
    )

    manifest_path = (
        output_root
        / "preprocessing_manifest.csv"
    )

    train_df.to_csv(
        train_csv_path,
        index=False,
    )

    val_df.to_csv(
        val_csv_path,
        index=False,
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    if failed_cases:
        pd.DataFrame(
            failed_cases
        ).to_csv(
            output_root
            / "failed_cases.csv",
            index=False,
        )

    if coordinate_failures:
        pd.DataFrame(
            coordinate_failures
        ).to_csv(
            output_root
            / "coordinate_failures.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "target_spacing_xyz_mm":
            list(TARGET_SPACING_XYZ),

        "validation_subset":
            args.val_subset,

        "processed_cases":
            len(manifest_rows),

        "failed_cases":
            len(failed_cases),

        "train_annotations":
            len(train_df),

        "val_annotations":
            len(val_df),

        "coordinate_failures":
            len(coordinate_failures),

        "annotation_size_policy":
            (
                "LUNA16 diameter_mm is "
                "replicated to w/h/d."
            ),

        "coordinate_policy":
            (
                "LUNA16 physical/world XYZ "
                "converted with "
                "SimpleITK "
                "TransformPhysicalPointTo"
                "ContinuousIndex on the "
                "1mm resampled CT."
            ),
    }

    with open(
        output_root
        / "preprocessing_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_header(
        "Preprocessing Summary"
    )

    train_cases = (
        manifest_df[
            manifest_df["split"] == "train"
        ]
        if len(manifest_df)
        else pd.DataFrame()
    )

    val_cases = (
        manifest_df[
            manifest_df["split"] == "val"
        ]
        if len(manifest_df)
        else pd.DataFrame()
    )

    print(
        f"Processed cases      : "
        f"{len(manifest_rows)}"
    )

    print(
        f"Train CT             : "
        f"{len(train_cases)}"
    )

    print(
        f"Validation CT        : "
        f"{len(val_cases)}"
    )

    print(
        f"Train annotations    : "
        f"{len(train_df)}"
    )

    print(
        f"Validation annotations: "
        f"{len(val_df)}"
    )

    print(
        f"Coordinate failures  : "
        f"{len(coordinate_failures)}"
    )

    print(
        f"Failed cases         : "
        f"{len(failed_cases)}"
    )

    print()
    print(
        f"train.csv : "
        f"{train_csv_path}"
    )

    print(
        f"val.csv   : "
        f"{val_csv_path}"
    )

    print(
        f"manifest  : "
        f"{manifest_path}"
    )

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    if (
        len(failed_cases) == 0
        and len(coordinate_failures) == 0
        and len(manifest_rows) > 0
    ):
        print()
        print(
            "LUNA16 preprocessing PASSED."
        )

        print()
        print(
            "Next step:"
        )

        print(
            "scripts/cpmnetv2/"
            "03_validate_preprocessed.py"
        )

        return 0

    print()
    print(
        "LUNA16 preprocessing completed "
        "with errors."
    )

    print(
        "Inspect failed_cases.csv and/or "
        "coordinate_failures.csv."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())