#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import SimpleITK as sitk


PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

DATA_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "luna16"
)

IMAGES_TR = DATA_ROOT / "imagesTr"
IMAGES_VA = DATA_ROOT / "imagesVa"

TRAIN_CSV = DATA_ROOT / "train.csv"
VAL_CSV = DATA_ROOT / "val.csv"
MANIFEST_CSV = DATA_ROOT / "preprocessing_manifest.csv"

EXPECTED_SPACING_XYZ = np.array(
    [1.0, 1.0, 1.0],
    dtype=np.float64,
)


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_result(name, success, detail=""):
    status = "PASS" if success else "FAIL"

    print(f"[{status}] {name}")

    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def validate_structure():
    print_header("1. Preprocessed Structure")

    checks = {
        "Data root": DATA_ROOT.exists(),
        "imagesTr": IMAGES_TR.exists(),
        "imagesVa": IMAGES_VA.exists(),
        "train.csv": TRAIN_CSV.exists(),
        "val.csv": VAL_CSV.exists(),
        "manifest": MANIFEST_CSV.exists(),
    }

    for name, result in checks.items():
        print_result(name, result)

    return all(checks.values())


def discover_images():
    print_header("2. Image Discovery")

    train_images = sorted(
        IMAGES_TR.glob("*.nii.gz")
    )

    val_images = sorted(
        IMAGES_VA.glob("*.nii.gz")
    )

    print(
        f"Train images      : {len(train_images)}"
    )

    print(
        f"Validation images : {len(val_images)}"
    )

    print_result(
        "At least one preprocessed CT",
        len(train_images) + len(val_images) > 0,
    )

    return train_images, val_images


def validate_sample_images(
    images,
    sample_count=5,
):
    print_header("3. Resampled CT QA")

    if not images:
        print_result(
            "Sample CT validation",
            False,
            "No images available.",
        )

        return False

    all_ok = True

    images = images[:sample_count]

    for index, path in enumerate(
        images,
        start=1,
    ):
        print()
        print("-" * 80)
        print(
            f"Sample {index}: {path.name}"
        )

        try:
            image = sitk.ReadImage(
                str(path)
            )

            arr = sitk.GetArrayFromImage(
                image
            ).astype(np.float32)

            spacing_xyz = np.asarray(
                image.GetSpacing(),
                dtype=np.float64,
            )

            size_xyz = image.GetSize()

            shape_zyx = arr.shape

            print(
                f"ITK size [X,Y,Z]     : "
                f"{size_xyz}"
            )

            print(
                f"NumPy shape [Z,Y,X]  : "
                f"{shape_zyx}"
            )

            print(
                f"Spacing [X,Y,Z]      : "
                f"{tuple(spacing_xyz)}"
            )

            print(
                f"HU min               : "
                f"{float(arr.min()):.2f}"
            )

            print(
                f"HU max               : "
                f"{float(arr.max()):.2f}"
            )

            print(
                f"HU mean              : "
                f"{float(arr.mean()):.2f}"
            )

            print(
                f"HU median            : "
                f"{float(np.median(arr)):.2f}"
            )

            p01 = float(
                np.percentile(arr, 1)
            )

            p50 = float(
                np.percentile(arr, 50)
            )

            p99 = float(
                np.percentile(arr, 99)
            )

            print(
                f"HU percentile 1%     : "
                f"{p01:.2f}"
            )

            print(
                f"HU percentile 50%    : "
                f"{p50:.2f}"
            )

            print(
                f"HU percentile 99%    : "
                f"{p99:.2f}"
            )

            spacing_ok = np.allclose(
                spacing_xyz,
                EXPECTED_SPACING_XYZ,
                atol=1e-6,
            )

            finite_ok = bool(
                np.isfinite(arr).all()
            )

            shape_ok = (
                arr.ndim == 3
                and all(v > 0 for v in arr.shape)
            )

            print_result(
                "1mm isotropic spacing",
                spacing_ok,
                tuple(spacing_xyz),
            )

            print_result(
                "3D array",
                shape_ok,
                shape_zyx,
            )

            print_result(
                "Finite voxel values",
                finite_ok,
            )

            if not (
                spacing_ok
                and finite_ok
                and shape_ok
            ):
                all_ok = False

        except Exception as e:
            print_result(
                "Read image",
                False,
                f"{type(e).__name__}: {e}",
            )

            all_ok = False

    return all_ok


def validate_csv(
    csv_path,
    image_dir,
    name,
):
    print_header(
        f"4. {name} Annotation QA"
    )

    df = load_csv(csv_path)

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
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        print_result(
            "CSV columns",
            False,
            f"Missing = {missing}",
        )

        return False

    print(
        f"Rows: {len(df)}"
    )

    if len(df) == 0:
        print(
            "INFO: CSV is empty."
        )

        return True

    finite_columns = [
        "coordX",
        "coordY",
        "coordZ",
        "w",
        "h",
        "d",
    ]

    arr = (
        df[finite_columns]
        .astype(float)
        .to_numpy()
    )

    finite_ok = bool(
        np.isfinite(arr).all()
    )

    size_ok = bool(
        (
            df[["w", "h", "d"]]
            .astype(float)
            > 0
        )
        .all()
        .all()
    )

    print_result(
        "Finite coordinates / sizes",
        finite_ok,
    )

    print_result(
        "w/h/d > 0",
        size_ok,
    )

    missing_image_count = 0
    outside_count = 0

    checked_rows = 0

    by_file = defaultdict(list)

    for idx, row in df.iterrows():
        by_file[str(row["seriesuid"])].append(
            (idx, row)
        )

    for filename, rows in by_file.items():
        image_path = image_dir / filename

        if not image_path.exists():
            missing_image_count += 1

            print(
                f"[WARNING] Missing image: "
                f"{filename}"
            )

            continue

        image = sitk.ReadImage(
            str(image_path)
        )

        size_xyz = np.asarray(
            image.GetSize(),
            dtype=np.float64,
        )

        for row_index, row in rows:
            xyz = np.asarray(
                [
                    float(row["coordX"]),
                    float(row["coordY"]),
                    float(row["coordZ"]),
                ]
            )

            inside = bool(
                np.all(xyz >= 0)
                and np.all(xyz < size_xyz)
            )

            checked_rows += 1

            if not inside:
                outside_count += 1

                print(
                    f"[WARNING] Coordinate outside CT"
                )

                print(
                    f"  row       : {row_index}"
                )

                print(
                    f"  file      : {filename}"
                )

                print(
                    f"  coord XYZ : {xyz}"
                )

                print(
                    f"  size XYZ  : {size_xyz}"
                )

    print()
    print_result(
        "CSV files exist",
        missing_image_count == 0,
        f"missing images = {missing_image_count}",
    )

    print_result(
        "Annotation centers inside CT",
        outside_count == 0,
        (
            f"checked = {checked_rows}\n"
            f"outside = {outside_count}"
        ),
    )

    return (
        finite_ok
        and size_ok
        and missing_image_count == 0
        and outside_count == 0
    )


def validate_manifest():
    print_header(
        "5. Manifest QA"
    )

    df = load_csv(
        MANIFEST_CSV
    )

    print(
        f"Manifest rows : {len(df)}"
    )

    if len(df) == 0:
        print_result(
            "Manifest non-empty",
            False,
        )

        return False

    train_count = int(
        (df["split"] == "train").sum()
    )

    val_count = int(
        (df["split"] == "val").sum()
    )

    positive_count = int(
        df["positive"].sum()
    )

    negative_count = int(
        (~df["positive"]).sum()
    )

    annotations = int(
        df["annotation_count"].sum()
    )

    print(
        f"Train CT       : {train_count}"
    )

    print(
        f"Validation CT  : {val_count}"
    )

    print(
        f"Positive CT    : {positive_count}"
    )

    print(
        f"Negative CT    : {negative_count}"
    )

    print(
        f"Annotations    : {annotations}"
    )

    spacing_columns = [
        "new_spacing_x",
        "new_spacing_y",
        "new_spacing_z",
    ]

    spacing = (
        df[spacing_columns]
        .astype(float)
        .to_numpy()
    )

    spacing_ok = bool(
        np.allclose(
            spacing,
            1.0,
            atol=1e-6,
        )
    )

    print_result(
        "Manifest spacing = 1mm",
        spacing_ok,
    )

    return spacing_ok


def evaluate_mra_normalization(
    images,
    sample_count=5,
):
    print_header(
        "6. MRA Normalization Compatibility Check"
    )

    if not images:
        print_result(
            "Normalization QA",
            False,
            "No CTs available.",
        )

        return False

    print(
        "Original CPMNetv2 dataset.py normalization:"
    )

    print(
        "  max_value = percentile(data, 99)"
    )

    print(
        "  min_value = 0"
    )

    print(
        "  data[data < 0] = 0"
    )

    print(
        "  data = data / max_value"
    )

    print()

    total_voxels = 0
    negative_voxels = 0

    for path in images[:sample_count]:
        image = sitk.ReadImage(
            str(path)
        )

        arr = sitk.GetArrayFromImage(
            image
        ).astype(np.float32)

        count = arr.size

        negatives = int(
            (arr < 0).sum()
        )

        ratio = (
            negatives / count
            if count
            else 0
        )

        total_voxels += count
        negative_voxels += negatives

        p99 = float(
            np.percentile(arr, 99)
        )

        print(
            f"{path.name}"
        )

        print(
            f"  negative HU ratio : "
            f"{ratio * 100:.2f}%"
        )

        print(
            f"  99th percentile   : "
            f"{p99:.2f}"
        )

    overall_ratio = (
        negative_voxels
        / total_voxels
        if total_voxels
        else 0
    )

    print()
    print(
        f"Overall negative-HU voxel ratio: "
        f"{overall_ratio * 100:.2f}%"
    )

    if overall_ratio > 0.5:
        print()
        print(
            "[WARNING] More than 50% of CT voxels "
            "are below 0 HU."
        )

        print(
            "The original MRA normalization would "
            "clip most lung/background voxels to 0."
        )

        print(
            "Do NOT use dataset.py __norm__ unchanged "
            "for the LUNA16 candidate."
        )

        return False

    print_result(
        "MRA normalization appears usable",
        True,
    )

    return True


def main():
    print_header(
        "CPMNetv2 Preprocessed LUNA16 Validation"
    )

    results = {}

    results["structure"] = (
        validate_structure()
    )

    if not results["structure"]:
        return 1

    train_images, val_images = (
        discover_images()
    )

    all_images = (
        train_images
        + val_images
    )

    results["ct_qa"] = (
        validate_sample_images(
            all_images,
            sample_count=5,
        )
    )

    results["train_csv"] = (
        validate_csv(
            TRAIN_CSV,
            IMAGES_TR,
            "Train",
        )
    )

    results["val_csv"] = (
        validate_csv(
            VAL_CSV,
            IMAGES_VA,
            "Validation",
        )
    )

    results["manifest"] = (
        validate_manifest()
    )

    # This is diagnostic, not a preprocessing failure.
    normalization_compatible = (
        evaluate_mra_normalization(
            all_images,
            sample_count=5,
        )
    )

    print_header(
        "7. Final Summary"
    )

    for key, value in results.items():
        print(
            f"{key:24s}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    print()
    print(
        "MRA normalization usable : "
        f"{'YES' if normalization_compatible else 'NO'}"
    )

    passed = sum(
        bool(v)
        for v in results.values()
    )

    total = len(results)

    print()
    print(
        f"Core QA passed: "
        f"{passed}/{total}"
    )

    if passed != total:
        print()
        print(
            "Preprocessed data QA FAILED."
        )

        return 1

    print()
    print(
        "Preprocessed LUNA16 QA PASSED."
    )

    if not normalization_compatible:
        print()
        print(
            "IMPORTANT:"
        )

        print(
            "The data itself is valid, but the "
            "original MRA intensity normalization "
            "must be replaced for LUNA16."
        )

    print()
    print(
        "Next step:"
    )

    print(
        "LUNA16 intensity normalization decision"
    )

    print(
        "then scripts/cpmnetv2/04_smoke_train.py"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())