#!/usr/bin/env python3

import sys
import csv
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import SimpleITK as sitk


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path("/mnt/c/AI/ct-project")
LUNA_ROOT = PROJECT_ROOT / "data" / "LUNA16"
RESULT_ROOT = PROJECT_ROOT / "results" / "cpmnetv2"


# =============================================================================
# Utility
# =============================================================================

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


def find_files_case_insensitive(root, filename):
    target = filename.lower()

    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.name.lower() == target
    ]


# =============================================================================
# 1. Root validation
# =============================================================================

def validate_root():
    print_header("1. LUNA16 Root Validation")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"LUNA16 root  : {LUNA_ROOT}")

    project_ok = PROJECT_ROOT.exists()
    luna_ok = LUNA_ROOT.exists()

    print_result(
        "Project root exists",
        project_ok,
        PROJECT_ROOT,
    )

    print_result(
        "LUNA16 root exists",
        luna_ok,
        LUNA_ROOT,
    )

    if luna_ok:
        try:
            children = sorted(LUNA_ROOT.iterdir())

            print()
            print("Top-level contents:")

            for path in children[:50]:
                marker = "DIR " if path.is_dir() else "FILE"
                print(f"  [{marker}] {path.name}")

            if len(children) > 50:
                print(f"  ... ({len(children) - 50} more)")

        except Exception as e:
            print(f"Could not list directory: {e}")

    return project_ok and luna_ok


# =============================================================================
# 2. MHD / RAW discovery
# =============================================================================

def discover_ct_files():
    print_header("2. CT File Discovery")

    if not LUNA_ROOT.exists():
        return False, [], [], {}

    mhd_files = sorted(LUNA_ROOT.rglob("*.mhd"))
    raw_files = sorted(LUNA_ROOT.rglob("*.raw"))

    print(f"MHD files : {len(mhd_files)}")
    print(f"RAW files : {len(raw_files)}")

    mhd_by_uid = {
        p.stem: p
        for p in mhd_files
    }

    raw_by_uid = {
        p.stem: p
        for p in raw_files
    }

    mhd_uids = set(mhd_by_uid)
    raw_uids = set(raw_by_uid)

    paired = mhd_uids & raw_uids
    missing_raw = mhd_uids - raw_uids
    orphan_raw = raw_uids - mhd_uids

    print()
    print(f"Paired MHD/RAW : {len(paired)}")
    print(f"MHD without RAW: {len(missing_raw)}")
    print(f"RAW without MHD: {len(orphan_raw)}")

    if missing_raw:
        print()
        print("Example MHD without RAW:")

        for uid in sorted(missing_raw)[:10]:
            print(f"  {uid}")

    if orphan_raw:
        print()
        print("Example RAW without MHD:")

        for uid in sorted(orphan_raw)[:10]:
            print(f"  {uid}")

    pair_ok = (
        len(mhd_files) > 0
        and len(paired) == len(mhd_files)
    )

    print_result(
        "MHD files discovered",
        len(mhd_files) > 0,
        f"count = {len(mhd_files)}",
    )

    print_result(
        "Every MHD has corresponding RAW",
        pair_ok,
        (
            f"paired = {len(paired)} / "
            f"MHD = {len(mhd_files)}"
        ),
    )

    return (
        pair_ok,
        mhd_files,
        raw_files,
        mhd_by_uid,
    )


# =============================================================================
# 3. annotations.csv
# =============================================================================

def find_annotations_csv():
    print_header("3. annotations.csv Validation")

    candidates = find_files_case_insensitive(
        LUNA_ROOT,
        "annotations.csv",
    )

    if len(candidates) == 0:
        print_result(
            "annotations.csv found",
            False,
            "No annotations.csv found under LUNA16 root",
        )

        return False, None, None

    print(f"Found {len(candidates)} annotations.csv candidate(s):")

    for p in candidates:
        print(f"  {p}")

    annotation_path = candidates[0]

    if len(candidates) > 1:
        print()
        print(
            "WARNING: Multiple annotations.csv files found. "
            "Using the first one for validation."
        )

    try:
        df = pd.read_csv(annotation_path)

    except Exception as e:
        print_result(
            "Read annotations.csv",
            False,
            f"{type(e).__name__}: {e}",
        )

        return False, annotation_path, None

    print()
    print(f"Selected annotations.csv : {annotation_path}")
    print(f"Rows                     : {len(df)}")
    print(f"Columns                  : {list(df.columns)}")

    expected_columns = [
        "seriesuid",
        "coordX",
        "coordY",
        "coordZ",
        "diameter_mm",
    ]

    missing_columns = [
        col
        for col in expected_columns
        if col not in df.columns
    ]

    columns_ok = len(missing_columns) == 0

    print_result(
        "Required annotation columns",
        columns_ok,
        (
            "required = "
            + ", ".join(expected_columns)
            + (
                ""
                if columns_ok
                else f"\nmissing = {missing_columns}"
            )
        ),
    )

    if not columns_ok:
        return False, annotation_path, df

    print()
    print("Annotation preview:")
    print(df.head().to_string(index=False))

    return True, annotation_path, df


# =============================================================================
# 4. Annotation integrity
# =============================================================================

def validate_annotations(df):
    print_header("4. Annotation Integrity")

    if df is None:
        return False

    required = [
        "seriesuid",
        "coordX",
        "coordY",
        "coordZ",
        "diameter_mm",
    ]

    if any(col not in df.columns for col in required):
        return False

    all_ok = True

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    null_counts = df[required].isnull().sum()

    print("Missing values:")
    for col in required:
        print(f"  {col:15s}: {int(null_counts[col])}")

    no_nulls = int(null_counts.sum()) == 0

    print_result(
        "No missing required annotation values",
        no_nulls,
    )

    if not no_nulls:
        all_ok = False

    # -------------------------------------------------------------------------
    # Numeric validation
    # -------------------------------------------------------------------------

    numeric_columns = [
        "coordX",
        "coordY",
        "coordZ",
        "diameter_mm",
    ]

    numeric_ok = True

    for col in numeric_columns:
        converted = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        invalid = int(converted.isna().sum())

        print_result(
            f"{col} numeric",
            invalid == 0,
            f"invalid = {invalid}",
        )

        if invalid != 0:
            numeric_ok = False

    if not numeric_ok:
        all_ok = False

    # -------------------------------------------------------------------------
    # Finite
    # -------------------------------------------------------------------------

    try:
        arr = df[numeric_columns].astype(float).to_numpy()

        finite_mask = np.isfinite(arr)

        finite_ok = bool(finite_mask.all())

        print_result(
            "Annotation coordinates / diameter finite",
            finite_ok,
        )

        if not finite_ok:
            all_ok = False

    except Exception as e:
        print_result(
            "Annotation coordinates / diameter finite",
            False,
            str(e),
        )

        all_ok = False

    # -------------------------------------------------------------------------
    # Diameter
    # -------------------------------------------------------------------------

    try:
        diameter = pd.to_numeric(
            df["diameter_mm"],
            errors="coerce",
        )

        positive_diameter = bool((diameter > 0).all())

        print_result(
            "All nodule diameters > 0",
            positive_diameter,
            (
                f"min = {diameter.min():.4f} mm\n"
                f"max = {diameter.max():.4f} mm\n"
                f"mean = {diameter.mean():.4f} mm"
            ),
        )

        if not positive_diameter:
            all_ok = False

    except Exception as e:
        print_result(
            "Diameter validation",
            False,
            str(e),
        )

        all_ok = False

    # -------------------------------------------------------------------------
    # General statistics
    # -------------------------------------------------------------------------

    unique_series = df["seriesuid"].astype(str).nunique()

    print()
    print(f"Total annotations      : {len(df)}")
    print(f"Annotated CT series    : {unique_series}")

    if unique_series > 0:
        counts = (
            df.groupby("seriesuid")
            .size()
            .sort_values(ascending=False)
        )

        print(
            f"Nodules / positive CT  : "
            f"mean={counts.mean():.2f}, "
            f"median={counts.median():.2f}, "
            f"max={counts.max()}"
        )

    return all_ok


# =============================================================================
# 5. Series UID matching
# =============================================================================

def validate_series_matching(mhd_by_uid, df):
    print_header("5. Series UID Matching")

    if df is None or "seriesuid" not in df.columns:
        return False

    ct_uids = set(mhd_by_uid.keys())
    annotation_uids = set(
        df["seriesuid"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    annotation_in_ct = annotation_uids & ct_uids
    annotation_missing_ct = annotation_uids - ct_uids
    ct_without_annotation = ct_uids - annotation_uids

    print(f"Total CT series             : {len(ct_uids)}")
    print(f"Annotated series            : {len(annotation_uids)}")
    print(f"Annotated series found in CT: {len(annotation_in_ct)}")
    print(f"Annotation UID missing CT   : {len(annotation_missing_ct)}")
    print(f"CT without annotations      : {len(ct_without_annotation)}")

    match_ok = len(annotation_missing_ct) == 0

    print_result(
        "All annotated series exist as MHD",
        match_ok,
        (
            f"{len(annotation_in_ct)} / "
            f"{len(annotation_uids)} matched"
        ),
    )

    if annotation_missing_ct:
        print()
        print("Example annotation series missing CT:")

        for uid in sorted(annotation_missing_ct)[:10]:
            print(f"  {uid}")

    # NOTE:
    # CT without annotations are NOT automatically an error.
    # They can serve as negative scans depending on the split / training design.

    if ct_without_annotation:
        print()
        print(
            "INFO: CT series without annotations were found."
        )
        print(
            "These are not treated as an error here."
        )

    return match_ok


# =============================================================================
# 6. Subset structure
# =============================================================================

def validate_subset_structure(mhd_files):
    print_header("6. LUNA16 Subset Structure")

    subset_counter = Counter()

    for path in mhd_files:
        subset_name = None

        for parent in path.parents:
            if parent.name.lower().startswith("subset"):
                subset_name = parent.name
                break

        if subset_name is None:
            subset_name = "<no subset directory>"

        subset_counter[subset_name] += 1

    print("MHD distribution:")

    for subset_name in sorted(subset_counter):
        print(
            f"  {subset_name:20s}: "
            f"{subset_counter[subset_name]}"
        )

    standard_subsets = {
        f"subset{i}"
        for i in range(10)
    }

    found_standard = {
        name.lower()
        for name in subset_counter
        if name.lower() in standard_subsets
    }

    print()
    print(
        f"Standard subset directories found: "
        f"{len(found_standard)} / 10"
    )

    if len(found_standard) == 10:
        print_result(
            "subset0 ~ subset9 present",
            True,
        )
        return True

    print_result(
        "subset0 ~ subset9 present",
        False,
        (
            "This is not necessarily fatal if the local LUNA16 "
            "dataset was reorganized.\n"
            "We only need to understand the actual structure "
            "before preprocessing."
        ),
    )

    return False


# =============================================================================
# 7. SimpleITK sample QA
# =============================================================================

def validate_sample_images(mhd_files, sample_count=5):
    print_header("7. Sample CT Read QA")

    if len(mhd_files) == 0:
        print_result(
            "Sample CT read",
            False,
            "No MHD files available",
        )

        return False

    sample_files = mhd_files[:sample_count]

    all_ok = True

    for i, path in enumerate(sample_files, start=1):
        print()
        print("-" * 80)
        print(f"Sample {i}: {path.name}")
        print(f"Path    : {path}")

        try:
            image = sitk.ReadImage(str(path))

            array = sitk.GetArrayFromImage(image)

            size_xyz = image.GetSize()
            spacing_xyz = image.GetSpacing()
            origin_xyz = image.GetOrigin()

            direction = image.GetDirection()

            print(f"ITK size [X,Y,Z]     : {size_xyz}")
            print(f"NumPy shape [Z,Y,X] : {array.shape}")
            print(f"Spacing [X,Y,Z]     : {spacing_xyz}")
            print(f"Origin [X,Y,Z]      : {origin_xyz}")

            print(
                f"Intensity range      : "
                f"{float(array.min()):.2f} ~ "
                f"{float(array.max()):.2f}"
            )

            shape_ok = (
                array.ndim == 3
                and all(v > 0 for v in array.shape)
            )

            spacing_ok = (
                len(spacing_xyz) == 3
                and all(
                    np.isfinite(v) and v > 0
                    for v in spacing_xyz
                )
            )

            finite_ok = bool(
                np.isfinite(array).all()
            )

            print_result(
                "3D image shape",
                shape_ok,
                array.shape,
            )

            print_result(
                "Positive spacing",
                spacing_ok,
                spacing_xyz,
            )

            print_result(
                "Finite voxel values",
                finite_ok,
            )

            if not (
                shape_ok
                and spacing_ok
                and finite_ok
            ):
                all_ok = False

        except Exception as e:
            print_result(
                "Read CT",
                False,
                f"{type(e).__name__}: {e}",
            )

            all_ok = False

    return all_ok


# =============================================================================
# 8. Annotation physical-coordinate QA
# =============================================================================

def validate_annotation_coordinates(
    mhd_by_uid,
    df,
    sample_series_count=5,
):
    print_header("8. Annotation Physical Coordinate QA")

    if df is None:
        return False

    required = {
        "seriesuid",
        "coordX",
        "coordY",
        "coordZ",
        "diameter_mm",
    }

    if not required.issubset(df.columns):
        return False

    grouped = df.groupby("seriesuid")

    checked = 0
    failures = 0

    for seriesuid, group in grouped:
        seriesuid = str(seriesuid).strip()

        if seriesuid not in mhd_by_uid:
            continue

        path = mhd_by_uid[seriesuid]

        try:
            image = sitk.ReadImage(str(path))

        except Exception as e:
            print(
                f"[FAIL] Could not read {seriesuid}: {e}"
            )

            failures += 1
            continue

        size_xyz = np.asarray(
            image.GetSize(),
            dtype=np.int64,
        )

        print()
        print("-" * 80)
        print(f"Series UID : {seriesuid}")
        print(f"CT         : {path}")
        print(f"ITK size   : {tuple(size_xyz)}")

        for row_index, row in group.iterrows():
            physical_point = (
                float(row["coordX"]),
                float(row["coordY"]),
                float(row["coordZ"]),
            )

            diameter = float(row["diameter_mm"])

            try:
                continuous_index = (
                    image.TransformPhysicalPointToContinuousIndex(
                        physical_point
                    )
                )

                idx = np.asarray(
                    continuous_index,
                    dtype=float,
                )

                inside = bool(
                    np.all(idx >= 0)
                    and np.all(
                        idx < size_xyz
                    )
                )

                print(
                    f"  Annotation row {row_index}:"
                )
                print(
                    f"    world XYZ    = "
                    f"({physical_point[0]:.3f}, "
                    f"{physical_point[1]:.3f}, "
                    f"{physical_point[2]:.3f})"
                )
                print(
                    f"    voxel XYZ    = "
                    f"({idx[0]:.3f}, "
                    f"{idx[1]:.3f}, "
                    f"{idx[2]:.3f})"
                )
                print(
                    f"    diameter     = "
                    f"{diameter:.3f} mm"
                )
                print(
                    f"    inside image = "
                    f"{inside}"
                )

                if not inside:
                    failures += 1

            except Exception as e:
                print(
                    f"    Transform FAILED: "
                    f"{type(e).__name__}: {e}"
                )

                failures += 1

        checked += 1

        if checked >= sample_series_count:
            break

    success = (
        checked > 0
        and failures == 0
    )

    print()
    print_result(
        "Sample annotation coordinates inside CT",
        success,
        (
            f"series checked = {checked}\n"
            f"failures = {failures}"
        ),
    )

    return success


# =============================================================================
# 9. Dataset summary
# =============================================================================

def print_dataset_summary(mhd_files, df):
    print_header("9. LUNA16 Dataset Summary")

    total_ct = len(mhd_files)

    if df is not None and "seriesuid" in df.columns:
        positive_ct = (
            df["seriesuid"]
            .astype(str)
            .nunique()
        )

        annotations = len(df)

    else:
        positive_ct = 0
        annotations = 0

    negative_or_unannotated = max(
        total_ct - positive_ct,
        0,
    )

    print(f"Total CT series            : {total_ct}")
    print(f"Annotated / positive CT    : {positive_ct}")
    print(f"Unannotated CT             : {negative_or_unannotated}")
    print(f"Total nodule annotations   : {annotations}")

    if annotations > 0:
        diameter = pd.to_numeric(
            df["diameter_mm"],
            errors="coerce",
        )

        print()
        print("Diameter statistics:")
        print(f"  min    : {diameter.min():.3f} mm")
        print(f"  median : {diameter.median():.3f} mm")
        print(f"  mean   : {diameter.mean():.3f} mm")
        print(f"  max    : {diameter.max():.3f} mm")

        print()
        print("Diameter groups:")

        groups = {
            "< 6 mm": diameter < 6,
            "6-10 mm": (
                (diameter >= 6)
                & (diameter < 10)
            ),
            ">= 10 mm": diameter >= 10,
        }

        for name, mask in groups.items():
            count = int(mask.sum())

            pct = (
                count / annotations * 100
                if annotations > 0
                else 0
            )

            print(
                f"  {name:10s}: "
                f"{count:4d} "
                f"({pct:5.1f}%)"
            )


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("CPMNetv2 LUNA16 Data Validation")

    results = {}

    # -------------------------------------------------------------------------
    # Root
    # -------------------------------------------------------------------------

    results["root"] = validate_root()

    if not results["root"]:
        print()
        print("LUNA16 root validation failed.")
        print("Cannot continue.")

        return 1

    # -------------------------------------------------------------------------
    # CT files
    # -------------------------------------------------------------------------

    (
        ct_pair_ok,
        mhd_files,
        raw_files,
        mhd_by_uid,
    ) = discover_ct_files()

    results["mhd_raw"] = ct_pair_ok

    # -------------------------------------------------------------------------
    # annotations.csv
    # -------------------------------------------------------------------------

    (
        annotation_csv_ok,
        annotation_path,
        df,
    ) = find_annotations_csv()

    results["annotations_csv"] = annotation_csv_ok

    # -------------------------------------------------------------------------
    # Annotation integrity
    # -------------------------------------------------------------------------

    results["annotation_integrity"] = (
        validate_annotations(df)
    )

    # -------------------------------------------------------------------------
    # UID matching
    # -------------------------------------------------------------------------

    results["series_uid_match"] = (
        validate_series_matching(
            mhd_by_uid,
            df,
        )
    )

    # -------------------------------------------------------------------------
    # subset structure
    #
    # Informational:
    # non-standard local structure does not block preprocessing.
    # -------------------------------------------------------------------------

    subset_standard = validate_subset_structure(
        mhd_files
    )

    # -------------------------------------------------------------------------
    # SimpleITK sample read
    # -------------------------------------------------------------------------

    results["sample_ct_read"] = (
        validate_sample_images(
            mhd_files,
            sample_count=5,
        )
    )

    # -------------------------------------------------------------------------
    # Physical annotation -> voxel coordinate check
    # -------------------------------------------------------------------------

    results["coordinate_qa"] = (
        validate_annotation_coordinates(
            mhd_by_uid,
            df,
            sample_series_count=5,
        )
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_dataset_summary(
        mhd_files,
        df,
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print_header("10. Final Summary")

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{name:24s}: {status}")

    print()
    print(
        f"subset0-subset9 standard : "
        f"{'YES' if subset_standard else 'NO'}"
    )

    passed = sum(
        bool(v)
        for v in results.values()
    )

    total = len(results)

    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print()
        print("LUNA16 data validation PASSED.")
        print()
        print(
            "Next step:"
        )
        print(
            "scripts/cpmnetv2/02_prepare_luna16.py"
        )
        print()
        print(
            "Before preprocessing, use this validation "
            "output to confirm:"
        )
        print(
            "  1. actual CT count"
        )
        print(
            "  2. annotations.csv path"
        )
        print(
            "  3. subset structure"
        )
        print(
            "  4. annotation coordinate matching"
        )

        return 0

    print()
    print("LUNA16 data validation FAILED.")
    print(
        "Resolve the FAIL items before preprocessing."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())