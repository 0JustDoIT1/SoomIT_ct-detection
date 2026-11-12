from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LUNA_ROOT = PROJECT_ROOT / "data" / "LUNA16"
RAW_ROOT = LUNA_ROOT / "raw"
ANNOTATIONS_CSV = LUNA_ROOT / "annotations.csv"


# ============================================================
# 1. Annotation 읽기
# ============================================================

annotations = pd.read_csv(
    ANNOTATIONS_CSV
)


# ============================================================
# 2. GT가 있는 실제 CT 하나 선택
# ============================================================

image_path = None
seriesuid = None

for uid in annotations["seriesuid"].dropna().unique():

    matches = list(
        RAW_ROOT.rglob(
            f"{uid}.mhd"
        )
    )

    if matches:
        image_path = matches[0]
        seriesuid = str(uid)
        break


if image_path is None:
    raise RuntimeError(
        "GT annotation이 있는 LUNA16 CT를 찾지 못했습니다."
    )


print("=" * 70)
print("LUNA16 Sample Inspection")
print("=" * 70)

print()
print("CT:")
print(image_path)

print()
print("SeriesUID:")
print(seriesuid)


# ============================================================
# 3. CT 읽기
# ============================================================

image = sitk.ReadImage(
    str(image_path)
)

volume = sitk.GetArrayFromImage(
    image
)


# SimpleITK
# array shape = (z, y, x)
# spacing     = (x, y, z)
# origin      = (x, y, z)

spacing = image.GetSpacing()
origin = image.GetOrigin()
direction = image.GetDirection()


print()
print("[CT 정보]")

print(
    "Array shape (z,y,x):",
    volume.shape
)

print(
    "Spacing (x,y,z):",
    spacing
)

print(
    "Origin (x,y,z):",
    origin
)

print(
    "Direction:",
    direction
)

print(
    "HU min:",
    volume.min()
)

print(
    "HU max:",
    volume.max()
)


# ============================================================
# 4. 해당 CT의 GT Annotation
# ============================================================

gt = annotations[
    annotations["seriesuid"] == seriesuid
].copy()


print()
print("[GT Annotation]")
print()

print(
    "GT nodule 수:",
    len(gt)
)


if len(gt) == 0:

    raise RuntimeError(
        "GT가 있는 CT를 선택했지만 annotation이 없습니다."
    )


# ============================================================
# 5. 각 GT 확인
# ============================================================

for nodule_idx, (_, row) in enumerate(
    gt.iterrows(),
    start=1
):

    x = float(
        row["coordX"]
    )

    y = float(
        row["coordY"]
    )

    z = float(
        row["coordZ"]
    )

    diameter = float(
        row["diameter_mm"]
    )

    radius = (
        diameter / 2.0
    )


    # --------------------------------------------------------
    # World coordinate bbox
    # --------------------------------------------------------

    x_min = x - radius
    x_max = x + radius

    y_min = y - radius
    y_max = y + radius

    z_min = z - radius
    z_max = z + radius


    # --------------------------------------------------------
    # World → voxel index
    #
    # SimpleITK index order = (x, y, z)
    # NumPy array order      = (z, y, x)
    # --------------------------------------------------------

    voxel_center = (
        image.TransformPhysicalPointToIndex(
            (
                x,
                y,
                z,
            )
        )
    )

    vx, vy, vz = voxel_center


    # --------------------------------------------------------
    # Voxel 범위 확인
    # --------------------------------------------------------

    inside_volume = (
        0 <= vz < volume.shape[0]
        and
        0 <= vy < volume.shape[1]
        and
        0 <= vx < volume.shape[2]
    )


    print(
        "-" * 70
    )

    print(
        f"Nodule #{nodule_idx}"
    )

    print(
        "Center world (mm):"
    )

    print(
        f"  x = {x:.3f}"
    )

    print(
        f"  y = {y:.3f}"
    )

    print(
        f"  z = {z:.3f}"
    )

    print(
        f"Diameter: {diameter:.3f} mm"
    )

    print()

    print(
        "Approx GT bbox world (mm):"
    )

    print(
        f"  x: {x_min:.3f} ~ {x_max:.3f}"
    )

    print(
        f"  y: {y_min:.3f} ~ {y_max:.3f}"
    )

    print(
        f"  z: {z_min:.3f} ~ {z_max:.3f}"
    )

    print()

    print(
        "Center voxel (x,y,z):",
        voxel_center
    )

    print(
        "Voxel center inside CT:",
        inside_volume
    )


    if not inside_volume:

        raise RuntimeError(
            "GT 중심 좌표가 CT volume 밖에 있습니다."
        )


print()
print("=" * 70)
print("✅ Sample inspection PASS")
print("=" * 70)