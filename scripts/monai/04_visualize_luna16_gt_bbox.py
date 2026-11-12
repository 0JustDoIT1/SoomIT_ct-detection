from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.patches import Rectangle


# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LUNA_ROOT = PROJECT_ROOT / "data" / "LUNA16"
RAW_ROOT = LUNA_ROOT / "raw"
ANNOTATIONS_CSV = LUNA_ROOT / "annotations.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "luna16_gt_bbox"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Annotation 읽기
# ============================================================

annotations = pd.read_csv(
    ANNOTATIONS_CSV
)


# ============================================================
# 2. 실제 파일이 존재하면서 GT가 있는 CT 찾기
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
print("LUNA16 GT BBox Visualization")
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
).astype(np.float32)


# SimpleITK
# index   = (x, y, z)
# spacing = (x, y, z)
#
# NumPy
# array   = (z, y, x)

spacing = image.GetSpacing()
origin = image.GetOrigin()
direction = image.GetDirection()


print()
print("[CT 정보]")

print(
    "Shape (z,y,x):",
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


# ============================================================
# 4. 이 CT의 모든 GT 가져오기
# ============================================================

gt_rows = annotations[
    annotations["seriesuid"] == seriesuid
].copy()


print()
print(
    "GT nodule 수:",
    len(gt_rows)
)


if len(gt_rows) == 0:

    raise RuntimeError(
        "선택된 CT에 GT annotation이 없습니다."
    )


# ============================================================
# 5. Lung window
# ============================================================

WINDOW_MIN = -1000
WINDOW_MAX = 400

volume_window = np.clip(
    volume,
    WINDOW_MIN,
    WINDOW_MAX,
)


# ============================================================
# 6. 각 GT 결절 시각화
# ============================================================

for nodule_idx, (_, row) in enumerate(
    gt_rows.iterrows(),
    start=1,
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


    # --------------------------------------------------------
    # World coordinate → continuous voxel coordinate
    #
    # SimpleITK index order = (x, y, z)
    # --------------------------------------------------------

    center_idx = (
        image.TransformPhysicalPointToContinuousIndex(
            (
                x,
                y,
                z,
            )
        )
    )

    cx, cy, cz_float = center_idx

    # 실제 표시할 axial slice
    cz = int(
        round(
            cz_float
        )
    )


    # --------------------------------------------------------
    # 좌표가 CT 내부인지 확인
    # --------------------------------------------------------

    inside_volume = (
        0 <= cx < volume.shape[2]
        and
        0 <= cy < volume.shape[1]
        and
        0 <= cz < volume.shape[0]
    )


    if not inside_volume:

        raise RuntimeError(
            f"Nodule #{nodule_idx}의 중심이 "
            "CT volume 밖에 있습니다."
        )


    # --------------------------------------------------------
    # diameter(mm) → voxel 크기
    #
    # LUNA16 annotation은 center + diameter이므로
    # central axial slice에서 정사각형 근사 bbox 생성
    # --------------------------------------------------------

    diameter_x_vox = (
        diameter
        / spacing[0]
    )

    diameter_y_vox = (
        diameter
        / spacing[1]
    )


    x_min = (
        cx
        - diameter_x_vox / 2.0
    )

    y_min = (
        cy
        - diameter_y_vox / 2.0
    )


    print()
    print("-" * 70)

    print(
        f"Nodule #{nodule_idx}"
    )

    print(
        "World center (mm):",
        (
            x,
            y,
            z,
        )
    )

    print(
        "Continuous voxel center (x,y,z):",
        (
            round(cx, 3),
            round(cy, 3),
            round(cz_float, 3),
        )
    )

    print(
        "Displayed slice z:",
        cz
    )

    print(
        "Diameter:",
        f"{diameter:.2f} mm"
    )

    print(
        "2D approximate bbox voxel:"
    )

    print(
        f"  x = "
        f"{x_min:.1f} ~ "
        f"{x_min + diameter_x_vox:.1f}"
    )

    print(
        f"  y = "
        f"{y_min:.1f} ~ "
        f"{y_min + diameter_y_vox:.1f}"
    )


    # ========================================================
    # 그림
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )


    ax.imshow(
        volume_window[cz],
        cmap="gray",
        vmin=WINDOW_MIN,
        vmax=WINDOW_MAX,
    )


    # --------------------------------------------------------
    # GT approximate bbox
    # --------------------------------------------------------

    rect = Rectangle(
        (
            x_min,
            y_min,
        ),
        diameter_x_vox,
        diameter_y_vox,
        fill=False,
        linewidth=2,
    )

    ax.add_patch(
        rect
    )


    # --------------------------------------------------------
    # GT 중심
    # --------------------------------------------------------

    ax.scatter(
        [cx],
        [cy],
        s=40,
        marker="x",
    )


    ax.set_title(
        f"LUNA16 GT Nodule #{nodule_idx}\n"
        f"Slice z={cz} | "
        f"Diameter={diameter:.1f} mm"
    )

    ax.axis(
        "off"
    )


    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    save_path = (
        OUTPUT_DIR
        / (
            f"{seriesuid}"
            f"_nodule{nodule_idx}"
            f"_z{cz}.png"
        )
    )


    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    print(
        "saved:",
        save_path
    )


print()
print("=" * 70)
print("✅ GT bbox visualization PASS")
print("=" * 70)