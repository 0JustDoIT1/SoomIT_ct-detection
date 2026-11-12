from pathlib import Path
import json
import math
import sys


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BUNDLE_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "monai"
    / "bundles"
    / "lung_nodule_ct_detection"
)

FOLD0_PATH = (
    BUNDLE_ROOT
    / "LUNA16_datasplit"
    / "mhd_original"
    / "dataset_fold0.json"
)

SMOKE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "candidate1_smoke"
)

PREDICTION_PATH = (
    SMOKE_ROOT
    / "output"
    / "candidate1_smoke_prediction.json"
)


# ============================================================
# Box conversion
#
# cccwhd:
# [cx, cy, cz, w, h, d]
#
# xyzxyz:
# [xmin, ymin, zmin, xmax, ymax, zmax]
# ============================================================

def cccwhd_to_xyzxyz(box):

    if len(box) != 6:
        raise ValueError(
            f"box length != 6: {box}"
        )

    cx, cy, cz, w, h, d = map(
        float,
        box,
    )

    return [
        cx - w / 2.0,
        cy - h / 2.0,
        cz - d / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
        cz + d / 2.0,
    ]


# ============================================================
# 3D IoU
# ============================================================

def box_iou_3d(
    box_a,
    box_b,
):

    ax1, ay1, az1, ax2, ay2, az2 = box_a
    bx1, by1, bz1, bx2, by2, bz2 = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    iz1 = max(
        az1,
        bz1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iz2 = min(
        az2,
        bz2,
    )


    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    id_ = max(
        0.0,
        iz2 - iz1,
    )


    intersection = (
        iw
        * ih
        * id_
    )


    volume_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    ) * max(
        0.0,
        az2 - az1,
    )


    volume_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    ) * max(
        0.0,
        bz2 - bz1,
    )


    union = (
        volume_a
        + volume_b
        - intersection
    )


    if union <= 0:
        return 0.0


    return (
        intersection
        / union
    )


# ============================================================
# Center distance
# ============================================================

def center_distance(
    box_a,
    box_b,
):

    ax = (
        box_a[0]
        + box_a[3]
    ) / 2.0

    ay = (
        box_a[1]
        + box_a[4]
    ) / 2.0

    az = (
        box_a[2]
        + box_a[5]
    ) / 2.0


    bx = (
        box_b[0]
        + box_b[3]
    ) / 2.0

    by = (
        box_b[1]
        + box_b[4]
    ) / 2.0

    bz = (
        box_b[2]
        + box_b[5]
    ) / 2.0


    return math.sqrt(
        (ax - bx) ** 2
        + (ay - by) ** 2
        + (az - bz) ** 2
    )


# ============================================================
# Find validation GT
# ============================================================

def find_gt_item(
    seriesuid,
):

    with open(
        FOLD0_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        fold0 = json.load(
            f
        )


    validation = fold0.get(
        "validation",
        []
    )


    for item in validation:

        image_name = Path(
            str(
                item.get(
                    "image",
                    ""
                )
            )
        ).name

        uid = (
            image_name[:-4]
            if image_name.endswith(
                ".mhd"
            )
            else Path(
                image_name
            ).stem
        )


        if uid == seriesuid:
            return item


    return None


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "Candidate 1 - Prediction vs GT"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    if not PREDICTION_PATH.exists():

        print(
            "❌ prediction JSON 없음:"
        )

        print(
            PREDICTION_PATH
        )

        sys.exit(
            1
        )


    if not FOLD0_PATH.exists():

        print(
            "❌ fold0 JSON 없음:"
        )

        print(
            FOLD0_PATH
        )

        sys.exit(
            1
        )


    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    with open(
        PREDICTION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        predictions = json.load(
            f
        )


    if (
        not isinstance(
            predictions,
            list,
        )
        or len(
            predictions
        ) == 0
    ):

        raise RuntimeError(
            "Prediction output이 비어 있습니다."
        )


    pred_case = predictions[0]


    image_value = str(
        pred_case.get(
            "image",
            ""
        )
    )


    image_name = Path(
        image_value
    ).name


    if image_name.endswith(
        ".mhd"
    ):

        seriesuid = (
            image_name[:-4]
        )

    else:

        seriesuid = Path(
            image_name
        ).stem


    pred_boxes = pred_case.get(
        "box",
        []
    )

    pred_scores = pred_case.get(
        "label_scores",
        pred_case.get(
            "score",
            [],
        ),
    )


    print()
    print(
        "[1] Prediction"
    )

    print()

    print(
        "SeriesUID:",
        seriesuid,
    )

    print(
        "Pred boxes:",
        len(
            pred_boxes
        ),
    )

    print(
        "Pred scores:",
        len(
            pred_scores
        ),
    )


    if len(
        pred_boxes
    ) != len(
        pred_scores
    ):

        raise RuntimeError(
            "Prediction box / score 개수 불일치"
        )


    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    gt_item = find_gt_item(
        seriesuid
    )


    if gt_item is None:

        raise RuntimeError(
            f"fold0 validation에서 GT를 찾지 못했습니다: {seriesuid}"
        )


    gt_boxes = gt_item.get(
        "box",
        []
    )


    print()
    print(
        "[2] Ground Truth"
    )

    print()

    print(
        "GT boxes:",
        len(
            gt_boxes
        ),
    )


    if len(
        gt_boxes
    ) == 0:

        raise RuntimeError(
            "GT box가 없습니다."
        )


    # --------------------------------------------------------
    # Validate numbers
    # --------------------------------------------------------

    for box in (
        pred_boxes
        + gt_boxes
    ):

        for value in box:

            value = float(
                value
            )

            if not math.isfinite(
                value
            ):

                raise RuntimeError(
                    "NaN / Inf box value 발견"
                )


    for score in pred_scores:

        score = float(
            score
        )

        if not math.isfinite(
            score
        ):

            raise RuntimeError(
                "NaN / Inf score 발견"
            )


    # --------------------------------------------------------
    # Convert boxes
    # --------------------------------------------------------

    gt_xyzxyz = [
        cccwhd_to_xyzxyz(
            box
        )
        for box in gt_boxes
    ]

    pred_xyzxyz = [
        cccwhd_to_xyzxyz(
            box
        )
        for box in pred_boxes
    ]


    # --------------------------------------------------------
    # Sort prediction by score
    # --------------------------------------------------------

    pred_indices = sorted(
        range(
            len(
                pred_scores
            )
        ),
        key=lambda i: float(
            pred_scores[i]
        ),
        reverse=True,
    )


    print()
    print(
        "[3] Predictions sorted by score"
    )

    print()


    for rank, pred_idx in enumerate(
        pred_indices,
        start=1,
    ):

        score = float(
            pred_scores[
                pred_idx
            ]
        )

        pred_box = (
            pred_xyzxyz[
                pred_idx
            ]
        )


        best_iou = 0.0
        best_gt_idx = None
        best_distance = None


        for gt_idx, gt_box in enumerate(
            gt_xyzxyz
        ):

            iou = box_iou_3d(
                pred_box,
                gt_box,
            )

            distance = center_distance(
                pred_box,
                gt_box,
            )


            if (
                best_gt_idx is None
                or iou > best_iou
            ):

                best_iou = iou
                best_gt_idx = gt_idx
                best_distance = distance


        print(
            f"Prediction #{rank}"
        )

        print(
            f"  Score         : {score:.6f}"
        )

        print(
            f"  Best GT       : #{best_gt_idx + 1}"
        )

        print(
            f"  Best IoU      : {best_iou:.6f}"
        )

        print(
            f"  Center dist   : {best_distance:.3f}"
        )

        print(
            f"  Box cccwhd    : {pred_boxes[pred_idx]}"
        )

        print()


    # --------------------------------------------------------
    # GT best prediction
    # --------------------------------------------------------

    print(
        "[4] Best prediction for each GT"
    )

    print()


    all_gt_matched = True


    for gt_idx, gt_box in enumerate(
        gt_xyzxyz
    ):

        best_pred_idx = None
        best_iou = -1.0
        best_score = None
        best_distance = None


        for pred_idx, pred_box in enumerate(
            pred_xyzxyz
        ):

            iou = box_iou_3d(
                gt_box,
                pred_box,
            )

            distance = center_distance(
                gt_box,
                pred_box,
            )


            if iou > best_iou:

                best_iou = iou
                best_pred_idx = pred_idx
                best_score = float(
                    pred_scores[
                        pred_idx
                    ]
                )
                best_distance = distance


        print(
            f"GT #{gt_idx + 1}"
        )

        print(
            f"  Best Pred     : #{best_pred_idx + 1}"
        )

        print(
            f"  Score         : {best_score:.6f}"
        )

        print(
            f"  IoU           : {best_iou:.6f}"
        )

        print(
            f"  Center dist   : {best_distance:.3f}"
        )

        print(
            f"  GT cccwhd     : {gt_boxes[gt_idx]}"
        )

        print(
            f"  Pred cccwhd   : {pred_boxes[best_pred_idx]}"
        )

        print()


        if best_iou <= 0.0:

            all_gt_matched = False


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "=" * 80
    )


    if all_gt_matched:

        print(
            "✅ Prediction / GT coordinate check PASS"
        )

        print()

        print(
            "모든 GT에 대해 IoU > 0 prediction 존재"
        )

    else:

        print(
            "⚠️ Prediction / GT coordinate check 필요"
        )

        print()

        print(
            "IoU=0인 GT가 있습니다."
        )

        print(
            "좌표계 / output box format / transformation을 다시 확인하세요."
        )


    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()