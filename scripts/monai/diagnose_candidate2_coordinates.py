#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "evaluation"
    / "candidate2_fold0_predictions.json"
)


def distance_mm(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def main():

    with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    all_nearest = []
    matched_radius = 0
    matched_5mm = 0
    matched_10mm = 0
    matched_20mm = 0

    total_gt = 0

    print("=" * 100)
    print("Candidate 2 Coordinate Diagnostic")
    print("=" * 100)

    for case in cases:

        image = case["image"]
        uid = Path(image).stem

        gts = case.get("gt_nodules", [])
        preds = case.get("predictions", [])

        for gt in gts:

            total_gt += 1

            gt_center = gt["center_xyz_mm"]
            diameter = float(gt["diameter_mm"])
            radius = diameter / 2.0

            if not preds:
                nearest_distance = np.inf
                nearest_score = None
                nearest_center = None

            else:
                distances = []

                for pred in preds:
                    d = distance_mm(
                        gt_center,
                        pred["center_xyz_mm"],
                    )

                    distances.append(
                        (
                            d,
                            float(pred["score"]),
                            pred["center_xyz_mm"],
                            pred.get("box_xyzxyz"),
                        )
                    )

                distances.sort(key=lambda x: x[0])

                (
                    nearest_distance,
                    nearest_score,
                    nearest_center,
                    nearest_box,
                ) = distances[0]

            all_nearest.append(
                {
                    "uid": uid,
                    "gt_center": gt_center,
                    "diameter": diameter,
                    "radius": radius,
                    "nearest_distance": nearest_distance,
                    "nearest_score": nearest_score,
                    "nearest_center": nearest_center,
                }
            )

            if nearest_distance <= radius:
                matched_radius += 1

            if nearest_distance <= 5:
                matched_5mm += 1

            if nearest_distance <= 10:
                matched_10mm += 1

            if nearest_distance <= 20:
                matched_20mm += 1

    distances = np.array(
        [
            x["nearest_distance"]
            for x in all_nearest
            if np.isfinite(x["nearest_distance"])
        ],
        dtype=float,
    )

    print()
    print("GT nodules:", total_gt)

    print()
    print("=" * 100)
    print("Nearest Prediction Distance")
    print("=" * 100)

    print(f"Mean   : {distances.mean():.2f} mm")
    print(f"Median : {np.median(distances):.2f} mm")
    print(f"Min    : {distances.min():.2f} mm")
    print(f"Max    : {distances.max():.2f} mm")

    for p in [25, 50, 75, 90, 95]:
        print(
            f"P{p:<2}    : "
            f"{np.percentile(distances, p):.2f} mm"
        )

    print()
    print("=" * 100)
    print("Distance-based Recall")
    print("=" * 100)

    print(
        f"GT radius match : "
        f"{matched_radius:3d}/{total_gt} "
        f"({matched_radius / total_gt:.4f})"
    )

    print(
        f"<= 5 mm         : "
        f"{matched_5mm:3d}/{total_gt} "
        f"({matched_5mm / total_gt:.4f})"
    )

    print(
        f"<= 10 mm        : "
        f"{matched_10mm:3d}/{total_gt} "
        f"({matched_10mm / total_gt:.4f})"
    )

    print(
        f"<= 20 mm        : "
        f"{matched_20mm:3d}/{total_gt} "
        f"({matched_20mm / total_gt:.4f})"
    )

    print()
    print("=" * 100)
    print("Closest 20 GTs")
    print("=" * 100)

    for x in sorted(
        all_nearest,
        key=lambda x: x["nearest_distance"],
    )[:20]:

        print(
            f'{x["uid"][-18:]:<20} '
            f'D={x["nearest_distance"]:7.2f} mm | '
            f'GTdia={x["diameter"]:6.2f} | '
            f'score={x["nearest_score"]}'
        )

    print()
    print("=" * 100)
    print("Worst 20 GTs")
    print("=" * 100)

    for x in sorted(
        all_nearest,
        key=lambda x: x["nearest_distance"],
        reverse=True,
    )[:20]:

        print(
            f'{x["uid"][-18:]:<20} '
            f'D={x["nearest_distance"]:7.2f} mm | '
            f'GTdia={x["diameter"]:6.2f} | '
            f'score={x["nearest_score"]}'
        )

    print()
    print("=" * 100)
    print("Quick Interpretation")
    print("=" * 100)

    median_distance = float(
        np.median(distances)
    )

    if median_distance < 10:
        print(
            "Prediction centers are generally close to GT."
        )
        print(
            "Matching threshold / FROC logic should be inspected."
        )

    elif median_distance < 30:
        print(
            "Predictions show some localization signal, "
            "but center accuracy is poor."
        )

    else:
        print(
            "Prediction centers are far from GT."
        )
        print(
            "Strongly inspect coordinate conversion, axis order, "
            "box decoding, or model training."
        )


if __name__ == "__main__":
    main()