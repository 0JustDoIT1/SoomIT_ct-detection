#!/usr/bin/env python3

import json
from itertools import permutations
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


AXIS_NAMES = ("x", "y", "z")


def dist(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def permute_center(center, order):
    """
    center는 현재 [x, y, z]라고 가정.
    order 예:
      ('x','y','z')
      ('z','y','x')
    """
    mapping = {
        "x": center[0],
        "y": center[1],
        "z": center[2],
    }

    return [
        float(mapping[a])
        for a in order
    ]


def evaluate_order(cases, order):
    nearest_distances = []

    count_5 = 0
    count_10 = 0
    count_20 = 0
    count_radius = 0

    total_gt = 0

    for case in cases:
        preds = case.get("predictions", [])
        gts = case.get("gt_nodules", [])

        pred_centers = []

        for p in preds:
            c = p.get("center_xyz_mm")
            if c is None:
                continue

            pred_centers.append(
                permute_center(c, order)
            )

        for gt in gts:
            gt_center = gt.get("center_xyz_mm")
            diameter = float(
                gt.get("diameter_mm", 0.0)
            )

            if gt_center is None:
                continue

            total_gt += 1

            if not pred_centers:
                nearest = np.inf
            else:
                nearest = min(
                    dist(gt_center, pc)
                    for pc in pred_centers
                )

            nearest_distances.append(nearest)

            if nearest <= diameter / 2.0:
                count_radius += 1

            if nearest <= 5:
                count_5 += 1

            if nearest <= 10:
                count_10 += 1

            if nearest <= 20:
                count_20 += 1

    finite = np.array(
        [
            x for x in nearest_distances
            if np.isfinite(x)
        ],
        dtype=float,
    )

    if len(finite) == 0:
        return {
            "order": "".join(order),
            "mean": np.inf,
            "median": np.inf,
            "p25": np.inf,
            "p75": np.inf,
            "radius": 0,
            "within_5": 0,
            "within_10": 0,
            "within_20": 0,
            "total_gt": total_gt,
        }

    return {
        "order": "".join(order),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "radius": count_radius,
        "within_5": count_5,
        "within_10": count_10,
        "within_20": count_20,
        "total_gt": total_gt,
    }


def main():
    with open(
        PREDICTION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        cases = json.load(f)

    orders = list(
        permutations(AXIS_NAMES)
    )

    results = []

    for order in orders:
        result = evaluate_order(
            cases,
            order,
        )
        results.append(result)

    results.sort(
        key=lambda x: x["median"]
    )

    print("=" * 110)
    print("Candidate 2 Axis Order Comparison")
    print("=" * 110)

    print(
        f"{'Order':<8}"
        f"{'Mean(mm)':>12}"
        f"{'Median(mm)':>14}"
        f"{'P25':>10}"
        f"{'P75':>10}"
        f"{'Radius':>10}"
        f"{'<=5mm':>10}"
        f"{'<=10mm':>10}"
        f"{'<=20mm':>10}"
    )

    print("-" * 110)

    for r in results:
        print(
            f"{r['order']:<8}"
            f"{r['mean']:12.2f}"
            f"{r['median']:14.2f}"
            f"{r['p25']:10.2f}"
            f"{r['p75']:10.2f}"
            f"{r['radius']:10d}"
            f"{r['within_5']:10d}"
            f"{r['within_10']:10d}"
            f"{r['within_20']:10d}"
        )

    best = results[0]

    print()
    print("=" * 110)
    print("Best Axis Order")
    print("=" * 110)

    print(
        f"Order          : {best['order']}"
    )

    print(
        f"Median distance: {best['median']:.2f} mm"
    )

    print(
        f"Mean distance  : {best['mean']:.2f} mm"
    )

    print(
        f"GT radius match: "
        f"{best['radius']}/{best['total_gt']}"
    )

    print(
        f"<= 5 mm        : "
        f"{best['within_5']}/{best['total_gt']}"
    )

    print(
        f"<= 10 mm       : "
        f"{best['within_10']}/{best['total_gt']}"
    )

    print(
        f"<= 20 mm       : "
        f"{best['within_20']}/{best['total_gt']}"
    )

    print()

    if (
        best["order"] != "xyz"
        and best["median"] < 0.5 * next(
            x["median"]
            for x in results
            if x["order"] == "xyz"
        )
    ):
        print(
            "RESULT: axis-order mismatch possibility is HIGH."
        )
        print(
            f"Current xyz보다 {best['order']} 순서에서 "
            "GT와 훨씬 가까워졌습니다."
        )

    elif best["median"] < 20:
        print(
            "RESULT: localization signal exists."
        )
        print(
            "Axis / coordinate conversion should still be inspected."
        )

    else:
        print(
            "RESULT: simple axis permutation does NOT explain the problem."
        )
        print(
            "Next inspect voxel-to-world conversion, origin/direction, "
            "box decoding, or model training."
        )


if __name__ == "__main__":
    main()