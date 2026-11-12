from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "candidate1_fold0"
)

CPM_CSV = (
    RESULT_ROOT
    / "cpm.csv"
)

METRICS_JSON = (
    RESULT_ROOT
    / "metrics.json"
)

OUTPUT_PATH = (
    RESULT_ROOT
    / "froc_curve.png"
)


# ============================================================
# Load CPM points
# ============================================================

fp_rates = []
sensitivities = []


with open(
    CPM_CSV,
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(
        f
    )

    for row in reader:

        fp_rates.append(
            float(
                row["fp_per_scan"]
            )
        )

        sensitivities.append(
            float(
                row["sensitivity"]
            )
        )


# ============================================================
# Load metrics
# ============================================================

with open(
    METRICS_JSON,
    "r",
    encoding="utf-8",
) as f:

    metrics = json.load(
        f
    )


cpm = float(
    metrics["cpm"]
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.plot(
    fp_rates,
    sensitivities,
    marker="o",
    linewidth=2,
)


ax.set_xscale(
    "log",
    base=2,
)


ax.set_xticks(
    fp_rates
)

ax.set_xticklabels(
    [
        "0.125",
        "0.25",
        "0.5",
        "1",
        "2",
        "4",
        "8",
    ]
)


ax.set_ylim(
    0,
    1,
)


ax.set_xlabel(
    "False Positives per Scan"
)

ax.set_ylabel(
    "Sensitivity"
)


ax.set_title(
    "Candidate 1 - FROC Curve\n"
    f"MONAI Pretrained RetinaNet | CPM={cpm:.4f}"
)


ax.grid(
    True,
    alpha=0.3,
)


# ============================================================
# Point labels
# ============================================================

for x, y in zip(
    fp_rates,
    sensitivities,
):

    ax.annotate(
        f"{y:.3f}",
        (
            x,
            y,
        ),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
    )


fig.tight_layout()


fig.savefig(
    OUTPUT_PATH,
    dpi=180,
    bbox_inches="tight",
)


plt.close(
    fig
)


print("=" * 70)
print("✅ Candidate 1 FROC visualization 완료")
print(OUTPUT_PATH)
print("=" * 70)