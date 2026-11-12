from pathlib import Path
import csv

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

PREDICTION_CSV = (
    RESULT_ROOT
    / "predictions.csv"
)

OUTPUT_PATH = (
    RESULT_ROOT
    / "score_distribution.png"
)


# ============================================================
# Load
# ============================================================

tp_scores = []
fp_scores = []


with open(
    PREDICTION_CSV,
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(
        f
    )

    for row in reader:

        score = float(
            row[
                "score"
            ]
        )

        status = row[
            "status"
        ]


        if status == "TP":

            tp_scores.append(
                score
            )

        elif status == "FP":

            fp_scores.append(
                score
            )


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.hist(
    tp_scores,
    bins=20,
    alpha=0.6,
    label="TP",
)


ax.hist(
    fp_scores,
    bins=20,
    alpha=0.6,
    label="FP",
)


ax.set_xlabel(
    "Confidence Score"
)

ax.set_ylabel(
    "Number of Candidates"
)


ax.set_title(
    "Candidate 1 - TP vs FP Score Distribution"
)


ax.legend()


ax.grid(
    True,
    alpha=0.3,
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
print("✅ Candidate 1 score distribution 완료")
print("TP:", len(tp_scores))
print("FP:", len(fp_scores))
print(OUTPUT_PATH)
print("=" * 70)