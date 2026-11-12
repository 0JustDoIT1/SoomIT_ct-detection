#!/usr/bin/env python3

"""
16_train_cpmnetv2_hard_negative_ratio025.py

CPMNetv2 Hard Negative Sampling - Reduced Ratio Candidate

Previous HN Candidate:
    Positive            75.0%
    Ordinary Negative   12.5%
    Hard Negative       12.5%

    HARD_NEGATIVE_RATIO = 0.50
    CPM = 0.636735

Baseline:
    Positive            75.0%
    Ordinary Negative   25.0%
    Hard Negative        0.0%

    CPM = 0.685714

Current Candidate:
    Positive            75.00%
    Ordinary Negative   18.75%
    Hard Negative        6.25%

    Negative pool 내부:
        Ordinary Negative = 75%
        Hard Negative     = 25%

Purpose:
    기존 HN 50% 설정에서 FP 억제는 좋아졌지만
    Recall / CPM이 감소했으므로 HN 강도를 절반으로 낮춘다.

Important:
    - 기존 13번 mining 결과를 그대로 사용한다.
    - mining 조건은 변경하지 않는다.
    - Train/Val split 변경 없음.
    - Validation에 HN 사용 안 함.
    - Architecture/Loss/LR/Batch/Crop 변경 없음.
    - Baseline 또는 HN checkpoint에서 resume하지 않는다.
    - 처음부터 새로 학습한다.

Implementation:
    검증된 14_train_cpmnetv2_hard_negative.py를 기반으로
    아래 두 항목만 변경한 임시 실행본을 생성한다.

    1. HARD_NEGATIVE_RATIO
       0.50 -> 0.25

    2. OUTPUT_ROOT
       hard_negative_training
       ->
       hard_negative_training_ratio025

이를 통해 14번과의 불필요한 코드 차이를 방지한다.
"""

import runpy
import shutil
import sys
from pathlib import Path


# =============================================================================
# PATH
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

SOURCE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
    / "14_train_cpmnetv2_hard_negative.py"
)

GENERATED_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
    / "_generated_16_hn_ratio025.py"
)

HARD_NEGATIVE_CSV = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_mining"
    / "hard_negatives.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_training_ratio025"
)

OLD_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_training"
)


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

OLD_HN_RATIO = 0.50
NEW_HN_RATIO = 0.25

POSITIVE_RATIO = 0.75
NEGATIVE_RATIO = 0.25

ABS_HN_RATIO = (
    NEGATIVE_RATIO
    * NEW_HN_RATIO
)

ABS_ORDINARY_NEG_RATIO = (
    NEGATIVE_RATIO
    * (
        1.0
        - NEW_HN_RATIO
    )
)


# =============================================================================
# BASELINE / PREVIOUS RESULTS
# =============================================================================

BASELINE_CPM = 0.685714

PREVIOUS_HN_CPM = 0.636735

PREVIOUS_HN_BEST_VAL_LOSS = 1.233689


# =============================================================================
# Utility
# =============================================================================

def header(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def fail(message):
    raise RuntimeError(
        message
    )


# =============================================================================
# Validate files
# =============================================================================

def validate_inputs():

    header(
        "Input Validation"
    )

    print(
        "Source training script:"
    )
    print(
        SOURCE_SCRIPT
    )

    print()
    print(
        "Hard Negative CSV:"
    )
    print(
        HARD_NEGATIVE_CSV
    )

    print()
    print(
        "Output:"
    )
    print(
        OUTPUT_ROOT
    )

    if not SOURCE_SCRIPT.exists():

        raise FileNotFoundError(
            "14번 HN training script를 "
            "찾지 못했습니다:\n"
            f"{SOURCE_SCRIPT}"
        )

    if not HARD_NEGATIVE_CSV.exists():

        raise FileNotFoundError(
            "Hard Negative mining CSV를 "
            "찾지 못했습니다:\n"
            f"{HARD_NEGATIVE_CSV}\n\n"
            "13_mine_cpmnetv2_hard_negatives.py "
            "결과가 필요합니다."
        )

    print()
    print(
        "[PASS] Source script exists"
    )

    print(
        "[PASS] Hard Negative CSV exists"
    )


# =============================================================================
# Build candidate source
# =============================================================================

def build_candidate_source():

    header(
        "Build Ratio-0.25 Candidate"
    )

    source = SOURCE_SCRIPT.read_text(
        encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # QA: expected HN ratio
    # -------------------------------------------------------------------------

    old_ratio_text = (
        "HARD_NEGATIVE_RATIO = 0.50"
    )

    new_ratio_text = (
        "HARD_NEGATIVE_RATIO = 0.25"
    )

    ratio_count = source.count(
        old_ratio_text
    )

    print(
        "HN ratio assignment matches:",
        ratio_count,
    )

    if ratio_count != 1:

        fail(
            "14번 script에서\n"
            "'HARD_NEGATIVE_RATIO = 0.50'\n"
            "정확히 1개를 예상했지만 "
            f"{ratio_count}개를 찾았습니다.\n\n"
            "14번 코드가 변경된 것으로 보이므로 "
            "자동 수정하지 않습니다."
        )

    # -------------------------------------------------------------------------
    # QA: expected output path
    # -------------------------------------------------------------------------

    old_output_text = (
        '/ "hard_negative_training"'
    )

    new_output_text = (
        '/ "hard_negative_training_ratio025"'
    )

    output_count = source.count(
        old_output_text
    )

    print(
        "Output path assignment matches:",
        output_count,
    )

    if output_count != 1:

        fail(
            "14번 script에서\n"
            "'/ \"hard_negative_training\"'\n"
            "정확히 1개를 예상했지만 "
            f"{output_count}개를 찾았습니다.\n\n"
            "기존 output을 덮어쓸 위험이 있으므로 "
            "실행을 중단합니다."
        )

    # -------------------------------------------------------------------------
    # Replace
    # -------------------------------------------------------------------------

    candidate = source.replace(
        old_ratio_text,
        new_ratio_text,
        1,
    )

    candidate = candidate.replace(
        old_output_text,
        new_output_text,
        1,
    )

    # -------------------------------------------------------------------------
    # Final QA
    # -------------------------------------------------------------------------

    if (
        "HARD_NEGATIVE_RATIO = 0.25"
        not in candidate
    ):

        fail(
            "HN ratio replacement 실패"
        )

    if (
        '/ "hard_negative_training_ratio025"'
        not in candidate
    ):

        fail(
            "Output path replacement 실패"
        )

    # 원래 output training path가
    # 실제 assignment로 남아 있으면 안 된다.

    if (
        '/ "hard_negative_training"'
        in candidate
    ):

        fail(
            "기존 hard_negative_training output "
            "assignment가 아직 남아 있습니다."
        )

    GENERATED_SCRIPT.write_text(
        candidate,
        encoding="utf-8",
    )

    print()
    print(
        "[PASS] HARD_NEGATIVE_RATIO"
    )
    print(
        "       0.50 -> 0.25"
    )

    print()
    print(
        "[PASS] OUTPUT_ROOT"
    )
    print(
        "       hard_negative_training"
    )
    print(
        "       -> hard_negative_training_ratio025"
    )

    print()
    print(
        "Generated runtime script:"
    )
    print(
        GENERATED_SCRIPT
    )

    return GENERATED_SCRIPT


# =============================================================================
# Inspect generated script
# =============================================================================

def validate_generated_script(
    generated_script,
):

    header(
        "Generated Script QA"
    )

    source = generated_script.read_text(
        encoding="utf-8"
    )

    required = [
        "HARD_NEGATIVE_RATIO = 0.25",
        "hard_negative_training_ratio025",
        "hard_negatives.csv",
    ]

    for text in required:

        if text not in source:

            fail(
                "Generated script에 "
                f"필수 설정이 없습니다: {text}"
            )

        print(
            "[PASS]",
            text,
        )

    forbidden = [
        "HARD_NEGATIVE_RATIO = 0.50",
    ]

    for text in forbidden:

        if text in source:

            fail(
                "Generated script에 "
                f"이전 설정이 남아 있습니다: {text}"
            )

    print()
    print(
        "[PASS] Previous HN ratio assignment removed"
    )


# =============================================================================
# Existing output check
# =============================================================================

def check_existing_output():

    header(
        "Output Safety"
    )

    if not OUTPUT_ROOT.exists():

        print(
            "[PASS] New output directory"
        )
        print(
            OUTPUT_ROOT
        )
        return

    files = list(
        OUTPUT_ROOT.rglob("*")
    )

    files = [
        path
        for path in files
        if path.is_file()
    ]

    if not files:

        print(
            "[PASS] Output directory exists but is empty"
        )
        return

    print(
        "[WARNING] Candidate output already contains files:"
    )

    for path in files[:20]:

        print(
            " -",
            path
        )

    if len(files) > 20:

        print(
            f" ... and {len(files) - 20} more"
        )

    print()

    answer = input(
        "기존 ratio025 결과를 삭제하고 "
        "처음부터 다시 학습할까요? [y/N]: "
    ).strip().lower()

    if answer not in (
        "y",
        "yes",
    ):

        print(
            "Training cancelled."
        )

        raise SystemExit(
            0
        )

    shutil.rmtree(
        OUTPUT_ROOT
    )

    print(
        "[PASS] Existing ratio025 output removed"
    )


# =============================================================================
# Experiment summary
# =============================================================================

def print_experiment():

    header(
        "CPMNetv2 Hard Negative Ratio 0.25"
    )

    print(
        "Previous HN Candidate"
    )

    print(
        f"  Positive          : "
        f"{POSITIVE_RATIO * 100:.2f}%"
    )

    print(
        "  Ordinary Negative : 12.50%"
    )

    print(
        "  Hard Negative     : 12.50%"
    )

    print(
        "  HN within Negative: 50.00%"
    )

    print(
        f"  CPM               : "
        f"{PREVIOUS_HN_CPM:.6f}"
    )

    print()

    print(
        "Current Candidate"
    )

    print(
        f"  Positive          : "
        f"{POSITIVE_RATIO * 100:.2f}%"
    )

    print(
        f"  Ordinary Negative : "
        f"{ABS_ORDINARY_NEG_RATIO * 100:.2f}%"
    )

    print(
        f"  Hard Negative     : "
        f"{ABS_HN_RATIO * 100:.2f}%"
    )

    print(
        f"  HN within Negative: "
        f"{NEW_HN_RATIO * 100:.2f}%"
    )

    print()

    print(
        "Baseline reference"
    )

    print(
        f"  CPM               : "
        f"{BASELINE_CPM:.6f}"
    )

    print()

    print(
        "Changed:"
    )

    print(
        "  HARD_NEGATIVE_RATIO"
    )

    print(
        "  0.50 -> 0.25"
    )

    print()

    print(
        "Unchanged:"
    )

    print(
        "  Train / Validation split"
    )

    print(
        "  Mined Hard Negative CSV"
    )

    print(
        "  HN mining score threshold"
    )

    print(
        "  Architecture"
    )

    print(
        "  Detection loss"
    )

    print(
        "  Crop size"
    )

    print(
        "  Spacing"
    )

    print(
        "  Batch size"
    )

    print(
        "  Train iterations"
    )

    print(
        "  Optimizer / LR"
    )

    print(
        "  Validation interval"
    )

    print(
        "  Early stopping"
    )

    print(
        "  Validation sampling"
    )

    print()

    print(
        "Training:"
    )

    print(
        "  From scratch"
    )

    print(
        "  No baseline resume"
    )

    print(
        "  No previous HN resume"
    )


# =============================================================================
# Run
# =============================================================================

def run_training(
    generated_script,
):

    header(
        "Training Control"
    )

    print(
        "Generated script:"
    )
    print(
        generated_script
    )

    print()

    print(
        "Output:"
    )
    print(
        OUTPUT_ROOT
    )

    print()

    answer = input(
        "Hard Negative Ratio 0.25 "
        "본학습을 시작할까요? [y/N]: "
    ).strip().lower()

    if answer not in (
        "y",
        "yes",
    ):

        print(
            "Training cancelled."
        )
        return

    header(
        "Training Start"
    )

    # __name__을 __main__으로 실행하면
    # 14번 script의 기존 main()을 그대로 사용한다.
    runpy.run_path(
        str(
            generated_script
        ),
        run_name="__main__",
    )


# =============================================================================
# Main
# =============================================================================

def main():

    header(
        "16 - CPMNetv2 Hard Negative Ratio 0.25 Candidate"
    )

    validate_inputs()

    check_existing_output()

    generated_script = (
        build_candidate_source()
    )

    validate_generated_script(
        generated_script
    )

    print_experiment()

    run_training(
        generated_script
    )


if __name__ == "__main__":
    main()