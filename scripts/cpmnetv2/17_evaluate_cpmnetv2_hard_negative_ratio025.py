#!/usr/bin/env python3

"""
17_evaluate_cpmnetv2_hard_negative_ratio025.py

CPMNetv2 Hard Negative Ratio 0.25
Full-volume Evaluation

Purpose
-------
기존 15_evaluate_cpmnetv2_hard_negative.py와
완전히 동일한 validation / inference / evaluation 설정으로

    HN 25% model

을 평가한다.

Only changes
------------
Checkpoint:
    results/cpmnetv2/hard_negative_training/best_val_loss.pt

->
    results/cpmnetv2/hard_negative_training_ratio025/best_val_loss.pt


Output:
    results/cpmnetv2/hard_negative_evaluation

->
    results/cpmnetv2/hard_negative_evaluation_ratio025


Evaluation settings:
    변경하지 않음.

Comparison:
    Baseline CPM = 0.685714
    HN 50% CPM   = 0.636735
    HN 25% CPM   = ?
"""

import runpy
import shutil
from pathlib import Path


# =============================================================================
# PATH
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

SOURCE_EVALUATOR = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
    / "15_evaluate_cpmnetv2_hard_negative.py"
)

GENERATED_EVALUATOR = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
    / "_generated_17_evaluate_hn_ratio025.py"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_training_ratio025"
    / "best_val_loss.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_evaluation_ratio025"
)


# =============================================================================
# REFERENCES
# =============================================================================

BASELINE_CPM = 0.685714
HN50_CPM = 0.636735


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
# Validate
# =============================================================================

def validate_inputs():

    header(
        "Input Validation"
    )

    print(
        "Source evaluator:"
    )
    print(
        SOURCE_EVALUATOR
    )

    print()
    print(
        "HN 25% checkpoint:"
    )
    print(
        CHECKPOINT
    )

    print()
    print(
        "Output:"
    )
    print(
        OUTPUT_ROOT
    )

    if not SOURCE_EVALUATOR.exists():

        raise FileNotFoundError(
            "15번 Hard Negative evaluator를 찾지 못했습니다:\n"
            f"{SOURCE_EVALUATOR}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "HN 25% best checkpoint를 찾지 못했습니다:\n"
            f"{CHECKPOINT}"
        )

    print()
    print(
        "[PASS] Source evaluator exists"
    )

    print(
        "[PASS] HN 25% checkpoint exists"
    )


# =============================================================================
# Build evaluator
# =============================================================================

def build_candidate_evaluator():

    header(
        "Build HN Ratio 0.25 Evaluator"
    )

    source = SOURCE_EVALUATOR.read_text(
        encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # Checkpoint path
    # -------------------------------------------------------------------------

    old_training_name = (
        "hard_negative_training"
    )

    new_training_name = (
        "hard_negative_training_ratio025"
    )

    training_count = source.count(
        old_training_name
    )

    print(
        "Training path occurrences:",
        training_count,
    )

    if training_count < 1:

        fail(
            "15번 evaluator에서 "
            "'hard_negative_training'을 찾지 못했습니다.\n"
            "Evaluator 구조가 예상과 다르므로 자동 변경하지 않습니다."
        )

    # -------------------------------------------------------------------------
    # Evaluation output path
    # -------------------------------------------------------------------------

    old_eval_name = (
        "hard_negative_evaluation"
    )

    new_eval_name = (
        "hard_negative_evaluation_ratio025"
    )

    eval_count = source.count(
        old_eval_name
    )

    print(
        "Evaluation path occurrences:",
        eval_count,
    )

    if eval_count < 1:

        fail(
            "15번 evaluator에서 "
            "'hard_negative_evaluation'을 찾지 못했습니다.\n"
            "기존 결과를 덮어쓸 위험이 있으므로 중단합니다."
        )

    # -------------------------------------------------------------------------
    # Replace
    # -------------------------------------------------------------------------

    candidate = source.replace(
        old_training_name,
        new_training_name,
    )

    candidate = candidate.replace(
        old_eval_name,
        new_eval_name,
    )

    # 출력 제목도 구분
    candidate = candidate.replace(
        "Hard Negative Full-volume Evaluation",
        "Hard Negative Ratio 0.25 Full-volume Evaluation",
    )

    candidate = candidate.replace(
        "Hard Negative Full-volume Evaluation Finished",
        "Hard Negative Ratio 0.25 Full-volume Evaluation Finished",
    )

    # -------------------------------------------------------------------------
    # QA
    # -------------------------------------------------------------------------

    if (
        "hard_negative_training_ratio025"
        not in candidate
    ):
        fail(
            "Checkpoint path replacement 실패"
        )

    if (
        "hard_negative_evaluation_ratio025"
        not in candidate
    ):
        fail(
            "Evaluation output path replacement 실패"
        )

    GENERATED_EVALUATOR.write_text(
        candidate,
        encoding="utf-8",
    )

    print()
    print(
        "[PASS] Checkpoint path changed"
    )

    print(
        "       hard_negative_training"
    )
    print(
        "       -> hard_negative_training_ratio025"
    )

    print()
    print(
        "[PASS] Evaluation output changed"
    )

    print(
        "       hard_negative_evaluation"
    )
    print(
        "       -> hard_negative_evaluation_ratio025"
    )

    print()
    print(
        "Generated evaluator:"
    )
    print(
        GENERATED_EVALUATOR
    )

    return GENERATED_EVALUATOR


# =============================================================================
# Generated QA
# =============================================================================

def validate_generated_evaluator(
    generated,
):

    header(
        "Generated Evaluator QA"
    )

    source = generated.read_text(
        encoding="utf-8"
    )

    required = [
        "hard_negative_training_ratio025",
        "hard_negative_evaluation_ratio025",
        "best_val_loss.pt",
    ]

    for text in required:

        if text not in source:

            fail(
                "Generated evaluator에 필수 문자열이 없습니다: "
                f"{text}"
            )

        print(
            "[PASS]",
            text,
        )

    # 실제 checkpoint가 존재해야 한다.
    if not CHECKPOINT.exists():

        fail(
            "Checkpoint disappeared:\n"
            f"{CHECKPOINT}"
        )


# =============================================================================
# Output safety
# =============================================================================

def check_output():

    header(
        "Output Safety"
    )

    if not OUTPUT_ROOT.exists():

        print(
            "[PASS] New evaluation output directory"
        )

        print(
            OUTPUT_ROOT
        )

        return

    files = [
        p
        for p in OUTPUT_ROOT.rglob("*")
        if p.is_file()
    ]

    if not files:

        print(
            "[PASS] Output directory exists but is empty"
        )
        return

    print(
        "[WARNING] Existing HN25 evaluation results:"
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
        "기존 HN25 evaluation 결과를 삭제하고 다시 평가할까요? [y/N]: "
    ).strip().lower()

    if answer not in (
        "y",
        "yes",
    ):

        print(
            "Evaluation cancelled."
        )

        raise SystemExit(
            0
        )

    shutil.rmtree(
        OUTPUT_ROOT
    )

    print(
        "[PASS] Existing evaluation output removed"
    )


# =============================================================================
# Summary
# =============================================================================

def print_summary():

    header(
        "Experiment Comparison"
    )

    print(
        "Baseline"
    )
    print(
        f"  CPM       : {BASELINE_CPM:.6f}"
    )

    print()

    print(
        "Hard Negative 50%"
    )
    print(
        f"  CPM       : {HN50_CPM:.6f}"
    )

    print()

    print(
        "Hard Negative 25%"
    )
    print(
        "  CPM       : ?"
    )

    print()
    print(
        "Checkpoint:"
    )
    print(
        CHECKPOINT
    )

    print()
    print(
        "Evaluation conditions:"
    )

    print(
        "  SAME as 15_evaluate_cpmnetv2_hard_negative.py"
    )

    print(
        "  Same validation split"
    )

    print(
        "  Same full-volume inference"
    )

    print(
        "  Same candidate threshold"
    )

    print(
        "  Same NMS"
    )

    print(
        "  Same top-k"
    )

    print(
        "  Same FROC targets"
    )

    print(
        "  Same center matching"
    )

    print(
        "  Same IoU AP"
    )

    print(
        "  Same size-wise recall"
    )


# =============================================================================
# Run
# =============================================================================

def run_evaluation(
    generated,
):

    header(
        "Evaluation Control"
    )

    print(
        "Generated evaluator:"
    )
    print(
        generated
    )

    print()

    answer = input(
        "HN Ratio 0.25 full-volume evaluation을 시작할까요? [y/N]: "
    ).strip().lower()

    if answer not in (
        "y",
        "yes",
    ):

        print(
            "Evaluation cancelled."
        )
        return

    header(
        "HN Ratio 0.25 Evaluation Start"
    )

    runpy.run_path(
        str(
            generated
        ),
        run_name="__main__",
    )

    header(
        "HN Ratio 0.25 Evaluation Finished"
    )

    print(
        f"Baseline CPM reference : {BASELINE_CPM:.6f}"
    )

    print(
        f"HN 50% CPM reference   : {HN50_CPM:.6f}"
    )

    print()

    print(
        "위 evaluator 출력의 HN 25% CPM과 "
        "직접 비교하면 됩니다."
    )


# =============================================================================
# Main
# =============================================================================

def main():

    header(
        "17 - CPMNetv2 Hard Negative Ratio 0.25 Evaluation"
    )

    validate_inputs()

    check_output()

    generated = (
        build_candidate_evaluator()
    )

    validate_generated_evaluator(
        generated
    )

    print_summary()

    run_evaluation(
        generated
    )


if __name__ == "__main__":
    main()