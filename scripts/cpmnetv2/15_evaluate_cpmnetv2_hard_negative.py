#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

BASELINE_EVALUATOR = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
    / "11_evaluate_cpmnetv2_full.py"
)

HARD_NEGATIVE_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_training"
    / "best_val_loss.pt"
)

HARD_NEGATIVE_EVAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_evaluation"
)

BASELINE_CPM = 0.685714


def header(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def load_evaluator():
    if not BASELINE_EVALUATOR.exists():
        raise FileNotFoundError(
            f"11번 evaluator를 찾지 못했습니다:\n"
            f"{BASELINE_EVALUATOR}"
        )

    spec = importlib.util.spec_from_file_location(
        "cpmnetv2_full_evaluator",
        BASELINE_EVALUATOR,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "11_evaluate_cpmnetv2_full.py import 실패"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def replace_path(path):
    text = str(path)

    baseline_checkpoint_patterns = [
        str(
            PROJECT_ROOT
            / "results"
            / "cpmnetv2"
            / "full_training"
            / "best_val_loss.pt"
        ),
        "/results/cpmnetv2/full_training/best_val_loss.pt",
    ]

    for pattern in baseline_checkpoint_patterns:
        if pattern in text:
            return HARD_NEGATIVE_CHECKPOINT

    evaluation_patterns = [
        "/results/cpmnetv2/full_evaluation",
        "/results/cpmnetv2/evaluation",
        "/results/cpmnetv2/full_eval",
    ]

    for pattern in evaluation_patterns:
        if pattern in text:
            suffix = text.split(pattern, 1)[1]

            return Path(
                str(HARD_NEGATIVE_EVAL_ROOT)
                + suffix
            )

    return path


def configure_evaluator(module):
    changed = []

    for name, value in list(
        vars(module).items()
    ):
        if not isinstance(value, Path):
            continue

        new_value = replace_path(
            value
        )

        if new_value != value:
            setattr(
                module,
                name,
                new_value,
            )

            changed.append(
                (
                    name,
                    value,
                    new_value,
                )
            )

    checkpoint_vars = [
        "CHECKPOINT",
        "CHECKPOINT_PATH",
        "BEST_CHECKPOINT",
        "MODEL_PATH",
        "WEIGHT_PATH",
    ]

    for name in checkpoint_vars:
        if hasattr(module, name):
            old_value = getattr(
                module,
                name
            )

            if isinstance(
                old_value,
                Path,
            ):
                if (
                    old_value
                    != HARD_NEGATIVE_CHECKPOINT
                ):
                    setattr(
                        module,
                        name,
                        HARD_NEGATIVE_CHECKPOINT,
                    )

                    changed.append(
                        (
                            name,
                            old_value,
                            HARD_NEGATIVE_CHECKPOINT,
                        )
                    )

    output_vars = [
        "OUTPUT_ROOT",
        "OUTPUT_DIR",
        "EVAL_ROOT",
        "EVALUATION_ROOT",
        "RESULT_ROOT",
        "RESULTS_DIR",
    ]

    for name in output_vars:
        if hasattr(module, name):
            old_value = getattr(
                module,
                name
            )

            if isinstance(
                old_value,
                Path,
            ):
                if (
                    "cpmnetv2"
                    in str(old_value)
                    and "evaluation"
                    in str(old_value).lower()
                    and old_value
                    != HARD_NEGATIVE_EVAL_ROOT
                ):
                    setattr(
                        module,
                        name,
                        HARD_NEGATIVE_EVAL_ROOT,
                    )

                    changed.append(
                        (
                            name,
                            old_value,
                            HARD_NEGATIVE_EVAL_ROOT,
                        )
                    )

    # OUTPUT_ROOT를 기준으로 파생 파일들이 이미 생성되어 있을 수 있으므로
    # 모든 Path global을 한 번 더 검사한다.
    for name, value in list(
        vars(module).items()
    ):
        if not isinstance(value, Path):
            continue

        text = str(value)

        if (
            "/results/cpmnetv2/full_evaluation/"
            in text
        ):
            suffix = text.split(
                "/results/cpmnetv2/full_evaluation",
                1,
            )[1]

            new_value = Path(
                str(
                    HARD_NEGATIVE_EVAL_ROOT
                )
                + suffix
            )

            setattr(
                module,
                name,
                new_value,
            )

            changed.append(
                (
                    name,
                    value,
                    new_value,
                )
            )

    return changed


def validate_configuration(module):
    header(
        "Hard Negative Evaluation Configuration"
    )

    print(
        "Baseline evaluator :",
        BASELINE_EVALUATOR,
    )

    print(
        "Checkpoint         :",
        HARD_NEGATIVE_CHECKPOINT,
    )

    print(
        "Output root        :",
        HARD_NEGATIVE_EVAL_ROOT,
    )

    print(
        "Baseline CPM ref   :",
        f"{BASELINE_CPM:.6f}",
    )

    if not HARD_NEGATIVE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\nHard Negative best checkpoint 없음:\n"
            f"{HARD_NEGATIVE_CHECKPOINT}"
        )

    checkpoint_matches = []

    for name, value in vars(
        module
    ).items():
        if not isinstance(
            value,
            Path,
        ):
            continue

        if (
            str(value)
            == str(
                HARD_NEGATIVE_CHECKPOINT
            )
        ):
            checkpoint_matches.append(
                name
            )

    if not checkpoint_matches:
        print()
        print(
            "[WARNING] evaluator global에서 "
            "checkpoint variable을 자동 확인하지 못했습니다."
        )
        print(
            "아래 Path configuration을 확인하십시오."
        )

        for name, value in vars(
            module
        ).items():
            if isinstance(
                value,
                Path,
            ):
                print(
                    f"  {name:<30} = {value}"
                )

        raise RuntimeError(
            "Checkpoint path replacement 확인 실패"
        )

    print()
    print(
        "[PASS] Hard Negative checkpoint:"
    )

    for name in checkpoint_matches:
        print(
            f"  {name}"
        )

    baseline_checkpoint_found = False

    for _, value in vars(
        module
    ).items():
        if not isinstance(
            value,
            Path,
        ):
            continue

        if (
            "full_training"
            in str(value)
            and "best_val_loss.pt"
            in str(value)
        ):
            baseline_checkpoint_found = True

    if baseline_checkpoint_found:
        raise RuntimeError(
            "Baseline checkpoint 경로가 evaluator에 "
            "남아 있습니다."
        )

    print(
        "[PASS] Baseline checkpoint reference 없음"
    )


def print_changed_paths(changed):
    header(
        "Replaced Paths"
    )

    if not changed:
        print(
            "자동 교체된 Path가 없습니다."
        )
        return

    seen = set()

    for name, old, new in changed:
        key = (
            name,
            str(old),
            str(new),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        print(
            name
        )
        print(
            "  OLD:",
            old,
        )
        print(
            "  NEW:",
            new,
        )


def main():
    module = load_evaluator()

    changed = configure_evaluator(
        module
    )

    print_changed_paths(
        changed
    )

    validate_configuration(
        module
    )

    HARD_NEGATIVE_EVAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    header(
        "Experiment Control"
    )

    print(
        "Model      : CPMNetv2 + Hard Negative Sampling"
    )
    print(
        "Checkpoint : best validation-loss checkpoint"
    )
    print(
        "Validation : same 88 scans as baseline"
    )
    print(
        "Evaluator  : EXACT 11_evaluate_cpmnetv2_full.py"
    )
    print()
    print(
        "Only changed:"
    )
    print(
        "  checkpoint"
    )
    print(
        "  output directory"
    )
    print()
    print(
        "Baseline reference CPM:",
        f"{BASELINE_CPM:.6f}",
    )

    print()

    answer = input(
        "Hard Negative full-volume 평가를 "
        "시작할까요? [y/N]: "
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
        "Hard Negative Full-volume Evaluation Start"
    )

    module.main()

    header(
        "Hard Negative Full-volume Evaluation Finished"
    )

    print()
    print(
        "Baseline CPM reference:",
        f"{BASELINE_CPM:.6f}",
    )
    print()
    print(
        "위 evaluator 출력의 Hard Negative CPM과 "
        "직접 비교하면 됩니다."
    )


if __name__ == "__main__":
    main()