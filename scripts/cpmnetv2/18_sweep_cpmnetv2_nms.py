#!/usr/bin/env python3

"""
18_sweep_cpmnetv2_nms.py

CPMNetv2 HN25 - FINAL NMS Sweep

Current best:
    CPMNetv2 + Hard Negative 25%
    CPM = 0.712925

Fixed:
    CROP_NMS_THRESHOLD = 0.05

Sweep:
    FINAL_NMS_THRESHOLD
        0.02
        0.05
        0.10
        0.15

Important
---------
- 11_evaluate_cpmnetv2_full.py를 기준으로 evaluator를 생성한다.
- checkpoint만 HN25 best checkpoint로 변경한다.
- output directory를 NMS 값별로 분리한다.
- CROP NMS는 절대 변경하지 않는다.
- FINAL NMS만 단일 변수로 변경한다.
- 각 설정마다 full-volume inference를 다시 수행한다.
"""

import json
import runpy
import shutil
import time
from pathlib import Path

import pandas as pd


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
    / "11_evaluate_cpmnetv2_full.py"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_training_ratio025"
    / "best_val_loss.pt"
)

GENERATED_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
    / "_generated_nms_sweep"
)

SWEEP_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "hard_negative_final_nms_sweep"
)


# =============================================================================
# CONFIG
# =============================================================================

BASELINE_CPM = 0.685714
HN50_CPM = 0.636735
HN25_CURRENT_CPM = 0.712925

FIXED_CROP_NMS = 0.05
CURRENT_FINAL_NMS = 0.05

FINAL_NMS_VALUES = [
    0.02,
    0.05,
    0.10,
    0.15,
]


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


def nms_tag(value):

    return (
        f"{value:.2f}"
        .replace(
            ".",
            "p",
        )
    )


# =============================================================================
# Validation
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
        "HN25 checkpoint:"
    )
    print(
        CHECKPOINT
    )

    if not SOURCE_EVALUATOR.exists():

        raise FileNotFoundError(
            f"11 evaluator 없음:\n"
            f"{SOURCE_EVALUATOR}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"HN25 checkpoint 없음:\n"
            f"{CHECKPOINT}"
        )

    print()

    print(
        "[PASS] Source evaluator exists"
    )

    print(
        "[PASS] HN25 checkpoint exists"
    )


# =============================================================================
# Verify source
# =============================================================================

def verify_source(
    source
):

    header(
        "Source Configuration QA"
    )

    required = [
        "CROP_NMS_THRESHOLD = 0.05",
        "FINAL_NMS_THRESHOLD = 0.05",
        '/ "full_training"',
        '/ "full_evaluation"',
    ]

    for text in required:

        count = (
            source.count(
                text
            )
        )

        print(
            f"{text:<40} : {count}"
        )

        if count != 1:

            fail(
                f"\nExpected exactly one occurrence:\n"
                f"{text}\n"
                f"Found: {count}\n\n"
                "11 evaluator가 예상 구조와 다릅니다."
            )

    print()

    print(
        "[PASS] 11 evaluator configuration verified"
    )


# =============================================================================
# Build variant
# =============================================================================

def build_variant(
    source,
    final_nms,
):

    tag = (
        nms_tag(
            final_nms
        )
    )

    output_name = (
        f"hard_negative_evaluation_"
        f"ratio025_finalnms_{tag}"
    )

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "cpmnetv2"
        / output_name
    )

    generated_script = (
        GENERATED_ROOT
        / (
            f"evaluate_hn25_"
            f"finalnms_{tag}.py"
        )
    )

    candidate = source

    # -------------------------------------------------------------------------
    # Training checkpoint path
    # -------------------------------------------------------------------------

    candidate = (
        candidate.replace(
            '/ "full_training"',
            '/ "hard_negative_training_ratio025"',
            1,
        )
    )

    # -------------------------------------------------------------------------
    # Evaluation output path
    # -------------------------------------------------------------------------

    candidate = (
        candidate.replace(
            '/ "full_evaluation"',
            f'/ "{output_name}"',
            1,
        )
    )

    # -------------------------------------------------------------------------
    # FINAL NMS ONLY
    # -------------------------------------------------------------------------

    candidate = (
        candidate.replace(
            "FINAL_NMS_THRESHOLD = 0.05",
            (
                "FINAL_NMS_THRESHOLD = "
                f"{final_nms:.2f}"
            ),
            1,
        )
    )

    # -------------------------------------------------------------------------
    # CRITICAL QA
    # CROP NMS must remain 0.05
    # -------------------------------------------------------------------------

    if (
        "CROP_NMS_THRESHOLD = 0.05"
        not in candidate
    ):

        fail(
            "CROP_NMS_THRESHOLD가 변경되었습니다."
        )

    expected_final = (
        "FINAL_NMS_THRESHOLD = "
        f"{final_nms:.2f}"
    )

    if (
        expected_final
        not in candidate
    ):

        fail(
            "FINAL_NMS_THRESHOLD 변경 실패"
        )

    if (
        "hard_negative_training_ratio025"
        not in candidate
    ):

        fail(
            "HN25 checkpoint path 변경 실패"
        )

    if (
        output_name
        not in candidate
    ):

        fail(
            "Output directory 변경 실패"
        )

    GENERATED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_script.write_text(
        candidate,
        encoding="utf-8",
    )

    return {
        "final_nms":
            final_nms,

        "tag":
            tag,

        "output_name":
            output_name,

        "output_dir":
            output_dir,

        "script":
            generated_script,
    }


# =============================================================================
# Existing output
# =============================================================================

def prepare_output(
    output_dir
):

    if not output_dir.exists():
        return True

    files = [
        p
        for p in output_dir.rglob("*")
        if p.is_file()
    ]

    if not files:
        return True

    # 이미 metrics까지 완성됐으면 재사용 가능
    metrics_path = (
        output_dir
        / "metrics_summary.json"
    )

    if metrics_path.exists():

        print(
            "[INFO] Existing completed evaluation found."
        )

        print(
            metrics_path
        )

        answer = input(
            "기존 결과를 재사용할까요? [Y/n]: "
        ).strip().lower()

        if answer in (
            "",
            "y",
            "yes",
        ):

            return False

    print()

    print(
        "[WARNING] Existing files:"
    )

    print(
        output_dir
    )

    answer = input(
        "삭제 후 다시 평가할까요? [y/N]: "
    ).strip().lower()

    if answer not in (
        "y",
        "yes",
    ):

        raise SystemExit(
            "Sweep cancelled."
        )

    shutil.rmtree(
        output_dir
    )

    return True


# =============================================================================
# Metric parsing
# =============================================================================

def load_metrics(
    output_dir
):

    metrics_path = (
        output_dir
        / "metrics_summary.json"
    )

    if not metrics_path.exists():

        raise FileNotFoundError(
            "metrics_summary.json 없음:\n"
            f"{metrics_path}"
        )

    with open(
        metrics_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    return (
        metrics_path,
        data,
    )


def search_numeric(
    obj,
    target_keys,
):

    """
    metrics_summary.json의 구조가 약간 달라도
    nested dict 안에서 원하는 numeric key를 찾는다.
    """

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            if (
                key in target_keys
                and isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
            ):

                return float(
                    value
                )

        for value in obj.values():

            found = search_numeric(
                value,
                target_keys,
            )

            if found is not None:

                return found

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            found = search_numeric(
                value,
                target_keys,
            )

            if found is not None:

                return found

    return None


def parse_metrics(
    metrics
):

    return {

        "cpm":
            search_numeric(
                metrics,
                {
                    "cpm",
                    "CPM",
                },
            ),

        "center_ap":
            search_numeric(
                metrics,
                {
                    "center_ap",
                    "center_AP",
                    "Center AP",
                },
            ),

        "best_f1":
            search_numeric(
                metrics,
                {
                    "best_f1",
                    "best_F1",
                    "f1",
                },
            ),

        "ap_0.10":
            search_numeric(
                metrics,
                {
                    "ap_0.10",
                    "ap10",
                    "AP@0.10",
                },
            ),

        "ap_0.30":
            search_numeric(
                metrics,
                {
                    "ap_0.30",
                    "ap30",
                    "AP@0.30",
                },
            ),

        "ap_0.50":
            search_numeric(
                metrics,
                {
                    "ap_0.50",
                    "ap50",
                    "AP@0.50",
                },
            ),
    }


# =============================================================================
# Main
# =============================================================================

def main():

    header(
        "18 - CPMNetv2 HN25 FINAL NMS Sweep"
    )

    validate_inputs()

    source = (
        SOURCE_EVALUATOR
        .read_text(
            encoding="utf-8"
        )
    )

    verify_source(
        source
    )

    header(
        "Experiment"
    )

    print(
        "Model:"
    )

    print(
        "  CPMNetv2 + HN 25%"
    )

    print()

    print(
        "Current CPM:"
    )

    print(
        f"  {HN25_CURRENT_CPM:.6f}"
    )

    print()

    print(
        "Fixed:"
    )

    print(
        f"  CROP_NMS_THRESHOLD  = "
        f"{FIXED_CROP_NMS:.2f}"
    )

    print()

    print(
        "Sweep:"
    )

    print(
        "  FINAL_NMS_THRESHOLD =",
        FINAL_NMS_VALUES,
    )

    print()

    print(
        "Other inference settings:"
    )

    print(
        "  Candidate threshold = 0.01"
    )

    print(
        "  Crop post top-k     = 120"
    )

    print(
        "  Crop post num top-k = 60"
    )

    print(
        "  Final top-k/scan    = 300"
    )

    print()

    answer = input(
        "FINAL NMS sweep을 시작할까요? [y/N]: "
    ).strip().lower()

    if answer not in (
        "y",
        "yes",
    ):

        print(
            "Sweep cancelled."
        )

        return

    SWEEP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_start = (
        time.time()
    )

    for index, final_nms in enumerate(
        FINAL_NMS_VALUES,
        start=1,
    ):

        header(
            f"[{index}/{len(FINAL_NMS_VALUES)}] "
            f"FINAL NMS = {final_nms:.2f}"
        )

        variant = (
            build_variant(
                source,
                final_nms,
            )
        )

        print(
            "Generated evaluator:"
        )

        print(
            variant[
                "script"
            ]
        )

        print()

        print(
            "Output:"
        )

        print(
            variant[
                "output_dir"
            ]
        )

        print()

        print(
            f"CROP NMS  = "
            f"{FIXED_CROP_NMS:.2f}"
        )

        print(
            f"FINAL NMS = "
            f"{final_nms:.2f}"
        )

        should_run = (
            prepare_output(
                variant[
                    "output_dir"
                ]
            )
        )

        start = (
            time.time()
        )

        if should_run:

            runpy.run_path(
                str(
                    variant[
                        "script"
                    ]
                ),
                run_name="__main__",
            )

        elapsed = (
            time.time()
            - start
        )

        metrics_path, metrics = (
            load_metrics(
                variant[
                    "output_dir"
                ]
            )
        )

        parsed = (
            parse_metrics(
                metrics
            )
        )

        row = {

            "crop_nms":
                FIXED_CROP_NMS,

            "final_nms":
                final_nms,

            "cpm":
                parsed[
                    "cpm"
                ],

            "center_ap":
                parsed[
                    "center_ap"
                ],

            "best_f1":
                parsed[
                    "best_f1"
                ],

            "ap_0.10":
                parsed[
                    "ap_0.10"
                ],

            "ap_0.30":
                parsed[
                    "ap_0.30"
                ],

            "ap_0.50":
                parsed[
                    "ap_0.50"
                ],

            "elapsed_minutes":
                elapsed
                / 60.0,

            "metrics_json":
                str(
                    metrics_path
                ),
        }

        rows.append(
            row
        )

        print()

        print(
            "Result:"
        )

        for key, value in row.items():

            print(
                f"  {key:<18}: "
                f"{value}"
            )


    # =========================================================================
    # Save
    # =========================================================================

    df = pd.DataFrame(
        rows
    )

    csv_path = (
        SWEEP_ROOT
        / "final_nms_sweep_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )


    # =========================================================================
    # Summary
    # =========================================================================

    header(
        "FINAL NMS Sweep Summary"
    )

    print(
        df.to_string(
            index=False
        )
    )

    valid = (
        df[
            df[
                "cpm"
            ].notna()
        ]
        .copy()
    )

    if len(
        valid
    ) == 0:

        print()

        print(
            "[WARNING] CPM을 JSON에서 "
            "자동 추출하지 못했습니다."
        )

        print(
            "각 evaluator 출력 CPM을 "
            "수동 비교하세요."
        )

    else:

        best_index = (
            valid[
                "cpm"
            ]
            .astype(float)
            .idxmax()
        )

        best = (
            valid.loc[
                best_index
            ]
        )

        best_nms = float(
            best[
                "final_nms"
            ]
        )

        best_cpm = float(
            best[
                "cpm"
            ]
        )

        header(
            "Best FINAL NMS"
        )

        print(
            f"FINAL NMS         : "
            f"{best_nms:.2f}"
        )

        print(
            f"Best CPM          : "
            f"{best_cpm:.6f}"
        )

        print(
            f"Current CPM       : "
            f"{HN25_CURRENT_CPM:.6f}"
        )

        print(
            f"Delta vs current  : "
            f"{best_cpm - HN25_CURRENT_CPM:+.6f}"
        )

        print(
            f"Delta vs baseline : "
            f"{best_cpm - BASELINE_CPM:+.6f}"
        )

        print()

        if (
            best_cpm
            > HN25_CURRENT_CPM
        ):

            print(
                "Decision:"
            )

            print(
                "  FINAL NMS TUNING IMPROVED CPM"
            )

        else:

            print(
                "Decision:"
            )

            print(
                "  KEEP FINAL_NMS_THRESHOLD = 0.05"
            )


    print()

    print(
        "Saved:"
    )

    print(
        csv_path
    )

    print()

    print(
        "Total elapsed:"
    )

    print(
        f"{(time.time() - total_start) / 3600.0:.2f} h"
    )


if __name__ == "__main__":

    main()