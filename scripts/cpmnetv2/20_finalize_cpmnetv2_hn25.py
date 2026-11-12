#!/usr/bin/env python3

"""
20_finalize_cpmnetv2_hn25.py

CPMNetv2 + HN25
Final Detection Research Artifact Packaging

목적
----
현재까지 확정된 detection 모델의 연구/검증 산출물을
하나의 final_detection_research 폴더에 보존한다.

이 스크립트는:
- 학습하지 않음
- inference하지 않음
- metric을 재계산하지 않음
- 원본 파일을 삭제하지 않음
- 기존 산출물을 복사하고
- 최종 config / summary / SHA256 / manifest를 생성함

Final model
-----------
Model              : CPMNetv2
HN within negative : 25%
Crop NMS           : 0.05
Final NMS          : 0.05
Validation CPM     : 0.712925
"""

from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime


# =============================================================================
# ROOT
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
)

SCRIPT_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "cpmnetv2"
)


# =============================================================================
# SOURCE PATHS
# =============================================================================

# -----------------------------------------------------------------------------
# Final checkpoint
# -----------------------------------------------------------------------------

CHECKPOINT = (
    RESULT_ROOT
    / "hard_negative_training_ratio025"
    / "best_val_loss.pt"
)


# -----------------------------------------------------------------------------
# Final evaluation
# -----------------------------------------------------------------------------

EVALUATION_DIR = (
    RESULT_ROOT
    / "hard_negative_evaluation_ratio025"
)

PREDICTIONS_CSV = (
    EVALUATION_DIR
    / "predictions.csv"
)

GROUND_TRUTH_CSV = (
    EVALUATION_DIR
    / "ground_truth.csv"
)

METRICS_JSON = (
    EVALUATION_DIR
    / "metrics_summary.json"
)

FROC_CSV = (
    EVALUATION_DIR
    / "froc_curve.csv"
)

CENTER_PR_CSV = (
    EVALUATION_DIR
    / "center_pr_curve.csv"
)

SIZE_RECALL_CSV = (
    EVALUATION_DIR
    / "size_recall.csv"
)


# -----------------------------------------------------------------------------
# Optional evaluation figures
# -----------------------------------------------------------------------------

FROC_PNG = (
    EVALUATION_DIR
    / "froc_curve.png"
)

CENTER_PR_PNG = (
    EVALUATION_DIR
    / "center_pr_curve.png"
)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

VISUALIZATION_DIR = (
    EVALUATION_DIR
    / "visualization"
)

EVENTS_CSV = (
    VISUALIZATION_DIR
    / "tp_fp_fn_events.csv"
)

VISUALIZATION_MANIFEST_CSV = (
    VISUALIZATION_DIR
    / "visualization_manifest.csv"
)


# -----------------------------------------------------------------------------
# Hard-negative mining
# -----------------------------------------------------------------------------

HARD_NEGATIVE_CSV = (
    RESULT_ROOT
    / "hard_negative_mining"
    / "hard_negatives.csv"
)


# -----------------------------------------------------------------------------
# Relevant scripts
# -----------------------------------------------------------------------------

TRAIN_SCRIPT = (
    SCRIPT_ROOT
    / "16_train_cpmnetv2_hard_negative_ratio025.py"
)

EVALUATE_SCRIPT = (
    SCRIPT_ROOT
    / "17_evaluate_cpmnetv2_hard_negative_ratio025.py"
)

NMS_SWEEP_SCRIPT = (
    SCRIPT_ROOT
    / "18_sweep_cpmnetv2_nms.py"
)

VISUALIZE_SCRIPT = (
    SCRIPT_ROOT
    / "19_visualize_cpmnetv2_hn25.py"
)


# =============================================================================
# OUTPUT
# =============================================================================

FINAL_ROOT = (
    RESULT_ROOT
    / "final_detection_research"
)

MODEL_DIR = (
    FINAL_ROOT
    / "model"
)

CONFIG_DIR = (
    FINAL_ROOT
    / "config"
)

EVALUATION_OUT_DIR = (
    FINAL_ROOT
    / "evaluation"
)

HARD_NEGATIVE_OUT_DIR = (
    FINAL_ROOT
    / "hard_negative"
)

VISUALIZATION_OUT_DIR = (
    FINAL_ROOT
    / "visualization"
)

SCRIPT_OUT_DIR = (
    FINAL_ROOT
    / "scripts"
)


# =============================================================================
# FINAL FIXED CONFIG
# =============================================================================

FINAL_CONFIG = {

    "model": {
        "name": "CPMNetv2",
        "backbone": "3D ResNet18",

        "n_channels": 1,

        "n_blocks": [
            2,
            3,
            3,
            3,
        ],

        "n_filters": [
            64,
            96,
            128,
            160,
        ],

        "stem_filters": 32,

        "norm_type": "batchnorm",
        "head_norm": "batchnorm",

        "activation": "ReLU",

        "se": False,

        "first_stride_zyx": [
            1,
            2,
            2,
        ],
    },

    "preprocessing": {
        "target_spacing_zyx_mm": [
            1.0,
            1.0,
            1.0,
        ],

        "hu_clip": [
            -1200.0,
            600.0,
        ],

        "normalization": "HU clip then scale to [-1, 1]",

        "crop_size_zyx": [
            64,
            128,
            128,
        ],

        "overlap_size_zyx": [
            16,
            32,
            32,
        ],
    },

    "training": {
        "optimizer": "AdamW",

        "learning_rate": 1e-4,

        "weight_decay": 1e-4,

        "batch_size": 2,

        "sample_num": 2,

        "effective_crops_per_iteration": 4,

        "iterations_per_epoch": 400,

        "positive_sampling_ratio": 0.75,

        "negative_sampling_ratio": 0.25,

        "hard_negative_ratio_within_negative": 0.25,

        "hard_negative_ratio_total_sampling": 0.0625,

        "ordinary_negative_ratio_total_sampling": 0.1875,

        "random_crop_pos_ratio": 0.90,

        "max_epoch": 100,

        "validation_every_epochs": 5,

        "minimum_epoch": 30,

        "early_stopping_patience_epochs": 20,

        "final_epoch": 35,

        "best_validation_loss": 1.146020,
    },

    "loss": {
        "classification_weight": 4.0,
        "shape_weight": 0.1,
        "offset_weight": 1.0,
        "iou_weight": 1.0,

        "topk": 5,
    },

    "hard_negative_mining": {
        "source_model": "baseline CPMNetv2",

        "train_scans": 800,

        "gt_nodules": 1081,

        "score_threshold": 0.90,

        "gt_safety_radius_multiplier": 1.5,

        "max_hard_negatives_per_scan": 10,

        "raw_predictions": 240000,

        "score_ge_0_90": 1669,

        "rejected_near_gt": 912,

        "safe_fp_candidates": 757,

        "selected_hard_negatives": 742,

        "cases_with_hard_negatives": 390,
    },

    "inference": {
        "infer_batch_size": 4,

        "crop_score_threshold": 0.01,

        "crop_post_topk": 120,

        "crop_post_num_topk": 60,

        "crop_nms_threshold": 0.05,

        "final_nms_threshold": 0.05,

        "final_topk_per_scan": 300,

        "prediction_format": [
            "score",
            "z",
            "y",
            "x",
            "d",
            "h",
            "w",
        ],
    },

    "validation": {
        "scans": 88,

        "gt_nodules": 105,

        "total_predictions": 26400,

        "center_ap": 0.654227,

        "center_map": 0.654227,

        "best_f1_threshold_exact": 0.9612025618553162,

        "best_f1": {
            "tp": 62,
            "fp": 24,
            "fn": 43,

            "precision": 0.720930,
            "recall": 0.590476,
            "f1": 0.649215,
        },

        "fixed_threshold_0_5": {
            "tp": 95,
            "fp": 548,
            "fn": 10,

            "precision": 0.147745,
            "recall": 0.904762,
            "f1": 0.254011,
        },

        "cpm": 0.712925,

        "froc": {
            "0.125_fp_per_scan": 0.419048,
            "0.25_fp_per_scan": 0.542857,
            "0.5_fp_per_scan": 0.647619,
            "1_fp_per_scan": 0.742857,
            "2_fp_per_scan": 0.828571,
            "4_fp_per_scan": 0.904762,
            "8_fp_per_scan": 0.904762,
        },

        "iou_ap": {
            "0.10": 0.654587,
            "0.30": 0.635789,
            "0.50": 0.584407,
        },

        "size_recall_best_f1": {
            "<6mm": 0.533333,
            "6-10mm": 0.529412,
            ">=10mm": 0.769231,
        },

        "size_recall_1_fp_per_scan": {
            "<6mm": 0.644444,
            "6-10mm": 0.735294,
            ">=10mm": 0.923077,
        },

        "size_recall_fixed_0_5": {
            "<6mm": 0.844444,
            "6-10mm": 0.911765,
            ">=10mm": 1.0,
        },

        "localization_best_f1_mm": {
            "count": 62,
            "mean": 0.7444113612,
            "median": 0.5544578124,
            "std": 0.8667770839,
            "min": 0.1236583767,
            "p25": 0.4016083906,
            "p75": 0.8079899410,
            "max": 6.8290733870,
        },
    },

    "comparison": {
        "baseline": {
            "center_ap": 0.602823,
            "best_f1": 0.589212,
            "precision": 0.522059,
            "recall": 0.676190,
            "cpm": 0.685714,
        },

        "hn50": {
            "center_ap": 0.551338,
            "best_f1": 0.599034,
            "precision": 0.607843,
            "recall": 0.590476,
            "cpm": 0.636735,
        },

        "hn25": {
            "center_ap": 0.654227,
            "best_f1": 0.649215,
            "precision": 0.720930,
            "recall": 0.590476,
            "cpm": 0.712925,
        },

        "cpm_absolute_gain_vs_baseline": 0.027211,

        "cpm_relative_gain_vs_baseline_percent": 3.97,
    },

    "nms_sweep": {
        "crop_nms_fixed": 0.05,

        "results": {
            "0.02": {
                "cpm": 0.712925,
                "best_f1": 0.649215,
                "ap_0_10": 0.654739,
                "ap_0_30": 0.635932,
                "ap_0_50": 0.584524,
            },

            "0.05": {
                "cpm": 0.712925,
                "best_f1": 0.649215,
                "ap_0_10": 0.654587,
                "ap_0_30": 0.635789,
                "ap_0_50": 0.584407,
            },

            "0.10": {
                "cpm": 0.712925,
                "best_f1": 0.649215,
                "ap_0_10": 0.654539,
                "ap_0_30": 0.635745,
                "ap_0_50": 0.584369,
            },

            "0.15": {
                "cpm": 0.711565,
                "best_f1": 0.649215,
                "ap_0_10": 0.653529,
                "ap_0_30": 0.641595,
                "ap_0_50": 0.589796,
            },
        },

        "decision": "retain final_nms_threshold=0.05",
    },

    "final_decision": {
        "selected_model": "CPMNetv2 + HN25",

        "status": "LOCKED",

        "crop_nms_threshold": 0.05,

        "final_nms_threshold": 0.05,

        "validation_cpm": 0.712925,

        "reason": (
            "HN25 improved Center AP, Best F1, precision, CPM, "
            "and IoU AP over baseline. Final NMS sweep did not "
            "produce a meaningful CPM improvement, therefore "
            "the existing NMS settings were retained."
        ),
    },
}


# =============================================================================
# UTILITY
# =============================================================================

def header(title):

    print()

    print(
        "=" * 110
    )

    print(
        title
    )

    print(
        "=" * 110
    )


def sha256_file(
    path,
):

    sha256 = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                1024
                * 1024
            )

            if not chunk:

                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def copy_file(
    source,
    destination,
    required=True,
):

    if not source.exists():

        if required:

            raise FileNotFoundError(
                f"Required file 없음:\n"
                f"{source}"
            )

        print(
            "[SKIP] Optional file 없음:"
        )

        print(
            source
        )

        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    print(
        "[COPY]"
    )

    print(
        f"  {source}"
    )

    print(
        "  ->"
    )

    print(
        f"  {destination}"
    )

    return True


# =============================================================================
# VALIDATE
# =============================================================================

def validate_required_artifacts():

    header(
        "Required Artifact Validation"
    )

    required_files = [
        CHECKPOINT,
        PREDICTIONS_CSV,
        GROUND_TRUTH_CSV,
        METRICS_JSON,
        FROC_CSV,
        CENTER_PR_CSV,
        SIZE_RECALL_CSV,
        HARD_NEGATIVE_CSV,
        EVENTS_CSV,
        VISUALIZATION_MANIFEST_CSV,
    ]

    missing = []

    for path in required_files:

        if path.exists():

            print(
                f"[PASS] {path}"
            )

        else:

            print(
                f"[MISSING] {path}"
            )

            missing.append(
                path
            )

    if missing:

        missing_text = "\n".join(
            str(
                path
            )
            for path in missing
        )

        raise FileNotFoundError(
            "\n필수 산출물이 없습니다:\n"
            f"{missing_text}"
        )

    print()

    print(
        "[PASS] All required artifacts exist"
    )


# =============================================================================
# PREPARE OUTPUT
# =============================================================================

def prepare_output_directories():

    directories = [
        FINAL_ROOT,
        MODEL_DIR,
        CONFIG_DIR,
        EVALUATION_OUT_DIR,
        HARD_NEGATIVE_OUT_DIR,
        SCRIPT_OUT_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# PACKAGE MODEL
# =============================================================================

def package_model():

    header(
        "Package Final Model"
    )

    destination = (
        MODEL_DIR
        / "best_val_loss.pt"
    )

    copy_file(
        CHECKPOINT,
        destination,
        required=True,
    )

    checkpoint_sha256 = sha256_file(
        destination
    )

    sha_file = (
        MODEL_DIR
        / "best_val_loss.sha256"
    )

    sha_file.write_text(
        (
            f"{checkpoint_sha256}"
            f"  best_val_loss.pt\n"
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Checkpoint SHA256:"
    )

    print(
        checkpoint_sha256
    )

    return checkpoint_sha256


# =============================================================================
# PACKAGE EVALUATION
# =============================================================================

def package_evaluation():

    header(
        "Package Evaluation Results"
    )

    required_files = [
        PREDICTIONS_CSV,
        GROUND_TRUTH_CSV,
        METRICS_JSON,
        FROC_CSV,
        CENTER_PR_CSV,
        SIZE_RECALL_CSV,
    ]

    optional_files = [
        FROC_PNG,
        CENTER_PR_PNG,
    ]

    for source in required_files:

        copy_file(
            source,
            EVALUATION_OUT_DIR
            / source.name,
            required=True,
        )

    for source in optional_files:

        copy_file(
            source,
            EVALUATION_OUT_DIR
            / source.name,
            required=False,
        )


# =============================================================================
# PACKAGE HARD NEGATIVES
# =============================================================================

def package_hard_negatives():

    header(
        "Package Hard Negative Mining Result"
    )

    copy_file(
        HARD_NEGATIVE_CSV,
        HARD_NEGATIVE_OUT_DIR
        / "hard_negatives.csv",
        required=True,
    )


# =============================================================================
# PACKAGE VISUALIZATION
# =============================================================================

def package_visualization():

    header(
        "Package Validation Visualization"
    )

    if not VISUALIZATION_DIR.exists():

        raise FileNotFoundError(
            f"Visualization directory 없음:\n"
            f"{VISUALIZATION_DIR}"
        )

    if VISUALIZATION_OUT_DIR.exists():

        shutil.rmtree(
            VISUALIZATION_OUT_DIR
        )

    shutil.copytree(
        VISUALIZATION_DIR,
        VISUALIZATION_OUT_DIR,
    )

    print(
        "[COPY TREE]"
    )

    print(
        VISUALIZATION_DIR
    )

    print(
        "->"
    )

    print(
        VISUALIZATION_OUT_DIR
    )


# =============================================================================
# PACKAGE SCRIPTS
# =============================================================================

def package_scripts():

    header(
        "Package Reproducibility Scripts"
    )

    scripts = [
        TRAIN_SCRIPT,
        EVALUATE_SCRIPT,
        NMS_SWEEP_SCRIPT,
        VISUALIZE_SCRIPT,
    ]

    for script in scripts:

        copy_file(
            script,
            SCRIPT_OUT_DIR
            / script.name,
            required=False,
        )


# =============================================================================
# SAVE CONFIG
# =============================================================================

def save_final_config(
    checkpoint_sha256,
):

    header(
        "Save Final Config"
    )

    config = json.loads(
        json.dumps(
            FINAL_CONFIG
        )
    )

    config[
        "artifact"
    ] = {
        "checkpoint_relative_path":
            "model/best_val_loss.pt",

        "checkpoint_sha256":
            checkpoint_sha256,

        "packaged_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "source_project_root":
            str(
                PROJECT_ROOT
            ),
    }

    output_path = (
        CONFIG_DIR
        / "final_detection_config.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[SAVE]"
    )

    print(
        output_path
    )


# =============================================================================
# SAVE SUMMARY
# =============================================================================

def save_final_summary(
    checkpoint_sha256,
):

    summary = f"""
CPMNetv2 FINAL DETECTION RESEARCH PACKAGE
=========================================

STATUS
------
LOCKED


FINAL MODEL
-----------
Model:
CPMNetv2 + Hard Negative 25%

Checkpoint:
model/best_val_loss.pt

Checkpoint SHA256:
{checkpoint_sha256}


MODEL ARCHITECTURE
------------------
Backbone:
3D ResNet18

n_blocks:
[2, 3, 3, 3]

n_filters:
[64, 96, 128, 160]

stem_filters:
32

normalization:
BatchNorm

activation:
ReLU

first_stride:
[1, 2, 2]


PREPROCESSING
-------------
Target spacing:
1 x 1 x 1 mm

HU clip:
[-1200, 600]

Normalization:
HU clip then scale to [-1, 1]

Crop size:
64 x 128 x 128

Overlap:
16 x 32 x 32


TRAINING
--------
Optimizer:
AdamW

Learning rate:
1e-4

Weight decay:
1e-4

Batch size:
2

Sample num:
2

Effective crops / iteration:
4

Train iterations / epoch:
400

Positive sampling:
75%

Ordinary negative:
18.75%

Hard negative:
6.25%

HN ratio within negative:
25%

Final epoch:
35

Best validation loss:
1.146020


HARD NEGATIVE MINING
--------------------
Train scans:
800

GT nodules:
1081

Mining score threshold:
0.90

GT safety radius multiplier:
1.5

Maximum HN / scan:
10

Safe FP candidates:
757

Selected HN:
742

Cases with HN:
390


FINAL INFERENCE SETTINGS
------------------------
Crop score threshold:
0.01

Crop post top-k:
120

Crop post num top-k:
60

Crop NMS:
0.05

Final NMS:
0.05

Final top-k / scan:
300


VALIDATION SET
--------------
Scans:
88

GT nodules:
105

Total retained predictions:
26400


FINAL VALIDATION METRICS
------------------------
Center AP:
0.654227

Best-F1 threshold:
0.9612025618553162

TP:
62

FP:
24

FN:
43

Precision:
0.720930

Recall:
0.590476

F1:
0.649215

CPM:
0.712925


FROC
----
0.125 FP/scan:
0.419048

0.25 FP/scan:
0.542857

0.5 FP/scan:
0.647619

1 FP/scan:
0.742857

2 FP/scan:
0.828571

4 FP/scan:
0.904762

8 FP/scan:
0.904762


IOU AP
------
AP@0.10:
0.654587

AP@0.30:
0.635789

AP@0.50:
0.584407


SIZE RECALL @ 1 FP/SCAN
-----------------------
<6 mm:
0.644444

6-10 mm:
0.735294

>=10 mm:
0.923077


LOCALIZATION @ BEST F1
----------------------
Mean:
0.744411 mm

Median:
0.554458 mm

Std:
0.866777 mm

Min:
0.123658 mm

P25:
0.401608 mm

P75:
0.807990 mm

Max:
6.829073 mm


BASELINE COMPARISON
-------------------
Baseline CPM:
0.685714

HN50 CPM:
0.636735

HN25 CPM:
0.712925

Absolute gain vs baseline:
+0.027211

Relative gain vs baseline:
approximately +3.97%


FINAL NMS SWEEP
---------------
Final NMS 0.02:
CPM 0.712925

Final NMS 0.05:
CPM 0.712925

Final NMS 0.10:
CPM 0.712925

Final NMS 0.15:
CPM 0.711565

Decision:
No meaningful CPM improvement.

Retained:
Crop NMS  = 0.05
Final NMS = 0.05


VALIDATION VISUALIZATION
------------------------
TP / FP / FN matching:
62 / 24 / 43

Representative visualization:
TP 5
FP 5
FN 5

tp_fp_fn_events.csv:
Full Best-F1 TP/FP/FN event table.

visualization_manifest.csv:
Manifest for generated representative PNG files.


IMPORTANT
---------
This package is the locked RESEARCH / VALIDATION artifact package.

It is NOT yet the final deployment inference package.

The following will be created in later steps:

21:
model.py
preprocessing.py
inference.py

22:
postprocessing.py
coordinate_transform.py

23:
operational visualization.py
viewer annotation output
2D / 3D bounding-box visualization

24:
malignancy patch adapter
50 mm physical cube
64 x 64 x 64
0.78125 mm isotropic
HU [-1000, 400] -> [0, 1]

25:
requirements
config
schema
README
VERSION

26:
end-to-end QA

27:
final deploy-ready artifact package
""".strip()

    output_path = (
        FINAL_ROOT
        / "FINAL_SUMMARY.txt"
    )

    output_path.write_text(
        summary
        + "\n",
        encoding="utf-8",
    )

    print(
        "[SAVE]"
    )

    print(
        output_path
    )


# =============================================================================
# MANIFEST
# =============================================================================

def save_artifact_manifest():

    header(
        "Save Artifact Manifest"
    )

    manifest_rows = []

    # 현재 manifest 파일 자체는 아직 없으므로
    # 이 시점의 모든 파일을 기록한다.
    for path in sorted(
        FINAL_ROOT.rglob("*")
    ):

        if not path.is_file():

            continue

        relative_path = (
            path.relative_to(
                FINAL_ROOT
            )
        )

        manifest_rows.append(
            {
                "path":
                    str(
                        relative_path
                    ),

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    output_path = (
        FINAL_ROOT
        / "artifact_manifest.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest_rows,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[SAVE]"
    )

    print(
        output_path
    )

    print()

    print(
        "Artifacts recorded:",
        len(
            manifest_rows
        ),
    )


# =============================================================================
# TREE
# =============================================================================

def print_final_tree():

    header(
        "Final Package Tree"
    )

    print(
        FINAL_ROOT
    )

    print()

    for path in sorted(
        FINAL_ROOT.rglob("*")
    ):

        relative = (
            path.relative_to(
                FINAL_ROOT
            )
        )

        depth = (
            len(
                relative.parts
            )
            - 1
        )

        indent = (
            "    "
            * depth
        )

        if path.is_dir():

            print(
                f"{indent}"
                f"{path.name}/"
            )

        else:

            print(
                f"{indent}"
                f"{path.name}"
            )


# =============================================================================
# FINAL QA
# =============================================================================

def final_qa():

    header(
        "Final Package QA"
    )

    required_outputs = [
        MODEL_DIR
        / "best_val_loss.pt",

        MODEL_DIR
        / "best_val_loss.sha256",

        CONFIG_DIR
        / "final_detection_config.json",

        EVALUATION_OUT_DIR
        / "predictions.csv",

        EVALUATION_OUT_DIR
        / "ground_truth.csv",

        EVALUATION_OUT_DIR
        / "metrics_summary.json",

        EVALUATION_OUT_DIR
        / "froc_curve.csv",

        EVALUATION_OUT_DIR
        / "center_pr_curve.csv",

        EVALUATION_OUT_DIR
        / "size_recall.csv",

        HARD_NEGATIVE_OUT_DIR
        / "hard_negatives.csv",

        VISUALIZATION_OUT_DIR
        / "tp_fp_fn_events.csv",

        VISUALIZATION_OUT_DIR
        / "visualization_manifest.csv",

        FINAL_ROOT
        / "FINAL_SUMMARY.txt",

        FINAL_ROOT
        / "artifact_manifest.json",
    ]

    failures = []

    for path in required_outputs:

        if path.exists():

            print(
                f"[PASS] {path}"
            )

        else:

            print(
                f"[FAIL] {path}"
            )

            failures.append(
                path
            )

    if failures:

        raise RuntimeError(
            "Final package QA failed."
        )

    print()

    print(
        "[PASS] Final research package complete"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "20 - Finalize CPMNetv2 HN25 Research Artifacts"
    )

    print(
        "Selected model:"
    )

    print(
        "CPMNetv2 + HN25"
    )

    print()

    print(
        "Final validation CPM:"
    )

    print(
        "0.712925"
    )

    print()

    print(
        "Output:"
    )

    print(
        FINAL_ROOT
    )

    # -------------------------------------------------------------------------
    # Validate existing artifacts
    # -------------------------------------------------------------------------

    validate_required_artifacts()

    # -------------------------------------------------------------------------
    # Create destination
    # -------------------------------------------------------------------------

    prepare_output_directories()

    # -------------------------------------------------------------------------
    # Package
    # -------------------------------------------------------------------------

    checkpoint_sha256 = (
        package_model()
    )

    package_evaluation()

    package_hard_negatives()

    package_visualization()

    package_scripts()

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    save_final_config(
        checkpoint_sha256
    )

    save_final_summary(
        checkpoint_sha256
    )

    save_artifact_manifest()

    # -------------------------------------------------------------------------
    # QA
    # -------------------------------------------------------------------------

    final_qa()

    print_final_tree()

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------

    header(
        "Finalization Complete"
    )

    print(
        "[PASS] CPMNetv2 + HN25 research artifacts LOCKED"
    )

    print()

    print(
        "Final checkpoint:"
    )

    print(
        MODEL_DIR
        / "best_val_loss.pt"
    )

    print()

    print(
        "Final config:"
    )

    print(
        CONFIG_DIR
        / "final_detection_config.json"
    )

    print()

    print(
        "Final summary:"
    )

    print(
        FINAL_ROOT
        / "FINAL_SUMMARY.txt"
    )

    print()

    print(
        "Artifact manifest:"
    )

    print(
        FINAL_ROOT
        / "artifact_manifest.json"
    )

    print()

    print(
        "Next:"
    )

    print(
        "21 - deployment model / preprocessing / inference"
    )


if __name__ == "__main__":

    main()