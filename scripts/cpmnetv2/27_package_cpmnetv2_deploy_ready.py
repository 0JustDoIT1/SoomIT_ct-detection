#!/usr/bin/env python3

"""
27_package_cpmnetv2_deploy_ready.py

Final deploy-ready packaging for CPMNetv2 + HN25.

Purpose
-------
Collect and lock:

- final checkpoint
- operational inference code
- preprocessing/postprocessing
- coordinate transform
- visualization
- malignancy patch adapter
- configs
- docs
- E2E QA report
- representative sample outputs
- SHA256 hashes
- artifact manifest
- final summary

No training.
No inference.
No modification of source artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/mnt/c/AI/ct-project"
)

RESEARCH_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "final_detection_research"
)

DEPLOYMENT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "final_detection_deployment"
)

E2E_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "e2e_qa_reference"
)

FINAL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "cpmnetv2"
    / "final_detection_deploy_ready"
)


# =============================================================================
# FIXED MODEL INFORMATION
# =============================================================================

MODEL_NAME = "CPMNetv2 + HN25"
MODEL_VERSION = "cpmnetv2-hn25-v1.0.0"

FINAL_CPM = 0.712925

CROP_NMS_THRESHOLD = 0.05
FINAL_NMS_THRESHOLD = 0.05
FINAL_TOPK = 300

BEST_F1_THRESHOLD = (
    0.9612025618553162
)

REFERENCE_CASE_ID = (
    "1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "102681962408431413578140925249"
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def copy_file(
    source: Path,
    destination: Path,
) -> Path:

    if not source.exists():

        raise FileNotFoundError(
            f"Required artifact not found:\n{source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


def copy_tree(
    source: Path,
    destination: Path,
) -> None:

    if not source.exists():

        raise FileNotFoundError(
            f"Required directory not found:\n{source}"
        )

    if destination.exists():

        shutil.rmtree(
            destination
        )

    shutil.copytree(
        source,
        destination,
    )


def collect_files(
    root: Path,
) -> List[Path]:

    return sorted(
        path
        for path
        in root.rglob("*")
        if path.is_file()
    )


# =============================================================================
# MAIN PACKAGING
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "27 - CPMNetv2 HN25 Final Deploy-Ready Packaging"
    )

    print(
        "=" * 100
    )

    # -------------------------------------------------------------------------
    # Safety / clean output
    # -------------------------------------------------------------------------

    if FINAL_ROOT.exists():

        shutil.rmtree(
            FINAL_ROOT
        )

    FINAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Target directories
    # -------------------------------------------------------------------------

    model_dir = (
        FINAL_ROOT
        / "model"
    )

    code_dir = (
        FINAL_ROOT
        / "code"
    )

    config_dir = (
        FINAL_ROOT
        / "config"
    )

    docs_dir = (
        FINAL_ROOT
        / "docs"
    )

    qa_dir = (
        FINAL_ROOT
        / "qa"
    )

    sample_dir = (
        FINAL_ROOT
        / "sample_output"
    )

    for directory in [
        model_dir,
        code_dir,
        config_dir,
        docs_dir,
        qa_dir,
        sample_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================================
    # MODEL
    # =========================================================================

    print()
    print(
        "[1/7] Copy model"
    )

    checkpoint_source = (
        RESEARCH_ROOT
        / "model"
        / "best_val_loss.pt"
    )

    checkpoint_dest = (
        model_dir
        / "best_val_loss.pt"
    )

    copy_file(
        checkpoint_source,
        checkpoint_dest,
    )

    print(
        checkpoint_dest
    )

    # =========================================================================
    # CODE
    # =========================================================================

    print()
    print(
        "[2/7] Copy operational code"
    )

    code_files = [
        "model.py",
        "preprocessing.py",
        "inference.py",
        "coordinate_transform.py",
        "postprocessing.py",
        "visualization.py",
        "patch_extraction.py",
        "e2e_qa.py",
    ]

    for filename in code_files:

        source = (
            DEPLOYMENT_ROOT
            / filename
        )

        destination = (
            code_dir
            / filename
        )

        copy_file(
            source,
            destination,
        )

        print(
            destination
        )

    # =========================================================================
    # CONFIG
    # =========================================================================

    print()
    print(
        "[3/7] Copy config"
    )

    config_files = [
        "inference_config.json",
        "malignancy_patch_config.json",
        "output_schema.json",
        "requirements.txt",
        "VERSION",
    ]

    for filename in config_files:

        source = (
            DEPLOYMENT_ROOT
            / filename
        )

        destination = (
            config_dir
            / filename
        )

        copy_file(
            source,
            destination,
        )

        print(
            destination
        )

    # =========================================================================
    # DOCS
    # =========================================================================

    print()
    print(
        "[4/7] Copy docs"
    )

    doc_files = [
        "README.md",
        "COORDINATE_SYSTEM.md",
    ]

    for filename in doc_files:

        source = (
            DEPLOYMENT_ROOT
            / filename
        )

        destination = (
            docs_dir
            / filename
        )

        copy_file(
            source,
            destination,
        )

        print(
            destination
        )

    # =========================================================================
    # QA
    # =========================================================================

    print()
    print(
        "[5/7] Copy E2E QA"
    )

    qa_report_source = (
        E2E_ROOT
        / "qa_report.json"
    )

    qa_report_dest = (
        qa_dir
        / "qa_report.json"
    )

    copy_file(
        qa_report_source,
        qa_report_dest,
    )

    print(
        qa_report_dest
    )

    # Copy representative visualization QA
    visualization_source = (
        E2E_ROOT
        / "visualization"
    )

    visualization_dest = (
        qa_dir
        / "visualization"
    )

    copy_tree(
        visualization_source,
        visualization_dest,
    )

    # =========================================================================
    # SAMPLE OUTPUT
    # =========================================================================

    print()
    print(
        "[6/7] Copy representative sample outputs"
    )

    sample_sources = {
        (
            E2E_ROOT
            / "inference"
            / "preprocessing_metadata.json"
        ):
            (
                sample_dir
                / "preprocessing_metadata.json"
            ),

        (
            E2E_ROOT
            / "inference"
            / "raw_candidates.json"
        ):
            (
                sample_dir
                / "raw_candidates.json"
            ),

        (
            E2E_ROOT
            / "inference"
            / "raw_candidates.npz"
        ):
            (
                sample_dir
                / "raw_candidates.npz"
            ),

        (
            E2E_ROOT
            / "inference"
            / "final_detections.json"
        ):
            (
                sample_dir
                / "final_detections.json"
            ),

        (
            E2E_ROOT
            / "inference"
            / "final_candidates.npz"
        ):
            (
                sample_dir
                / "final_candidates.npz"
            ),

        (
            E2E_ROOT
            / "visualization"
            / "viewer_annotations.json"
        ):
            (
                sample_dir
                / "viewer_annotations.json"
            ),

        (
            E2E_ROOT
            / "visualization"
            / "visualization_manifest.json"
        ):
            (
                sample_dir
                / "visualization_manifest.json"
            ),

        (
            E2E_ROOT
            / "malignancy"
            / "patch_metadata.json"
        ):
            (
                sample_dir
                / "patch_metadata.json"
            ),

        (
            E2E_ROOT
            / "malignancy"
            / "patches"
            / "NODULE_001.npy"
        ):
            (
                sample_dir
                / "NODULE_001.npy"
            ),
    }

    for source, destination in (
        sample_sources.items()
    ):

        copy_file(
            source,
            destination,
        )

        print(
            destination
        )

    # =========================================================================
    # READ QA REPORT
    # =========================================================================

    with open(
        qa_report_dest,
        "r",
        encoding="utf-8",
    ) as file:

        qa_report = json.load(
            file
        )

    if qa_report.get(
        "status"
    ) != "PASS":

        raise RuntimeError(
            "E2E QA report is not PASS."
        )

    qa_summary = (
        qa_report.get(
            "qa_summary",
            {}
        )
    )

    failed_checks = int(
        qa_summary.get(
            "failed",
            -1
        )
    )

    if failed_checks != 0:

        raise RuntimeError(
            "E2E QA contains failed checks."
        )

    # =========================================================================
    # HASHES
    # =========================================================================

    print()
    print(
        "[7/7] Generate SHA256 + manifest"
    )

    files_before_manifest = (
        collect_files(
            FINAL_ROOT
        )
    )

    checksum_lines = []

    manifest_files = []

    for path in files_before_manifest:

        relative = (
            path.relative_to(
                FINAL_ROOT
            )
        )

        digest = (
            sha256_file(
                path
            )
        )

        size_bytes = (
            path.stat().st_size
        )

        checksum_lines.append(
            f"{digest}  {relative.as_posix()}"
        )

        manifest_files.append(
            {
                "path":
                    relative.as_posix(),

                "size_bytes":
                    int(
                        size_bytes
                    ),

                "sha256":
                    digest,
            }
        )

    sha_path = (
        FINAL_ROOT
        / "SHA256SUMS.txt"
    )

    with open(
        sha_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                checksum_lines
            )
        )

        file.write(
            "\n"
        )

    # =========================================================================
    # MANIFEST
    # =========================================================================

    checkpoint_sha = (
        sha256_file(
            checkpoint_dest
        )
    )

    manifest = {
        "artifact":
            "CPMNetv2_HN25_deploy_ready",

        "model_name":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "LOCKED",

        "checkpoint": {
            "path":
                "model/best_val_loss.pt",

            "sha256":
                checkpoint_sha,
        },

        "final_metrics": {
            "validation_cpm":
                FINAL_CPM,

            "validation_best_f1_threshold":
                BEST_F1_THRESHOLD,
        },

        "final_postprocessing": {
            "crop_nms_threshold":
                CROP_NMS_THRESHOLD,

            "final_nms_threshold":
                FINAL_NMS_THRESHOLD,

            "final_topk_per_scan":
                FINAL_TOPK,
        },

        "e2e_qa": {
            "reference_case_id":
                REFERENCE_CASE_ID,

            "status":
                qa_report.get(
                    "status"
                ),

            "total_checks":
                qa_summary.get(
                    "total_checks"
                ),

            "passed":
                qa_summary.get(
                    "passed"
                ),

            "failed":
                qa_summary.get(
                    "failed"
                ),

            "reference_comparison":
                qa_report.get(
                    "reference_comparison"
                ),
        },

        "artifact_count":
            len(
                manifest_files
            ),

        "files":
            manifest_files,
    }

    manifest_path = (
        FINAL_ROOT
        / "artifact_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    summary_path = (
        FINAL_ROOT
        / "FINAL_SUMMARY.txt"
    )

    reference_comparison = (
        qa_report.get(
            "reference_comparison",
            {}
        )
    )

    top1_diff = (
        reference_comparison.get(
            "top1_abs_diff",
            {}
        )
    )

    summary = f"""
CPMNetv2 + HN25 FINAL DEPLOY-READY ARTIFACT
============================================================

MODEL
-----
Name:
{MODEL_NAME}

Version:
{MODEL_VERSION}

Checkpoint:
model/best_val_loss.pt

Checkpoint SHA256:
{checkpoint_sha}


FINAL VALIDATION RESULT
-----------------------
CPM:
{FINAL_CPM:.6f}

Validation Best-F1 Threshold:
{BEST_F1_THRESHOLD:.16f}


FINAL POSTPROCESSING
--------------------
Crop NMS:
{CROP_NMS_THRESHOLD}

Final NMS:
{FINAL_NMS_THRESHOLD}

Final Top-K / scan:
{FINAL_TOPK}


DETECTION PREPROCESSING
-----------------------
Target spacing:
1.0 x 1.0 x 1.0 mm

HU:
[-1200, 600]

Normalization:
[-1, 1]


MALIGNANCY PATCH ADAPTER
------------------------
Detection input:
Nodule center in patient physical/world XYZ mm

Physical FOV:
50 x 50 x 50 mm

Output:
64 x 64 x 64

Target spacing:
0.78125 mm isotropic

HU:
[-1000, 400]

Normalization:
[0, 1]

Tensor:
[1, 64, 64, 64] float32


COORDINATE SYSTEM
-----------------
Internal volume:
ZYX

Detection:
center ZYX / size DHW

SimpleITK:
XYZ

Patient physical/world:
XYZ mm


END-TO-END QA
-------------
Reference case:
{REFERENCE_CASE_ID}

Status:
{qa_report.get("status")}

Checks:
{qa_summary.get("total_checks")}

Passed:
{qa_summary.get("passed")}

Failed:
{qa_summary.get("failed")}

Reference candidates:
{reference_comparison.get("reference_count")}

Deployment candidates:
{reference_comparison.get("deployment_count")}

Top-1 absolute differences:
score = {top1_diff.get("score")}
z     = {top1_diff.get("z")}
y     = {top1_diff.get("y")}
x     = {top1_diff.get("x")}
d     = {top1_diff.get("d")}
h     = {top1_diff.get("h")}
w     = {top1_diff.get("w")}


PACKAGE CONTENT
---------------
model/
    Final checkpoint

code/
    Operational model / preprocessing / inference
    Postprocessing / coordinate transform
    Visualization
    Malignancy patch adapter
    E2E QA

config/
    Inference configuration
    Patch configuration
    Output contract
    Requirements
    Version

docs/
    README
    Coordinate-system specification

qa/
    E2E QA report
    Operational visualization samples

sample_output/
    Representative inference / viewer / patch outputs

SHA256SUMS.txt
    Artifact integrity hashes

artifact_manifest.json
    Machine-readable artifact manifest


STATUS
------
LOCKED

This package contains deploy-ready model-side artifacts.
It does NOT include an API server, Docker image, or Cloud Run deployment.
Those can be added later around this package without changing the
validated model pipeline.
""".strip()

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            summary
        )

        file.write(
            "\n"
        )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL PACKAGE"
    )

    print(
        "=" * 100
    )

    print(
        "Path:"
    )

    print(
        FINAL_ROOT
    )

    print()

    print(
        "Checkpoint SHA256:"
    )

    print(
        checkpoint_sha
    )

    print()

    print(
        "Artifacts hashed:",
        len(
            manifest_files
        ),
    )

    print()

    print(
        "E2E QA:",
        qa_report.get(
            "status"
        ),
        f"({qa_summary.get('passed')}/"
        f"{qa_summary.get('total_checks')})",
    )

    print()

    print(
        "[PASS] CPMNetv2 + HN25 deploy-ready artifacts LOCKED"
    )


if __name__ == "__main__":
    main()