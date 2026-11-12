from pathlib import Path
from datetime import datetime
import importlib.util
import json
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_diagnostic_module():
    path = (
        ROOT
        / "scripts/monai/23_diagnose_candidate2_previous.py"
    )

    spec = importlib.util.spec_from_file_location(
        "previous_diagnostic",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"불러올 수 없는 파일: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    diagnostic = load_diagnostic_module()

    pipeline = diagnostic.load_module(
        "validation_pipeline",
        "19_validate_candidate2_training_pipeline.py",
    )

    evaluator = diagnostic.load_module(
        "validation_evaluator",
        "22_evaluate_candidate2_fold0.py",
    )

    # 이전 모델의 학습 조건 확인
    checkpoint = torch.load(
        diagnostic.CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    patch_size = tuple(checkpoint["patch_size"])
    spacing = tuple(checkpoint["spacing"])
    epoch = checkpoint["epoch"]

    if patch_size != (96, 96, 96):
        raise RuntimeError(f"예상과 다른 patch_size: {patch_size}")

    if not np.allclose(spacing, (1.0, 1.0, 1.0)):
        raise RuntimeError(f"예상과 다른 spacing: {spacing}")

    del checkpoint

    # 메모리에서만 설정 변경: 기존 파일은 수정하지 않음
    pipeline.PATCH_SIZE = patch_size
    pipeline.TARGET_SPACING = spacing
    evaluator.BEST_CHECKPOINT = diagnostic.CHECKPOINT

    # 3개 CT 진단 때와 같은 후처리 조건
    evaluator.SCORE_THRESH = 0.02
    evaluator.TOPK_CANDIDATES_PER_LEVEL = 1000
    evaluator.NMS_THRESH = 0.22
    evaluator.DETECTIONS_PER_IMAGE = 300
    evaluator.OVERLAP = 0.25
    evaluator.SW_BATCH_SIZE = 1

    with diagnostic.SPLIT_PATH.open(encoding="utf-8") as f:
        split = json.load(f)

    # 양성/음성 모두 포함
    samples = split["validation"]

    validation_ids = {sample["image"] for sample in samples}
    test_ids = {sample["image"] for sample in split["test"]}

    if len(samples) != 42 or len(validation_ids) != 42:
        raise RuntimeError("중복 없는 validation CT 42개가 필요합니다.")

    if validation_ids & test_ids:
        raise RuntimeError("Validation과 test에 중복 CT가 있습니다.")

    positive_count = sum(
        len(sample.get("box", [])) > 0
        for sample in samples
    )
    negative_count = len(samples) - positive_count
    expected_gt = sum(
        len(sample.get("box", []))
        for sample in samples
    )

    output_dir = (
        ROOT
        / "results/candidate2/diagnostics"
        / datetime.now().strftime(
            "previous_validation42_%Y%m%d_%H%M%S_%f"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "checkpoint": str(diagnostic.CHECKPOINT),
        "checkpoint_epoch": epoch,
        "dataset": "internal_validation",
        "num_scans": len(samples),
        "positive_scans": positive_count,
        "negative_scans": negative_count,
        "total_gt": expected_gt,
        "patch_size": list(patch_size),
        "spacing": list(spacing),
        "score_threshold": evaluator.SCORE_THRESH,
        "nms_threshold": evaluator.NMS_THRESH,
        "detections_per_image": evaluator.DETECTIONS_PER_IMAGE,
        "topk_candidates_per_level": (
            evaluator.TOPK_CANDIDATES_PER_LEVEL
        ),
        "sliding_window_overlap": evaluator.OVERLAP,
        "sw_batch_size": evaluator.SW_BATCH_SIZE,
        "inference_dtype": "float32",
        "cpm_method": (
            "Existing project evaluator: best sensitivity "
            "at FP/scan <= target; not official LUNA16 evaluation"
        ),
    }

    diagnostic.save_json(output_dir / "config.json", config)

    print(f"Checkpoint epoch: {epoch}")
    print(f"Patch: {patch_size}, spacing: {spacing}")
    print(
        f"Validation: {len(samples)} CT "
        f"(양성 {positive_count}, 음성 {negative_count})"
    )
    print(f"GT 결절: {expected_gt}")
    print(f"Output: {output_dir}", flush=True)

    device = torch.device("cuda")

    detector, loaded_checkpoint = evaluator.build_detector(
        pipeline,
        device,
    )
    del loaded_checkpoint

    scans = []
    started = time.time()

    for index, sample in enumerate(samples, start=1):
        case_started = time.time()

        print(
            f"\n[{index}/{len(samples)}] {sample['image']}",
            flush=True,
        )

        scan = evaluator.infer_one_scan(
            sample,
            pipeline,
            detector,
            device,
        )
        scans.append(scan)

        # CT 하나가 끝날 때마다 중간 결과 저장
        diagnostic.save_json(
            output_dir / "predictions.json",
            scans,
        )

        print(
            f"GT={len(scan['gt_nodules'])}, "
            f"예측={len(scan['predictions'])}, "
            f"소요={time.time() - case_started:.1f}초",
            flush=True,
        )

    # 전체 CT의 예측을 점수순으로 정렬하고 TP/FP 판정
    events, total_gt = evaluator.build_detection_events(scans)

    if total_gt != expected_gt:
        raise RuntimeError("평가 과정에서 GT 개수가 달라졌습니다.")

    # 음성 CT까지 포함한 42개가 FP/scan의 분모
    froc = evaluator.calculate_froc(
        events,
        num_scans=len(scans),
        total_gt=total_gt,
    )

    per_scan = [
        diagnostic.summarize_scan(scan, events, index)
        for index, scan in enumerate(scans)
    ]

    total_tp = sum(row["tp"] for row in per_scan)
    total_fp = sum(row["fp"] for row in per_scan)

    summary = {
        **config,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_gt - total_tp,
        "sensitivity_at_saved_predictions": total_tp / total_gt,
        "cpm": froc["cpm"],
        "cpm_fp_rates": froc["cpm_fp_rates"],
        "sensitivities_at_cpm_rates": (
            froc["sensitivities_at_cpm_rates"]
        ),
        "elapsed_seconds": time.time() - started,
        "per_scan": per_scan,
    }

    # 기존 FROC의 시작 threshold는 무한대이므로
    # 표준 JSON으로 저장할 수 있도록 null로 변환
    froc["curve_threshold"] = [
        value if np.isfinite(value) else None
        for value in froc["curve_threshold"]
    ]

    diagnostic.save_json(
        output_dir / "detection_events.json",
        events,
    )
    diagnostic.save_json(output_dir / "froc.json", froc)
    diagnostic.save_json(output_dir / "summary.json", summary)

    print("\n===== Validation 42 CT 평가 결과 =====")
    print(f"전체 후보에서 검출한 결절: {total_tp}/{total_gt}")
    print(f"전체 후보의 오탐 수: {total_fp}")

    print("\nSensitivity @ FP/scan")

    for rate, sensitivity in zip(
        froc["cpm_fp_rates"],
        froc["sensitivities_at_cpm_rates"],
    ):
        print(f"  {rate:>5g} FP/scan: {sensitivity:.4f}")

    print(f"\nValidation CPM: {froc['cpm']:.4f}")
    print(f"결과 파일: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()