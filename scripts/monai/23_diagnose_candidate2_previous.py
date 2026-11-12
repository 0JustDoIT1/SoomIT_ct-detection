from pathlib import Path
from datetime import datetime
import importlib.util
import json

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    ROOT / "results/candidate2/training/checkpoints/best_val_loss.pt"
)

SPLIT_PATH = (
    ROOT / "results/candidate2/training_setup/candidate2_fold0_split.json"
)

OUTPUT_DIR = (
    ROOT
    / "results/candidate2/diagnostics"
    / datetime.now().strftime("previous_model_%Y%m%d_%H%M%S_%f")
)


def load_module(name, filename):
    path = ROOT / "scripts/monai" / filename
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_scan(scan, events, scan_index):
    scan_events = [
        event for event in events
        if event["scan_index"] == scan_index
    ]

    tp_events = [
        event for event in scan_events
        if event["is_tp"]
    ]

    # 각 정답 결절에서 가장 가까운 예측까지의 거리.
    # 검출이 안 됐을 때 위치가 얼마나 벗어났는지 확인합니다.
    predictions = scan["predictions"]
    nearest_distances = []

    for gt in scan["gt_nodules"]:
        nearest = min(
            (
                float(
                    np.linalg.norm(
                        np.asarray(pred["center_xyz_mm"])
                        - np.asarray(gt["center_xyz_mm"])
                    )
                )
                for pred in predictions
            ),
            default=None,
        )

        nearest_distances.append({
            "gt_index": gt["gt_index"],
            "radius_mm": gt["radius_mm"],
            "nearest_prediction_distance_mm": nearest,
        })

    return {
        "image": scan["image"],
        "gt_count": len(scan["gt_nodules"]),
        "prediction_count": len(predictions),
        "tp": len(tp_events),
        "fp": len(scan_events) - len(tp_events),
        "fn": len(scan["gt_nodules"]) - len(tp_events),
        "max_prediction_score": max(
            (pred["score"] for pred in predictions),
            default=None,
        ),
        "tp_scores": [event["score"] for event in tp_events],
        "nearest_distances": nearest_distances,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. monai-luna 환경을 확인하세요."
        )

    pipeline = load_module(
        "candidate2_diagnostic_pipeline",
        "19_validate_candidate2_training_pipeline.py",
    )
    evaluator = load_module(
        "candidate2_diagnostic_evaluator",
        "22_evaluate_candidate2_fold0.py",
    )

    # 이전 체크포인트의 실제 학습 조건을 확인합니다.
    checkpoint = torch.load(
        CHECKPOINT,
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

    # 실행 중인 모듈의 설정만 바꿉니다.
    # 원본 Python 파일과 체크포인트에는 기록하지 않습니다.
    pipeline.PATCH_SIZE = patch_size
    pipeline.TARGET_SPACING = spacing
    evaluator.BEST_CHECKPOINT = CHECKPOINT

    # 현재 평가와 같은 후처리 조건을 유지합니다.
    evaluator.SCORE_THRESH = 0.02
    evaluator.TOPK_CANDIDATES_PER_LEVEL = 1000
    evaluator.NMS_THRESH = 0.22
    evaluator.DETECTIONS_PER_IMAGE = 300
    evaluator.OVERLAP = 0.25
    evaluator.SW_BATCH_SIZE = 1

    with SPLIT_PATH.open(encoding="utf-8") as f:
        split = json.load(f)

    # 앞서 패치 검사에 사용한 것과 같은 선택 방식입니다.
    # 최종 test 데이터는 사용하지 않습니다.
    samples = [
        sample
        for sample in split["validation"]
        if len(sample.get("box", [])) > 0
    ][:3]

    if len(samples) != 3:
        raise RuntimeError("양성 validation CT 3개를 확보하지 못했습니다.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Epoch: {epoch}")
    print(f"Patch: {patch_size}")
    print(f"Spacing: {spacing}")
    print(f"Output: {OUTPUT_DIR}")

    device = torch.device("cuda")
    detector, loaded_checkpoint = evaluator.build_detector(
        pipeline, device
    )
    del loaded_checkpoint

    scans = []

    for index, sample in enumerate(samples, start=1):
        print(f"\n[{index}/{len(samples)}] {sample['image']}", flush=True)

        scan = evaluator.infer_one_scan(
            sample,
            pipeline,
            detector,
            device,
        )
        scans.append(scan)

        # 중간에 중단돼도 완료한 CT 결과는 남깁니다.
        save_json(OUTPUT_DIR / "predictions.json", scans)

        print(
            f"GT={len(scan['gt_nodules'])}, "
            f"예측={len(scan['predictions'])}",
            flush=True,
        )

    events, total_gt = evaluator.build_detection_events(scans)

    per_scan = [
        summarize_scan(scan, events, index)
        for index, scan in enumerate(scans)
    ]

    total_tp = sum(row["tp"] for row in per_scan)
    total_fp = sum(row["fp"] for row in per_scan)

    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": epoch,
        "patch_size": list(patch_size),
        "spacing": list(spacing),
        "dataset": "first_3_positive_validation_scans",
        "score_threshold": evaluator.SCORE_THRESH,
        "nms_threshold": evaluator.NMS_THRESH,
        "detections_per_image": evaluator.DETECTIONS_PER_IMAGE,
        "sliding_window_overlap": evaluator.OVERLAP,
        "num_scans": len(scans),
        "total_gt": total_gt,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_gt - total_tp,
        "sensitivity_at_saved_predictions": total_tp / total_gt,
        "per_scan": per_scan,
    }

    save_json(OUTPUT_DIR / "detection_events.json", events)
    save_json(OUTPUT_DIR / "summary.json", summary)

    print("\n===== 전체 CT 진단 결과 =====")

    for row in per_scan:
        print(
            f"{row['image']}\n"
            f"  GT={row['gt_count']} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']}\n"
            f"  정답 예측 점수={row['tp_scores']}"
        )

    print(f"\n전체 결절 검출: {total_tp}/{total_gt}")
    print(f"결과 파일: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()