from pathlib import Path
from datetime import datetime
import importlib.util
import json

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

# 방금 완료한 validation 42개 평가 결과
SOURCE_DIR = (
    ROOT
    / "results/candidate2/diagnostics"
    / "previous_validation42_20260909_205826_520168"
)


def load_diagnostic():
    path = ROOT / "scripts/monai/23_diagnose_candidate2_previous.py"
    spec = importlib.util.spec_from_file_location("diagnostic", path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_centered_patch(volume, target_box, pipeline):
    """결절 중심으로 자르되 CT 바깥은 기존 pipeline 방식으로 패딩."""
    patch_size = np.asarray(pipeline.PATCH_SIZE, dtype=np.int64)
    volume_size = np.asarray(volume.shape, dtype=np.int64)

    center = (target_box[:3] + target_box[3:]) / 2
    start = np.floor(center - patch_size / 2).astype(np.int64)

    start = np.clip(
        start,
        0,
        np.maximum(volume_size - patch_size, 0),
    )
    end = np.minimum(start + patch_size, volume_size)

    patch = volume[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]

    if tuple(patch.shape) != tuple(pipeline.PATCH_SIZE):
        patch = pipeline.pad_to_patch_size(patch)

    return np.ascontiguousarray(patch, dtype=np.float32), start


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    diagnostic = load_diagnostic()

    pipeline = diagnostic.load_module(
        "missed_pipeline",
        "19_validate_candidate2_training_pipeline.py",
    )
    evaluator = diagnostic.load_module(
        "missed_evaluator",
        "22_evaluate_candidate2_fold0.py",
    )

    with (SOURCE_DIR / "summary.json").open(encoding="utf-8") as f:
        source_summary = json.load(f)

    with (SOURCE_DIR / "predictions.json").open(encoding="utf-8") as f:
        scans = json.load(f)

    with diagnostic.SPLIT_PATH.open(encoding="utf-8") as f:
        split = json.load(f)

    # 잘못된 평가 결과나 모델을 섞지 않도록 확인
    if source_summary["dataset"] != "internal_validation":
        raise RuntimeError("Internal validation 결과가 아닙니다.")

    if source_summary["checkpoint_epoch"] != 21:
        raise RuntimeError("예상한 이전 모델의 평가 결과가 아닙니다.")

    if tuple(source_summary["patch_size"]) != (96, 96, 96):
        raise RuntimeError("원본 평가의 patch_size가 다릅니다.")

    if not np.allclose(source_summary["spacing"], (1, 1, 1)):
        raise RuntimeError("원본 평가의 spacing이 다릅니다.")

    validation = {
        sample["image"]: sample
        for sample in split["validation"]
    }

    if len(scans) != 42 or {s["image"] for s in scans} != set(validation):
        raise RuntimeError("Validation 42개 전체의 결과가 필요합니다.")

    checkpoint = torch.load(
        diagnostic.CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint["epoch"] != 21
        or tuple(checkpoint["patch_size"]) != (96, 96, 96)
        or not np.allclose(checkpoint["spacing"], (1, 1, 1))
    ):
        raise RuntimeError("체크포인트 조건이 예상과 다릅니다.")

    del checkpoint

    pipeline.PATCH_SIZE = (96, 96, 96)
    pipeline.TARGET_SPACING = (1.0, 1.0, 1.0)
    evaluator.BEST_CHECKPOINT = diagnostic.CHECKPOINT

    # 전체 CT 평가와 동일한 후보 선택 조건
    evaluator.SCORE_THRESH = source_summary["score_threshold"]
    evaluator.NMS_THRESH = source_summary["nms_threshold"]
    evaluator.DETECTIONS_PER_IMAGE = source_summary["detections_per_image"]
    evaluator.TOPK_CANDIDATES_PER_LEVEL = source_summary[
        "topk_candidates_per_level"
    ]

    # 저장된 전체 CT 예측에서 실제 미검출 GT 목록 재구성
    events, total_gt = evaluator.build_detection_events(scans)

    matched = {
        (event["scan_index"], event["matched_gt_index"])
        for event in events
        if event["is_tp"]
    }

    missed = [
        (scan_index, gt)
        for scan_index, scan in enumerate(scans)
        for gt in scan["gt_nodules"]
        if (scan_index, gt["gt_index"]) not in matched
    ]

    if len(missed) != source_summary["total_fn"]:
        raise RuntimeError("원본 결과와 미검출 결절 수가 다릅니다.")

    output_dir = (
        ROOT
        / "results/candidate2/diagnostics"
        / datetime.now().strftime("missed_patch_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"전체 GT: {total_gt}")
    print(f"중심 패치 검사 대상: {len(missed)}개")
    print(f"Output: {output_dir}", flush=True)

    device = torch.device("cuda")
    detector, loaded = evaluator.build_detector(pipeline, device)
    del loaded

    records = []
    cached_scan_index = None
    volume = None
    voxel_boxes = None
    spacing = np.asarray(pipeline.TARGET_SPACING)

    for index, (scan_index, gt) in enumerate(missed, start=1):
        scan = scans[scan_index]
        sample = validation[scan["image"]]

        # 같은 CT에 미검출 결절이 여러 개면 영상 로딩 재사용
        if cached_scan_index != scan_index:
            volume = None
            voxel_boxes = None

            (
                _,
                _,
                _,
                volume,
                voxel_boxes,
                _,
            ) = pipeline.load_real_sample(sample)

            cached_scan_index = scan_index

        gt_index = gt["gt_index"]

        # split의 GT와 저장된 평가 GT가 같은 순서인지 확인
        if not np.allclose(
            sample["box"][gt_index][:3],
            gt["center_xyz_mm"],
        ):
            raise RuntimeError("GT 순서 또는 좌표가 일치하지 않습니다.")

        target_box = voxel_boxes[gt_index]
        patch, start = make_centered_patch(
            volume,
            target_box,
            pipeline,
        )

        target_center = (
            (target_box[:3] + target_box[3:]) / 2 - start
        )

        image_tensor = (
            torch.from_numpy(patch)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

        # 이미 96³ 패치이므로 sliding-window를 사용하지 않음
        with torch.no_grad():
            prediction = detector(
                [image_tensor],
                use_inferer=False,
            )[0]

        boxes = (
            prediction[detector.target_box_key]
            .detach().cpu().numpy()
        )
        scores = (
            prediction[detector.pred_score_key]
            .detach().cpu().numpy()
        )

        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]

        centers = (boxes[:, :3] + boxes[:, 3:]) / 2
        distances = np.linalg.norm(
            (centers - target_center) * spacing,
            axis=1,
        )

        # 전체 CT 평가와 같은 기준:
        # 예측 중심과 GT 중심 거리 <= GT 반지름
        hit_indices = np.flatnonzero(
            distances <= gt["radius_mm"]
        )
        detected = len(hit_indices) > 0

        full_nearest = min(
            (
                float(np.linalg.norm(
                    np.asarray(pred["center_xyz_mm"])
                    - np.asarray(gt["center_xyz_mm"])
                ))
                for pred in scan["predictions"]
            ),
            default=None,
        )

        record = {
            "image": scan["image"],
            "gt_index": gt_index,
            "diameter_mm": gt["diameter_mm"],
            "radius_mm": gt["radius_mm"],
            "full_volume_detected": False,
            "full_volume_nearest_prediction_mm": full_nearest,
            "patch_detected": detected,
            "patch_prediction_count": len(scores),
            "patch_nearest_prediction_mm": (
                float(distances.min()) if len(distances) else None
            ),
            "patch_best_matching_score": (
                float(scores[hit_indices[0]]) if detected else None
            ),
            "patch_best_matching_rank": (
                int(hit_indices[0]) + 1 if detected else None
            ),
            "patch_start_xyz": start.tolist(),
            "target_center_in_patch_xyz": target_center.tolist(),
            # 이후 위치 시각화에 사용할 수 있도록 저장
            "patch_prediction_boxes": boxes.tolist(),
            "patch_prediction_scores": scores.tolist(),
        }
        records.append(record)

        diagnostic.save_json(
            output_dir / "nodule_results.json",
            records,
        )

        status = "패치에서 검출" if detected else "패치에서도 미검출"

        print(
            f"[{index}/{len(missed)}] {status} | "
            f"직경={gt['diameter_mm']:.2f}mm | "
            f"정답 후보 순위={record['patch_best_matching_rank']} | "
            f"점수={record['patch_best_matching_score']}",
            flush=True,
        )

    recovered = sum(record["patch_detected"] for record in records)

    summary = {
        "source_evaluation": str(SOURCE_DIR),
        "checkpoint": str(diagnostic.CHECKPOINT),
        "checkpoint_epoch": 21,
        "patch_size": list(pipeline.PATCH_SIZE),
        "spacing": list(pipeline.TARGET_SPACING),
        "score_threshold": evaluator.SCORE_THRESH,
        "nms_threshold": evaluator.NMS_THRESH,
        "detections_per_image": evaluator.DETECTIONS_PER_IMAGE,
        "topk_candidates_per_level": (
            evaluator.TOPK_CANDIDATES_PER_LEVEL
        ),
        "missed_in_full_volume": len(missed),
        "detected_in_centered_patch": recovered,
        "still_missed_in_centered_patch": len(missed) - recovered,
    }

    diagnostic.save_json(output_dir / "summary.json", summary)

    print("\n===== 미검출 결절 중심 패치 검사 =====")
    print(f"전체 CT에서 놓친 결절: {len(missed)}개")
    print(f"중심 패치에서는 검출: {recovered}개")
    print(f"중심 패치에서도 미검출: {len(missed) - recovered}개")
    print(f"결과 파일: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()