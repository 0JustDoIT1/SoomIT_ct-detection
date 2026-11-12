"""Compare NMS thresholds on the validation nodule missed in script 29."""
from datetime import datetime
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
FINAL_LIMIT = 50000
TOPK_PER_LEVEL = 10000
NMS_THRESHOLDS = (0.22, 1.0)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Use the monai-luna environment.")

    # Reuse the previous diagnostic's target and matching criterion.
    path = ROOT / "scripts/monai/29_check_candidate2_final_limit.py"
    spec = importlib.util.spec_from_file_location("previous_limit_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    diagnostic = previous.load_diagnostic()
    pipeline = diagnostic.load_module("nms_pipeline", "19_validate_candidate2_training_pipeline.py")
    evaluator = diagnostic.load_module("nms_evaluator", "22_evaluate_candidate2_fold0.py")

    window_summary = previous.read_json(previous.WINDOW_RESULT_DIR / "summary.json")
    source_dir = Path(window_summary["source_evaluation"])
    source = previous.read_json(source_dir / "summary.json")
    if (
        source["dataset"] != "internal_validation"
        or source["checkpoint_epoch"] != 21
        or tuple(source["patch_size"]) != (96, 96, 96)
        or not np.allclose(source["spacing"], (1, 1, 1))
        or not np.isclose(source["nms_threshold"], NMS_THRESHOLDS[0])
    ):
        raise RuntimeError("Unexpected source evaluation settings.")

    split = previous.read_json(diagnostic.SPLIT_PATH)
    validation = {sample["image"]: sample for sample in split["validation"]}
    target_image = previous.TARGET_IMAGE
    gt_index = previous.TARGET_GT_INDEX
    if target_image not in validation:
        raise RuntimeError("Target is not an internal validation CT.")
    if target_image in {sample["image"] for sample in split["test"]}:
        raise RuntimeError("Validation/test overlap.")
    sample = validation[target_image]
    if not 0 <= gt_index < len(sample.get("box", [])):
        raise RuntimeError("Target GT index is invalid.")

    checkpoint = torch.load(diagnostic.CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        checkpoint["epoch"] != 21
        or tuple(checkpoint["patch_size"]) != (96, 96, 96)
        or not np.allclose(checkpoint["spacing"], (1, 1, 1))
    ):
        raise RuntimeError("Unexpected checkpoint settings.")
    del checkpoint

    pipeline.PATCH_SIZE = (96, 96, 96)
    pipeline.TARGET_SPACING = (1.0, 1.0, 1.0)
    evaluator.BEST_CHECKPOINT = diagnostic.CHECKPOINT
    evaluator.SCORE_THRESH = source["score_threshold"]
    evaluator.NMS_THRESH = NMS_THRESHOLDS[0]
    evaluator.OVERLAP = source["sliding_window_overlap"]
    evaluator.SW_BATCH_SIZE = 1
    evaluator.TOPK_CANDIDATES_PER_LEVEL = TOPK_PER_LEVEL
    evaluator.DETECTIONS_PER_IMAGE = FINAL_LIMIT

    output_dir = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("nms_compare_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "checkpoint": str(diagnostic.CHECKPOINT), "checkpoint_epoch": 21,
        "source_evaluation": str(source_dir), "dataset": "internal_validation",
        "target_image": target_image, "target_gt_index": gt_index,
        "patch_size": list(pipeline.PATCH_SIZE), "spacing": list(pipeline.TARGET_SPACING),
        "score_threshold": evaluator.SCORE_THRESH,
        "compared_nms_thresholds": list(NMS_THRESHOLDS),
        "topk_candidates_per_level": TOPK_PER_LEVEL, "detections_per_image": FINAL_LIMIT,
        "sliding_window_overlap": evaluator.OVERLAP, "sw_batch_size": 1,
        "inference_dtype": "float32",
    }
    diagnostic.save_json(output_dir / "config.json", config)
    print(f"Target: {target_image}, GT {gt_index}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    device = torch.device("cuda")
    detector, loaded = evaluator.build_detector(pipeline, device)
    del loaded
    checks = {}

    for threshold in NMS_THRESHOLDS:
        # Only NMS changes; both candidate limits and score threshold stay fixed.
        detector.set_box_selector_parameters(
            score_thresh=evaluator.SCORE_THRESH,
            topk_candidates_per_level=TOPK_PER_LEVEL,
            nms_thresh=threshold,
            detections_per_img=FINAL_LIMIT,
        )
        print(f"\nNMS threshold: {threshold}", flush=True)
        scan = evaluator.infer_one_scan(sample, pipeline, detector, device)
        diagnostic.save_json(output_dir / f"predictions_nms_{threshold:g}.json", scan)
        check = previous.check_target(scan, gt_index)
        check["final_limit_reached"] = check["prediction_count"] >= FINAL_LIMIT
        checks[str(threshold)] = check
        diagnostic.save_json(output_dir / "checks.json", checks)
        print(
            f"Detected={check['detected']}, predictions={check['prediction_count']}, "
            f"rank={check['best_matching_rank']}, score={check['best_matching_score']}\n"
            f"Nearest distance={check['nearest_prediction_mm']} mm; "
            f"GT radius={check['gt_radius_mm']} mm; "
            f"final limit reached={check['final_limit_reached']}", flush=True,
        )

    baseline, relaxed = checks["0.22"], checks["1.0"]
    if baseline["detected"]:
        verdict = "baseline_now_detected"
        explanation = "기준 설정에서도 검출됐습니다. 29번 결과와의 재현성 확인이 필요합니다."
    elif relaxed["detected"]:
        verdict = "recovered_after_nms_relaxation"
        explanation = "NMS 완화로 정답이 복구됐습니다. 이 조건에서 NMS 억제의 영향이 확인됐습니다."
    elif relaxed["final_limit_reached"]:
        verdict = "still_missed_and_final_limit_reached"
        explanation = "미검출이며 최종 제한에 도달했습니다. 최종 후보 제한 영향을 배제할 수 없습니다."
    else:
        verdict = "still_missed_without_nms_suppression"
        explanation = "NMS 완화로 복구되지 않았습니다. 레벨별 후보 제한, 점수 threshold, 출력 맵 병합 및 박스 생성을 조사해야 합니다."
    diagnostic.save_json(output_dir / "comparison.json", {
        **config, "nms_0.22": baseline, "nms_1.0": relaxed,
        "verdict": verdict, "explanation": explanation,
    })
    print(f"\n판정: {verdict}\n{explanation}", flush=True)
    print(f"결과 파일: {output_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
