from pathlib import Path
from datetime import datetime
import importlib.util
import json

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

WINDOW_RESULT_DIR = (
    ROOT
    / "results/candidate2/diagnostics"
    / "actual_windows_20260909_232828_194697"
)

TARGET_IMAGE = (
    "1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "112767175295249119452142211437.mhd"
)
TARGET_GT_INDEX = 4

TOPK_PER_LEVEL = 10000
FINAL_LIMITS = (3000, 50000)


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_diagnostic():
    path = (
        ROOT
        / "scripts/monai/23_diagnose_candidate2_previous.py"
    )

    spec = importlib.util.spec_from_file_location(
        "previous_diagnostic",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_target(scan, gt_index):
    """GT 반지름 안에 중심이 들어오는 예측의 순위와 점수."""
    gt = next(
        (
            item
            for item in scan["gt_nodules"]
            if item["gt_index"] == gt_index
        ),
        None,
    )

    if gt is None:
        raise RuntimeError(f"GT index {gt_index}가 없습니다.")

    predictions = sorted(
        scan["predictions"],
        key=lambda pred: pred["score"],
        reverse=True,
    )

    gt_center = np.asarray(
        gt["center_xyz_mm"],
        dtype=np.float64,
    )

    distances = [
        float(
            np.linalg.norm(
                np.asarray(
                    pred["center_xyz_mm"],
                    dtype=np.float64,
                )
                - gt_center
            )
        )
        for pred in predictions
    ]

    matching_indices = [
        index
        for index, distance in enumerate(distances)
        if distance <= gt["radius_mm"]
    ]

    best_index = (
        matching_indices[0]
        if matching_indices
        else None
    )

    return {
        "detected": best_index is not None,
        "prediction_count": len(predictions),
        "matching_prediction_count": len(matching_indices),
        "best_matching_rank": (
            best_index + 1
            if best_index is not None
            else None
        ),
        "best_matching_score": (
            predictions[best_index]["score"]
            if best_index is not None
            else None
        ),
        "best_matching_distance_mm": (
            distances[best_index]
            if best_index is not None
            else None
        ),
        "nearest_prediction_mm": min(distances, default=None),
        "gt_radius_mm": gt["radius_mm"],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. "
            "monai-luna 환경을 확인하세요."
        )

    diagnostic = load_diagnostic()

    pipeline = diagnostic.load_module(
        "final_limit_pipeline",
        "19_validate_candidate2_training_pipeline.py",
    )

    evaluator = diagnostic.load_module(
        "final_limit_evaluator",
        "22_evaluate_candidate2_fold0.py",
    )

    # 기존 전체 CT 평가 조건 확인
    window_summary = read_json(
        WINDOW_RESULT_DIR / "summary.json"
    )
    window_results = read_json(
        WINDOW_RESULT_DIR / "results.json"
    )

    source_dir = Path(
        window_summary["source_evaluation"]
    )
    source_config = read_json(
        source_dir / "summary.json"
    )

    if (
        source_config["dataset"] != "internal_validation"
        or source_config["checkpoint_epoch"] != 21
        or tuple(source_config["patch_size"]) != (96, 96, 96)
        or not np.allclose(
            source_config["spacing"],
            (1.0, 1.0, 1.0),
        )
    ):
        raise RuntimeError("원본 평가 조건이 예상과 다릅니다.")

    target_record = next(
        (
            row
            for row in window_results
            if (
                row["image"] == TARGET_IMAGE
                and row["gt_index"] == TARGET_GT_INDEX
            )
        ),
        None,
    )

    if target_record is None:
        raise RuntimeError("기존 window 검사에 대상 결절이 없습니다.")

    if (
        target_record["diagnosis"]
        != "window_detected_but_full_volume_missed"
    ):
        raise RuntimeError("대상 결절의 기존 진단 결과가 다릅니다.")

    # Validation에서만 대상 CT 선택
    split = read_json(diagnostic.SPLIT_PATH)

    validation = {
        sample["image"]: sample
        for sample in split["validation"]
    }

    if TARGET_IMAGE not in validation:
        raise RuntimeError("대상 CT가 validation에 없습니다.")

    if TARGET_IMAGE in {
        sample["image"] for sample in split["test"]
    }:
        raise RuntimeError("대상 CT가 test에도 포함되어 있습니다.")

    sample = validation[TARGET_IMAGE]

    # 이전 체크포인트 조건 확인
    checkpoint = torch.load(
        diagnostic.CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint["epoch"] != 21
        or tuple(checkpoint["patch_size"]) != (96, 96, 96)
        or not np.allclose(
            checkpoint["spacing"],
            (1.0, 1.0, 1.0),
        )
    ):
        raise RuntimeError("예상한 이전 체크포인트가 아닙니다.")

    del checkpoint

    # 원래 학습 조건 유지
    pipeline.PATCH_SIZE = (96, 96, 96)
    pipeline.TARGET_SPACING = (1.0, 1.0, 1.0)
    evaluator.BEST_CHECKPOINT = diagnostic.CHECKPOINT

    # 기존 평가와 같은 threshold, NMS, overlap
    evaluator.SCORE_THRESH = source_config["score_threshold"]
    evaluator.NMS_THRESH = source_config["nms_threshold"]
    evaluator.OVERLAP = source_config["sliding_window_overlap"]
    evaluator.SW_BATCH_SIZE = 1

    # 레벨별 후보 제한 고정
    evaluator.TOPK_CANDIDATES_PER_LEVEL = TOPK_PER_LEVEL
    evaluator.DETECTIONS_PER_IMAGE = FINAL_LIMITS[0]

    output_dir = (
        ROOT
        / "results/candidate2/diagnostics"
        / datetime.now().strftime(
            "final_limit_check_%Y%m%d_%H%M%S_%f"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "checkpoint": str(diagnostic.CHECKPOINT),
        "checkpoint_epoch": 21,
        "source_evaluation": str(source_dir),
        "source_window_diagnostic": str(WINDOW_RESULT_DIR),
        "target_image": TARGET_IMAGE,
        "target_gt_index": TARGET_GT_INDEX,
        "patch_size": list(pipeline.PATCH_SIZE),
        "spacing": list(pipeline.TARGET_SPACING),
        "score_threshold": evaluator.SCORE_THRESH,
        "nms_threshold": evaluator.NMS_THRESH,
        "topk_candidates_per_level": TOPK_PER_LEVEL,
        "compared_detection_limits": list(FINAL_LIMITS),
        "sliding_window_overlap": evaluator.OVERLAP,
        "sw_batch_size": evaluator.SW_BATCH_SIZE,
        "inference_dtype": "float32",
    }

    diagnostic.save_json(
        output_dir / "config.json",
        config,
    )

    print(f"대상 CT: {TARGET_IMAGE}")
    print(f"대상 GT: {TARGET_GT_INDEX}")
    print(f"레벨별 후보 제한: {TOPK_PER_LEVEL}")
    print(f"최종 후보 제한 비교: {FINAL_LIMITS}")
    print(f"Output: {output_dir}", flush=True)

    device = torch.device("cuda")

    detector, loaded = evaluator.build_detector(
        pipeline,
        device,
    )
    del loaded

    checks = {}

    for limit in FINAL_LIMITS:
        # 비교 중 변경하는 설정은 최종 후보 제한 하나뿐
        detector.set_box_selector_parameters(
            score_thresh=evaluator.SCORE_THRESH,
            topk_candidates_per_level=TOPK_PER_LEVEL,
            nms_thresh=evaluator.NMS_THRESH,
            detections_per_img=limit,
        )

        print(
            f"\n최종 후보 제한: {limit}",
            flush=True,
        )

        scan = evaluator.infer_one_scan(
            sample,
            pipeline,
            detector,
            device,
        )

        diagnostic.save_json(
            output_dir / f"predictions_final_{limit}.json",
            scan,
        )

        check = check_target(scan, TARGET_GT_INDEX)
        check["final_limit"] = limit
        check["limit_reached"] = (
            check["prediction_count"] >= limit
        )
        checks[limit] = check

        # 완료된 조건의 진단 결과를 중간 저장
        diagnostic.save_json(
            output_dir / "checks.json",
            checks,
        )

        print(
            f"GT {TARGET_GT_INDEX}: "
            f"검출={check['detected']}\n"
            f"  전체 후보={check['prediction_count']}\n"
            f"  정답 순위={check['best_matching_rank']}\n"
            f"  정답 점수={check['best_matching_score']}\n"
            f"  최근접 예측 거리={check['nearest_prediction_mm']} mm\n"
            f"  GT 반지름={check['gt_radius_mm']} mm\n"
            f"  최종 제한 도달={check['limit_reached']}",
            flush=True,
        )

    baseline = checks[3000]
    expanded = checks[50000]

    if baseline["detected"]:
        verdict = "baseline_now_detected"
        explanation = (
            "3000개 설정에서도 검출됐습니다. "
            "이전 결과와 달라 재현성 확인이 필요합니다."
        )
    elif expanded["detected"]:
        verdict = "recovered_after_final_limit_increase"
        explanation = (
            "최종 제한을 늘리자 정답이 복구됐습니다. "
            "레벨별 10000개 설정에서 최종 3000개 제한의 "
            "영향이 확인됐습니다."
        )
    elif expanded["limit_reached"]:
        verdict = "still_missed_and_final_limit_reached"
        explanation = (
            "여전히 미검출이며 최종 제한에도 도달했습니다. "
            "최종 제한의 영향을 아직 배제할 수 없습니다."
        )
    else:
        verdict = "still_missed_below_final_limit"
        explanation = (
            "최종 제한에 도달하지 않았는데도 미검출입니다. "
            "이 조건에서는 최종 개수 제한보다 앞 단계의 "
            "영향을 조사해야 합니다."
        )

    comparison = {
        **config,
        "final_3000": baseline,
        "final_50000": expanded,
        "verdict": verdict,
        "explanation": explanation,
    }

    diagnostic.save_json(
        output_dir / "comparison.json",
        comparison,
    )

    print("\n===== 최종 후보 제한 추가 검사 =====")
    print(f"최종 3000개 검출: {baseline['detected']}")
    print(f"최종 50000개 검출: {expanded['detected']}")
    print(f"판정: {verdict}")
    print(explanation)
    print(f"결과 파일: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()