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


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_diagnostic():
    path = ROOT / "scripts/monai/23_diagnose_candidate2_previous.py"
    spec = importlib.util.spec_from_file_location("diagnostic", path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_target(scan, gt_index):
    """특정 GT에 매칭되는 예측의 최고 점수와 전체 후보 순위."""
    gt = next(
        gt for gt in scan["gt_nodules"]
        if gt["gt_index"] == gt_index
    )

    predictions = sorted(
        scan["predictions"],
        key=lambda pred: pred["score"],
        reverse=True,
    )

    distances = [
        float(np.linalg.norm(
            np.asarray(pred["center_xyz_mm"])
            - np.asarray(gt["center_xyz_mm"])
        ))
        for pred in predictions
    ]

    hits = [
        index
        for index, distance in enumerate(distances)
        if distance <= gt["radius_mm"]
    ]

    return {
        "detected": bool(hits),
        "prediction_count": len(predictions),
        "best_matching_rank": hits[0] + 1 if hits else None,
        "best_matching_score": (
            predictions[hits[0]]["score"] if hits else None
        ),
        "nearest_prediction_mm": min(distances, default=None),
        "gt_radius_mm": gt["radius_mm"],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    diagnostic = load_diagnostic()

    pipeline = diagnostic.load_module(
        "limit_pipeline",
        "19_validate_candidate2_training_pipeline.py",
    )
    evaluator = diagnostic.load_module(
        "limit_evaluator",
        "22_evaluate_candidate2_fold0.py",
    )

    window_summary = read_json(WINDOW_RESULT_DIR / "summary.json")
    window_results = read_json(WINDOW_RESULT_DIR / "results.json")

    source_dir = Path(window_summary["source_evaluation"])
    source_config = read_json(source_dir / "summary.json")

    # 실제 window에서는 검출됐지만 전체 CT에서 사라진 결절
    targets = [
        row for row in window_results
        if row["diagnosis"] == "window_detected_but_full_volume_missed"
    ]

    if not targets:
        raise RuntimeError("비교 대상 결절이 없습니다.")

    target_images = list(dict.fromkeys(
        row["image"] for row in targets
    ))

    split = read_json(diagnostic.SPLIT_PATH)
    validation = {
        sample["image"]: sample
        for sample in split["validation"]
    }

    if any(image not in validation for image in target_images):
        raise RuntimeError("Validation에 없는 CT가 포함돼 있습니다.")

    if (
        source_config["dataset"] != "internal_validation"
        or source_config["checkpoint_epoch"] != 21
        or tuple(source_config["patch_size"]) != (96, 96, 96)
        or not np.allclose(source_config["spacing"], (1, 1, 1))
        or source_config["detections_per_image"] != 300
    ):
        raise RuntimeError("원본 평가 조건이 예상과 다릅니다.")

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
        raise RuntimeError("예상한 이전 체크포인트가 아닙니다.")

    del checkpoint

    pipeline.PATCH_SIZE = (96, 96, 96)
    pipeline.TARGET_SPACING = (1.0, 1.0, 1.0)
    evaluator.BEST_CHECKPOINT = diagnostic.CHECKPOINT

    evaluator.SCORE_THRESH = source_config["score_threshold"]
    evaluator.NMS_THRESH = source_config["nms_threshold"]
    evaluator.TOPK_CANDIDATES_PER_LEVEL = source_config[
        "topk_candidates_per_level"
    ]
    evaluator.OVERLAP = source_config["sliding_window_overlap"]
    evaluator.SW_BATCH_SIZE = 1
    evaluator.DETECTIONS_PER_IMAGE = 300

    output_dir = (
        ROOT
        / "results/candidate2/diagnostics"
        / datetime.now().strftime("limit_compare_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    diagnostic.save_json(
        output_dir / "config.json",
        {
            "checkpoint": str(diagnostic.CHECKPOINT),
            "source_evaluation": str(source_dir),
            "source_window_diagnostic": str(WINDOW_RESULT_DIR),
            "patch_size": list(pipeline.PATCH_SIZE),
            "spacing": list(pipeline.TARGET_SPACING),
            "score_threshold": evaluator.SCORE_THRESH,
            "nms_threshold": evaluator.NMS_THRESH,
            "topk_candidates_per_level": (
                evaluator.TOPK_CANDIDATES_PER_LEVEL
            ),
            "overlap": evaluator.OVERLAP,
            "compared_detection_limits": [300, 3000],
            "images": target_images,
        },
    )

    device = torch.device("cuda")
    detector, loaded = evaluator.build_detector(pipeline, device)
    del loaded

    comparisons = []

    print(
        f"\n검사 대상: {len(target_images)}개 CT, "
        f"{len(targets)}개 결절"
    )
    print(f"Output: {output_dir}", flush=True)

    for image in target_images:
        sample = validation[image]
        scan_targets = [
            row for row in targets
            if row["image"] == image
        ]

        checks_by_limit = {}

        for limit in (300, 3000):
            # 비교 중 바꾸는 값은 detections_per_img 하나뿐
            detector.set_box_selector_parameters(
                score_thresh=evaluator.SCORE_THRESH,
                topk_candidates_per_level=(
                    evaluator.TOPK_CANDIDATES_PER_LEVEL
                ),
                nms_thresh=evaluator.NMS_THRESH,
                detections_per_img=limit,
            )

            print(f"\nCT: {image}\n후보 제한: {limit}", flush=True)

            scan = evaluator.infer_one_scan(
                sample,
                pipeline,
                detector,
                device,
            )

            diagnostic.save_json(
                output_dir / f"{Path(image).stem}_limit{limit}.json",
                scan,
            )

            checks_by_limit[limit] = {
                row["gt_index"]: check_target(scan, row["gt_index"])
                for row in scan_targets
            }

            for gt_index, check in checks_by_limit[limit].items():
                print(
                    f"  GT {gt_index}: "
                    f"검출={check['detected']}, "
                    f"정답 순위={check['best_matching_rank']}, "
                    f"점수={check['best_matching_score']}, "
                    f"전체 후보={check['prediction_count']}",
                    flush=True,
                )

        for row in scan_targets:
            gt_index = row["gt_index"]
            baseline = checks_by_limit[300][gt_index]
            expanded = checks_by_limit[3000][gt_index]

            if baseline["detected"]:
                verdict = "baseline_now_detected"
            elif expanded["detected"]:
                verdict = "recovered_after_limit_increase"
            else:
                verdict = "still_missed"

            comparisons.append({
                "image": image,
                "gt_index": gt_index,
                "limit_300": baseline,
                "limit_3000": expanded,
                "verdict": verdict,
            })

        diagnostic.save_json(
            output_dir / "comparison.json",
            comparisons,
        )

    print("\n===== 최종 후보 수 비교 =====")

    for row in comparisons:
        print(
            f"{row['image']} / GT {row['gt_index']}\n"
            f"  300개: {row['limit_300']['detected']}\n"
            f"  3000개: {row['limit_3000']['detected']}\n"
            f"  판정: {row['verdict']}"
        )

    print(f"\n결과 파일: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()