from pathlib import Path
from datetime import datetime
from collections import defaultdict
import importlib.util
import json

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

MISSED_DIR = (
    ROOT
    / "results/candidate2/diagnostics"
    / "missed_patch_20260909_210651_785537"
)

# 먼저 3개만 검사합니다.
# 전체 대상으로 확대하려면 None으로 바꾸세요.
MAX_NODULES = 3


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_diagnostic():
    path = ROOT / "scripts/monai/23_diagnose_candidate2_previous.py"
    spec = importlib.util.spec_from_file_location("diagnostic", path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"불러올 수 없습니다: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_prediction(output, detector, target_center, spacing, radius):
    boxes = output[detector.target_box_key].detach().cpu().numpy()
    scores = output[detector.pred_score_key].detach().cpu().numpy()

    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]

    centers = (boxes[:, :3] + boxes[:, 3:]) / 2
    distances = np.linalg.norm(
        (centers - target_center) * spacing,
        axis=1,
    )
    hits = np.flatnonzero(distances <= radius)

    return {
        "detected": bool(len(hits)),
        "prediction_count": len(scores),
        "nearest_distance_mm": (
            float(distances.min()) if len(distances) else None
        ),
        "best_matching_score": (
            float(scores[hits[0]]) if len(hits) else None
        ),
        "best_matching_rank": (
            int(hits[0]) + 1 if len(hits) else None
        ),
    }


class WindowRecorder:
    """
    MONAI가 실제 생성한 window 좌표와 입력을 기록합니다.
    원래 predictor의 출력은 변경하지 않습니다.
    """

    def __init__(self, inferer, targets, roi_size):
        self.inferer = inferer
        self.targets = targets
        self.roi_size = np.asarray(roi_size)
        self.windows = []

    def __call__(self, images, predictor, *args, **kwargs):
        # Detector는 뒤쪽에 padding을 추가합니다.
        # Inferer가 ROI보다 작은 영상을 추가 padding하는 경우만
        # 앞쪽 padding 좌표를 보정합니다.
        image_size = np.asarray(images.shape[-3:])
        pad_left = np.maximum(self.roi_size - image_size, 0) // 2

        if not hasattr(self.inferer, "with_coord"):
            raise RuntimeError(
                "현재 MONAI inferer가 with_coord를 지원하지 않습니다."
            )

        previous_with_coord = self.inferer.with_coord
        self.inferer.with_coord = True

        def record_predictor(patches, coordinates, *p_args, **p_kwargs):
            for batch_index, coordinate in enumerate(coordinates):
                slices = coordinate[-3:]

                start = np.asarray(
                    [part.start for part in slices],
                    dtype=np.int64,
                ) - pad_left

                end = start + np.asarray(patches.shape[-3:])

                included = [
                    target["gt_index"]
                    for target in self.targets
                    if (
                        np.all(target["center"] >= start)
                        and np.all(target["center"] < end)
                    )
                ]

                if included:
                    self.windows.append({
                        "start": start,
                        "gt_indices": included,
                        "patch": (
                            patches[batch_index]
                            .detach().cpu().clone()
                        ),
                    })

            return predictor(patches, *p_args, **p_kwargs)

        try:
            return self.inferer(
                images,
                record_predictor,
                *args,
                **kwargs,
            )
        finally:
            self.inferer.with_coord = previous_with_coord


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    diagnostic = load_diagnostic()

    pipeline = diagnostic.load_module(
        "actual_window_pipeline",
        "19_validate_candidate2_training_pipeline.py",
    )
    evaluator = diagnostic.load_module(
        "actual_window_evaluator",
        "22_evaluate_candidate2_fold0.py",
    )

    missed_summary = read_json(MISSED_DIR / "summary.json")
    nodule_results = read_json(MISSED_DIR / "nodule_results.json")

    source_dir = Path(missed_summary["source_evaluation"])
    source_summary = read_json(source_dir / "summary.json")

    if source_summary["dataset"] != "internal_validation":
        raise RuntimeError("Validation 평가 결과가 아닙니다.")

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

    evaluator.SCORE_THRESH = source_summary["score_threshold"]
    evaluator.NMS_THRESH = source_summary["nms_threshold"]
    evaluator.DETECTIONS_PER_IMAGE = source_summary["detections_per_image"]
    evaluator.TOPK_CANDIDATES_PER_LEVEL = source_summary[
        "topk_candidates_per_level"
    ]
    evaluator.OVERLAP = source_summary["sliding_window_overlap"]
    evaluator.SW_BATCH_SIZE = 1

    # 전체 CT에서는 놓쳤지만 중심 패치에서는 찾았던 결절
    selected = [
        row for row in nodule_results
        if row["patch_detected"]
    ]

    if MAX_NODULES is not None:
        selected = selected[:MAX_NODULES]

    if not selected:
        raise RuntimeError("검사할 결절이 없습니다.")

    grouped = defaultdict(list)
    for row in selected:
        grouped[row["image"]].append(row)

    split = read_json(diagnostic.SPLIT_PATH)
    validation = {
        sample["image"]: sample
        for sample in split["validation"]
    }

    output_dir = (
        ROOT
        / "results/candidate2/diagnostics"
        / datetime.now().strftime("actual_windows_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda")
    spacing = np.asarray(pipeline.TARGET_SPACING)

    detector, loaded = evaluator.build_detector(pipeline, device)
    del loaded

    original_inferer = detector.inferer
    results = []

    print(
        f"\n검사 대상: {len(selected)}개 결절 / {len(grouped)}개 CT"
    )
    print(f"Output: {output_dir}", flush=True)

    for image_name, rows in grouped.items():
        print(f"\nCT: {image_name}", flush=True)

        sample = validation[image_name]

        (
            _,
            _,
            _,
            volume,
            voxel_boxes,
            _,
        ) = pipeline.load_real_sample(sample)

        targets = []

        for row in rows:
            box = voxel_boxes[row["gt_index"]]
            targets.append({
                "gt_index": row["gt_index"],
                "center": (box[:3] + box[3:]) / 2,
                "radius": row["radius_mm"],
                "centered_patch_score": row["patch_best_matching_score"],
            })

        recorder = WindowRecorder(
            original_inferer,
            targets,
            pipeline.PATCH_SIZE,
        )

        image_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(volume, dtype=np.float32)
            )
            .unsqueeze(0)
            .to(device)
        )

        # 전체 CT 추론을 그대로 실행하면서 실제 입력 window 기록
        detector.inferer = recorder

        try:
            with torch.no_grad():
                full_output = detector(
                    [image_tensor],
                    use_inferer=True,
                )[0]
        finally:
            detector.inferer = original_inferer

        full_checks = {
            target["gt_index"]: measure_prediction(
                full_output,
                detector,
                target["center"],
                spacing,
                target["radius"],
            )
            for target in targets
        }

        del image_tensor, full_output, volume

        window_checks = defaultdict(list)

        # 기록했던 실제 window를 그대로 단독 추론
        for window in recorder.windows:
            with torch.no_grad():
                output = detector(
                    [window["patch"].to(device)],
                    use_inferer=False,
                )[0]

            for target in targets:
                if target["gt_index"] not in window["gt_indices"]:
                    continue

                local_center = target["center"] - window["start"]

                check = measure_prediction(
                    output,
                    detector,
                    local_center,
                    spacing,
                    target["radius"],
                )

                edge_distance = np.minimum(
                    local_center,
                    np.asarray(pipeline.PATCH_SIZE) - local_center,
                ) * spacing

                window_checks[target["gt_index"]].append({
                    "window_start_xyz": window["start"].tolist(),
                    "target_center_in_window_xyz": local_center.tolist(),
                    "nearest_window_edge_mm": float(edge_distance.min()),
                    **check,
                })

            del output

        for target in targets:
            gt_index = target["gt_index"]
            checks = window_checks[gt_index]
            full = full_checks[gt_index]
            successful = sum(check["detected"] for check in checks)

            if not checks:
                diagnosis = "no_window_recorded"
            elif full["detected"]:
                diagnosis = "full_volume_now_detected"
            elif successful:
                diagnosis = "window_detected_but_full_volume_missed"
            else:
                diagnosis = "actual_windows_also_missed"

            result = {
                "image": image_name,
                "gt_index": gt_index,
                "centered_patch_score": target["centered_patch_score"],
                "full_volume": full,
                "containing_window_count": len(checks),
                "detecting_window_count": successful,
                "diagnosis": diagnosis,
                "windows": checks,
            }
            results.append(result)

            print(
                f"GT {gt_index}: "
                f"전체 CT 검출={full['detected']}, "
                f"실제 window 검출={successful}/{len(checks)}\n"
                f"  판정: {diagnosis}",
                flush=True,
            )

        diagnostic.save_json(
            output_dir / "results.json",
            results,
        )

        del recorder

    counts = {}
    for result in results:
        key = result["diagnosis"]
        counts[key] = counts.get(key, 0) + 1

    diagnostic.save_json(
        output_dir / "summary.json",
        {
            "source_evaluation": str(source_dir),
            "source_missed_diagnostic": str(MISSED_DIR),
            "checkpoint": str(diagnostic.CHECKPOINT),
            "num_nodules": len(results),
            "counts": counts,
        },
    )

    print("\n===== 실제 window 비교 결과 =====")
    for key, count in counts.items():
        print(f"{key}: {count}개")

    print(f"상세 결과: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()