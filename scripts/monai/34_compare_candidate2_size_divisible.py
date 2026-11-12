"""Compare full-volume padding alignment with fixed weights and filtering."""
from datetime import datetime
from pathlib import Path
import importlib.util

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/monai" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordWindows:
    def __init__(self, inferer, report):
        self.inferer, self.report = inferer, report

    def __call__(self, images, predictor, *args, **kwargs):
        self.report["padded_input_shape"] = list(images.shape)
        self.report["window_starts_xyz"] = []
        previous = self.inferer.with_coord
        self.inferer.with_coord = True

        def record(patches, coordinates, *p_args, **p_kwargs):
            self.report["window_starts_xyz"].extend(
                [[int(s.start) for s in coordinate[-3:]] for coordinate in coordinates]
            )
            return predictor(patches, *p_args, **p_kwargs)

        try:
            return self.inferer(images, record, *args, **kwargs)
        finally:
            self.inferer.with_coord = previous


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    previous = load_script("limit_check", "29_check_candidate2_final_limit.py")
    raw = load_script("before_filtering", "31_inspect_candidate2_before_filtering.py")
    diagnostic = previous.load_diagnostic()
    pipeline = diagnostic.load_module("alignment_pipeline", "19_validate_candidate2_training_pipeline.py")
    evaluator = diagnostic.load_module("alignment_evaluator", "22_evaluate_candidate2_fold0.py")
    window_summary = previous.read_json(previous.WINDOW_RESULT_DIR / "summary.json")
    source_dir = Path(window_summary["source_evaluation"])
    source = previous.read_json(source_dir / "summary.json")
    if source["dataset"] != "internal_validation" or source["checkpoint_epoch"] != 21:
        raise RuntimeError("Unexpected source evaluation.")
    split = previous.read_json(diagnostic.SPLIT_PATH)
    sample = next(s for s in split["validation"] if s["image"] == previous.TARGET_IMAGE)
    if any(s["image"] == sample["image"] for s in split["test"]):
        raise RuntimeError("Validation/test overlap.")
    checkpoint = torch.load(diagnostic.CHECKPOINT, map_location="cpu", weights_only=False)
    if (checkpoint["epoch"] != 21 or tuple(checkpoint["patch_size"]) != (96, 96, 96)
            or not np.allclose(checkpoint["spacing"], (1, 1, 1))):
        raise RuntimeError("Unexpected checkpoint preprocessing.")
    del checkpoint
    pipeline.PATCH_SIZE = (96, 96, 96)
    pipeline.TARGET_SPACING = (1.0, 1.0, 1.0)
    evaluator.BEST_CHECKPOINT = diagnostic.CHECKPOINT
    evaluator.SCORE_THRESH = source["score_threshold"]
    evaluator.TOPK_CANDIDATES_PER_LEVEL = 10000
    evaluator.DETECTIONS_PER_IMAGE = 50000
    evaluator.NMS_THRESH = 1.0
    evaluator.OVERLAP = source["sliding_window_overlap"]
    evaluator.SW_BATCH_SIZE = 1
    # Obtain the same resampled-coordinate GT used by inference.
    import SimpleITK as sitk
    original = sitk.ReadImage(str(pipeline.resolve_mhd_path(sample["image"])))
    resampled = pipeline.resample_ct(original, pipeline.TARGET_SPACING)
    gt_index = previous.TARGET_GT_INDEX
    center = np.asarray(resampled.TransformPhysicalPointToContinuousIndex(
        tuple(map(float, sample["box"][gt_index][:3]))))
    radius = float(max(sample["box"][gt_index][3:]) / 2)
    del original, resampled
    output = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("size_divisible_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "checkpoint": str(diagnostic.CHECKPOINT), "epoch": 21,
        "image": sample["image"], "gt_index": gt_index, "source_evaluation": str(source_dir),
        "patch_size": [96, 96, 96], "spacing": [1, 1, 1],
        "compared_size_divisible": [4, 32], "score_threshold": evaluator.SCORE_THRESH,
        "topk_candidates_per_level": 10000, "detections_per_image": 50000,
        "nms_threshold": 1.0, "overlap": evaluator.OVERLAP,
        "note": "Padding also changes sliding-window positions and blended outputs. This is not an anchor-only intervention or an accuracy benchmark.",
    }
    diagnostic.save_json(output / "config.json", config)
    reports = []
    print(f"Output: {output}", flush=True)
    device = torch.device("cuda")
    for divisible in (4, 32):
        # Fresh detector avoids retaining any anchor cache across input geometries.
        detector, loaded = evaluator.build_detector(pipeline, device)
        del loaded
        detector.size_divisible = (divisible,) * 3
        detector.network.size_divisible = (divisible,) * 3
        report = {"size_divisible": divisible}
        original_inferer = detector.inferer
        detector.inferer = RecordWindows(original_inferer, report)
        generate = detector.generate_anchors
        postprocess = detector.postprocess_detections

        def capture_geometry(images, heads):
            result = generate(images, heads)
            levels = []
            offset = 0
            for level, feature in enumerate(heads[detector.cls_key]):
                shape = np.asarray(feature.shape[-3:])
                count = int(np.prod(shape)) * detector.num_anchors_per_loc
                anchors = detector.anchors[0][offset:offset + count]
                centers = (anchors[:, :3] + anchors[:, 3:]) / 2
                steps = []
                for axis in range(3):
                    values = torch.unique(centers[:, axis], sorted=True)
                    steps.append(float((values[1:]-values[:-1]).min()) if len(values)>1 else None)
                levels.append({"level": level, "feature_shape_xyz": shape.tolist(),
                               "input_feature_ratio_xyz": (np.asarray(images.shape[-3:])/shape).tolist(),
                               "actual_anchor_step_voxel_xyz": steps, "anchor_count": count})
                offset += count
            report["geometry"] = levels
            return result

        def capture_raw(heads, anchors, sizes, locations, *args, **kwargs):
            report["before_filtering"] = raw.inspect_heads(
                detector, heads, anchors, sizes, locations, center, radius)
            return postprocess(heads, anchors, sizes, locations, *args, **kwargs)

        detector.generate_anchors = capture_geometry
        detector.postprocess_detections = capture_raw
        try:
            print(f"\nsize_divisible={divisible}", flush=True)
            scan = evaluator.infer_one_scan(sample, pipeline, detector, device)
        finally:
            detector.generate_anchors = generate
            detector.postprocess_detections = postprocess
            detector.inferer = original_inferer
        report["after_filtering"] = previous.check_target(scan, gt_index)
        diagnostic.save_json(output / f"predictions_divisible_{divisible}.json", scan)
        reports.append(report)
        diagnostic.save_json(output / "comparison.json", {**config, "reports": reports})
        print(f"Padded input: {report['padded_input_shape']}; windows={len(report['window_starts_xyz'])}", flush=True)
        for geometry, stats in zip(report["geometry"], report["before_filtering"]):
            hit = stats["clipped_valid"]
            print(f"level={geometry['level']} shape={geometry['feature_shape_xyz']} "
                  f"anchor_step={geometry['actual_anchor_step_voxel_xyz']} "
                  f"matching={hit['matching_count']} score={hit['best_matching_score']} "
                  f"rank>={hit['best_matching_rank_lower_bound_in_level']}", flush=True)
        print(f"Final target: {report['after_filtering']}", flush=True)
        del detector, generate, postprocess, original_inferer, scan
        torch.cuda.empty_cache()
    starts_4 = {tuple(x) for x in reports[0]["window_starts_xyz"]}
    starts_32 = {tuple(x) for x in reports[1]["window_starts_xyz"]}
    comparison = {**config, "reports": reports,
                  "shared_window_starts": len(starts_4 & starts_32),
                  "only_divisible_4_window_starts": len(starts_4 - starts_32),
                  "only_divisible_32_window_starts": len(starts_32 - starts_4)}
    diagnostic.save_json(output / "comparison.json", comparison)
    print("\n완료: 검출 변화만으로 CPM 개선이나 원인 해결을 확정하지 않습니다.", flush=True)
    print(f"Results: {output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
