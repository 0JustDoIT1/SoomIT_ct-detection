"""Inspect all decoded boxes before score/top-k/NMS filtering; no training."""
from datetime import datetime
import importlib.util
from pathlib import Path

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


def inspect_heads(detector, heads, anchors, image_sizes, locations, center, radius):
    """Decode every anchor in chunks; do not apply any candidate filters.

    Report both raw and image-clipped boxes. Ranks use strictly higher scores,
    so ties at a top-k boundary must be interpreted separately.
    """
    logits = heads[detector.cls_key][0].detach()
    regression = heads[detector.box_reg_key][0].detach()
    if logits.shape[-1] != 1:
        raise RuntimeError("This diagnostic requires the single-class model.")
    results = []
    offset = 0
    for level, locations_in_level in enumerate(locations):
        count = int(locations_in_level) * detector.num_anchors_per_loc
        scores = logits[offset:offset + count, 0].sigmoid()
        device = regression.device
        target = torch.as_tensor(center, dtype=torch.float32, device=device)
        size = torch.as_tensor(image_sizes[0], dtype=torch.float32, device=device)
        states = {
            key: {"matching_count": 0, "best_matching_score": None,
                  "nearest_distance_mm": None, "nearest_box_score": None}
            for key in ("raw", "clipped_valid")
        }
        for begin in range(0, count, 65536):
            end = min(begin + 65536, count)
            boxes = detector.box_coder.decode_single(
                regression[offset + begin:offset + end].float(),
                anchors[0][offset + begin:offset + end],
            ).reshape(-1, 6).to(logits.dtype)
            chunk_scores = scores[begin:end]
            for key in states:
                candidate = boxes
                if key == "clipped_valid":
                    candidate = torch.cat((
                        boxes[:, :3].clamp(min=0).minimum(size),
                        boxes[:, 3:].clamp(min=0).minimum(size),
                    ), dim=1)
                valid = torch.isfinite(candidate).all(dim=1) & torch.isfinite(chunk_scores)
                valid &= (candidate[:, 3:] > candidate[:, :3]).all(dim=1)
                distance = torch.linalg.vector_norm(
                    (candidate[:, :3] + candidate[:, 3:]) / 2 - target, dim=1
                )  # Model spacing is verified as 1 mm on every axis.
                state = states[key]
                if valid.any():
                    masked = distance.masked_fill(~valid, float("inf"))
                    nearest_index = int(masked.argmin())
                    nearest = float(masked[nearest_index])
                    if state["nearest_distance_mm"] is None or nearest < state["nearest_distance_mm"]:
                        state["nearest_distance_mm"] = nearest
                        state["nearest_box_score"] = float(chunk_scores[nearest_index])
                hit = valid & (distance <= radius)
                state["matching_count"] += int(hit.sum())
                if hit.any():
                    best = float(chunk_scores[hit].max())
                    if state["best_matching_score"] is None or best > state["best_matching_score"]:
                        state["best_matching_score"] = best
                        state["best_matching_box_xyzxyz"] = candidate[hit][chunk_scores[hit].argmax()].cpu().tolist()
        for state in states.values():
            best = state["best_matching_score"]
            state["best_matching_rank_lower_bound_in_level"] = (
                int((scores > best).sum()) + 1 if best is not None else None
            )
            state["equal_score_count_in_level"] = int((scores == best).sum()) if best is not None else None
        results.append({"level": level, "anchor_count": count, **states})
        offset += count
    if offset != len(logits):
        raise RuntimeError("Anchor/level count mismatch.")
    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    previous = load_script("previous_limit", "29_check_candidate2_final_limit.py")
    window_module = load_script("actual_windows", "26_diagnose_candidate2_actual_windows.py")
    diagnostic = previous.load_diagnostic()
    pipeline = diagnostic.load_module("raw_pipeline", "19_validate_candidate2_training_pipeline.py")
    evaluator = diagnostic.load_module("raw_evaluator", "22_evaluate_candidate2_fold0.py")
    window_summary = previous.read_json(previous.WINDOW_RESULT_DIR / "summary.json")
    source = previous.read_json(Path(window_summary["source_evaluation"]) / "summary.json")
    if source["dataset"] != "internal_validation" or source["checkpoint_epoch"] != 21:
        raise RuntimeError("Unexpected source evaluation.")
    split = previous.read_json(diagnostic.SPLIT_PATH)
    sample = next(s for s in split["validation"] if s["image"] == previous.TARGET_IMAGE)
    if any(s["image"] == previous.TARGET_IMAGE for s in split["test"]):
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
    output_dir = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("before_filtering_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    diagnostic.save_json(output_dir / "config.json", {
        "checkpoint": str(diagnostic.CHECKPOINT), "epoch": 21,
        "image": previous.TARGET_IMAGE, "gt_index": previous.TARGET_GT_INDEX,
        "patch_size": [96, 96, 96], "spacing": [1, 1, 1],
        "score_threshold": evaluator.SCORE_THRESH, "topk_per_level": 10000,
        "final_limit": 50000, "nms_threshold": 1.0, "overlap": evaluator.OVERLAP,
        "note": "Raw box statistics precede all candidate filtering. Matching is target-specific, not CPM.",
    })
    print(f"Output: {output_dir}", flush=True)
    device = torch.device("cuda")
    detector, loaded = evaluator.build_detector(pipeline, device)
    del loaded
    _, _, _, volume, boxes, _ = pipeline.load_real_sample(sample)
    gt_index = previous.TARGET_GT_INDEX
    center = (boxes[gt_index, :3] + boxes[gt_index, 3:]) / 2
    radius = max(sample["box"][gt_index][3:]) / 2
    target = {"gt_index": gt_index, "center": center}
    original_inferer = detector.inferer
    recorder = window_module.WindowRecorder(original_inferer, [target], pipeline.PATCH_SIZE)
    original_postprocess = detector.postprocess_detections
    reports = []
    context = {"name": "full_volume", "center": center, "start": [0, 0, 0]}

    def inspect_then_postprocess(heads, anchors, image_sizes, locations, *args, **kwargs):
        levels = inspect_heads(detector, heads, anchors, image_sizes, locations, context["center"], radius)
        report = {"name": context["name"], "start_xyz": list(context["start"]),
                  "target_center_xyz": context["center"].tolist(), "radius_mm": radius, "levels": levels}
        reports.append(report)
        diagnostic.save_json(output_dir / "reports.json", reports)
        for row in levels:
            state = row["clipped_valid"]
            print(f"{context['name']} level={row['level']} matching={state['matching_count']} "
                  f"best_score={state['best_matching_score']} "
                  f"level_rank>={state['best_matching_rank_lower_bound_in_level']} "
                  f"nearest_mm={state['nearest_distance_mm']}", flush=True)
        output = original_postprocess(heads, anchors, image_sizes, locations, *args, **kwargs)
        report["after_filtering"] = window_module.measure_prediction(
            output[0], detector, context["center"], np.ones(3), radius
        )
        diagnostic.save_json(output_dir / "reports.json", reports)
        return output

    detector.postprocess_detections = inspect_then_postprocess
    try:
        detector.inferer = recorder
        tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            detector([tensor], use_inferer=True)
        del tensor, volume
        detector.inferer = original_inferer
        if not recorder.windows:
            raise RuntimeError("No actual windows containing the GT were recorded.")
        for index, window in enumerate(recorder.windows):
            context.update(name=f"window_{index}", center=center-window["start"], start=window["start"].tolist())
            with torch.no_grad():
                detector([window["patch"].to(device)], use_inferer=False)
    finally:
        detector.inferer = original_inferer
        detector.postprocess_detections = original_postprocess
    print(f"\nFinished. Full volume + {len(recorder.windows)} actual windows.", flush=True)
    print(f"Results: {output_dir / 'reports.json'}", flush=True)


if __name__ == "__main__":
    main()
