"""Audit saved training history and fixed positive/negative patches; no updates."""
import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random

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


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def distribution(values):
    values = values.detach().float()
    if not len(values):
        return {"count": 0}
    q = torch.quantile(values, torch.tensor([0., .5, .9, .99, 1.], device=values.device))
    return dict(zip(("min", "median", "p90", "p99", "max"), q.cpu().tolist()),
                count=len(values), mean=float(values.mean()),
                count_above_0_02=int((values > .02).sum()),
                count_above_0_5=int((values > .5).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-per-group", type=int, default=3)
    parser.add_argument("--history-only", action="store_true", help="Skip GPU patch diagnostics")
    args = parser.parse_args()
    if args.cases_per_group < 1:
        parser.error("cases-per-group must be positive")
    training_dir = ROOT / "results/candidate2/training"
    history = json.loads((training_dir / "history.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "results/candidate2/training_setup/candidate2_fold0_split.json").read_text(encoding="utf-8"))
    if not history:
        raise RuntimeError("Empty training history.")
    output = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("training_dynamics_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    epochs = len(history)
    counts = Counter(len(s.get("box", [])) for s in split["training"])
    timeline = [{"epoch": h["epoch"], "lr_before": h["lr_before_scheduler"],
                 "lr_after": h["lr_after_scheduler"], "train_cls": h["train"]["cls_loss"],
                 "train_box": h["train"]["box_loss"], "val_cls": h["validation"]["cls_loss"],
                 "val_box": h["validation"]["box_loss"], "improved": h["improved"],
                 "epochs_without_improvement": h["epochs_without_improvement"]} for h in history]
    reductions = [h for h in timeline if h["lr_after"] < h["lr_before"]]
    report = {
        "history_source": str(training_dir / "history.json"), "epochs": epochs,
        "best_total_val_loss_epoch": min(history, key=lambda h: h["validation"]["loss"])["epoch"],
        "best_val_cls_epoch": min(history, key=lambda h: h["validation"]["cls_loss"])["epoch"],
        "lr_reductions": reductions, "timeline": timeline,
        "ct_counts_by_gt_count": dict(sorted(counts.items())),
        "target_selection_expectations": [
            {"gt_per_ct": n, "ct_count": count,
             "expected_target_selections_per_gt": epochs / n,
             "probability_never_selected_as_target": (1 - 1 / n) ** epochs}
            for n, count in sorted(counts.items()) if n > 0],
        "limitations": [
            "Historical per-nodule crop selections and positive/negative loss breakdown were not logged.",
            "Selection expectations assume one independent uniform GT choice per positive CT per epoch; not actual exposure counts. Other GTs can share the crop.",
            "Patch measurements below are new fixed train crops evaluated with epoch-31 weights, not historical training measurements.",
            "ATSS score groups include all matched/background anchors BEFORE hard-negative sampling, not the sampler's selected loss subset.",
            "A correlation between LR changes and stalled loss is not proof that scheduling caused the failure.",
        ],
        "patches": [],
    }
    save(output / "report.json", report)
    print(f"Output: {output}\nHistory epochs: {epochs}", flush=True)
    for h in reductions:
        print(f"LR reduction epoch={h['epoch']}: {h['lr_before']} -> {h['lr_after']}; "
              f"train_cls={h['train_cls']:.6f}, val_cls={h['val_cls']:.6f}", flush=True)
    print(f"First/last train classification loss: {timeline[0]['train_cls']:.6f} / {timeline[-1]['train_cls']:.6f}", flush=True)
    if args.history_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("History was saved. GPU diagnostics require CUDA; use --history-only otherwise.")
    pipeline = load_script("dynamics_pipeline", "19_validate_candidate2_training_pipeline.py")
    trainer = load_script("dynamics_trainer", "21_train_candidate2.py")
    overfit = load_script("fixed_patch", "35_overfit_candidate2_fixed_training_patch.py")
    checkpoint = torch.load(training_dir / "best_val_loss.pt", map_location="cpu", weights_only=False)
    preprocessing = checkpoint["config"]["preprocessing"]
    if (checkpoint["epoch"] != 31 or tuple(preprocessing["patch_size_xyz"]) != (256, 256, 96)
            or not np.allclose(preprocessing["spacing_xyz"], (.703125, .703125, 1.25))):
        raise RuntimeError("Unexpected checkpoint.")
    pipeline.PATCH_SIZE = tuple(preprocessing["patch_size_xyz"])
    pipeline.TARGET_SPACING = tuple(preprocessing["spacing_xyz"])
    device = torch.device("cuda")
    detector = pipeline.build_detector(device)
    detector.network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    detector.set_box_selector_parameters(score_thresh=.02, topk_candidates_per_level=1000,
                                         nms_thresh=.22, detections_per_img=300)
    original_loss = detector.compute_loss
    current = {}

    def inspect_loss(heads, targets, anchors, locations):
        matched = detector.compute_anchor_matched_idxs(anchors, targets, locations)[0]
        logits = heads[detector.cls_key][0]
        if logits.shape[-1] != 1:
            raise RuntimeError("Expected a single-class model.")
        scores = logits[:, 0].sigmoid()
        current["matched_positive_scores"] = distribution(scores[matched >= 0])
        current["background_scores"] = distribution(scores[matched == -1])
        current["ignored_anchor_count"] = int((matched == -2).sum())
        current["positive_anchors_per_gt"] = [int((matched == i).sum()) for i in range(len(targets[0]["box"]))]
        return original_loss(heads, targets, anchors, locations)

    detector.compute_loss = inspect_loss
    forbidden = {s["image"] for key in ("validation", "test") for s in split[key]}
    groups = {
        "positive": [s for s in split["training"] if s.get("box")][:args.cases_per_group],
        "negative": [s for s in split["training"] if not s.get("box")][:args.cases_per_group],
    }
    report["probe_config"] = {"checkpoint": str(training_dir / "best_val_loss.pt"), "epoch": 31,
                              "patch_size": list(pipeline.PATCH_SIZE), "spacing": list(pipeline.TARGET_SPACING),
                              "cases_per_group": args.cases_per_group, "seed_base": 42,
                              "loss_mode": "detector train with BatchNorm eval, no_grad; same mode as script 21 validation",
                              "augmentation": False}
    try:
        for group, samples in groups.items():
            for i, sample in enumerate(samples):
                if sample["image"] in forbidden:
                    raise RuntimeError("Training/held-out overlap.")
                random.seed(42+i)
                np.random.seed(42+i)
                torch.manual_seed(42+i)
                _, _, _, volume, boxes, labels = pipeline.load_real_sample(sample)
                if group == "positive":
                    patch, boxes, labels = pipeline.make_positive_crop(volume, boxes, labels)
                else:
                    patch, boxes, labels = pipeline.make_negative_crop(volume)
                del volume
                images, targets = pipeline.to_detector_batch(patch, boxes, labels, device)
                current = {"group": group, "image": sample["image"], "seed": 42+i, "crop_gt_count": len(boxes)}
                detector.train()
                detector.apply(trainer.set_batchnorm_eval)
                with torch.no_grad():
                    loss, cls, box = trainer.compute_loss(detector, images, targets)
                current.update(loss=float(loss), cls_loss=float(cls), box_loss=float(box))
                current["detection"] = overfit.evaluate(detector, images, boxes, np.asarray(pipeline.TARGET_SPACING))
                report["patches"].append(current)
                save(output / "report.json", report)
                print(f"{group}[{i}] cls={float(cls):.6f}, box={float(box):.6f}; "
                      f"positive_scores={current['matched_positive_scores']}; "
                      f"background_scores={current['background_scores']}; detection={current['detection']}", flush=True)
        report["group_means"] = {
            group: {key: float(np.mean([r[key] for r in report["patches"] if r["group"] == group]))
                    for key in ("loss", "cls_loss", "box_loss")}
            for group in groups if any(r["group"] == group for r in report["patches"])}
        save(output / "report.json", report)
    finally:
        detector.compute_loss = original_loss
    print(f"Finished; no optimizer steps or checkpoint writes. Results: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
