"""Diagnostic training on one fixed TRAIN patch; never overwrite source weights."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = ROOT / "results/candidate2/training/best_val_loss.pt"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/monai" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def evaluate(detector, images, boxes, spacing):
    detector.eval()
    with torch.no_grad():
        output = detector(images, use_inferer=False)[0]
    predicted = output[detector.target_box_key].detach().cpu().numpy()
    scores = output[detector.pred_score_key].detach().cpu().numpy()
    order = np.argsort(-scores)
    predicted, scores = predicted[order], scores[order]
    centers = (predicted[:, :3] + predicted[:, 3:]) / 2
    gt_centers = (boxes[:, :3] + boxes[:, 3:]) / 2
    radii = ((boxes[:, 3:] - boxes[:, :3]) * spacing).max(axis=1) / 2
    # Global score order, one-to-one assignment, as in the project evaluator.
    matched = set()
    per_gt = [{"gt_index": i, "detected": False, "score": None, "rank": None}
              for i in range(len(boxes))]
    for rank, (center, score) in enumerate(zip(centers, scores), 1):
        distance = np.linalg.norm((gt_centers - center) * spacing, axis=1)
        candidates = [i for i in range(len(boxes)) if i not in matched and distance[i] <= radii[i]]
        if candidates:
            i = min(candidates, key=lambda j: distance[j])
            matched.add(i)
            per_gt[i].update(detected=True, score=float(score), rank=rank,
                             distance_mm=float(distance[i]))
    return {"detected_gt": len(matched), "total_gt": len(boxes),
            "prediction_count": len(scores), "max_prediction_score": float(scores.max()) if len(scores) else None,
            "per_gt": per_gt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--positive-index", type=int, default=0,
                        help="Index among positive TRAIN scans, not validation scans")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or not np.isfinite(args.lr) or args.lr <= 0 or args.positive_index < 0:
        parser.error("steps/eval-every/lr must be positive; positive-index must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    pipeline = load_script("overfit_pipeline", "19_validate_candidate2_training_pipeline.py")
    trainer = load_script("overfit_trainer", "21_train_candidate2.py")
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    preprocessing = checkpoint["config"]["preprocessing"]
    if (checkpoint["epoch"] != 31 or tuple(preprocessing["patch_size_xyz"]) != (256, 256, 96)
            or not np.allclose(preprocessing["spacing_xyz"], (0.703125, 0.703125, 1.25))):
        raise RuntimeError("Expected the current epoch-31 checkpoint and its preprocessing.")
    pipeline.PATCH_SIZE = tuple(preprocessing["patch_size_xyz"])
    pipeline.TARGET_SPACING = tuple(preprocessing["spacing_xyz"])
    split = json.loads(pipeline.SPLIT_PATH.read_text(encoding="utf-8"))
    positives = [s for s in split["training"] if s.get("box")]
    if args.positive_index >= len(positives):
        raise RuntimeError("positive-index exceeds the positive training set size.")
    sample = positives[args.positive_index]
    forbidden = {s["image"] for key in ("validation", "test") for s in split[key]}
    if sample["image"] in forbidden:
        raise RuntimeError("Training/held-out overlap.")
    _, _, _, volume, boxes, labels = pipeline.load_real_sample(sample)
    # Create ONCE. No augmentation or resampling of the crop during training.
    patch, crop_boxes, crop_labels = pipeline.make_positive_crop(volume, boxes, labels)
    del volume
    device = torch.device("cuda")
    detector = pipeline.build_detector(device)
    detector.network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    detector.set_box_selector_parameters(score_thresh=0.02, topk_candidates_per_level=1000,
                                         nms_thresh=0.22, detections_per_img=300)
    images, targets = pipeline.to_detector_batch(patch, crop_boxes, crop_labels, device)
    parameters = [(name, p) for name, p in detector.network.named_parameters() if p.requires_grad]
    # Track one parameter in the backbone and one in each head, when present.
    tracked = {}
    for group, keywords in (("backbone", ("backbone",)), ("classification", ("classification", "cls")),
                            ("regression", ("regression", "box_reg"))):
        candidates = [(name, p) for name, p in parameters if any(k in name.lower() for k in keywords)]
        if candidates:
            tracked[group] = candidates[-1]
    initial = {group: p.detach().clone() for group, (_, p) in tracked.items()}
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=args.lr, weight_decay=1e-4)
    output = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("fixed_patch_overfit_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {"source_checkpoint": str(SOURCE_CHECKPOINT), "source_epoch": 31,
              "training_image": sample["image"], "positive_index": args.positive_index,
              "patch_size": list(pipeline.PATCH_SIZE), "spacing": list(pipeline.TARGET_SPACING),
              "steps": args.steps, "eval_every": args.eval_every, "learning_rate": args.lr,
              "optimizer": "fresh AdamW", "weight_decay": 1e-4, "gradient_clip_norm": 5.0,
              "batch_size": 1, "augmentation": False, "batchnorm": "train during updates; eval for detection",
              "score_threshold": 0.02, "topk_per_level": 1000, "nms_threshold": 0.22, "final_limit": 300,
              "tracked_parameters": {g: name for g, (name, _) in tracked.items()},
              "note": "Positive-only diagnostic, not a replacement training recipe or generalization metric. Failed memorization within the step budget is not proof of a code bug."}
    save_json(output / "config.json", config)
    np.savez_compressed(output / "fixed_patch.npz", image=patch, boxes=crop_boxes, labels=crop_labels)
    history = [{"step": 0, "evaluation": evaluate(detector, images, crop_boxes, np.asarray(pipeline.TARGET_SPACING))}]
    save_json(output / "history.json", history)
    print(f"Source: {SOURCE_CHECKPOINT}\nTraining CT: {sample['image']}\nOutput: {output}", flush=True)
    print(f"BASELINE: {history[0]['evaluation']}", flush=True)
    for step in range(1, args.steps + 1):
        detector.train()
        optimizer.zero_grad(set_to_none=True)
        loss, cls_loss, box_loss = trainer.compute_loss(detector, images, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 5.0, error_if_nonfinite=True)
        tracked_grads = {g: float(p.grad.norm()) if p.grad is not None else None for g, (_, p) in tracked.items()}
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach()), "cls_loss": float(cls_loss.detach()),
                  "box_loss": float(box_loss.detach()), "gradient_norm_before_clip": float(grad_norm),
                  "tracked_gradient_norm_after_clip": tracked_grads}
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            record["max_abs_change_from_source"] = {g: float((p.detach()-initial[g]).abs().max()) for g, (_, p) in tracked.items()}
            record["evaluation"] = evaluate(detector, images, crop_boxes, np.asarray(pipeline.TARGET_SPACING))
            print(f"step={step} loss={record['loss']:.6f} cls={record['cls_loss']:.6f} "
                  f"box={record['box_loss']:.6f} grad={record['gradient_norm_before_clip']:.6f}\n"
                  f"  changes={record['max_abs_change_from_source']}\n  detection={record['evaluation']}", flush=True)
        history.append(record)
        if "evaluation" in record:
            save_json(output / "history.json", history)
    # Save diagnostic weights only in the new output directory.
    torch.save({"model_state_dict": detector.network.state_dict(), "step": args.steps,
                "diagnostic_only": True, "config": config}, output / "diagnostic_last.pt")
    save_json(output / "summary.json", {"config": config, "baseline": history[0], "final": history[-1]})
    print(f"Finished. Results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
