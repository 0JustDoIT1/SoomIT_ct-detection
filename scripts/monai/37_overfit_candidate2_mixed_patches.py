"""Controlled memorization of four positive and four negative TRAIN patches."""
import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import json
import random

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCORE_POINTS = (0.02, 0.1, 0.25, 0.5, 0.9)


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/monai" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_patch(detector, images, boxes, spacing):
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
    matched = set()
    per_gt = [{"gt_index": i, "detected": False, "score": None, "rank": None} for i in range(len(boxes))]
    is_tp = np.zeros(len(scores), dtype=bool)
    for rank, (center, score) in enumerate(zip(centers, scores), 1):
        distance = np.linalg.norm((gt_centers - center) * spacing, axis=1)
        eligible = [i for i in range(len(boxes)) if i not in matched and distance[i] <= radii[i]]
        if eligible:
            i = min(eligible, key=lambda j: distance[j])
            matched.add(i)
            is_tp[rank - 1] = True
            per_gt[i].update(detected=True, score=float(score), rank=rank, distance_mm=float(distance[i]))
    return {
        "total_gt": len(boxes), "detected_gt": len(matched), "prediction_count": len(scores),
        "max_prediction_score": float(scores.max()) if len(scores) else None,
        "per_gt": per_gt,
        "by_score": {str(t): {"tp": int((is_tp & (scores >= t)).sum()),
                               "fp": int((~is_tp & (scores >= t)).sum())} for t in SCORE_POINTS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps, eval-every and lr must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    helper = load_script("single_patch_helper", "35_overfit_candidate2_fixed_training_patch.py")
    pipeline = helper.load_script("mixed_pipeline", "19_validate_candidate2_training_pipeline.py")
    trainer = helper.load_script("mixed_trainer", "21_train_candidate2.py")
    checkpoint = torch.load(helper.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    preprocessing = checkpoint["config"]["preprocessing"]
    if (checkpoint["epoch"] != 31 or tuple(preprocessing["patch_size_xyz"]) != (256, 256, 96)
            or not np.allclose(preprocessing["spacing_xyz"], (.703125, .703125, 1.25))):
        raise RuntimeError("Expected the original current epoch-31 checkpoint, not diagnostic weights.")
    pipeline.PATCH_SIZE = tuple(preprocessing["patch_size_xyz"])
    pipeline.TARGET_SPACING = tuple(preprocessing["spacing_xyz"])
    spacing = np.asarray(pipeline.TARGET_SPACING)
    split = json.loads(pipeline.SPLIT_PATH.read_text(encoding="utf-8"))
    groups = {"positive": [s for s in split["training"] if s.get("box")][:4],
              "negative": [s for s in split["training"] if not s.get("box")][:4]}
    if any(len(samples) != 4 for samples in groups.values()):
        raise RuntimeError("Four positive and four negative TRAIN CTs are required.")
    selected_ids = [s["image"] for samples in groups.values() for s in samples]
    forbidden = {s["image"] for key in ("validation", "test") for s in split[key]}
    if len(set(selected_ids)) != 8 or set(selected_ids) & forbidden:
        raise RuntimeError("Duplicate or held-out CT in diagnostic selection.")
    output = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("mixed_patch_overfit_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "source_checkpoint": str(helper.SOURCE_CHECKPOINT), "source_epoch": 31,
        "patch_size": list(pipeline.PATCH_SIZE), "spacing": spacing.tolist(),
        "steps": args.steps, "eval_every": args.eval_every, "learning_rate": args.lr,
        "optimizer": "fresh AdamW", "weight_decay": 1e-4, "gradient_clip_norm": 5.0,
        "batch_size": 1, "augmentation": False, "scheduler": None,
        "sampling": "Shuffle all 8 fixed patches once per cycle; each patch exactly once per complete cycle",
        "batchnorm": "train during updates; running statistics in eval mode for detection",
        "score_threshold": .02, "topk_per_level": 1000, "nms_threshold": .22, "final_limit": 300,
        "limitations": ["Training-set memorization only; not validation CPM or a new production model.",
                        "Only postprocessed predictions are counted, capped at 300 per patch.",
                        "Failure within this budget does not prove a code bug. BatchNorm mode remains a possible confound."],
        "patches": [],
    }
    helper.save_json(output / "config.json", config)
    patches = []
    for group, samples in groups.items():
        for index, sample in enumerate(samples):
            random.seed(42 + index)
            _, _, _, volume, boxes, labels = pipeline.load_real_sample(sample)
            if group == "positive":
                patch, boxes, labels = pipeline.make_positive_crop(volume, boxes, labels)
            else:
                patch, boxes, labels = pipeline.make_negative_crop(volume)
            del volume
            if bool(len(boxes)) != (group == "positive"):
                raise RuntimeError("Unexpected patch group.")
            tag = f"{group}_{index}"
            # Exact float32 inputs retained on CPU and saved once; no augmentation.
            np.savez_compressed(output / f"{tag}.npz", image=patch, boxes=boxes, labels=labels)
            patches.append({"id": tag, "group": group, "image": patch, "boxes": boxes, "labels": labels})
            config["patches"].append({"id": tag, "source_image": sample["image"], "gt_count": len(boxes), "seed": 42+index})
            helper.save_json(output / "config.json", config)
            print(f"Prepared {tag}: GT={len(boxes)}", flush=True)
    device = torch.device("cuda")
    detector = pipeline.build_detector(device)
    detector.network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    detector.set_box_selector_parameters(score_thresh=.02, topk_candidates_per_level=1000,
                                         nms_thresh=.22, detections_per_img=300)
    parameters = [p for p in detector.network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    history, evaluations = [], []
    visits = {p["id"]: 0 for p in patches}

    def evaluate_all(step):
        rows = []
        for patch in patches:
            images, _ = pipeline.to_detector_batch(patch["image"], patch["boxes"], patch["labels"], device)
            result = evaluate_patch(detector, images, patch["boxes"], spacing)
            rows.append({"id": patch["id"], "group": patch["group"], **result})
            print(f"eval step={step} {patch['id']}: GT={result['detected_gt']}/{result['total_gt']} "
                  f"max={result['max_prediction_score']} FP@0.5={result['by_score']['0.5']['fp']} "
                  f"GT scores/ranks={[(g['score'], g['rank']) for g in result['per_gt']]}", flush=True)
        positive = [r for r in rows if r["group"] == "positive"]
        negative = [r for r in rows if r["group"] == "negative"]
        item = {"step": step, "patches": rows,
                "positive_detected_gt": sum(r["detected_gt"] for r in positive),
                "positive_total_gt": sum(r["total_gt"] for r in positive),
                "positive_tp_by_score": {str(t): sum(r["by_score"][str(t)]["tp"] for r in positive) for t in SCORE_POINTS},
                "negative_fp_by_score": {str(t): sum(r["by_score"][str(t)]["fp"] for r in negative) for t in SCORE_POINTS}}
        evaluations.append(item)
        helper.save_json(output / "evaluations.json", evaluations)
        print(f"SUMMARY step={step}: positive TP={item['positive_tp_by_score']}; negative FP={item['negative_fp_by_score']}", flush=True)

    print(f"\nSource: {helper.SOURCE_CHECKPOINT}\nOutput: {output}", flush=True)
    evaluate_all(0)
    order = []
    order_rng = random.Random(42)
    interval_start = 0
    for step in range(1, args.steps + 1):
        if (step - 1) % len(patches) == 0:
            order = list(range(len(patches)))
            order_rng.shuffle(order)
        patch = patches[order[(step - 1) % len(patches)]]
        visits[patch["id"]] += 1
        images, targets = pipeline.to_detector_batch(patch["image"], patch["boxes"], patch["labels"], device)
        detector.train()
        optimizer.zero_grad(set_to_none=True)
        loss, cls_loss, box_loss = trainer.compute_loss(detector, images, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "patch_id": patch["id"], "group": patch["group"],
                        "loss": float(loss.detach()), "cls_loss": float(cls_loss.detach()),
                        "box_loss": float(box_loss.detach()), "gradient_norm_before_clip": float(norm)})
        if step % 20 == 0 or step == 1:
            print(f"step={step}/{args.steps} {patch['id']} loss={float(loss.detach()):.6f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            recent = history[interval_start:]
            group_losses = {g: {key: float(np.mean([r[key] for r in recent if r["group"] == g]))
                               for key in ("loss", "cls_loss", "box_loss")}
                            for g in groups if any(r["group"] == g for r in recent)}
            helper.save_json(output / "history.json", history)
            print(f"Interval group loss: {group_losses}", flush=True)
            evaluate_all(step)
            evaluations[-1]["interval_group_losses"] = group_losses
            helper.save_json(output / "evaluations.json", evaluations)
            interval_start = len(history)
    torch.save({"model_state_dict": detector.network.state_dict(), "step": args.steps,
                "diagnostic_only": True, "config": config}, output / "diagnostic_last.pt")
    helper.save_json(output / "summary.json", {"config": config, "visits_per_patch": visits,
                                              "baseline": evaluations[0], "final": evaluations[-1]})
    print(f"Finished. Source weights unchanged. Results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
