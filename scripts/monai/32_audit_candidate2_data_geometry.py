"""Read-only source-data audit and visual transform checks. No model inference."""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
REGIMES = {
    "previous_96_1mm": ((96, 96, 96), (1.0, 1.0, 1.0)),
    "current_256_anisotropic": ((256, 256, 96), (0.703125, 0.703125, 1.25)),
}


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/monai" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def plot_stages(path, stages):
    fig, axes = plt.subplots(len(stages), 3, figsize=(12, 4 * len(stages)))
    for row, (name, volume, boxes, spacing) in enumerate(stages):
        target = boxes[0]
        center = (target[:3] + target[3:]) / 2
        for col, (u, v, sliced) in enumerate(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
            index = int(np.clip(np.rint(center[sliced]), 0, volume.shape[sliced] - 1))
            plane = np.take(volume, index, axis=sliced)
            ax = axes[row, col]
            ax.imshow(plane.T, origin="lower", cmap="gray", vmin=0, vmax=1,
                      aspect=spacing[v] / spacing[u])
            for box in boxes:
                if box[sliced] <= index <= box[sliced + 3]:
                    ax.add_patch(Rectangle((box[u], box[v]), box[u + 3] - box[u],
                                           box[v + 3] - box[v], fill=False, edgecolor="red", linewidth=1))
            ax.set_title(f"{name}: axes {u}/{v}, slice {index}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    pipeline = load_script("audit_pipeline", "19_validate_candidate2_training_pipeline.py")
    trainer = load_script("audit_trainer", "21_train_candidate2.py")
    split = json.loads(pipeline.SPLIT_PATH.read_text(encoding="utf-8"))
    output = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("data_geometry_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    # Test image pixels are not used. Only split IDs are checked for overlap.
    id_sets = {key: {x["image"] for x in split[key]} for key in ("training", "validation", "test")}
    overlaps = {f"{a}/{b}": sorted(id_sets[a] & id_sets[b]) for a, b in
                (("training", "validation"), ("training", "test"), ("validation", "test"))}
    summary = {"overlaps": overlaps, "issues": [], "splits": {}, "cases": []}
    selected = [("validation", x) for x in split["validation"] if x.get("box")][:3]
    reversed_case = None
    for key in ("training", "validation"):
        directions = {}
        for sample in split[key]:
            boxes = np.asarray(sample.get("box", []), dtype=float).reshape(-1, 6)
            labels = sample.get("label", [])
            if len(labels) != len(boxes) or any(label != 0 for label in labels):
                summary["issues"].append({"image": sample["image"], "issue": "labels"})
            if not np.isfinite(boxes).all() or (len(boxes) and np.any(boxes[:, 3:] <= 0)):
                summary["issues"].append({"image": sample["image"], "issue": "invalid_boxes"})
                continue
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(pipeline.resolve_mhd_path(sample["image"])))
            reader.ReadImageInformation()
            direction = np.asarray(reader.GetDirection()).reshape(3, 3)
            origin = np.asarray(reader.GetOrigin())
            spacing = np.asarray(reader.GetSpacing())
            direction_key = str(tuple(np.round(direction.ravel(), 6)))
            directions[direction_key] = directions.get(direction_key, 0) + 1
            if len(boxes):
                centers = np.linalg.solve(direction, (boxes[:, :3] - origin).T).T / spacing
                if np.any((centers < 0) | (centers > np.asarray(reader.GetSize()) - 1)):
                    summary["issues"].append({"image": sample["image"], "issue": "center_outside_image"})
                if reversed_case is None and (direction[0, 0] < 0 or direction[1, 1] < 0):
                    reversed_case = (key, sample)
        summary["splits"][key] = {"ct_count": len(split[key]), "directions": directions}
    if reversed_case and all(x[1]["image"] != reversed_case[1]["image"] for x in selected):
        selected.append(reversed_case)
    save(output / "summary.json", summary)

    for case_index, (split_name, sample) in enumerate(selected):
        original = sitk.ReadImage(str(pipeline.resolve_mhd_path(sample["image"])))
        original_xyz = pipeline.normalize_ct(sitk.GetArrayFromImage(original).transpose(2, 1, 0))
        original_boxes = pipeline.physical_boxes_to_voxel_xyzxyz(original, sample["box"])
        for regime, (patch_size, spacing) in REGIMES.items():
            pipeline.PATCH_SIZE, pipeline.TARGET_SPACING = patch_size, spacing
            resampled = pipeline.resample_ct(original, spacing)
            volume = pipeline.normalize_ct(sitk.GetArrayFromImage(resampled).transpose(2, 1, 0))
            boxes = pipeline.physical_boxes_to_voxel_xyzxyz(resampled, sample["box"])
            labels = np.asarray(sample["label"], dtype=np.int64)
            reconstructed = np.asarray([resampled.TransformContinuousIndexToPhysicalPoint(tuple(map(float, (b[:3]+b[3:])/2))) for b in boxes])
            roundtrip = float(np.max(np.linalg.norm(reconstructed - np.asarray(sample["box"])[:, :3], axis=1)))
            random.seed(42)
            patch, crop_boxes, crop_labels = pipeline.make_positive_crop(volume, boxes, labels)
            flipped, flipped_boxes = trainer.flip_patch_and_boxes(patch, crop_boxes, 0)
            restored, restored_boxes = trainer.flip_patch_and_boxes(flipped, flipped_boxes, 0)
            tag = f"case{case_index:02d}_{regime}"
            np.savez_compressed(output / f"{tag}.npz", image=patch.astype(np.float16),
                                boxes=crop_boxes, labels=crop_labels, spacing=np.asarray(spacing))
            plot_stages(output / f"{tag}.png", [
                ("original", original_xyz, original_boxes, original.GetSpacing()),
                ("resampled", volume, boxes, spacing),
                ("positive crop (current code)", patch, crop_boxes, spacing),
                ("flip axis 0 (current code)", flipped, flipped_boxes, spacing),
            ])
            record = {"image": sample["image"], "split": split_name, "regime": regime,
                      "patch_file": f"{tag}.npz", "figure": f"{tag}.png",
                      "roundtrip_max_mm": roundtrip,
                      "flip_twice_image_equal": bool(np.array_equal(patch, restored)),
                      "flip_twice_boxes_equal": bool(np.allclose(crop_boxes, restored_boxes)),
                      "crop_gt_count": len(crop_boxes),
                      "note": "Current 19/21 transforms under two spacings; not a reconstruction of historical cache creation. Visual inspection is still required."}
            summary["cases"].append(record)
            save(output / "summary.json", summary)
            print(f"{tag}: roundtrip={roundtrip:.6f} mm; GT={len(crop_boxes)}", flush=True)
    print(f"Results: {output / 'summary.json'}\nInspect PNG overlays before judging label correctness.", flush=True)


if __name__ == "__main__":
    main()
