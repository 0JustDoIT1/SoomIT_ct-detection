"""Compare geometry and ATSS matching on identical patches, not model accuracy."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.utils.detector_utils import preprocess_images
from monai.networks.nets.resnet import resnet50


ROOT = Path(__file__).resolve().parents[2]


def build_reference(device):
    # Architecture in the local MONAI bundle. Random weights: geometry only.
    backbone = resnet50(spatial_dims=3, n_input_channels=1,
                        conv1_t_stride=[2, 2, 1], conv1_t_size=[7, 7, 7])
    features = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
    network = RetinaNet(spatial_dims=3, num_classes=1, num_anchors=3,
                        feature_extractor=features, size_divisible=[16, 16, 8])
    anchors = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4], base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]])
    detector = RetinaNetDetector(network=network, anchor_generator=anchors, debug=False)
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_target_keys(box_key="box", label_key="label")
    return detector.to(device).eval()


def analyze(detector, image, boxes, spacing):
    padded, _ = preprocess_images([image], detector.spatial_dims, detector.size_divisible)
    with torch.no_grad():
        maps = detector.network.feature_extractor(padded)
        features = list(maps.values())
        anchors = detector.anchor_generator(padded, features)
    locations = [int(np.prod(feature.shape[-3:])) for feature in features]
    targets = [{"box": boxes, "label": torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)}]
    matched = detector.compute_anchor_matched_idxs(anchors, targets, locations)[0]
    max_iou = torch.zeros(len(boxes), device=boxes.device)
    for begin in range(0, len(anchors[0]), 65536):
        overlap = detector.box_overlap_metric(boxes, anchors[0][begin:begin+65536])
        max_iou = torch.maximum(max_iou, overlap.max(dim=1).values)
    levels = []
    offset = 0
    for name, feature, count in zip(maps.keys(), features, locations):
        anchor_count = count * detector.num_anchors_per_loc
        shape = np.asarray(feature.shape[-3:])
        ratio = np.asarray(padded.shape[-3:]) / shape
        level_anchors = anchors[0][offset:offset + anchor_count]
        widths = (level_anchors[:detector.num_anchors_per_loc, 3:] - level_anchors[:detector.num_anchors_per_loc, :3]).cpu().numpy()
        centers = (level_anchors[:, :3] + level_anchors[:, 3:]) / 2
        grid_steps = []
        for axis in range(3):
            values = torch.unique(centers[:, axis], sorted=True)
            grid_steps.append(float((values[1:] - values[:-1]).min()) if len(values) > 1 else None)
        assignments = matched[offset:offset + anchor_count]
        levels.append({
            "name": name, "feature_shape": list(feature.shape),
            "input_to_feature_ratio_xyz": ratio.tolist(),
            "ratio_is_integer": bool(np.allclose(ratio, np.round(ratio))),
            "actual_anchor_grid_step_voxel_xyz": grid_steps,
            "actual_anchor_grid_step_mm_xyz": [None if v is None else v*float(spacing[i]) for i,v in enumerate(grid_steps)],
            "anchor_shapes_mm": (widths * spacing).tolist(), "anchor_count": anchor_count,
            "positive_anchors_per_gt": [int((assignments == i).sum()) for i in range(len(boxes))],
        })
        offset += anchor_count
    return {"input_shape": list(image.shape), "padded_shape": list(padded.shape),
            "size_divisible": detector.size_divisible,
            "gt_sizes_mm": ((boxes[:, 3:] - boxes[:, :3]).cpu().numpy()*spacing).tolist(),
            "max_anchor_iou_per_gt": max_iou.cpu().tolist(),
            "positive_anchors_per_gt": [int((matched == i).sum()) for i in range(len(boxes))],
            "levels": levels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="32번 결과 폴더. 생략하면 가장 최근 폴더 사용")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_audit", ROOT / "scripts/monai/32_audit_candidate2_data_geometry.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if args.data_dir is None:
        candidates = sorted((ROOT / "results/candidate2/diagnostics").glob("data_geometry_*/summary.json"))
        if not candidates:
            raise RuntimeError("Run script 32 first.")
        data_dir = candidates[-1].parent
    else:
        data_dir = args.data_dir
    source = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not source["cases"]:
        raise RuntimeError("No completed patches in script 32 output.")
    output = ROOT / "results/candidate2/diagnostics" / datetime.now().strftime("backbone_anchors_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"source_data": str(data_dir), "note": "Geometry/ATSS comparison only. No learned detector checkpoint or accuracy comparison. Same GT crops for both architectures; this does not reproduce official training.", "cases": [], "odd_size_probes": []}
    device = torch.device("cuda")
    pipeline = helper.load_script("geometry_pipeline", "19_validate_candidate2_training_pipeline.py")
    print(f"Data: {data_dir}\nOutput: {output}", flush=True)
    for model_name in ("med3d_custom_fpn", "monai_bundle_fpn"):
        detector = pipeline.build_detector(device).eval() if model_name == "med3d_custom_fpn" else build_reference(device)
        for case in source["cases"]:
            with np.load(data_dir / case["patch_file"]) as patch:
                image = torch.from_numpy(patch["image"].astype(np.float32)).unsqueeze(0).to(device)
                boxes = torch.from_numpy(patch["boxes"].astype(np.float32)).to(device)
                spacing = patch["spacing"].copy()
            result = analyze(detector, image, boxes, spacing)
            report["cases"].append({"model": model_name, "patch_file": case["patch_file"], **result})
            helper.save(output / "report.json", report)
            print(f"{model_name} {case['patch_file']}: positives={result['positive_anchors_per_gt']}, maxIoU={result['max_anchor_iou_per_gt']}", flush=True)
            del image, boxes
        # A synthetic non-stride-aligned input reveals padding/grid differences.
        image = torch.zeros((1, 100, 100, 100), device=device)
        boxes = torch.tensor([[47., 47., 47., 53., 53., 53.]], device=device)
        report["odd_size_probes"].append({"model": model_name, **analyze(detector, image, boxes, np.ones(3))})
        helper.save(output / "report.json", report)
        del detector, image, boxes
        torch.cuda.empty_cache()
    print(f"Results: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
