#!/usr/bin/env python3

import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path("/mnt/c/AI/ct-project")

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "monai"
    / "21_train_candidate2.py"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "training_setup"
    / "candidate2_fold0_split.json"
)

# 너무 많이 돌릴 필요 없음.
# positive training case 20개만 우선 진단.
MAX_POSITIVE_CASES = 20

SEED = 42


# =============================================================================
# UTILS
# =============================================================================

def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_module_from_path(
    module_name,
    path,
):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Module load 실패: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def find_training_list(
    split_data,
):
    """
    candidate2_fold0_split.json의
    training list key를 유연하게 찾는다.
    """

    if isinstance(split_data, list):
        return split_data

    if not isinstance(
        split_data,
        dict,
    ):
        raise RuntimeError(
            "Split JSON 구조를 해석할 수 없습니다."
        )

    possible_keys = [
        "training",
        "train",
        "train_data",
        "training_data",
    ]

    for key in possible_keys:

        value = split_data.get(key)

        if isinstance(
            value,
            list,
        ):
            return value

    print(
        "Split keys:",
        list(
            split_data.keys()
        ),
    )

    raise RuntimeError(
        "training list를 찾을 수 없습니다."
    )


def sample_has_gt(
    sample,
):
    boxes = sample.get(
        "box",
        []
    )

    return (
        isinstance(
            boxes,
            list,
        )
        and len(boxes) > 0
    )


# =============================================================================
# MATCH CAPTURE
# =============================================================================

class AnchorMatchCapture:

    def __init__(
        self,
        detector,
    ):
        self.detector = detector

        self.original = (
            detector
            .compute_anchor_matched_idxs
        )

        self.last = None

    def install(self):

        def wrapped(
            anchors,
            targets,
            num_anchor_locs_per_level,
        ):

            matched = self.original(
                anchors,
                targets,
                num_anchor_locs_per_level,
            )

            self.last = {
                "anchors": [
                    x.detach().cpu()
                    for x in anchors
                ],

                "targets": [
                    {
                        k: (
                            v.detach().cpu()
                            if torch.is_tensor(v)
                            else v
                        )
                        for k, v
                        in target.items()
                    }
                    for target in targets
                ],

                "num_anchor_locs_per_level": [
                    int(x)
                    for x
                    in num_anchor_locs_per_level
                ],

                "matched": [
                    x.detach().cpu()
                    for x in matched
                ],
            }

            return matched

        self.detector.compute_anchor_matched_idxs = (
            wrapped
        )

    def restore(self):
        self.detector.compute_anchor_matched_idxs = (
            self.original
        )


# =============================================================================
# ANALYZE ONE MATCH
# =============================================================================

def analyze_match(
    captured,
    detector,
):

    if captured is None:
        raise RuntimeError(
            "Anchor matching 결과를 capture하지 못했습니다."
        )

    if len(
        captured["matched"]
    ) != 1:
        raise RuntimeError(
            "현재 진단은 batch size 1 기준입니다."
        )

    matched = (
        captured[
            "matched"
        ][0]
        .numpy()
        .astype(
            np.int64
        )
    )

    anchors = (
        captured[
            "anchors"
        ][0]
    )

    target = (
        captured[
            "targets"
        ][0]
    )

    gt_boxes = target[
        detector.target_box_key
    ]

    total_anchors = int(
        len(matched)
    )

    positive_mask = (
        matched >= 0
    )

    background_mask = (
        matched == -1
    )

    ignored_mask = (
        matched == -2
    )

    positive = int(
        positive_mask.sum()
    )

    background = int(
        background_mask.sum()
    )

    ignored = int(
        ignored_mask.sum()
    )

    # ---------------------------------------------------------
    # GT별 positive anchor 수
    # ---------------------------------------------------------

    gt_positive_counts = []

    for gt_index in range(
        len(gt_boxes)
    ):

        count = int(
            (
                matched
                == gt_index
            ).sum()
        )

        gt_positive_counts.append(
            count
        )

    # ---------------------------------------------------------
    # Level별 positive anchor 수
    #
    # num_anchor_locs_per_level은 spatial location 수.
    # location당 anchor 개수를 곱해서 실제 anchor 수 계산.
    # ---------------------------------------------------------

    anchors_per_location = (
        detector
        .anchor_generator
        .num_anchors_per_location()
    )

    if len(
        set(
            anchors_per_location
        )
    ) == 1:

        num_anchor_per_loc = int(
            anchors_per_location[0]
        )

        level_anchor_counts = [
            int(loc)
            * num_anchor_per_loc
            for loc
            in captured[
                "num_anchor_locs_per_level"
            ]
        ]

        level_positive_counts = []

        start = 0

        for level_index, count in enumerate(
            level_anchor_counts
        ):

            end = (
                start
                + count
            )

            level_matched = (
                matched[
                    start:end
                ]
            )

            level_positive_counts.append(
                int(
                    (
                        level_matched
                        >= 0
                    ).sum()
                )
            )

            start = end

    else:

        level_anchor_counts = []
        level_positive_counts = []

    # ---------------------------------------------------------
    # GT box sizes
    # StandardMode:
    # [xmin, ymin, zmin, xmax, ymax, zmax]
    # ---------------------------------------------------------

    gt_sizes = []

    for box in gt_boxes:

        box = (
            box
            .numpy()
            .astype(
                np.float64
            )
        )

        if len(box) == 6:

            size = (
                box[3:]
                - box[:3]
            )

            gt_sizes.append(
                [
                    float(v)
                    for v in size
                ]
            )

    return {
        "total_anchors":
            total_anchors,

        "positive":
            positive,

        "background":
            background,

        "ignored":
            ignored,

        "gt_count":
            int(
                len(gt_boxes)
            ),

        "gt_positive_counts":
            gt_positive_counts,

        "gt_sizes":
            gt_sizes,

        "level_anchor_counts":
            level_anchor_counts,

        "level_positive_counts":
            level_positive_counts,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    header(
        "Candidate 2 Positive Anchor Diagnostic"
    )

    print(
        "Project:",
        PROJECT_ROOT,
    )

    print(
        "Samples:",
        MAX_POSITIVE_CASES,
    )

    # -------------------------------------------------------------------------
    # Load training module
    # -------------------------------------------------------------------------

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            TRAIN_SCRIPT
        )

    train_module = (
        load_module_from_path(
            "candidate2_train_module",
            TRAIN_SCRIPT,
        )
    )

    pipeline = (
        train_module
        .load_pipeline_module()
    )

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

    split_data = load_json(
        SPLIT_PATH
    )

    training = find_training_list(
        split_data
    )

    positive_samples = [
        sample
        for sample
        in training
        if sample_has_gt(
            sample
        )
    ]

    print()
    print(
        "Training cases        :",
        len(training),
    )

    print(
        "Positive training cases:",
        len(positive_samples),
    )

    if len(
        positive_samples
    ) == 0:

        raise RuntimeError(
            "Positive training sample이 없습니다."
        )

    selected = (
        positive_samples[
            :MAX_POSITIVE_CASES
        ]
    )

    # -------------------------------------------------------------------------
    # Detector
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    detector = (
        pipeline
        .build_detector(
            device
        )
    )

    detector.train()

    print()
    print(
        "Anchor shapes:"
    )

    try:
        print(
            detector
            .anchor_generator
            .cell_anchors
        )
    except Exception:
        print(
            "  cell_anchors: "
            "direct display unavailable"
        )

    print(
        "Anchors/location:",
        detector
        .anchor_generator
        .num_anchors_per_location(),
    )

    # -------------------------------------------------------------------------
    # Hook matcher
    # -------------------------------------------------------------------------

    capture = AnchorMatchCapture(
        detector
    )

    capture.install()

    results = []

    try:

        for case_index, sample in enumerate(
            selected,
            start=1,
        ):

            # -------------------------------------------------------------
            # 동일한 training preprocessing / positive crop 사용
            # -------------------------------------------------------------

            (
                patch,
                crop_boxes,
                crop_labels,
            ) = (
                train_module
                .prepare_training_sample(
                    sample,
                    pipeline,
                )
            )

            if len(
                crop_boxes
            ) == 0:

                print(
                    f"[{case_index:02d}] "
                    "WARNING: positive sample인데 "
                    "crop GT=0"
                )

                continue

            images, targets = (
                pipeline
                .to_detector_batch(
                    patch,
                    crop_boxes,
                    crop_labels,
                    device,
                )
            )

            # matcher가 실행되도록 training forward
            with torch.no_grad():

                _ = detector(
                    images,
                    targets,
                )

            result = analyze_match(
                capture.last,
                detector,
            )

            result[
                "case_index"
            ] = case_index

            result[
                "image"
            ] = sample.get(
                "image",
                "unknown",
            )

            results.append(
                result
            )

            gt_pos_text = ",".join(
                str(x)
                for x
                in result[
                    "gt_positive_counts"
                ]
            )

            print(
                f"[{case_index:02d}] "
                f"GT={result['gt_count']:2d} | "
                f"anchors={result['total_anchors']:7d} | "
                f"POS={result['positive']:4d} | "
                f"BG={result['background']:7d} | "
                f"IGN={result['ignored']:5d} | "
                f"GT→POS=[{gt_pos_text}]"
            )

            # 메모리 정리
            del images
            del targets

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

    finally:
        capture.restore()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    header(
        "Summary"
    )

    if not results:

        raise RuntimeError(
            "분석 가능한 결과가 없습니다."
        )

    total_gt = sum(
        r["gt_count"]
        for r
        in results
    )

    total_positive = sum(
        r["positive"]
        for r
        in results
    )

    gt_positive_counts = [
        count
        for result
        in results
        for count
        in result[
            "gt_positive_counts"
        ]
    ]

    zero_positive_gt = sum(
        1
        for x
        in gt_positive_counts
        if x == 0
    )

    print(
        f"Cases analyzed      : {len(results)}"
    )

    print(
        f"GT boxes            : {total_gt}"
    )

    print(
        f"Total positive anchor: {total_positive}"
    )

    print(
        f"GT with 0 positives : "
        f"{zero_positive_gt}/{total_gt}"
    )

    if gt_positive_counts:

        arr = np.asarray(
            gt_positive_counts,
            dtype=np.float64,
        )

        print()
        print(
            "Positive anchors / GT"
        )

        print(
            f"Mean   : {arr.mean():.2f}"
        )

        print(
            f"Median : {np.median(arr):.2f}"
        )

        print(
            f"Min    : {arr.min():.0f}"
        )

        print(
            f"Max    : {arr.max():.0f}"
        )

        print(
            f"P25    : "
            f"{np.percentile(arr, 25):.2f}"
        )

        print(
            f"P75    : "
            f"{np.percentile(arr, 75):.2f}"
        )

    # -------------------------------------------------------------------------
    # Level summary
    # -------------------------------------------------------------------------

    if (
        results[0][
            "level_positive_counts"
        ]
    ):

        num_levels = len(
            results[0][
                "level_positive_counts"
            ]
        )

        print()
        print(
            "Positive anchors by FPN level"
        )

        for level in range(
            num_levels
        ):

            positives = sum(
                r[
                    "level_positive_counts"
                ][level]
                for r
                in results
            )

            anchor_count = (
                results[0][
                    "level_anchor_counts"
                ][level]
            )

            print(
                f"Level {level}: "
                f"POS={positives} | "
                f"anchors/sample≈{anchor_count}"
            )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    header(
        "Interpretation"
    )

    zero_ratio = (
        zero_positive_gt
        / max(
            total_gt,
            1,
        )
    )

    median_positive = float(
        np.median(
            gt_positive_counts
        )
    )

    if zero_ratio >= 0.5:

        print(
            "FAIL: 절반 이상의 GT가 positive anchor를 하나도 받지 못했습니다."
        )

        print(
            "→ Anchor shape / ATSS matcher 설정이 1순위 원인입니다."
        )

    elif median_positive <= 1:

        print(
            "WARNING: GT당 positive anchor 수가 매우 적습니다."
        )

        print(
            "→ Anchor shape / ATSS num_candidates / center_in_gt를 "
            "우선 점검해야 합니다."
        )

    else:

        print(
            "PASS: GT에 positive anchor가 정상적으로 할당되고 있습니다."
        )

        print(
            "→ 다음 단계는 classification head / box regression / "
            "gradient 학습 상태를 확인합니다."
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "results"
        / "candidate2"
        / "diagnostics"
        / "positive_anchor_diagnostic.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable = {
        "cases_analyzed":
            len(results),

        "gt_boxes":
            total_gt,

        "total_positive_anchors":
            total_positive,

        "gt_with_zero_positive":
            zero_positive_gt,

        "positive_anchors_per_gt": {
            "mean":
                float(
                    np.mean(
                        gt_positive_counts
                    )
                ),

            "median":
                float(
                    np.median(
                        gt_positive_counts
                    )
                ),

            "min":
                int(
                    np.min(
                        gt_positive_counts
                    )
                ),

            "max":
                int(
                    np.max(
                        gt_positive_counts
                    )
                ),
        },

        "cases":
            results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serializable,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()