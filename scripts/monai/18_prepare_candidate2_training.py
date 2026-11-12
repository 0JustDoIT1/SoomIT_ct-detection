from pathlib import Path
import json
import random


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# MONAI official fold0
#
# Positive LUNA16 detection cases
# ------------------------------------------------------------

SOURCE_OFFICIAL_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "LUNA16"
    / "monai_official_split"
    / "LUNA16_datasplit"
    / "mhd_original"
    / "dataset_fold0.json"
)


# ------------------------------------------------------------
# 기존 experiment split
#
# 여기에서 box == [] 인 negative CT만 가져옴
# ------------------------------------------------------------

NEGATIVE_SOURCE_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "LUNA16"
    / "experiment_split"
    / "train_val_test.json"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate2"
    / "training_setup"
)

OUTPUT_SPLIT = (
    OUTPUT_DIR
    / "candidate2_fold0_split.json"
)

OUTPUT_CONFIG = (
    OUTPUT_DIR
    / "candidate2_training_config.json"
)


# ============================================================
# Fixed official fold0
# ============================================================

EXPECTED_FOLD0_TRAIN = 534

EXPECTED_FINAL_TEST = 67

EXPECTED_FINAL_GT = 112


# ============================================================
# Positive split
#
# 기존 Candidate 2 정책 유지
# ============================================================

POSITIVE_TRAIN_COUNT = 507

POSITIVE_INTERNAL_VAL_COUNT = 27


# ============================================================
# Negative split
#
# official training positive split과 동일한 약 95/5 비율 적용
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Candidate 2 configuration
# ============================================================

CONFIG = {

    "candidate": 2,

    "model": (
        "Med3D pretrained ResNet50 "
        "+ Custom Multi-scale FPN "
        "+ MONAI 3D RetinaNet"
    ),

    "backbone": (
        "MedicalNet ResNet50"
    ),

    "pretrained_weight": (
        "weights/med3d/"
        "resnet_50_23dataset.pth"
    ),

    "data_source": {
        "positive": (
            "MONAI official LUNA16 "
            "fold0 mhd_original"
        ),

        "negative": (
            "LUNA16 experiment_split "
            "box-empty scans"
        ),

        "final_test": (
            "MONAI official LUNA16 "
            "fold0 validation"
        ),
    },

    "preprocessing": {

        "spacing_xyz": [
            0.703125,
            0.703125,
            1.25,
        ],

        "patch_size_xyz": [
            256,
            256,
            96,
        ],

        "intensity": {
            "min_hu": -1024.0,
            "max_hu": 300.0,
            "output_min": 0.0,
            "output_max": 1.0,
        },
    },

    "sampling": {
        "positive_negative_training": True,
        "random_seed": RANDOM_SEED,
    },

    "final_test_used_during_training": False,
}


# ============================================================
# Helpers
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"파일 없음:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(
    path,
    obj,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


def get_image_id(
    item,
):

    image = str(
        item.get(
            "image",
            ""
        )
    )

    return Path(
        image
    ).name


def has_box(
    item,
):

    boxes = item.get(
        "box",
        []
    )

    return (
        boxes is not None
        and len(boxes) > 0
    )


def count_boxes(
    items,
):

    return sum(
        len(
            item.get(
                "box",
                []
            )
        )
        for item in items
    )


def count_positive_negative(
    items,
):

    positive = sum(
        1
        for item in items
        if has_box(item)
    )

    negative = (
        len(items)
        - positive
    )

    return (
        positive,
        negative,
    )


def validate_unique_ids(
    items,
    name,
):

    ids = [
        get_image_id(
            item
        )
        for item in items
    ]

    empty = [
        value
        for value in ids
        if not value
    ]

    duplicate_count = (
        len(ids)
        - len(
            set(ids)
        )
    )

    print(
        f"{name} empty image IDs:",
        len(empty),
    )

    print(
        f"{name} duplicates:",
        duplicate_count,
    )

    if empty:

        raise RuntimeError(
            f"{name}: 빈 image ID 발견"
        )

    if duplicate_count != 0:

        raise RuntimeError(
            f"{name}: duplicate image 발견"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "Candidate 2 - "
        "Candidate 1 Aligned Training Split Preparation"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # 1. Official fold0
    # ========================================================

    print()
    print(
        "[1] Load MONAI official fold0 - MHD original"
    )

    print(
        SOURCE_OFFICIAL_SPLIT
    )


    official = load_json(
        SOURCE_OFFICIAL_SPLIT
    )


    if "training" not in official:

        raise RuntimeError(
            "Official split에 training 없음"
        )

    if "validation" not in official:

        raise RuntimeError(
            "Official split에 validation 없음"
        )


    fold0_training = official[
        "training"
    ]

    final_test = official[
        "validation"
    ]


    print(
        "Official training:",
        len(
            fold0_training
        ),
    )

    print(
        "Official validation:",
        len(
            final_test
        ),
    )


    if (
        len(
            fold0_training
        )
        != EXPECTED_FOLD0_TRAIN
    ):

        raise RuntimeError(
            "Official training count 이상: "
            f"{len(fold0_training)} "
            f"!= {EXPECTED_FOLD0_TRAIN}"
        )


    if (
        len(
            final_test
        )
        != EXPECTED_FINAL_TEST
    ):

        raise RuntimeError(
            "Official final test count 이상: "
            f"{len(final_test)} "
            f"!= {EXPECTED_FINAL_TEST}"
        )


    print(
        "✅ Official fold0 count PASS"
    )


    # ========================================================
    # 2. Check official positive structure
    # ========================================================

    print()
    print(
        "[2] Validate official positive cases"
    )


    official_train_pos, official_train_neg = (
        count_positive_negative(
            fold0_training
        )
    )

    final_pos, final_neg = (
        count_positive_negative(
            final_test
        )
    )


    print(
        "Official training positive:",
        official_train_pos,
    )

    print(
        "Official training negative:",
        official_train_neg,
    )

    print(
        "Final test positive:",
        final_pos,
    )

    print(
        "Final test negative:",
        final_neg,
    )


    if official_train_neg != 0:

        print(
            "⚠️ Official training에 "
            "negative case가 포함되어 있음"
        )


    final_gt = count_boxes(
        final_test
    )


    print(
        "Final test GT nodules:",
        final_gt,
    )


    if final_gt != EXPECTED_FINAL_GT:

        raise RuntimeError(
            "Final test GT count 이상: "
            f"{final_gt} "
            f"!= {EXPECTED_FINAL_GT}"
        )


    print(
        "✅ Final 67 CT / 112 GT 확인"
    )


    # ========================================================
    # 3. Positive internal split
    # ========================================================

    print()
    print(
        "[3] Build positive train / internal validation"
    )


    positive_train = fold0_training[
        :POSITIVE_TRAIN_COUNT
    ]

    positive_internal_val = (
        fold0_training[
            POSITIVE_TRAIN_COUNT:
        ]
    )


    print(
        "Positive Train:",
        len(
            positive_train
        ),
    )

    print(
        "Positive Internal Val:",
        len(
            positive_internal_val
        ),
    )


    if (
        len(
            positive_train
        )
        != POSITIVE_TRAIN_COUNT
    ):

        raise RuntimeError(
            "Positive train count 이상"
        )


    if (
        len(
            positive_internal_val
        )
        != POSITIVE_INTERNAL_VAL_COUNT
    ):

        raise RuntimeError(
            "Positive internal val count 이상"
        )


    print(
        "✅ Positive 507 / 27 split PASS"
    )


    # ========================================================
    # 4. Load negative source
    # ========================================================

    print()
    print(
        "[4] Collect negative CT pool"
    )

    print(
        NEGATIVE_SOURCE_SPLIT
    )


    negative_source = load_json(
        NEGATIVE_SOURCE_SPLIT
    )


    all_negative_source_items = []


    for key in [
        "training",
        "validation",
        "test",
    ]:

        if key not in negative_source:

            raise RuntimeError(
                f"Negative source에 {key} 없음"
            )


        all_negative_source_items.extend(
            negative_source[
                key
            ]
        )


    negative_pool = [
        item
        for item
        in all_negative_source_items
        if not has_box(
            item
        )
    ]


    print(
        "Experiment split total:",
        len(
            all_negative_source_items
        ),
    )

    print(
        "Negative candidates:",
        len(
            negative_pool
        ),
    )


    if len(
        negative_pool
    ) == 0:

        raise RuntimeError(
            "Negative CT를 찾지 못함"
        )


    # ========================================================
    # 5. Deduplicate negative pool
    # ========================================================

    print()
    print(
        "[5] Deduplicate negative pool"
    )


    negative_by_id = {}


    for item in negative_pool:

        image_id = get_image_id(
            item
        )

        if not image_id:

            continue

        if image_id not in (
            negative_by_id
        ):

            negative_by_id[
                image_id
            ] = item


    negative_pool = list(
        negative_by_id.values()
    )


    print(
        "Unique negative CT:",
        len(
            negative_pool
        ),
    )


    # ========================================================
    # 6. Final test leakage removal
    # ========================================================

    print()
    print(
        "[6] Remove final-test leakage from negative pool"
    )


    final_test_ids = {
        get_image_id(
            item
        )
        for item in final_test
    }


    before = len(
        negative_pool
    )


    negative_pool = [
        item
        for item in negative_pool
        if get_image_id(
            item
        )
        not in final_test_ids
    ]


    removed = (
        before
        - len(
            negative_pool
        )
    )


    print(
        "Removed due to final-test overlap:",
        removed,
    )

    print(
        "Remaining negative CT:",
        len(
            negative_pool
        ),
    )


    # ========================================================
    # 7. Negative train / val split
    #
    # Positive와 동일한 507 / 534 비율
    # ========================================================

    print()
    print(
        "[7] Build negative train / internal validation split"
    )


    random_generator = random.Random(
        RANDOM_SEED
    )


    random_generator.shuffle(
        negative_pool
    )


    train_ratio = (
        POSITIVE_TRAIN_COUNT
        / EXPECTED_FOLD0_TRAIN
    )


    negative_train_count = int(
        round(
            len(
                negative_pool
            )
            * train_ratio
        )
    )


    negative_train_count = min(
        negative_train_count,
        len(
            negative_pool
        ),
    )


    negative_train = (
        negative_pool[
            :negative_train_count
        ]
    )


    negative_internal_val = (
        negative_pool[
            negative_train_count:
        ]
    )


    print(
        "Negative Train:",
        len(
            negative_train
        ),
    )

    print(
        "Negative Internal Val:",
        len(
            negative_internal_val
        ),
    )


    if len(
        negative_train
    ) == 0:

        raise RuntimeError(
            "Negative train 없음"
        )


    if len(
        negative_internal_val
    ) == 0:

        raise RuntimeError(
            "Negative internal val 없음"
        )


    print(
        "✅ Negative split PASS"
    )


    # ========================================================
    # 8. Combine positive + negative
    # ========================================================

    print()
    print(
        "[8] Combine training datasets"
    )


    train = (
        positive_train
        + negative_train
    )


    internal_val = (
        positive_internal_val
        + negative_internal_val
    )


    # --------------------------------------------------------
    # Shuffle combined datasets
    #
    # Positive → Negative 순서 그대로 두지 않음
    # --------------------------------------------------------

    random_generator.shuffle(
        train
    )

    random_generator.shuffle(
        internal_val
    )


    train_pos, train_neg = (
        count_positive_negative(
            train
        )
    )


    val_pos, val_neg = (
        count_positive_negative(
            internal_val
        )
    )


    print()
    print(
        "TRAIN"
    )

    print(
        " Total:",
        len(
            train
        ),
    )

    print(
        " Positive:",
        train_pos,
    )

    print(
        " Negative:",
        train_neg,
    )


    print()
    print(
        "INTERNAL VALIDATION"
    )

    print(
        " Total:",
        len(
            internal_val
        ),
    )

    print(
        " Positive:",
        val_pos,
    )

    print(
        " Negative:",
        val_neg,
    )


    print()
    print(
        "FINAL TEST"
    )

    print(
        " Total:",
        len(
            final_test
        ),
    )

    print(
        " GT nodules:",
        final_gt,
    )


    # ========================================================
    # 9. Image ID validation
    # ========================================================

    print()
    print(
        "[9] Image ID validation"
    )


    validate_unique_ids(
        train,
        "Train",
    )

    validate_unique_ids(
        internal_val,
        "Internal Val",
    )

    validate_unique_ids(
        final_test,
        "Final Test",
    )


    print(
        "✅ Image ID validation PASS"
    )


    # ========================================================
    # 10. Leakage check
    # ========================================================

    print()
    print(
        "[10] Leakage check"
    )


    train_ids = {
        get_image_id(
            item
        )
        for item in train
    }


    val_ids = {
        get_image_id(
            item
        )
        for item in internal_val
    }


    test_ids = {
        get_image_id(
            item
        )
        for item in final_test
    }


    train_val_overlap = (
        train_ids
        & val_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    val_test_overlap = (
        val_ids
        & test_ids
    )


    print(
        "Train ∩ Internal Val:",
        len(
            train_val_overlap
        ),
    )

    print(
        "Train ∩ Final Test:",
        len(
            train_test_overlap
        ),
    )

    print(
        "Internal Val ∩ Final Test:",
        len(
            val_test_overlap
        ),
    )


    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):

        raise RuntimeError(
            "Data leakage 발견"
        )


    print(
        "✅ Data leakage 없음"
    )


    # ========================================================
    # 11. Annotation validation
    # ========================================================

    print()
    print(
        "[11] Annotation structure validation"
    )


    all_items = (
        train
        + internal_val
        + final_test
    )


    missing_image = 0
    missing_box = 0
    missing_label = 0

    invalid_box = 0

    box_label_mismatch = 0


    for item in all_items:

        if not item.get(
            "image"
        ):

            missing_image += 1


        if "box" not in item:

            missing_box += 1

            boxes = []

        else:

            boxes = item[
                "box"
            ]


        if "label" not in item:

            missing_label += 1

            labels = []

        else:

            labels = item[
                "label"
            ]


        for box in boxes:

            if len(
                box
            ) != 6:

                invalid_box += 1


        if len(
            boxes
        ) != len(
            labels
        ):

            box_label_mismatch += 1


    print(
        "Missing image:",
        missing_image,
    )

    print(
        "Missing box:",
        missing_box,
    )

    print(
        "Missing label:",
        missing_label,
    )

    print(
        "Invalid bbox:",
        invalid_box,
    )

    print(
        "Box / label mismatch:",
        box_label_mismatch,
    )


    if any(
        [
            missing_image,
            missing_box,
            missing_label,
            invalid_box,
            box_label_mismatch,
        ]
    ):

        raise RuntimeError(
            "Annotation structure 이상"
        )


    print(
        "✅ Annotation structure PASS"
    )


    # ========================================================
    # 12. Save split
    # ========================================================

    print()
    print(
        "[12] Save Candidate 2 aligned split"
    )


    candidate2_split = {

        "source": (
            "MONAI official fold0 positive "
            "+ LUNA16 negative CT"
        ),

        "training": (
            train
        ),

        "validation": (
            internal_val
        ),

        "test": (
            final_test
        ),
    }


    save_json(
        OUTPUT_SPLIT,
        candidate2_split,
    )


    print(
        OUTPUT_SPLIT
    )


    # ========================================================
    # 13. Save config
    # ========================================================

    print()
    print(
        "[13] Save Candidate 2 config"
    )


    config = dict(
        CONFIG
    )


    config[
        "source_official_split"
    ] = str(
        SOURCE_OFFICIAL_SPLIT
    )


    config[
        "negative_source_split"
    ] = str(
        NEGATIVE_SOURCE_SPLIT
    )


    config[
        "candidate2_split"
    ] = str(
        OUTPUT_SPLIT
    )


    config[
        "counts"
    ] = {

        "training": {
            "total": len(
                train
            ),

            "positive": (
                train_pos
            ),

            "negative": (
                train_neg
            ),
        },

        "validation": {
            "total": len(
                internal_val
            ),

            "positive": (
                val_pos
            ),

            "negative": (
                val_neg
            ),
        },

        "test": {
            "total": len(
                final_test
            ),

            "gt_nodules": (
                final_gt
            ),
        },
    }


    save_json(
        OUTPUT_CONFIG,
        config,
    )


    print(
        OUTPUT_CONFIG
    )


    # ========================================================
    # Final
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "✅ Candidate 2 Aligned Training Setup PASS"
    )

    print()
    print(
        "TRAIN"
    )

    print(
        f"Total     : {len(train)}"
    )

    print(
        f"Positive  : {train_pos}"
    )

    print(
        f"Negative  : {train_neg}"
    )


    print()
    print(
        "INTERNAL VALIDATION"
    )

    print(
        f"Total     : {len(internal_val)}"
    )

    print(
        f"Positive  : {val_pos}"
    )

    print(
        f"Negative  : {val_neg}"
    )


    print()
    print(
        "FINAL HELD-OUT TEST"
    )

    print(
        f"CT        : {len(final_test)}"
    )

    print(
        f"GT Nodules: {final_gt}"
    )

    print(
        "Training 사용: NO"
    )


    print()
    print(
        "Candidate 1 aligned:"
    )

    print(
        "Positive + Negative CT         ✅"
    )

    print(
        "MHD / RAW                      ✅"
    )

    print(
        "Final official fold0 67 CT     ✅"
    )

    print(
        "Final GT 112 nodules           ✅"
    )

    print(
        "No final-test leakage          ✅"
    )


    print()
    print(
        "Next:"
    )

    print(
        "19_validate_candidate2_training_pipeline.py"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()