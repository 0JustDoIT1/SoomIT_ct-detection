from pathlib import Path
import csv
import json
import sys


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LUNA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "LUNA16"
)

RAW_ROOT = (
    LUNA_ROOT
    / "raw"
)

ANNOTATIONS_PATH = (
    LUNA_ROOT
    / "annotations.csv"
)

BUNDLE_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "monai"
    / "bundles"
    / "lung_nodule_ct_detection"
)

FOLD0_PATH = (
    BUNDLE_ROOT
    / "LUNA16_datasplit"
    / "mhd_original"
    / "dataset_fold0.json"
)


# ============================================================
# Expected values
# ============================================================

EXPECTED_LUNA_SCANS = 888
EXPECTED_POSITIVE_SCANS = 601

EXPECTED_FOLD0_TRAIN = 534
EXPECTED_FOLD0_VALIDATION = 67


# ============================================================
# Collect actual LUNA16 scans
# ============================================================

def collect_scan_paths():

    scan_paths = {}

    for subset_idx in range(10):

        subset_dir = (
            RAW_ROOT
            / f"subset{subset_idx}"
        )

        if not subset_dir.exists():
            continue

        for mhd_path in subset_dir.glob(
            "*.mhd"
        ):

            uid = mhd_path.stem

            if uid in scan_paths:
                raise RuntimeError(
                    f"중복 SeriesUID 발견: {uid}"
                )

            scan_paths[uid] = mhd_path

    return scan_paths


# ============================================================
# Collect annotation positive UIDs
# ============================================================

def collect_positive_ids():

    positive_ids = set()

    with open(
        ANNOTATIONS_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            uid = row.get(
                "seriesuid"
            )

            if uid:
                positive_ids.add(
                    uid
                )

    return positive_ids


# ============================================================
# Extract UID from datalist item
# ============================================================

def get_seriesuid(item):

    image = item.get(
        "image",
        ""
    )

    if isinstance(
        image,
        list,
    ):
        image = image[0]

    image_name = Path(
        str(image)
    ).name

    if image_name.endswith(
        ".mhd"
    ):
        return image_name[:-4]

    return Path(
        image_name
    ).stem


# ============================================================
# Validate item
# ============================================================

def validate_item(
    item,
    scan_paths,
):

    errors = []

    uid = get_seriesuid(
        item
    )

    image = item.get(
        "image"
    )

    boxes = item.get(
        "box"
    )

    labels = item.get(
        "label"
    )


    # --------------------------------------------------------
    # image
    # --------------------------------------------------------

    if not image:

        errors.append(
            "image 없음"
        )


    if uid not in scan_paths:

        errors.append(
            "실제 .mhd 없음"
        )


    # --------------------------------------------------------
    # box
    # --------------------------------------------------------

    if not isinstance(
        boxes,
        list,
    ):

        errors.append(
            "box가 list가 아님"
        )

        boxes = []


    if len(
        boxes
    ) == 0:

        errors.append(
            "box 없음"
        )


    for box_idx, box in enumerate(
        boxes
    ):

        if (
            not isinstance(
                box,
                list,
            )
            or len(
                box
            ) != 6
        ):

            errors.append(
                f"box[{box_idx}] shape 이상"
            )


    # --------------------------------------------------------
    # label
    # --------------------------------------------------------

    if not isinstance(
        labels,
        list,
    ):

        errors.append(
            "label이 list가 아님"
        )

        labels = []


    if len(
        boxes
    ) != len(
        labels
    ):

        errors.append(
            "box / label 개수 불일치"
        )


    if any(
        label != 0
        for label in labels
    ):

        errors.append(
            "label에 0 이외 값 존재"
        )


    return (
        uid,
        errors,
    )


# ============================================================
# Validate split
# ============================================================

def validate_split(
    name,
    items,
    scan_paths,
    positive_ids,
):

    ids = []

    invalid_items = []

    total_boxes = 0


    for index, item in enumerate(
        items
    ):

        uid, errors = validate_item(
            item,
            scan_paths,
        )

        ids.append(
            uid
        )


        boxes = item.get(
            "box",
            [],
        )

        if isinstance(
            boxes,
            list,
        ):

            total_boxes += len(
                boxes
            )


        if errors:

            invalid_items.append(
                {
                    "index": index,
                    "uid": uid,
                    "errors": errors,
                }
            )


    unique_ids = set(
        ids
    )

    duplicate_count = (
        len(ids)
        - len(unique_ids)
    )

    positive_count = len(
        unique_ids
        & positive_ids
    )

    negative_count = (
        len(unique_ids)
        - positive_count
    )


    return {
        "name": name,
        "items": len(items),
        "ids": unique_ids,
        "unique": len(unique_ids),
        "duplicates": duplicate_count,
        "positive": positive_count,
        "negative": negative_count,
        "boxes": total_boxes,
        "invalid": invalid_items,
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "MONAI Official LUNA16 fold0 Validation"
    )

    print(
        "mhd_original"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "fold0 JSON:"
    )

    print(
        FOLD0_PATH
    )

    print()


    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    if not FOLD0_PATH.exists():

        print(
            "❌ mhd_original dataset_fold0.json이 없습니다."
        )

        sys.exit(
            1
        )


    if not ANNOTATIONS_PATH.exists():

        print(
            "❌ annotations.csv가 없습니다."
        )

        sys.exit(
            1
        )


    # --------------------------------------------------------
    # Load actual scans
    # --------------------------------------------------------

    scan_paths = (
        collect_scan_paths()
    )

    positive_ids = (
        collect_positive_ids()
    )


    all_ok = True


    print(
        "[1] 실제 LUNA16 데이터"
    )

    print()

    print(
        "Actual CT scans       :",
        len(
            scan_paths
        ),
    )

    print(
        "Annotated positive CT :",
        len(
            positive_ids
        ),
    )


    if len(
        scan_paths
    ) != EXPECTED_LUNA_SCANS:

        print(
            f"❌ CT 수 이상 "
            f"(expected={EXPECTED_LUNA_SCANS})"
        )

        all_ok = False


    if len(
        positive_ids
    ) != EXPECTED_POSITIVE_SCANS:

        print(
            f"❌ Positive CT 수 이상 "
            f"(expected={EXPECTED_POSITIVE_SCANS})"
        )

        all_ok = False


    # --------------------------------------------------------
    # Load fold0
    # --------------------------------------------------------

    with open(
        FOLD0_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        fold0 = json.load(
            f
        )


    print()
    print(
        "[2] fold0 JSON 구조"
    )

    print()

    print(
        "Top-level keys:",
        list(
            fold0.keys()
        ),
    )


    if (
        "training" not in fold0
        or "validation" not in fold0
    ):

        print(
            "❌ training / validation key 없음"
        )

        sys.exit(
            1
        )


    training = fold0[
        "training"
    ]

    validation = fold0[
        "validation"
    ]


    # --------------------------------------------------------
    # Validate splits
    # --------------------------------------------------------

    train_stats = validate_split(
        "TRAINING",
        training,
        scan_paths,
        positive_ids,
    )

    val_stats = validate_split(
        "VALIDATION",
        validation,
        scan_paths,
        positive_ids,
    )


    print()
    print(
        "[3] fold0 split 통계"
    )

    print()


    for stats in [
        train_stats,
        val_stats,
    ]:

        print(
            stats[
                "name"
            ]
        )

        print(
            "  Items      :",
            stats[
                "items"
            ],
        )

        print(
            "  Unique CT  :",
            stats[
                "unique"
            ],
        )

        print(
            "  Positive   :",
            stats[
                "positive"
            ],
        )

        print(
            "  Negative   :",
            stats[
                "negative"
            ],
        )

        print(
            "  Total boxes:",
            stats[
                "boxes"
            ],
        )

        print(
            "  Duplicates :",
            stats[
                "duplicates"
            ],
        )

        print(
            "  Invalid    :",
            len(
                stats[
                    "invalid"
                ]
            ),
        )

        print()


    # --------------------------------------------------------
    # Expected counts
    # --------------------------------------------------------

    if train_stats[
        "items"
    ] != EXPECTED_FOLD0_TRAIN:

        print(
            "❌ fold0 training 수 이상:",
            train_stats[
                "items"
            ],
        )

        all_ok = False


    if val_stats[
        "items"
    ] != EXPECTED_FOLD0_VALIDATION:

        print(
            "❌ fold0 validation 수 이상:",
            val_stats[
                "items"
            ],
        )

        all_ok = False


    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    if train_stats[
        "duplicates"
    ] != 0:

        print(
            "❌ Training UID 중복 존재"
        )

        all_ok = False


    if val_stats[
        "duplicates"
    ] != 0:

        print(
            "❌ Validation UID 중복 존재"
        )

        all_ok = False


    # --------------------------------------------------------
    # Positive-only check
    # --------------------------------------------------------

    if train_stats[
        "negative"
    ] != 0:

        print(
            "❌ Training에 annotation 없는 CT 존재"
        )

        all_ok = False


    if val_stats[
        "negative"
    ] != 0:

        print(
            "❌ Validation에 annotation 없는 CT 존재"
        )

        all_ok = False


    # --------------------------------------------------------
    # Item structure
    # --------------------------------------------------------

    invalid_items = (
        train_stats[
            "invalid"
        ]
        + val_stats[
            "invalid"
        ]
    )


    if invalid_items:

        print()
        print(
            "[4] 잘못된 datalist item"
        )

        print()

        for item in invalid_items[
            :20
        ]:

            print(
                item
            )

        all_ok = False

    else:

        print(
            "[4] Datalist item 구조"
        )

        print()

        print(
            "✅ 모든 image가 실제 .mhd에 대응"
        )

        print(
            "✅ 모든 box가 6개 값"
        )

        print(
            "✅ box / label 개수 일치"
        )

        print(
            "✅ 모든 label = 0"
        )


    # --------------------------------------------------------
    # Train / Validation overlap
    # --------------------------------------------------------

    overlap = (
        train_stats[
            "ids"
        ]
        & val_stats[
            "ids"
        ]
    )


    print()
    print(
        "[5] Training / Validation overlap"
    )

    print()

    print(
        "Overlap:",
        len(
            overlap
        ),
    )


    if overlap:

        print(
            "❌ overlap 발견"
        )

        for uid in sorted(
            overlap
        )[:20]:

            print(
                "  -",
                uid
            )

        all_ok = False

    else:

        print(
            "✅ overlap 없음"
        )


    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    fold0_ids = (
        train_stats[
            "ids"
        ]
        | val_stats[
            "ids"
        ]
    )


    print()
    print(
        "[6] fold0 coverage"
    )

    print()

    print(
        "fold0 unique CT:",
        len(
            fold0_ids
        ),
    )

    print(
        "Positive CT total:",
        len(
            positive_ids
        ),
    )

    print(
        "Positive CT not in fold0:",
        len(
            positive_ids
            - fold0_ids
        ),
    )


    if fold0_ids != positive_ids:

        print(
            "❌ fold0가 전체 601 positive CT와 일치하지 않음"
        )

        all_ok = False

    else:

        print(
            "✅ fold0 = 전체 601 positive CT"
        )


    # --------------------------------------------------------
    # Official internal split
    # --------------------------------------------------------

    internal_train_count = int(
        0.95
        * len(
            training
        )
    )

    internal_val_count = (
        len(
            training
        )
        - internal_train_count
    )


    print()
    print(
        "[7] MONAI 공식 train.json 내부 split"
    )

    print()

    print(
        "fold0 training:",
        len(
            training
        ),
    )

    print(
        "Internal Train:",
        internal_train_count,
    )

    print(
        "Internal Val  :",
        internal_val_count,
    )

    print(
        "Final Test    :",
        len(
            validation
        ),
    )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )


    if all_ok:

        print(
            "✅ MONAI 공식 fold0 mhd_original 검증 PASS"
        )

        print()

        print(
            "Experiment protocol:"
        )

        print()

        print(
            f"Internal Train : {internal_train_count}"
        )

        print(
            f"Internal Val   : {internal_val_count}"
        )

        print(
            f"Final Test     : {len(validation)}"
        )

        print()

        print(
            "Candidate 1:"
        )

        print(
            "  공식 pretrained model 평가"
        )

        print()

        print(
            "Candidate 2/3:"
        )

        print(
            "  fold0 training의 동일 95/5 split 사용"
        )

        print(
            "  fold0 validation 67개는 최종 평가에만 사용"
        )


    else:

        print(
            "❌ MONAI 공식 fold0 검증 FAIL"
        )


    print(
        "=" * 80
    )


    if not all_ok:

        sys.exit(
            1
        )


if __name__ == "__main__":

    main()