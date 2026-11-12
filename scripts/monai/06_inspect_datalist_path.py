from pathlib import Path
import json


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BUNDLE_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "monai"
    / "bundles"
    / "lung_nodule_ct_detection"
)

SPLIT_ROOT = (
    BUNDLE_ROOT
    / "LUNA16_datasplit"
)


# ============================================================
# Datalist targets
# ============================================================

TARGET_DATALISTS = {
    "resampled_default": (
        SPLIT_ROOT
        / "dataset_fold0.json"
    ),

    "mhd_original": (
        SPLIT_ROOT
        / "mhd_original"
        / "dataset_fold0.json"
    ),

    "dicom_original": (
        SPLIT_ROOT
        / "dicom_original"
        / "dataset_fold0.json"
    ),
}


# ============================================================
# Item inspection
# ============================================================

def inspect_item(
    item,
    index,
):

    print(
        f"  Item #{index}"
    )

    print(
        "    image :",
        item.get(
            "image"
        )
    )

    print(
        "    box   :",
        item.get(
            "box"
        )
    )

    print(
        "    label :",
        item.get(
            "label"
        )
    )


# ============================================================
# Main
# ============================================================

print(
    "=" * 80
)

print(
    "MONAI LUNA16 Datalist Path Inspection"
)

print(
    "=" * 80
)


for name, path in TARGET_DATALISTS.items():

    print()
    print(
        "#" * 80
    )

    print(
        name
    )

    print(
        "#" * 80
    )

    print(
        "Path:"
    )

    print(
        path
    )

    print()


    if not path.exists():

        print(
            "❌ 파일 없음"
        )

        continue


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )


    print(
        "Top-level keys:",
        list(
            data.keys()
        )
    )


    training = data.get(
        "training",
        []
    )

    validation = data.get(
        "validation",
        []
    )


    print()

    print(
        "Training count:",
        len(
            training
        )
    )

    print(
        "Validation count:",
        len(
            validation
        )
    )


    # --------------------------------------------------------
    # Training sample
    # --------------------------------------------------------

    print()

    print(
        "[Training sample]"
    )

    if training:

        inspect_item(
            training[0],
            0,
        )

    else:

        print(
            "  없음"
        )


    # --------------------------------------------------------
    # Validation sample
    # --------------------------------------------------------

    print()

    print(
        "[Validation sample]"
    )

    if validation:

        inspect_item(
            validation[0],
            0,
        )

    else:

        print(
            "  없음"
        )


print()
print(
    "=" * 80
)

print(
    "✅ Datalist inspection 완료"
)

print(
    "=" * 80
)