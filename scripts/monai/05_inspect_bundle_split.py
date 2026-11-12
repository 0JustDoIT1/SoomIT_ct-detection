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

CONFIG_DIR = (
    BUNDLE_ROOT
    / "configs"
)


# ============================================================
# Inspection targets
# ============================================================

TARGET_FILES = [
    "train.json",
    "evaluate.json",
    "inference.json",
    "metadata.json",
]


KEYWORDS = [
    "fold",
    "split",
    "datalist",
    "dataset",
    "data_list",
    "dataset_dir",
    "training",
    "validation",
]


# ============================================================
# Recursive search
# ============================================================

def search_dict(
    obj,
    path="root",
):

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            current_path = (
                f"{path}.{key}"
            )

            key_lower = (
                key.lower()
            )

            if any(
                keyword in key_lower
                for keyword in KEYWORDS
            ):

                print()
                print(
                    current_path
                )

                print(
                    "=",
                    value,
                )

            search_dict(
                value,
                current_path,
            )


    elif isinstance(
        obj,
        list,
    ):

        for i, value in enumerate(
            obj
        ):

            search_dict(
                value,
                f"{path}[{i}]",
            )


# ============================================================
# Main
# ============================================================

print(
    "=" * 80
)

print(
    "MONAI Lung Nodule Detection Bundle"
)

print(
    "Train / Validation / Fold Inspection"
)

print(
    "=" * 80
)


# ------------------------------------------------------------
# Config inspection
# ------------------------------------------------------------

for filename in TARGET_FILES:

    path = (
        CONFIG_DIR
        / filename
    )

    print()
    print()

    print(
        "#" * 80
    )

    print(
        filename
    )

    print(
        "#" * 80
    )


    if not path.exists():

        print(
            f"⚠️ 없음: {path}"
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


    search_dict(
        data,
        filename,
    )


# ------------------------------------------------------------
# Dataset / fold related files
# ------------------------------------------------------------

print()
print()

print(
    "=" * 80
)

print(
    "Bundle 내 dataset 관련 파일"
)

print(
    "=" * 80
)


found_files = []


for p in BUNDLE_ROOT.rglob(
    "*"
):

    if not p.is_file():
        continue


    name = (
        p.name.lower()
    )


    if (
        "dataset" in name
        or "fold" in name
        or "split" in name
        or "datalist" in name
    ):

        found_files.append(
            p
        )


if found_files:

    for p in sorted(
        found_files
    ):

        print(
            p.relative_to(
                BUNDLE_ROOT
            )
        )

else:

    print(
        "관련 파일 없음"
    )


print()
print(
    "=" * 80
)

print(
    "✅ Bundle inspection 완료"
)

print(
    "=" * 80
)