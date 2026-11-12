from pathlib import Path
import json
import shutil
import subprocess
import sys


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

INFERENCE_CONFIG = (
    BUNDLE_ROOT
    / "configs"
    / "inference.json"
)

MODEL_PATH = (
    BUNDLE_ROOT
    / "models"
    / "model.pt"
)

FOLD0_PATH = (
    BUNDLE_ROOT
    / "LUNA16_datasplit"
    / "mhd_original"
    / "dataset_fold0.json"
)

LUNA_RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "LUNA16"
    / "raw"
)

SMOKE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "candidate1_smoke"
)

SMOKE_DATA_DIR = (
    SMOKE_ROOT
    / "data"
)

SMOKE_DATALIST = (
    SMOKE_ROOT
    / "dataset_smoke.json"
)

SMOKE_CONFIG = (
    SMOKE_ROOT
    / "inference_smoke.json"
)

OUTPUT_DIR = (
    SMOKE_ROOT
    / "output"
)


# ============================================================
# Find raw MHD
# ============================================================

def find_mhd(seriesuid):

    matches = list(
        LUNA_RAW_ROOT.rglob(
            f"{seriesuid}.mhd"
        )
    )

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"중복 MHD 발견: {seriesuid}"
        )

    return matches[0]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("Candidate 1 - MONAI Pretrained RetinaNet Smoke Test")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Required files
    # --------------------------------------------------------

    print()
    print("[1] Required files")

    required = [
        INFERENCE_CONFIG,
        MODEL_PATH,
        FOLD0_PATH,
    ]

    for path in required:

        print(
            f"{path.name:20s}:",
            path,
        )

        if not path.exists():

            print(
                "❌ 파일 없음:",
                path,
            )

            sys.exit(1)

    print(
        "✅ inference config / model.pt / fold0 존재"
    )


    # --------------------------------------------------------
    # 2. Load fold0 validation
    # --------------------------------------------------------

    with open(
        FOLD0_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        fold0 = json.load(f)


    validation = fold0.get(
        "validation",
        []
    )


    if len(validation) == 0:

        raise RuntimeError(
            "fold0 validation이 비어 있습니다."
        )


    sample = validation[0]

    image_name = str(
        sample["image"]
    )

    seriesuid = Path(
        image_name
    ).stem


    print()
    print("[2] Smoke sample")

    print(
        "SeriesUID:",
        seriesuid,
    )

    print(
        "GT boxes:",
        len(
            sample.get(
                "box",
                []
            )
        ),
    )


    # --------------------------------------------------------
    # 3. Find actual MHD / RAW
    # --------------------------------------------------------

    mhd_path = find_mhd(
        seriesuid
    )


    if mhd_path is None:

        raise RuntimeError(
            f"실제 MHD를 찾지 못했습니다: {seriesuid}"
        )


    raw_path = (
        mhd_path.with_suffix(
            ".raw"
        )
    )


    if not raw_path.exists():

        raise RuntimeError(
            f"RAW 파일 없음: {raw_path}"
        )


    print()
    print("[3] Raw files")

    print(
        "MHD:",
        mhd_path,
    )

    print(
        "RAW:",
        raw_path,
    )

    print(
        "✅ .mhd / .raw 존재"
    )


    # --------------------------------------------------------
    # 4. Prepare smoke directory
    #
    # MONAI datalist은 image filename만 가지고 있기 때문에
    # 1개 케이스 전용 directory를 만든다.
    # --------------------------------------------------------

    if SMOKE_ROOT.exists():

        shutil.rmtree(
            SMOKE_ROOT
        )


    SMOKE_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    smoke_mhd = (
        SMOKE_DATA_DIR
        / mhd_path.name
    )

    smoke_raw = (
        SMOKE_DATA_DIR
        / raw_path.name
    )


    shutil.copy2(
        mhd_path,
        smoke_mhd,
    )

    shutil.copy2(
        raw_path,
        smoke_raw,
    )


    print()
    print("[4] Smoke data prepared")

    print(
        smoke_mhd
    )

    print(
        smoke_raw
    )


    # --------------------------------------------------------
    # 5. Create one-case datalist
    # --------------------------------------------------------

    smoke_dataset = {
        "training": [],
        "validation": [
            {
                "image": smoke_mhd.name,
                "box": sample.get(
                    "box",
                    [],
                ),
                "label": sample.get(
                    "label",
                    [],
                ),
            }
        ],
    }


    with open(
        SMOKE_DATALIST,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            smoke_dataset,
            f,
            indent=2,
        )


    print()
    print("[5] Smoke datalist")

    print(
        SMOKE_DATALIST
    )


    # --------------------------------------------------------
    # 6. Create smoke inference config
    # --------------------------------------------------------

    with open(
        INFERENCE_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)


    # Raw LUNA16 mode
    config[
        "whether_raw_luna16"
    ] = True


    # Important:
    # bundle_root는 실제 bundle을 유지해야
    # models/model.pt, scripts 등을 그대로 찾을 수 있음
    config[
        "bundle_root"
    ] = str(
        BUNDLE_ROOT
    )


    config[
        "data_list_file_path"
    ] = str(
        SMOKE_DATALIST
    )


    config[
        "dataset_dir"
    ] = str(
        SMOKE_DATA_DIR
    )


    config[
        "output_dir"
    ] = str(
        OUTPUT_DIR
    )


    config[
        "output_filename"
    ] = (
        "candidate1_smoke_prediction.json"
    )


    # pretrained checkpoint 사용
    config[
        "load_pretrain"
    ] = True


    with open(
        SMOKE_CONFIG,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )


    print()
    print("[6] Smoke inference config")

    print(
        SMOKE_CONFIG
    )


    # --------------------------------------------------------
    # 7. Run MONAI bundle inference
    # --------------------------------------------------------

    print()
    print("[7] Run pretrained RetinaNet inference")

    print()
    print(
        "이 단계는 GPU inference라 시간이 조금 걸릴 수 있습니다."
    )
    print()


    command = [
        sys.executable,
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        str(
            SMOKE_CONFIG
        ),
    ]


    print(
        "COMMAND:"
    )

    print(
        " ".join(
            command
        )
    )

    print()


    result = subprocess.run(
        command,
        cwd=str(
            BUNDLE_ROOT
        ),
    )


    if result.returncode != 0:

        print()
        print(
            "❌ MONAI inference 실행 실패"
        )

        sys.exit(
            result.returncode
        )


    # --------------------------------------------------------
    # 8. Check prediction output
    # --------------------------------------------------------

    prediction_path = (
        OUTPUT_DIR
        / "candidate1_smoke_prediction.json"
    )


    print()
    print("[8] Prediction output")


    if not prediction_path.exists():

        print(
            "❌ prediction JSON이 생성되지 않았습니다."
        )

        print(
            "예상 경로:",
            prediction_path,
        )

        sys.exit(1)


    print(
        "Prediction JSON:"
    )

    print(
        prediction_path
    )


    with open(
        prediction_path,
        "r",
        encoding="utf-8",
    ) as f:

        predictions = json.load(
            f
        )


    print()
    print(
        "Output type:",
        type(
            predictions
        ).__name__,
    )


    if isinstance(
        predictions,
        list,
    ):

        print(
            "Output cases:",
            len(
                predictions
            ),
        )


        if len(
            predictions
        ) > 0:

            first = predictions[0]

            print()
            print(
                "Prediction keys:",
                list(
                    first.keys()
                ),
            )


            boxes = first.get(
                "box",
                []
            )

            scores = first.get(
                "label_scores",
                first.get(
                    "score",
                    [],
                ),
            )


            print(
                "Pred boxes:",
                len(
                    boxes
                ),
            )

            print(
                "Pred scores:",
                len(
                    scores
                ),
            )


            if len(
                scores
            ) > 0:

                numeric_scores = [
                    float(
                        score
                    )
                    for score in scores
                ]

                print(
                    "Score min:",
                    min(
                        numeric_scores
                    ),
                )

                print(
                    "Score max:",
                    max(
                        numeric_scores
                    ),
                )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "✅ Candidate 1 pretrained RetinaNet smoke test PASS"
    )

    print()

    print(
        "확인 완료:"
    )

    print(
        "  - 공식 model.pt 존재"
    )

    print(
        "  - raw LUNA16 .mhd/.raw 로딩"
    )

    print(
        "  - MONAI bundle network 생성"
    )

    print(
        "  - pretrained checkpoint load"
    )

    print(
        "  - validation CT 1개 inference"
    )

    print(
        "  - prediction JSON 생성"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()