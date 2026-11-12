"""
공식 LUNA16 데이터 다운로드

다운로드 대상:
- subset0 ~ subset9
- annotations.csv

구조:
ct-project/
├── data/
│   └── LUNA16/
│       ├── raw/
│       │   ├── subset0/
│       │   ├── subset1/
│       │   ├── ...
│       │   └── subset9/
│       └── annotations.csv
└── scripts/
    └── monai/
        └── 01_download_luna16_official.py

특징:
- 공식 Zenodo LUNA16 최신 배포본 사용
- subset0 ~ subset6: Zenodo 3723295
- subset7 ~ subset9: Zenodo 4121926
- aria2c 멀티커넥션 다운로드
- MD5 검증
- Python zipfile 무결성 검사
- Python zipfile 압축 해제
- 완료된 subset은 marker로 스킵
"""

from pathlib import Path
import hashlib
import subprocess
import zipfile
import shutil


# ============================================================
# 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LUNA_DIR = PROJECT_ROOT / "data" / "LUNA16"
RAW_DIR = LUNA_DIR / "raw"
ANNOTATIONS_PATH = LUNA_DIR / "annotations.csv"


# ============================================================
# 다운로드 정보
# ============================================================

FILES = {
    "subset0.zip": {
        "url": "https://zenodo.org/records/3723295/files/subset0.zip?download=1",
        "md5": "1065b0f42b8c25cf29260fd924a3c3a2",
    },

    "subset1.zip": {
        "url": "https://zenodo.org/records/3723295/files/subset1.zip?download=1",
        "md5": "2eda28bb123543074f5cd849499b87ff",
    },

    "subset2.zip": {
        "url": "https://zenodo.org/records/3723295/files/subset2.zip?download=1",
        "md5": "685b484b085bcdb81c525f3b1bf97c51",
    },

    "subset3.zip": {
        "url": "https://zenodo.org/records/3723295/files/subset3.zip?download=1",
        "md5": "54707e9e8954af29326eed60f38ea321",
    },

    "subset4.zip": {
        "url": "https://zenodo.org/records/3723295/files/subset4.zip?download=1",
        "md5": "98225ccb1a41fc434631f63f4284796a",
    },

    "subset5.zip": {
        "url": "https://zenodo.org/records/3723295/files/subset5.zip?download=1",
        "md5": "395ec714c4123cf213fb7b08f05ee9cc",
    },

    "subset6.zip": {
        "url": "https://zenodo.org/records/3723295/files/subset6.zip?download=1",
        "md5": "d162df1444a6f2674ab8273c7f9e1520",
    },

    "subset7.zip": {
        "url": "https://zenodo.org/records/4121926/files/subset7.zip?download=1",
        "md5": "e99b8921990d1414bb5e92130371e8a3",
    },

    "subset8.zip": {
        "url": "https://zenodo.org/records/4121926/files/subset8.zip?download=1",
        "md5": "38260ca9a8741888997bcebedfabc3f1",
    },

    "subset9.zip": {
        "url": "https://zenodo.org/records/4121926/files/subset9.zip?download=1",
        "md5": "e55c473bbeebd712eb2224a58998be8e",
    },
}


ANNOTATIONS_URL = (
    "https://zenodo.org/records/3723295/files/"
    "annotations.csv?download=1"
)

ANNOTATIONS_MD5 = "f4404df491aee6445a4485061bfdffdd"


# ============================================================
# MD5
# ============================================================

def calculate_md5(path, chunk_size=16 * 1024 * 1024):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


def verify_md5(path, expected):

    print(f"🔍 MD5 검사: {path.name}")

    actual = calculate_md5(path)

    print(f"   expected: {expected}")
    print(f"   actual  : {actual}")

    if actual.lower() != expected.lower():

        print("❌ MD5 불일치")
        return False

    print("✅ MD5 정상")
    return True


# ============================================================
# aria2c 확인
# ============================================================

def check_aria2():

    result = subprocess.run(
        ["which", "aria2c"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "aria2c가 설치되어 있지 않습니다.\n"
            "sudo apt install -y aria2\n"
            "또는\n"
            "conda install -c conda-forge aria2"
        )

    print(
        f"✅ aria2c 확인: "
        f"{result.stdout.strip()}"
    )


# ============================================================
# 다운로드
# ============================================================

def download_file(url, output_path):

    print(
        f"\n🔍 다운로드 시작: "
        f"{output_path.name}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "aria2c",

            # 서버당 최대 연결 수
            "-x", "16",

            # 파일 분할 수
            "-s", "16",

            # 분할 최소 크기
            "-k", "10M",

            # 이어받기
            "--continue=true",

            # 기존 파일 처리
            "--allow-overwrite=true",
            "--auto-file-renaming=false",

            # 재시도
            "--max-tries=20",
            "--retry-wait=5",
            "--timeout=60",
            "--connect-timeout=30",

            # 출력 위치
            "-d", str(output_path.parent),
            "-o", output_path.name,

            url,
        ],
        check=True,
    )

    print(
        f"✅ 다운로드 완료: "
        f"{output_path.name}"
    )


# ============================================================
# ZIP 검사 + 압축해제
# ============================================================

def extract_zip(zip_path):

    print(
        f"🔍 ZIP 검사: "
        f"{zip_path.name}"
    )

    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zf:

            bad_file = zf.testzip()

            if bad_file is not None:

                raise RuntimeError(
                    f"ZIP 손상 파일 발견: "
                    f"{bad_file}"
                )

            print(
                "✅ ZIP 정상"
            )

            print(
                f"🔍 압축 해제: "
                f"{zip_path.name}"
            )

            zf.extractall(
                RAW_DIR
            )

    except Exception as e:

        raise RuntimeError(
            f"ZIP 검사/압축 해제 실패: "
            f"{zip_path.name}"
        ) from e

    print(
        "✅ 압축 해제 완료"
    )


# ============================================================
# 압축 해제 결과 검증
# ============================================================

def validate_subset(subset_name):

    subset_dir = RAW_DIR / subset_name

    if not subset_dir.exists():

        return False

    mhd_files = list(
        subset_dir.glob("*.mhd")
    )

    raw_files = list(
        subset_dir.glob("*.raw")
    )

    if len(mhd_files) == 0:

        return False

    if len(raw_files) == 0:

        return False

    if len(mhd_files) != len(raw_files):

        return False

    return True


# ============================================================
# annotations.csv
# ============================================================

def ensure_annotations():

    if ANNOTATIONS_PATH.exists():

        if verify_md5(
            ANNOTATIONS_PATH,
            ANNOTATIONS_MD5,
        ):

            print(
                "✅ annotations.csv 이미 존재"
            )

            return

        print(
            "⚠️ 기존 annotations.csv 삭제"
        )

        ANNOTATIONS_PATH.unlink()

    download_file(
        ANNOTATIONS_URL,
        ANNOTATIONS_PATH,
    )

    if not verify_md5(
        ANNOTATIONS_PATH,
        ANNOTATIONS_MD5,
    ):

        ANNOTATIONS_PATH.unlink()

        raise RuntimeError(
            "annotations.csv MD5 검증 실패"
        )

    print(
        "✅ annotations.csv 준비 완료"
    )


# ============================================================
# subset 처리
# ============================================================

def process_subset(filename, info):

    subset_name = filename.replace(
        ".zip",
        "",
    )

    zip_path = RAW_DIR / filename

    subset_dir = RAW_DIR / subset_name

    marker_path = (
        RAW_DIR /
        f".{subset_name}_extracted.ok"
    )

    print()

    print(
        "=" * 60
    )

    print(
        f"처리 대상: {subset_name}"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # 이미 완료된 subset
    # --------------------------------------------------------

    if marker_path.exists():

        if validate_subset(
            subset_name
        ):

            print(
                f"✅ 이미 완료됨, 스킵: "
                f"{subset_name}"
            )

            return

        print(
            "⚠️ marker는 있지만 "
            "실제 데이터 검증 실패"
        )

        marker_path.unlink()


    # --------------------------------------------------------
    # 기존 ZIP
    # --------------------------------------------------------

    if zip_path.exists():

        print(
            f"✅ ZIP 이미 존재: "
            f"{filename}"
        )

        if not verify_md5(
            zip_path,
            info["md5"],
        ):

            print(
                "⚠️ 기존 ZIP 삭제"
            )

            zip_path.unlink()

            aria_control = Path(
                str(zip_path) + ".aria2"
            )

            if aria_control.exists():

                aria_control.unlink()


    # --------------------------------------------------------
    # 다운로드
    # --------------------------------------------------------

    if not zip_path.exists():

        download_file(
            info["url"],
            zip_path,
        )


    # --------------------------------------------------------
    # MD5 검증
    # --------------------------------------------------------

    if not verify_md5(
        zip_path,
        info["md5"],
    ):

        if zip_path.exists():

            zip_path.unlink()

        aria_control = Path(
            str(zip_path) + ".aria2"
        )

        if aria_control.exists():

            aria_control.unlink()

        raise RuntimeError(
            f"{filename} MD5 검증 실패"
        )


    # --------------------------------------------------------
    # 이전 실패 데이터 제거
    # --------------------------------------------------------

    if subset_dir.exists():

        print(
            f"⚠️ 기존 subset 폴더 제거: "
            f"{subset_name}"
        )

        shutil.rmtree(
            subset_dir
        )


    # --------------------------------------------------------
    # 압축 해제
    # --------------------------------------------------------

    extract_zip(
        zip_path
    )


    # --------------------------------------------------------
    # 결과 검증
    # --------------------------------------------------------

    if not validate_subset(
        subset_name
    ):

        raise RuntimeError(
            f"{subset_name} "
            "압축 해제 결과 검증 실패"
        )


    # --------------------------------------------------------
    # marker 생성
    # --------------------------------------------------------

    marker_path.write_text(
        "done\n"
    )


    # --------------------------------------------------------
    # ZIP 삭제
    # --------------------------------------------------------

    zip_path.unlink()

    aria_control = Path(
        str(zip_path) + ".aria2"
    )

    if aria_control.exists():

        aria_control.unlink()

    print(
        f"🗑️ ZIP 삭제: "
        f"{filename}"
    )

    print(
        f"✅ 완료: "
        f"{subset_name}"
    )


# ============================================================
# main
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "LUNA16 공식 데이터 다운로드"
    )

    print(
        "=" * 60
    )

    print(
        f"PROJECT ROOT: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"LUNA16 DATA : "
        f"{LUNA_DIR}"
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # aria2c 설치 확인
    check_aria2()

    # annotations.csv
    ensure_annotations()

    # subset0 ~ subset9
    for filename, info in FILES.items():

        process_subset(
            filename,
            info,
        )


    # --------------------------------------------------------
    # 전체 최종 검증
    # --------------------------------------------------------

    print()

    print(
        "=" * 60
    )

    print(
        "최종 검증"
    )

    print(
        "=" * 60
    )

    for i in range(10):

        subset_name = (
            f"subset{i}"
        )

        if not validate_subset(
            subset_name
        ):

            raise RuntimeError(
                f"{subset_name} "
                "최종 검증 실패"
            )

        print(
            f"✅ {subset_name}"
        )


    print()

    print(
        "✅ LUNA16 공식 데이터 "
        "다운로드 완료"
    )

    print(
        f"RAW: "
        f"{RAW_DIR}"
    )

    print(
        f"Annotations: "
        f"{ANNOTATIONS_PATH}"
    )


if __name__ == "__main__":

    main()