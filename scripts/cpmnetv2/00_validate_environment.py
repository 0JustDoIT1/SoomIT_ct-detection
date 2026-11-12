#!/usr/bin/env python3

import os
import sys
import platform
import traceback
from pathlib import Path


PROJECT_ROOT = Path("/mnt/c/AI/ct-project")
CPMNET_ROOT = PROJECT_ROOT / "external" / "CPMNetv2"
RESULT_DIR = PROJECT_ROOT / "results" / "cpmnetv2"


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_result(name, success, detail=""):
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {name}")

    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")


def get_version(module):
    return getattr(module, "__version__", "unknown")


def validate_paths():
    print_header("1. Project Path Validation")

    checks = [
        ("Project root", PROJECT_ROOT),
        ("CPMNetv2 root", CPMNET_ROOT),
        ("CPMNetv2 networks", CPMNET_ROOT / "networks"),
        ("CPMNetv2 network file", CPMNET_ROOT / "networks" / "ResNet_3D_CPM.py"),
        ("CPMNetv2 infer.py", CPMNET_ROOT / "infer.py"),
        ("CPMNetv2 preprocessing", CPMNET_ROOT / "luna16_preprocessing.py"),
        ("Result directory", RESULT_DIR),
    ]

    all_ok = True

    for name, path in checks:
        exists = path.exists()

        print_result(
            name,
            exists,
            str(path),
        )

        if not exists:
            all_ok = False

    return all_ok


def validate_python():
    print_header("2. Python Environment")

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version}")
    print(f"Platform          : {platform.platform()}")
    print(f"Working directory : {os.getcwd()}")

    version_ok = sys.version_info >= (3, 7)

    print_result(
        "Python >= 3.7",
        version_ok,
        platform.python_version(),
    )

    return version_ok


def validate_packages():
    print_header("3. Python Package Validation")

    package_results = {}
    all_ok = True

    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("skimage", "scikit-image"),
        ("SimpleITK", "SimpleITK"),
        ("torch", "PyTorch"),
    ]

    for import_name, display_name in packages:
        try:
            module = __import__(import_name)

            version = get_version(module)

            if import_name == "SimpleITK":
                try:
                    version = module.Version_VersionString()
                except Exception:
                    pass

            print_result(
                display_name,
                True,
                f"version = {version}",
            )

            package_results[import_name] = module

        except Exception as e:
            print_result(
                display_name,
                False,
                f"{type(e).__name__}: {e}",
            )

            all_ok = False

    return all_ok, package_results


def validate_torch(torch):
    print_header("4. PyTorch / CUDA / GPU Validation")

    print(f"PyTorch version    : {torch.__version__}")
    print(f"CUDA build version : {torch.version.cuda}")
    print(f"CUDA available     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print_result(
            "CUDA available",
            False,
            "torch.cuda.is_available() == False",
        )

        return False

    print_result(
        "CUDA available",
        True,
    )

    device_count = torch.cuda.device_count()

    print(f"GPU count          : {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)

        total_memory_gb = props.total_memory / (1024 ** 3)

        print()
        print(f"GPU {i}")
        print(f"  Name             : {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory     : {total_memory_gb:.2f} GB")

    try:
        device = torch.device("cuda:0")

        x = torch.randn(
            1,
            1,
            16,
            32,
            32,
            device=device,
        )

        y = x * 2.0 + 1.0

        torch.cuda.synchronize()

        assert y.is_cuda
        assert torch.isfinite(y).all()

        print_result(
            "Basic GPU tensor operation",
            True,
            f"shape = {tuple(y.shape)}",
        )

    except Exception as e:
        print_result(
            "Basic GPU tensor operation",
            False,
            f"{type(e).__name__}: {e}",
        )

        traceback.print_exc()

        return False

    return True


def validate_cpmnet_import(torch):
    print_header("5. CPMNetv2 Import Validation")

    if not CPMNET_ROOT.exists():
        print_result(
            "CPMNetv2 root",
            False,
            str(CPMNET_ROOT),
        )
        return False

    cpmnet_path = str(CPMNET_ROOT)

    if cpmnet_path not in sys.path:
        sys.path.insert(0, cpmnet_path)

    try:
        import networks.ResNet_3D_CPM as cpm_module

        print_result(
            "Import networks.ResNet_3D_CPM",
            True,
            str(cpm_module.__file__),
        )

    except Exception as e:
        print_result(
            "Import networks.ResNet_3D_CPM",
            False,
            f"{type(e).__name__}: {e}",
        )

        traceback.print_exc()

        return False

    print()
    print("Available public symbols:")

    symbols = [
        name
        for name in dir(cpm_module)
        if not name.startswith("_")
    ]

    for name in symbols:
        print(f"  - {name}")

    return True


def validate_gpu_conv3d(torch):
    print_header("6. 3D GPU Forward Test")

    if not torch.cuda.is_available():
        print_result(
            "3D Conv GPU forward",
            False,
            "CUDA unavailable",
        )
        return False

    try:
        device = torch.device("cuda:0")

        model = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm3d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
        ).to(device)

        model.eval()

        x = torch.randn(
            1,
            1,
            32,
            64,
            64,
            device=device,
        )

        with torch.no_grad():
            y = model(x)

        torch.cuda.synchronize()

        finite = torch.isfinite(y).all().item()

        print(f"Input shape  : {tuple(x.shape)}")
        print(f"Output shape : {tuple(y.shape)}")

        print_result(
            "3D Conv GPU forward",
            bool(finite),
            f"finite = {finite}",
        )

        del x
        del y
        del model

        torch.cuda.empty_cache()

        return bool(finite)

    except Exception as e:
        print_result(
            "3D Conv GPU forward",
            False,
            f"{type(e).__name__}: {e}",
        )

        traceback.print_exc()

        return False


def print_reference_versions():
    print_header("7. CPMNetv2 README Reference Versions")

    print(
        """
These are reference versions from the original CPMNetv2 environment.

Python       >= 3.7
PyTorch      >= 1.13.1
CUDA         >= 11.6
SimpleITK    2.2.1
pandas       1.3.5
scikit-image 0.19.3
scipy        1.7.3

IMPORTANT:
Do NOT downgrade or reinstall packages just because the versions differ.

The purpose of this script is to verify whether the CURRENT monai-luna
environment can run the CPMNetv2 pipeline.

Create a separate environment only if an actual incompatibility occurs.
""".strip()
    )


def main():
    print_header("CPMNetv2 Environment Validation")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"CPMNet root  : {CPMNET_ROOT}")
    print(f"Result dir   : {RESULT_DIR}")

    results = {}

    results["paths"] = validate_paths()
    results["python"] = validate_python()

    packages_ok, modules = validate_packages()

    results["packages"] = packages_ok

    torch = modules.get("torch")

    if torch is not None:
        results["cuda_gpu"] = validate_torch(torch)
        results["cpmnet_import"] = validate_cpmnet_import(torch)
        results["gpu_3d_forward"] = validate_gpu_conv3d(torch)

    else:
        results["cuda_gpu"] = False
        results["cpmnet_import"] = False
        results["gpu_3d_forward"] = False

    print_reference_versions()

    print_header("8. Final Summary")

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{name:20s}: {status}")

    total = len(results)
    passed = sum(bool(v) for v in results.values())

    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print()
        print("Environment validation PASSED.")
        print(
            "Next step: scripts/cpmnetv2/01_validate_luna16_data.py"
        )
        return 0

    print()
    print("Environment validation FAILED.")
    print(
        "Check the FAIL items above before proceeding to LUNA16 preprocessing."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())