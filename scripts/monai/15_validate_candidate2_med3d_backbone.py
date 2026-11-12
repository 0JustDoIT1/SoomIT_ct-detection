from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_ROOT = (
    PROJECT_ROOT
    / "external"
)

MEDICALNET_ROOT = (
    EXTERNAL_ROOT
    / "MedicalNet"
)

WEIGHT_PATH = (
    PROJECT_ROOT
    / "weights"
    / "med3d"
    / "resnet_50_23dataset.pth"
)


MEDICALNET_GIT = (
    "https://github.com/Tencent/MedicalNet.git"
)


# ============================================================
# Utility
# ============================================================

def run_command(command):

    print()
    print("COMMAND:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
    )


# ============================================================
# MedicalNet source
# ============================================================

def prepare_medicalnet():

    print()
    print("[1] MedicalNet source")

    EXTERNAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MEDICALNET_ROOT.exists():

        print(
            "MedicalNet already exists:"
        )

        print(
            MEDICALNET_ROOT
        )

    else:

        print(
            "MedicalNet source 없음"
        )

        print(
            "공식 repository clone..."
        )

        run_command(
            [
                "git",
                "clone",
                MEDICALNET_GIT,
                str(
                    MEDICALNET_ROOT
                ),
            ]
        )


    resnet_path = (
        MEDICALNET_ROOT
        / "models"
        / "resnet.py"
    )


    if not resnet_path.exists():

        raise FileNotFoundError(
            f"MedicalNet resnet.py 없음: {resnet_path}"
        )


    print(
        "✅ MedicalNet source 확인"
    )


# ============================================================
# Import MedicalNet
# ============================================================

def import_medicalnet():

    if str(
        MEDICALNET_ROOT
    ) not in sys.path:

        sys.path.insert(
            0,
            str(
                MEDICALNET_ROOT
            ),
        )


    try:

        from models import resnet

    except Exception as e:

        raise RuntimeError(
            f"MedicalNet models.resnet import 실패: {e}"
        )


    return resnet


# ============================================================
# Build Med3D ResNet50
# ============================================================

def build_med3d_resnet50(
    resnet_module,
):

    print()
    print(
        "[2] Build MedicalNet ResNet50"
    )


    model = resnet_module.resnet50(
        sample_input_W=96,
        sample_input_H=96,
        sample_input_D=96,
        shortcut_type="B",
        no_cuda=True,
        num_seg_classes=2,
    )


    print(
        "✅ MedicalNet 3D ResNet50 생성"
    )

    return model


# ============================================================
# Check checkpoint
# ============================================================

def load_checkpoint():

    print()
    print(
        "[3] Check pretrained weight"
    )


    if not WEIGHT_PATH.exists():

        raise FileNotFoundError(
            f"Weight 없음: {WEIGHT_PATH}"
        )


    size_mb = (
        WEIGHT_PATH.stat().st_size
        / 1024
        / 1024
    )


    print(
        "Weight:"
    )

    print(
        WEIGHT_PATH
    )

    print(
        f"Size: {size_mb:.2f} MB"
    )


    try:

        checkpoint = torch.load(
            WEIGHT_PATH,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as e:

        raise RuntimeError(
            f"Checkpoint load 실패: {e}"
        )


    print(
        "Checkpoint type:",
        type(
            checkpoint
        ),
    )


    if isinstance(
        checkpoint,
        dict,
    ):

        print(
            "Checkpoint keys:"
        )

        for key in list(
            checkpoint.keys()
        )[:20]:

            print(
                " -",
                key
            )


    print(
        "✅ Weight 파일 읽기 성공"
    )


    return checkpoint


# ============================================================
# Extract state dict
# ============================================================

def extract_state_dict(
    checkpoint,
):

    print()
    print(
        "[4] Extract state_dict"
    )


    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint가 dict 형태가 아닙니다."
        )


    if "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

        print(
            "state_dict key 사용"
        )

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print(
            "model_state_dict key 사용"
        )

    else:

        # checkpoint 자체가 state_dict인 경우
        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(
                value
            )
        ]


        if len(
            tensor_values
        ) > 0:

            state_dict = checkpoint

            print(
                "Checkpoint 자체를 state_dict로 사용"
            )

        else:

            raise RuntimeError(
                "state_dict를 찾지 못했습니다."
            )


    print(
        "Raw state_dict keys:",
        len(
            state_dict
        ),
    )


    return state_dict


# ============================================================
# Normalize key prefix
# ============================================================

def normalize_state_dict(
    state_dict,
):

    normalized = {}


    for key, value in state_dict.items():

        new_key = key


        # DataParallel prefix
        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len(
                    "module."
                ):
            ]


        normalized[
            new_key
        ] = value


    return normalized


# ============================================================
# Load compatible pretrained weights
# ============================================================

def load_pretrained(
    model,
    state_dict,
):

    print()
    print(
        "[5] Load Med3D pretrained weights"
    )


    state_dict = normalize_state_dict(
        state_dict
    )


    model_dict = model.state_dict()


    compatible = {}

    shape_mismatch = []

    missing_in_model = []


    for key, value in state_dict.items():

        if key not in model_dict:

            missing_in_model.append(
                key
            )

            continue


        if (
            model_dict[
                key
            ].shape
            != value.shape
        ):

            shape_mismatch.append(
                (
                    key,
                    tuple(
                        value.shape
                    ),
                    tuple(
                        model_dict[
                            key
                        ].shape
                    ),
                )
            )

            continue


        compatible[
            key
        ] = value


    print(
        "Model parameters:",
        len(
            model_dict
        ),
    )

    print(
        "Checkpoint parameters:",
        len(
            state_dict
        ),
    )

    print(
        "Compatible parameters:",
        len(
            compatible
        ),
    )

    print(
        "Unknown checkpoint keys:",
        len(
            missing_in_model
        ),
    )

    print(
        "Shape mismatch:",
        len(
            shape_mismatch
        ),
    )


    if shape_mismatch:

        print()
        print(
            "Shape mismatch examples:"
        )

        for item in shape_mismatch[:10]:

            print(
                item
            )


    if len(
        compatible
    ) == 0:

        raise RuntimeError(
            "로드 가능한 pretrained parameter가 없습니다."
        )


    model_dict.update(
        compatible
    )


    model.load_state_dict(
        model_dict
    )


    load_ratio = (
        len(
            compatible
        )
        / len(
            model_dict
        )
    )


    print()
    print(
        f"Load ratio: {load_ratio:.3f}"
    )


    if load_ratio < 0.8:

        print(
            "⚠️ Load ratio가 낮습니다."
        )

        print(
            "Checkpoint 구조를 추가 확인하세요."
        )

    else:

        print(
            "✅ Med3D pretrained weight load 정상"
        )


    return compatible


# ============================================================
# Backbone wrapper
#
# MedicalNet 원본 ResNet은 segmentation/classification
# 형태의 최종 layer까지 포함한다.
#
# RetinaNet에 필요한 것은 중간 feature maps이므로
# conv1 → layer1 → layer2 → layer3 → layer4
# 출력만 추출한다.
# ============================================================

class Med3DBackbone(
    nn.Module):

    def __init__(
        self,
        medicalnet_model,
    ):

        super().__init__()


        self.conv1 = (
            medicalnet_model.conv1
        )

        self.bn1 = (
            medicalnet_model.bn1
        )

        self.relu = (
            medicalnet_model.relu
        )

        self.maxpool = (
            medicalnet_model.maxpool
        )


        self.layer1 = (
            medicalnet_model.layer1
        )

        self.layer2 = (
            medicalnet_model.layer2
        )

        self.layer3 = (
            medicalnet_model.layer3
        )

        self.layer4 = (
            medicalnet_model.layer4
        )


    def forward(
        self,
        x,
    ):

        x = self.conv1(
            x
        )

        x = self.bn1(
            x
        )

        x = self.relu(
            x
        )

        x = self.maxpool(
            x
        )


        c1 = self.layer1(
            x
        )

        c2 = self.layer2(
            c1
        )

        c3 = self.layer3(
            c2
        )

        c4 = self.layer4(
            c3
        )


        return {
            "layer1": c1,
            "layer2": c2,
            "layer3": c3,
            "layer4": c4,
        }


# ============================================================
# Dummy forward test
# ============================================================

def test_forward(
    backbone,
):

    print()
    print(
        "[6] Dummy backbone forward"
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    backbone = backbone.to(
        device
    )

    backbone.eval()


    # 너무 큰 tensor를 쓰면 12GB에서도
    # 불필요하게 메모리를 많이 먹으므로
    # 구조 검증용 96^3 사용
    x = torch.randn(
        1,
        1,
        96,
        96,
        96,
        device=device,
    )


    print(
        "Input:",
        tuple(
            x.shape
        )
    )


    with torch.no_grad():

        outputs = backbone(
            x
        )


    print()
    print(
        "Feature shapes:"
    )


    for name, feature in outputs.items():

        print(
            f"{name:8s}: "
            f"{tuple(feature.shape)}"
        )


    expected_channels = {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
    }


    for name, channels in expected_channels.items():

        actual = outputs[
            name
        ].shape[1]


        if actual != channels:

            raise RuntimeError(
                f"{name} channel 이상: "
                f"expected={channels}, "
                f"actual={actual}"
            )


    print()
    print(
        "✅ ResNet50 feature channel 검증 PASS"
    )


    return outputs


# ============================================================
# Gradient check
# ============================================================

def check_trainable(
    backbone,
):

    print()
    print(
        "[7] Trainable parameter check"
    )


    total_params = sum(
        parameter.numel()
        for parameter
        in backbone.parameters()
    )


    trainable_params = sum(
        parameter.numel()
        for parameter
        in backbone.parameters()
        if parameter.requires_grad
    )


    print(
        "Total parameters:",
        f"{total_params:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}",
    )


    if trainable_params == 0:

        raise RuntimeError(
            "Backbone parameter가 모두 freeze 상태입니다."
        )


    print(
        "✅ Backbone fine-tuning 가능"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "Candidate 2 - Med3D ResNet50 Backbone Validation"
    )

    print(
        "=" * 80
    )


    print()
    print(
        "Torch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.cuda.is_available(),
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )


    # --------------------------------------------------------
    # 1. MedicalNet source
    # --------------------------------------------------------

    prepare_medicalnet()


    # --------------------------------------------------------
    # 2. Import / build
    # --------------------------------------------------------

    resnet_module = (
        import_medicalnet()
    )


    model = build_med3d_resnet50(
        resnet_module
    )


    # --------------------------------------------------------
    # 3. Weight
    # --------------------------------------------------------

    checkpoint = (
        load_checkpoint()
    )


    state_dict = (
        extract_state_dict(
            checkpoint
        )
    )


    compatible = (
        load_pretrained(
            model,
            state_dict,
        )
    )


    # --------------------------------------------------------
    # 4. Backbone
    # --------------------------------------------------------

    backbone = Med3DBackbone(
        model
    )


    # --------------------------------------------------------
    # 5. Forward
    # --------------------------------------------------------

    outputs = test_forward(
        backbone
    )


    # --------------------------------------------------------
    # 6. Trainable
    # --------------------------------------------------------

    check_trainable(
        backbone
    )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "✅ Candidate 2 Med3D Backbone Validation PASS"
    )

    print()
    print(
        "Architecture:"
    )

    print(
        "Med3D pretrained 3D ResNet50"
    )

    print(
        "        ↓"
    )

    print(
        "Feature maps:"
    )

    print(
        "layer1 = 256 channels"
    )

    print(
        "layer2 = 512 channels"
    )

    print(
        "layer3 = 1024 channels"
    )

    print(
        "layer4 = 2048 channels"
    )

    print()
    print(
        "Next:"
    )

    print(
        "Med3D backbone → FPN → MONAI 3D RetinaNet 연결"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()