from pathlib import Path
import sys

import torch
import torch.nn as nn

from monai.apps.detection.networks.retinanet_network import RetinaNet
from monai.networks.blocks.backbone_fpn_utils import BackboneWithFPN


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEDICALNET_ROOT = (
    PROJECT_ROOT
    / "external"
    / "MedicalNet"
)

MED3D_WEIGHT = (
    PROJECT_ROOT
    / "weights"
    / "med3d"
    / "resnet_50_23dataset.pth"
)


# ============================================================
# Configuration
# ============================================================

SPATIAL_DIMS = 3

NUM_CLASSES = 1

NUM_ANCHORS = 3

FPN_OUT_CHANNELS = 256


# MedicalNet ResNet50 feature channels
RETURN_LAYERS = {
    "layer1": "0",
    "layer2": "1",
    "layer3": "2",
    "layer4": "3",
}

IN_CHANNELS_LIST = [
    256,
    512,
    1024,
    2048,
]


# ============================================================
# MedicalNet import
# ============================================================

def import_medicalnet():

    if not MEDICALNET_ROOT.exists():

        raise FileNotFoundError(
            f"MedicalNet source 없음:\n"
            f"{MEDICALNET_ROOT}"
        )


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
            f"MedicalNet resnet import 실패: {e}"
        )


    return resnet


# ============================================================
# Build Med3D ResNet50
# ============================================================

def build_med3d_backbone():

    resnet = import_medicalnet()


    model = resnet.resnet50(
        sample_input_D=96,
        sample_input_H=96,
        sample_input_W=96,
        num_seg_classes=2,
        shortcut_type="B",
        no_cuda=False,
    )


    return model


# ============================================================
# Load pretrained weights
# ============================================================

def load_med3d_weights(
    model,
):

    if not MED3D_WEIGHT.exists():

        raise FileNotFoundError(
            f"Med3D weight 없음:\n"
            f"{MED3D_WEIGHT}"
        )


    checkpoint = torch.load(
        MED3D_WEIGHT,
        map_location="cpu",
        weights_only=False,
    )


    if isinstance(
        checkpoint,
        dict,
    ) and (
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint


    model_state = (
        model.state_dict()
    )


    compatible = {}


    for key, value in state_dict.items():

        clean_key = key


        if clean_key.startswith(
            "module."
        ):

            clean_key = clean_key[
                len(
                    "module."
                ):
            ]


        if (
            clean_key
            in model_state
            and model_state[
                clean_key
            ].shape
            == value.shape
        ):

            compatible[
                clean_key
            ] = value


    missing, unexpected = (
        model.load_state_dict(
            compatible,
            strict=False,
        )
    )


    print(
        "Compatible parameters:",
        len(
            compatible
        ),
    )

    print(
        "Missing parameters:",
        len(
            missing
        ),
    )

    print(
        "Unexpected parameters:",
        len(
            unexpected
        ),
    )


    load_ratio = (
        len(
            compatible
        )
        / len(
            model_state
        )
    )


    print(
        "Load ratio:",
        f"{load_ratio:.3f}",
    )


    if load_ratio < 0.90:

        raise RuntimeError(
            "Med3D pretrained weight "
            "load ratio가 너무 낮습니다."
        )


    return model


# ============================================================
# Remove classifier / segmentation head dependency
# ============================================================

class MedicalNetBackbone(
    nn.Module
):

    def __init__(
        self,
        model,
    ):

        super().__init__()


        self.conv1 = model.conv1

        self.bn1 = model.bn1

        self.relu = model.relu

        self.maxpool = model.maxpool


        self.layer1 = model.layer1

        self.layer2 = model.layer2

        self.layer3 = model.layer3

        self.layer4 = model.layer4


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
# Build FPN
# ============================================================

def build_fpn(
    backbone,
):

    feature_extractor = (
        BackboneWithFPN(
            backbone=backbone,
            return_layers=RETURN_LAYERS,
            in_channels_list=(
                IN_CHANNELS_LIST
            ),
            out_channels=(
                FPN_OUT_CHANNELS
            ),
            spatial_dims=3,
        )
    )


    return feature_extractor


# ============================================================
# Build RetinaNet
# ============================================================

def build_retinanet(
    feature_extractor,
):

    # MedicalNet:
    #
    # conv1 stride 2
    # maxpool stride 2
    #
    # layer1 output ≈ input / 4
    #
    # RetinaNet 자체에서는 여기서
    # input divisibility만 확인하면 됨.
    #
    # Dummy QA는 96^3 사용.

    model = RetinaNet(
        spatial_dims=SPATIAL_DIMS,

        num_classes=NUM_CLASSES,

        num_anchors=NUM_ANCHORS,

        feature_extractor=(
            feature_extractor
        ),

        size_divisible=4,
    )


    return model


# ============================================================
# Print tensor shapes
# ============================================================

def print_feature_shapes(
    features,
):

    print()
    print(
        "FPN feature maps:"
    )


    if isinstance(
        features,
        dict,
    ):

        for key, value in (
            features.items()
        ):

            print(
                f"{key:8s}: "
                f"{tuple(value.shape)}"
            )

    else:

        for index, value in enumerate(
            features
        ):

            print(
                f"P{index}: "
                f"{tuple(value.shape)}"
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
        "Med3D ResNet50 + FPN + "
        "MONAI 3D RetinaNet Validation"
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


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ========================================================
    # 1. Backbone
    # ========================================================

    print()
    print(
        "[1] Build Med3D ResNet50"
    )


    med3d = (
        build_med3d_backbone()
    )


    print(
        "✅ MedicalNet ResNet50 생성"
    )


    # ========================================================
    # 2. Weight
    # ========================================================

    print()
    print(
        "[2] Load Med3D pretrained weight"
    )


    med3d = (
        load_med3d_weights(
            med3d
        )
    )


    print(
        "✅ Med3D pretrained weight load"
    )


    # ========================================================
    # 3. Detection backbone wrapper
    # ========================================================

    print()
    print(
        "[3] Build detection backbone"
    )


    backbone = (
        MedicalNetBackbone(
            med3d
        )
    )


    print(
        "✅ Detection backbone wrapper 생성"
    )


    # ========================================================
    # 4. FPN
    # ========================================================

    print()
    print(
        "[4] Build MONAI 3D FPN"
    )


    feature_extractor = (
        build_fpn(
            backbone
        )
    )


    feature_extractor = (
        feature_extractor.to(
            device
        )
    )


    print(
        "FPN output channels:",
        feature_extractor.out_channels,
    )


    print(
        "✅ MONAI FPN 생성"
    )


    # ========================================================
    # 5. FPN dummy forward
    # ========================================================

    print()
    print(
        "[5] Dummy FPN forward"
    )


    dummy = torch.randn(
        1,
        1,
        96,
        96,
        96,
        device=device,
    )


    feature_extractor.eval()


    with torch.no_grad():

        features = (
            feature_extractor(
                dummy
            )
        )


    print(
        "Input:",
        tuple(
            dummy.shape
        ),
    )


    print_feature_shapes(
        features
    )


    if len(
        features
    ) < 2:

        raise RuntimeError(
            "FPN feature level이 "
            "충분하지 않습니다."
        )


    print(
        "✅ FPN forward PASS"
    )


    # ========================================================
    # 6. RetinaNet
    # ========================================================

    print()
    print(
        "[6] Build MONAI 3D RetinaNet"
    )


    retinanet = build_retinanet(
        feature_extractor
    )


    retinanet = (
        retinanet.to(
            device
        )
    )


    retinanet.eval()


    print(
        "✅ RetinaNet 생성"
    )


    # ========================================================
    # 7. Full forward
    # ========================================================

    print()
    print(
        "[7] Full RetinaNet dummy forward"
    )


    with torch.no_grad():

        outputs = (
            retinanet(
                dummy
            )
        )


    if not isinstance(
        outputs,
        dict,
    ):

        raise RuntimeError(
            "RetinaNet output이 "
            "dict가 아닙니다."
        )


    if (
        "classification"
        not in outputs
    ):

        raise RuntimeError(
            "classification output 없음"
        )


    if (
        "box_regression"
        not in outputs
    ):

        raise RuntimeError(
            "box_regression output 없음"
        )


    cls_outputs = outputs[
        "classification"
    ]


    box_outputs = outputs[
        "box_regression"
    ]


    print()
    print(
        "Classification maps:"
    )


    for index, tensor in enumerate(
        cls_outputs
    ):

        print(
            f"Level {index}: "
            f"{tuple(tensor.shape)}"
        )


    print()
    print(
        "Box regression maps:"
    )


    for index, tensor in enumerate(
        box_outputs
    ):

        print(
            f"Level {index}: "
            f"{tuple(tensor.shape)}"
        )


    # ========================================================
    # 8. Output checks
    # ========================================================

    print()
    print(
        "[8] Output validation"
    )


    if (
        len(
            cls_outputs
        )
        != len(
            box_outputs
        )
    ):

        raise RuntimeError(
            "classification / bbox "
            "feature level 수 불일치"
        )


    for index, (
        cls_tensor,
        box_tensor,
    ) in enumerate(
        zip(
            cls_outputs,
            box_outputs,
        )
    ):

        expected_cls_channels = (
            NUM_ANCHORS
            * NUM_CLASSES
        )


        expected_box_channels = (
            NUM_ANCHORS
            * 2
            * SPATIAL_DIMS
        )


        if (
            cls_tensor.shape[1]
            != expected_cls_channels
        ):

            raise RuntimeError(
                f"Level {index} "
                f"classification channel 이상: "
                f"{cls_tensor.shape[1]} "
                f"!= "
                f"{expected_cls_channels}"
            )


        if (
            box_tensor.shape[1]
            != expected_box_channels
        ):

            raise RuntimeError(
                f"Level {index} "
                f"bbox channel 이상: "
                f"{box_tensor.shape[1]} "
                f"!= "
                f"{expected_box_channels}"
            )


    print(
        "✅ Classification head PASS"
    )

    print(
        "✅ Box regression head PASS"
    )


    # ========================================================
    # 9. Parameter check
    # ========================================================

    print()
    print(
        "[9] Parameter check"
    )


    total_params = sum(
        parameter.numel()
        for parameter
        in retinanet.parameters()
    )


    trainable_params = sum(
        parameter.numel()
        for parameter
        in retinanet.parameters()
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


    if (
        trainable_params
        == 0
    ):

        raise RuntimeError(
            "Trainable parameter 없음"
        )


    print(
        "✅ Fine-tuning 가능"
    )


    # ========================================================
    # Final
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "✅ Candidate 2 RetinaNet "
        "Architecture Validation PASS"
    )

    print()
    print(
        "Architecture:"
    )

    print(
        "Med3D pretrained "
        "3D ResNet50"
    )

    print(
        "        ↓"
    )

    print(
        "MONAI 3D FPN "
        "(256 channels)"
    )

    print(
        "        ↓"
    )

    print(
        "MONAI 3D RetinaNet"
    )

    print(
        "        ↓"
    )

    print(
        "Classification + "
        "3D Box Regression"
    )

    print()
    print(
        "Next:"
    )

    print(
        "Candidate 2 "
        "507 train / 27 val "
        "학습 구성"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()