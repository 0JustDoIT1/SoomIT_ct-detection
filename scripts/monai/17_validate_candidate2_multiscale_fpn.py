from pathlib import Path
from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.apps.detection.networks.retinanet_network import RetinaNet


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


# ============================================================
# MedicalNet import
# ============================================================

def import_medicalnet():

    if not MEDICALNET_ROOT.exists():

        raise FileNotFoundError(
            f"MedicalNet source 없음:\n"
            f"{MEDICALNET_ROOT}"
        )


    if str(MEDICALNET_ROOT) not in sys.path:

        sys.path.insert(
            0,
            str(MEDICALNET_ROOT),
        )


    try:

        from models import resnet

    except Exception as e:

        raise RuntimeError(
            f"MedicalNet import 실패: {e}"
        )


    return resnet


# ============================================================
# Build MedicalNet ResNet50
# ============================================================

def build_med3d():

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
            MED3D_WEIGHT
        )


    checkpoint = torch.load(
        MED3D_WEIGHT,
        map_location="cpu",
        weights_only=False,
    )


    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint


    model_state = model.state_dict()

    compatible = {}


    for key, value in state_dict.items():

        clean_key = key


        if clean_key.startswith(
            "module."
        ):

            clean_key = clean_key[
                len("module.") :
            ]


        if (
            clean_key in model_state
            and model_state[
                clean_key
            ].shape == value.shape
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


    load_ratio = (
        len(compatible)
        / len(model_state)
    )


    print(
        "Compatible parameters:",
        len(compatible),
    )

    print(
        "Missing parameters:",
        len(missing),
    )

    print(
        "Unexpected parameters:",
        len(unexpected),
    )

    print(
        "Load ratio:",
        f"{load_ratio:.3f}",
    )


    if load_ratio < 0.90:

        raise RuntimeError(
            "Med3D pretrained weight "
            "load ratio too low"
        )


    return model


# ============================================================
# Med3D feature backbone
# ============================================================

class MedicalNetFeatureBackbone(
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

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)


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


        return (
            c1,
            c2,
            c3,
            c4,
        )


# ============================================================
# Custom Multi-scale FPN
# ============================================================

class Med3DMultiScaleFPN(
    nn.Module
):

    def __init__(
        self,
        backbone,
        out_channels=256,
    ):

        super().__init__()


        self.backbone = backbone

        self.out_channels = (
            out_channels
        )


        # ----------------------------------------------------
        # Med3D channels
        #
        # layer1 = 256
        # layer2 = 512
        # layer3 = 1024
        # layer4 = 2048
        # ----------------------------------------------------

        self.lateral_c1 = nn.Conv3d(
            256,
            out_channels,
            kernel_size=1,
        )


        self.lateral_c2 = nn.Conv3d(
            512,
            out_channels,
            kernel_size=1,
        )


        self.lateral_c4 = nn.Conv3d(
            2048,
            out_channels,
            kernel_size=1,
        )


        # ----------------------------------------------------
        # FPN smoothing
        # ----------------------------------------------------

        self.smooth_p2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )


        self.smooth_p3 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )


        # ----------------------------------------------------
        # Extra RetinaNet pyramid levels
        #
        # 12³ -> 6³ -> 3³
        # ----------------------------------------------------

        self.p4_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )


        self.p5_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )


    def forward(
        self,
        x,
    ):

        (
            c1,
            c2,
            c3,
            c4,
        ) = self.backbone(
            x
        )


        # ====================================================
        # P3
        #
        # c2와 c4는 spatial size가 모두 12³.
        #
        # c2:
        # 상대적으로 localization 정보가 강함
        #
        # c4:
        # high-level semantic feature
        #
        # 같은 resolution에서 fusion
        # ====================================================

        p3 = (
            self.lateral_c2(
                c2
            )
            +
            self.lateral_c4(
                c4
            )
        )


        p3 = self.smooth_p3(
            p3
        )


        # ====================================================
        # P2
        #
        # 12³ P3를 24³로 upsample 후
        # layer1 feature와 fusion
        # ====================================================

        p3_up = F.interpolate(
            p3,
            size=c1.shape[
                -3:
            ],
            mode="trilinear",
            align_corners=False,
        )


        p2 = (
            self.lateral_c1(
                c1
            )
            +
            p3_up
        )


        p2 = self.smooth_p2(
            p2
        )


        # ====================================================
        # P4
        #
        # 12³ -> 6³
        # ====================================================

        p4 = self.p4_conv(
            F.relu(
                p3
            )
        )


        # ====================================================
        # P5
        #
        # 6³ -> 3³
        # ====================================================

        p5 = self.p5_conv(
            F.relu(
                p4
            )
        )


        return OrderedDict(
            [
                (
                    "p2",
                    p2,
                ),

                (
                    "p3",
                    p3,
                ),

                (
                    "p4",
                    p4,
                ),

                (
                    "p5",
                    p5,
                ),
            ]
        )


# ============================================================
# Build RetinaNet
# ============================================================

def build_retinanet(
    feature_extractor,
):

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
# Print feature shapes
# ============================================================

def print_features(
    features,
):

    print()
    print(
        "Multi-scale FPN feature maps:"
    )


    for key, value in (
        features.items()
    ):

        print(
            f"{key:5s}: "
            f"{tuple(value.shape)}"
        )


# ============================================================
# Validate pyramid
# ============================================================

def validate_pyramid(
    features,
):

    expected = {
        "p2": (
            24,
            24,
            24,
        ),

        "p3": (
            12,
            12,
            12,
        ),

        "p4": (
            6,
            6,
            6,
        ),

        "p5": (
            3,
            3,
            3,
        ),
    }


    for key, expected_shape in (
        expected.items()
    ):

        if key not in features:

            raise RuntimeError(
                f"{key} 없음"
            )


        actual = tuple(
            features[
                key
            ].shape[
                -3:
            ]
        )


        if (
            actual
            != expected_shape
        ):

            raise RuntimeError(
                f"{key} shape 이상: "
                f"{actual} != "
                f"{expected_shape}"
            )


        channels = (
            features[
                key
            ].shape[1]
        )


        if (
            channels
            != FPN_OUT_CHANNELS
        ):

            raise RuntimeError(
                f"{key} channel 이상: "
                f"{channels}"
            )


    print()
    print(
        "✅ True multi-scale pyramid PASS"
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
        "Med3D Multi-scale FPN "
        "+ MONAI 3D RetinaNet Validation"
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


    if (
        torch.cuda.is_available()
    ):

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
    # 1. MedicalNet
    # ========================================================

    print()
    print(
        "[1] Build Med3D ResNet50"
    )


    med3d = build_med3d()


    print(
        "✅ MedicalNet ResNet50 생성"
    )


    # ========================================================
    # 2. Pretrained weight
    # ========================================================

    print()
    print(
        "[2] Load Med3D pretrained weight"
    )


    med3d = load_med3d_weights(
        med3d
    )


    print(
        "✅ Med3D weight load"
    )


    # ========================================================
    # 3. Backbone
    # ========================================================

    print()
    print(
        "[3] Build feature backbone"
    )


    backbone = (
        MedicalNetFeatureBackbone(
            med3d
        )
    )


    print(
        "✅ Med3D feature backbone 생성"
    )


    # ========================================================
    # 4. Custom Multi-scale FPN
    # ========================================================

    print()
    print(
        "[4] Build custom multi-scale FPN"
    )


    feature_extractor = (
        Med3DMultiScaleFPN(
            backbone,
            out_channels=(
                FPN_OUT_CHANNELS
            ),
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
        "✅ Multi-scale FPN 생성"
    )


    # ========================================================
    # 5. Dummy FPN forward
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


    print_features(
        features
    )


    validate_pyramid(
        features
    )


    # ========================================================
    # 6. Build RetinaNet
    # ========================================================

    print()
    print(
        "[6] Build MONAI 3D RetinaNet"
    )


    retinanet = build_retinanet(
        feature_extractor
    )


    retinanet = retinanet.to(
        device
    )


    retinanet.eval()


    print(
        "✅ RetinaNet 생성"
    )


    # ========================================================
    # 7. RetinaNet forward
    # ========================================================

    print()
    print(
        "[7] Full RetinaNet dummy forward"
    )


    with torch.no_grad():

        outputs = retinanet(
            dummy
        )


    if not isinstance(
        outputs,
        dict,
    ):

        raise RuntimeError(
            "RetinaNet output dict 아님"
        )


    cls_outputs = outputs.get(
        "classification"
    )


    box_outputs = outputs.get(
        "box_regression"
    )


    if cls_outputs is None:

        raise RuntimeError(
            "classification output 없음"
        )


    if box_outputs is None:

        raise RuntimeError(
            "box_regression output 없음"
        )


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
    # 8. Validate RetinaNet outputs
    # ========================================================

    print()
    print(
        "[8] RetinaNet output validation"
    )


    if (
        len(cls_outputs)
        != 4
    ):

        raise RuntimeError(
            "Classification pyramid "
            f"level 수 이상: "
            f"{len(cls_outputs)}"
        )


    if (
        len(box_outputs)
        != 4
    ):

        raise RuntimeError(
            "Box pyramid "
            f"level 수 이상: "
            f"{len(box_outputs)}"
        )


    expected_cls_channels = (
        NUM_ANCHORS
        * NUM_CLASSES
    )


    expected_box_channels = (
        NUM_ANCHORS
        * 2
        * SPATIAL_DIMS
    )


    expected_spatial = [
        (
            24,
            24,
            24,
        ),

        (
            12,
            12,
            12,
        ),

        (
            6,
            6,
            6,
        ),

        (
            3,
            3,
            3,
        ),
    ]


    for index, (
        cls_tensor,
        box_tensor,
    ) in enumerate(
        zip(
            cls_outputs,
            box_outputs,
        )
    ):

        # ----------------------------------------------------
        # Channels
        # ----------------------------------------------------

        if (
            cls_tensor.shape[1]
            != expected_cls_channels
        ):

            raise RuntimeError(
                f"Level {index} "
                "classification channel 이상"
            )


        if (
            box_tensor.shape[1]
            != expected_box_channels
        ):

            raise RuntimeError(
                f"Level {index} "
                "box channel 이상"
            )


        # ----------------------------------------------------
        # Spatial shape
        # ----------------------------------------------------

        cls_spatial = tuple(
            cls_tensor.shape[
                -3:
            ]
        )


        box_spatial = tuple(
            box_tensor.shape[
                -3:
            ]
        )


        if (
            cls_spatial
            != expected_spatial[
                index
            ]
        ):

            raise RuntimeError(
                f"Level {index} "
                f"classification spatial 이상: "
                f"{cls_spatial}"
            )


        if (
            box_spatial
            != expected_spatial[
                index
            ]
        ):

            raise RuntimeError(
                f"Level {index} "
                f"box spatial 이상: "
                f"{box_spatial}"
            )


    print(
        "✅ Classification head PASS"
    )

    print(
        "✅ Box regression head PASS"
    )

    print(
        "✅ RetinaNet multi-scale PASS"
    )


    # ========================================================
    # 9. Parameter check
    # ========================================================

    print()
    print(
        "[9] Parameter check"
    )


    total_params = sum(
        p.numel()
        for p in (
            retinanet.parameters()
        )
    )


    trainable_params = sum(
        p.numel()
        for p in (
            retinanet.parameters()
        )
        if p.requires_grad
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
        "✅ Candidate 2 "
        "Multi-scale RetinaNet "
        "Validation PASS"
    )

    print()
    print(
        "Final architecture:"
    )

    print()
    print(
        "Med3D pretrained "
        "3D ResNet50"
    )

    print(
        "        ↓"
    )

    print(
        "Custom Multi-scale FPN"
    )

    print(
        "        ↓"
    )

    print(
        "P2 = 24³"
    )

    print(
        "P3 = 12³"
    )

    print(
        "P4 = 6³"
    )

    print(
        "P5 = 3³"
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
        "BBox + Confidence"
    )

    print()
    print(
        "Next:"
    )

    print(
        "Candidate 2 "
        "507 train / 27 val "
        "training setup"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()