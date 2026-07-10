"""
模型导出: PyTorch .pth → ONNX
- 支持动态 batch size
- 导出后可跨平台推理 (CPU/GPU, onnxruntime)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
import torch
from models.model import CropDiseaseClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_to_onnx(
    model_path="checkpoints/best_model.pth",
    output_path="checkpoints/model.onnx",
    dynamic_batch=True,
):
    cfg = load_config()
    device = torch.device("cpu")

    # 加载模型
    model = CropDiseaseClassifier(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 导出 ONNX
    dummy_input = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=12,
        do_constant_folding=True,
    )

    print(f"[OK] ONNX 模型已导出: {output_path}")
    print(f"   输入: [batch, 3, {cfg['data']['image_size']}, {cfg['data']['image_size']}]")
    print(f"   输出: [batch, {cfg['data']['num_classes']}]")
    print(f"   动态batch: {'[OK]' if dynamic_batch else '❌'}")

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"[OK] ONNX 模型结构验证通过")
    except ImportError:
        print("[WARN]  安装 onnx 库后可验证模型结构: pip install onnx")

    return output_path


if __name__ == "__main__":
    export_to_onnx()
