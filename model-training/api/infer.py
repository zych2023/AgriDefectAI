"""
病虫害推理接口 — 供外部系统调用
支持 ONNX (CPU/GPU) 和 PyTorch 两种后端
支持相似病例检索

用法:
    predictor = DiseasePredictor("checkpoints/model.onnx")
    result = predictor.predict("leaf.jpg")
    # → {"disease": "苹果黑星病一般", "confidence": 0.93, "similar_cases": [...], ...}
"""
import sys
import json
from pathlib import Path
# 将 model-training/ 根目录加入 path，以支持 from models.model import ...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
from PIL import Image


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DiseasePredictor:
    """
    农作物病害预测器

    - 支持 ONNX Runtime (默认) 和 PyTorch 两种后端
    - 自动加载类别名称和种植建议
    - 单张 / 批量推理
    """

    def __init__(self, model_path=None, backend="onnx",
                 config_path=None, knowledge_path=None,
                 index_path=None):
        # 默认路径均相对于 model-training/ 根目录（即本文件 ../）
        _root = Path(__file__).resolve().parent.parent
        if model_path is None:
            model_path = str(_root / "checkpoints" / "model.onnx")
        if config_path is None:
            config_path = str(_root / "config.yaml")
        if knowledge_path is None:
            knowledge_path = str(_root / "knowledge_base.json")
        if index_path is None:
            index_path = str(_root / "checkpoints" / "feature_index.pt")
        self.cfg = load_config(config_path)
        self.image_size = self.cfg["data"]["image_size"]
        self.num_classes = self.cfg["data"]["num_classes"]
        self.class_names = {int(k): v for k, v in self.cfg["class_names"].items()}

        # 加载知识库
        kb_path = Path(knowledge_path)
        if kb_path.exists():
            with open(kb_path, "r", encoding="utf-8") as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {}

        self.backend = backend
        self.device = "cpu"

        if backend == "onnx":
            self._init_onnx(model_path)
        elif backend == "pytorch":
            self._init_pytorch(model_path)
        else:
            raise ValueError(f"不支持的backend: {backend}，可选 onnx | pytorch")

        # 加载相似图片特征库
        self._load_index(index_path)

    def _init_onnx(self, model_path):
        """初始化 ONNX Runtime 推理后端"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请先安装 onnxruntime: pip install onnxruntime")

        # 尝试 GPU，降级 CPU
        providers = ["CPUExecutionProvider"]
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[OK] ONNX 推理器已初始化 [{model_path}]")

    def _init_pytorch(self, model_path):
        """初始化 PyTorch 推理后端"""
        import torch
        from models.model import CropDiseaseClassifier

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CropDiseaseClassifier(
            num_classes=self.num_classes, pretrained=False,
            dropout=self.cfg["model"]["dropout"]
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"[OK] PyTorch 推理器已初始化 [{self.device}]")

    def _load_index(self, index_path):
        """加载预计算的特征库（如果存在）"""
        idx_file = Path(index_path)
        if idx_file.exists():
            import torch
            data = torch.load(idx_file, map_location="cpu", weights_only=True)
            self.index_features = data["features"]       # [N, 2048] L2归一化
            self.index_paths = data["image_paths"]       # list of str
            self.index_labels = data["labels"]           # list of int
            self.index_class_names = data.get("class_names", self.class_names)
            print(f"[OK] 特征库已加载: {len(self.index_paths)} 张参考图")
            self._has_index = True
        else:
            self.index_features = None
            self.index_paths = []
            self.index_labels = []
            self._has_index = False
            print(f"[Info] 特征库未找到 ({index_path})，相似样例功能暂不可用")
            print(f"       请先运行: python build_index.py")

    def extract_features(self, image_input):
        """
        提取图片的 2048 维特征向量（用于相似度检索）

        Args:
            image_input: 图片路径(str) 或 numpy数组[H,W,3](np.ndarray)
        Returns:
            numpy array [2048]  L2归一化
        """
        if self.backend != "pytorch":
            raise RuntimeError("特征提取仅支持 PyTorch 后端")

        import torch

        if isinstance(image_input, str):
            input_tensor = self._preprocess(image_input)
        elif isinstance(image_input, np.ndarray):
            input_tensor = self._preprocess_from_array(image_input)
        else:
            raise TypeError(f"image_input 应为 str 或 numpy 数组, 收到 {type(image_input)}")

        tensor = torch.from_numpy(input_tensor).to(self.device)
        with torch.no_grad():
            features = self.model.extract_features(tensor)  # [1, 2048]
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()[0]

    def find_similar(self, query_features, top_k=5):
        """
        在特征库中搜索最相似的图片

        Args:
            query_features: numpy array [2048] (已 L2 归一化)
            top_k: 返回数量
        Returns:
            list of dict: [{"image_path", "class_id", "disease", "similarity"}, ...]
        """
        if not self._has_index:
            return []

        import torch
        query = torch.from_numpy(query_features).unsqueeze(0)   # [1, 2048]
        similarity = torch.mm(query, self.index_features.T)      # [1, N] 余弦相似度(=点积)
        similarity = similarity.squeeze(0)                       # [N]

        top_scores, top_indices = torch.topk(similarity, k=min(top_k, len(similarity)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            class_id = self.index_labels[idx]
            full_path = self.index_paths[idx]
            filename = Path(full_path).name if full_path else ""
            results.append({
                "image_path": full_path,
                "filename": filename,
                "image_url": f"/images/{filename}",
                "class_id": int(class_id) if isinstance(class_id, (int, float)) else -1,
                "disease": self.index_class_names.get(class_id, f"class_{class_id}"),
                "similarity": round(float(score), 4),
            })
        return results

    def _preprocess(self, image_path):
        """图片预处理: 读取→缩放→转tensor→归一化"""
        image = Image.open(image_path).convert("RGB")
        # Resize to match training pipeline
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # 转 numpy 并归一化
        img_np = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)  # HWC → CHW
        img_np = np.expand_dims(img_np, axis=0)  # [C,H,W] → [1,C,H,W]
        return img_np.astype(np.float32)

    def _preprocess_from_array(self, image_array):
        """numpy uint8 数组预处理 [H,W,3]"""
        img = Image.fromarray(image_array).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np.astype(np.float32)

    def _postprocess(self, outputs):
        """将模型输出转为结构化结果"""
        if isinstance(outputs, list):
            logits = outputs[0]  # ONNX 返回列表
        else:
            logits = outputs

        probs = self._softmax(logits[0])
        top5_idx = np.argsort(probs)[::-1][:5]

        results = []
        for rank, idx in enumerate(top5_idx):
            class_id = int(idx)
            disease_name = self.class_names.get(class_id, f"class_{class_id}")
            confidence = float(probs[idx])
            advice = self.knowledge_base.get(str(class_id), {})
            results.append({
                "rank": rank + 1,
                "class_id": class_id,
                "disease": disease_name,
                "confidence": round(confidence, 4),
                "advice": advice.get("advice", ""),
                "pesticide": advice.get("pesticide", ""),
            })

        return {
            "top_prediction": results[0],
            "top5": results,
        }

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, image_input):
        """
        单张图片推理

        Args:
            image_input: 图片路径(str) 或 numpy数组[H,W,3](np.ndarray)

        Returns:
            dict: {"top_prediction": {...}, "top5": [...], "similar_cases": [...]}
        """
        if isinstance(image_input, str):
            input_tensor = self._preprocess(image_input)
        elif isinstance(image_input, np.ndarray):
            input_tensor = self._preprocess_from_array(image_input)
        else:
            raise TypeError(f"image_input 应为 文件路径(str) 或 numpy 数组, 收到 {type(image_input)}")

        if self.backend == "onnx":
            outputs = self.session.run([self.session.get_outputs()[0].name],
                                       {self.input_name: input_tensor})
            result = self._postprocess(outputs)
            # ONNX 不支持直接提取特征，跳过相似搜索
            result["similar_cases"] = []
            return result
        elif self.backend == "pytorch":
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(input_tensor).to(self.device)
                outputs = self.model(tensor).cpu().numpy()
                features = self.model.extract_features(tensor)  # [1, 2048]
                features = torch.nn.functional.normalize(features, p=2, dim=1)

            result = self._postprocess(outputs)
            result["similar_cases"] = self.find_similar(features.cpu().numpy()[0], top_k=5)
            return result

    def predict_batch(self, image_paths):
        """批量推理 (仅 PyTorch 后端)"""
        if self.backend != "pytorch":
            raise RuntimeError("批量推理仅支持 PyTorch 后端")

        import torch
        batch_tensors = []
        for path in image_paths:
            tensor = self._preprocess(path)
            batch_tensors.append(torch.from_numpy(tensor))

        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch).cpu().numpy()

        results = []
        for i, path in enumerate(image_paths):
            probs = self._softmax(outputs[i])
            top1_idx = int(np.argmax(probs))
            results.append({
                "image": path,
                "class_id": top1_idx,
                "disease": self.class_names.get(top1_idx, f"class_{top1_idx}"),
                "confidence": round(float(probs[top1_idx]), 4),
            })
        return results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 默认用 PyTorch 后端（支持相似样例搜索）
    _root = Path(__file__).resolve().parent.parent
    predictor = DiseasePredictor(
        model_path=str(_root / "checkpoints" / "best_model.pth"),
        backend="pytorch",
    )

    # 找一张验证集图片做测试
    cfg = load_config(str(_root / "config.yaml"))
    val_images = Path(cfg["data"]["val_images"])
    test_images = list(val_images.glob("*.jpg"))[:3]

    if test_images:
        for img in test_images:
            result = predictor.predict(str(img))
            top = result["top_prediction"]
            print(f"\n{'='*60}")
            print(f"[IMG] {img.name}")
            print(f"  识别结果: {top['disease']}")
            print(f"  置信度:   {top['confidence']}")
            if top["advice"]:
                print(f"  防治建议: {top['advice']}")

            # 相似样例
            sim = result.get("similar_cases", [])
            if sim:
                print(f"\n  --- 相似病例 (Top {len(sim)}) ---")
                for s in sim:
                    fname = Path(s["image_path"]).name
                    print(f"  {s['disease']:20s}  相似度: {s['similarity']:.4f}  [{fname}]")
    else:
        print("[WARN] 未找到测试图片，请先训练模型")
