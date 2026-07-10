"""
病虫害识别 — FastAPI 推理服务

启动:
    python api.py
    # → http://localhost:8000/docs 查看 Swagger 文档

API:
    POST /predict    — 上传图片 → 返回识别结果
    GET  /health     — 服务健康检查
    GET  /classes    — 返回支持的61类列表
"""
import sys
import base64
import io
from pathlib import Path

# 将 model-training/ 根目录加入 path
_MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MODEL_ROOT))

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from infer import DiseasePredictor

# ---- FastAPI 应用 ----
app = FastAPI(
    title="智慧农业AI病虫害识别",
    description="基于 ResNet50 的 36 类农作物病害识别服务",
    version="1.0.0",
)

# ---- 静态文件 ----
_STATIC_DIR = str(_MODEL_ROOT / "api" / "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ---- 全局推理器 (服务启动时加载一次) ----
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        # 优先使用 PyTorch 模型（支持相似样例搜索）
        pytorch_path = str(_MODEL_ROOT / "checkpoints" / "best_model.pth")
        onnx_path = str(_MODEL_ROOT / "checkpoints" / "model.onnx")

        if Path(pytorch_path).exists():
            backend = "pytorch"
            model_path = pytorch_path
        elif Path(onnx_path).exists():
            backend = "onnx"
            model_path = onnx_path
        else:
            raise FileNotFoundError(
                "未找到模型文件! 请先训练模型或将模型放到 checkpoints/ 目录"
            )

        predictor = DiseasePredictor(model_path=model_path, backend=backend)
    return predictor


# ---- 响应模型 ----
class PredictionItem(BaseModel):
    rank: int
    class_id: int
    disease: str
    confidence: float
    advice: str = ""
    pesticide: str = ""


class SimilarCase(BaseModel):
    image_path: str
    filename: str = ""
    image_url: str = ""
    class_id: int
    disease: str
    similarity: float


class PredictResponse(BaseModel):
    success: bool
    top_prediction: PredictionItem
    top5: list[PredictionItem]
    similar_cases: list[SimilarCase] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    similarity_enabled: bool = False


# ---- API 路由 ----
@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(str(_MODEL_ROOT / "api" / "static" / "index.html"))


@app.on_event("startup")
async def startup():
    """服务启动: 预加载模型"""
    print("[Loading] 正在加载模型...")
    get_predictor()
    print("[OK] 模型加载完成，服务就绪")


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    try:
        p = get_predictor()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            backend=p.backend,
            similarity_enabled=p._has_index,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            backend="none",
        )


@app.get("/classes")
async def list_classes():
    """返回所有支持的病害类别"""
    p = get_predictor()
    return {
        "num_classes": p.num_classes,
        "classes": [{"id": k, "name": v} for k, v in sorted(p.class_names.items())],
    }


@app.get("/images/{filename}")
async def serve_image(filename: str):
    """返回参考图片（用于相似样例展示）"""
    p = get_predictor()
    # 在训练集图片目录中查找
    train_dir = Path(p.cfg["data"]["train_images"])
    image_path = train_dir / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"图片未找到: {filename}")
    return FileResponse(str(image_path), media_type="image/jpeg")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    上传农作物叶片图片，返回病虫害识别结果

    - 支持 jpg/jpeg/png/bmp 格式
    - 返回 Top-5 预测及置信度
    - 返回对应防治建议
    """
    # 校验文件类型
    valid_types = {"image/jpeg", "image/png", "image/bmp", "image/jpg", "image/webp"}
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式: {file.content_type}。仅支持: jpg, png, bmp"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败: {str(e)}")

    try:
        p = get_predictor()
        result = p.predict(image_np)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

    return PredictResponse(success=True, **result)


# ---- 辅助: base64 输入 (方便前端调用) ----
class Base64Request(BaseModel):
    image: str  # base64 编码字符串


@app.post("/predict_base64")
async def predict_base64(req: Base64Request):
    """
    通过 Base64 编码图片进行推理 (适合前端直接调用)
    """
    try:
        # 移除 data:image/...;base64, 前缀 (如果有)
        b64_str = req.image
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        image_data = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 解码失败: {str(e)}")

    try:
        p = get_predictor()
        result = p.predict(image_np)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

    return {"success": True, **result}


# ---- 启动入口 ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
