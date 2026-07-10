from fastapi import APIRouter, UploadFile, File
from app.common.response import ApiResponse
from app.modules.disease.schemas import RecognizeResponse
from app.modules.disease.service import DiseaseService

router = APIRouter(prefix="/api/v1/disease", tags=["disease"])


@router.post("/recognize", response_model=ApiResponse)
async def recognize(file: UploadFile = File(...)):
    result = await DiseaseService.recognize(file)
    return ApiResponse.success(data=result, message="Recognition completed")
