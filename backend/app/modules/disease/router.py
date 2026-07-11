from fastapi import APIRouter, Depends, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.deps import get_current_user, get_optional_user
from app.common.response import ApiResponse, PaginatedData
from app.modules.disease.schemas import RecognizeResponse
from app.modules.disease.service import DiseaseService

router = APIRouter(prefix="/api/v1/disease", tags=["disease"])


@router.post("/recognize", response_model=ApiResponse)
async def recognize(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_optional_user),
):
    """上传叶片图片识别病虫害。无需登录；登录后自动保存识别历史。"""
    result = await DiseaseService.recognize(
        file,
        db=db if user else None,
        farmer_id=user.id if user and user.role == "farmer" else None,
    )
    return ApiResponse.success(data=result, message="Recognition completed")


@router.get("/records", response_model=ApiResponse[PaginatedData])
async def list_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """查询当前农户的识别历史"""
    result = await DiseaseService.get_records(db, farmer_id=user.id, page=page, page_size=page_size)
    return ApiResponse.success(data=result)
