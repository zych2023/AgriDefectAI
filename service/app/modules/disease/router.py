from fastapi import APIRouter, Depends, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.deps import get_current_user
from app.common.response import ApiResponse, PaginatedData
from app.modules.disease.schemas import FieldPestRecordResponse
from app.modules.disease.service import DiseaseService

router = APIRouter(prefix="/api/v1/disease", tags=["disease"])


@router.post("/recognize", response_model=ApiResponse[FieldPestRecordResponse])
async def recognize(
    field_id: int = Query(..., description="Field ID"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await DiseaseService.recognize(db, file, field_id=field_id, user_id=user.id)
    return ApiResponse.success(data=result, message="Recognition completed")


@router.get("/records", response_model=ApiResponse[PaginatedData])
async def list_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await DiseaseService.get_records(db, user_id=user.id, page=page, page_size=page_size)
    return ApiResponse.success(data=result)


@router.get("/records/{record_id}", response_model=ApiResponse[FieldPestRecordResponse])
async def get_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await DiseaseService.get_record_detail(db, record_id, user_id=user.id)
    return ApiResponse.success(data=result)
