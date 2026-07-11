from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.deps import get_current_user, require_role
from app.common.response import ApiResponse, PaginatedData
from app.modules.business.schemas import (
    FieldCreate, FieldUpdate, FieldResponse,
    CropCreate, CropUpdate, CropResponse,
    PestCreate, PestUpdate, PestResponse,
    FieldPestRecordResponse,
)
from app.modules.business.service import BusinessService

router = APIRouter(prefix="/api/v1/business", tags=["business"])

# ---- Field ----
@router.post("/fields", response_model=ApiResponse[FieldResponse])
async def create_field(data: FieldCreate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.create_field(db, data, farmer_id=user.id)
    return ApiResponse.success(data=result)

@router.get("/fields", response_model=ApiResponse[PaginatedData])
async def list_fields(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    farmer_id: int | None = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    if user.role == "farmer":
        farmer_id = user.id
    result = await BusinessService.get_fields(db, page=page, page_size=page_size, farmer_id=farmer_id)
    return ApiResponse.success(data=result)

@router.put("/fields/{field_id}", response_model=ApiResponse[FieldResponse])
async def update_field(field_id: int, data: FieldUpdate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.update_field(db, field_id, data)
    return ApiResponse.success(data=result)


# ---- Crop ----
@router.post("/crops", response_model=ApiResponse[CropResponse])
async def create_crop(data: CropCreate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.create_crop(db, data)
    return ApiResponse.success(data=result)

@router.get("/crops", response_model=ApiResponse[PaginatedData])
async def list_crops(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    result = await BusinessService.get_crops(db, page=page, page_size=page_size)
    return ApiResponse.success(data=result)

@router.put("/crops/{crop_id}", response_model=ApiResponse[CropResponse])
async def update_crop(crop_id: int, data: CropUpdate, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    result = await BusinessService.update_crop(db, crop_id, data)
    return ApiResponse.success(data=result)


# ---- Pest (admin/expert for CUD) ----
@router.post("/pests", response_model=ApiResponse[PestResponse])
async def create_pest(data: PestCreate, db: AsyncSession = Depends(get_db), user=Depends(require_role("admin", "expert"))):
    result = await BusinessService.create_pest(db, data)
    return ApiResponse.success(data=result)

@router.get("/pests", response_model=ApiResponse[PaginatedData])
async def list_pests(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    type: str | None = Query(None, alias="type"),
    db: AsyncSession = Depends(get_db),
):
    result = await BusinessService.get_pests(db, pest_type=type, page=page, page_size=page_size)
    return ApiResponse.success(data=result)

@router.put("/pests/{pest_id}", response_model=ApiResponse[PestResponse])
async def update_pest(pest_id: int, data: PestUpdate, db: AsyncSession = Depends(get_db), user=Depends(require_role("admin", "expert"))):
    result = await BusinessService.update_pest(db, pest_id, data)
    return ApiResponse.success(data=result)

@router.delete("/pests/{pest_id}", response_model=ApiResponse)
async def delete_pest(pest_id: int, db: AsyncSession = Depends(get_db), user=Depends(require_role("admin"))):
    await BusinessService.delete_pest(db, pest_id)
    return ApiResponse.success(message="Deleted successfully")


# ---- FieldPestRecord (admin/expert view) ----
@router.get("/field-pest-records", response_model=ApiResponse[PaginatedData])
async def list_field_pest_records(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    severity: str | None = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "expert")),
):
    result = await BusinessService.get_field_pest_records(db, severity=severity, page=page, page_size=page_size)
    return ApiResponse.success(data=result)
