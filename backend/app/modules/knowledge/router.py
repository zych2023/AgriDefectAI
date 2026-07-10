from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.common.response import ApiResponse, PaginatedData
from app.modules.knowledge.schemas import PestResponse, SearchResult
from app.modules.knowledge.service import KnowledgeService

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.get("/search", response_model=ApiResponse[SearchResult])
async def search(
    q: str = Query(..., min_length=1, description="Search keyword"),
    db: AsyncSession = Depends(get_db),
):
    result = await KnowledgeService.search(db, q)
    return ApiResponse.success(data=result)


@router.get("/catalog", response_model=ApiResponse[PaginatedData])
async def list_catalog(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    type: str | None = Query(None, alias="type", description="Filter: disease, pest, weed"),
    db: AsyncSession = Depends(get_db),
):
    result = await KnowledgeService.get_catalog_list(
        db, pest_type=type, page=page, page_size=page_size
    )
    return ApiResponse.success(data=result)


@router.get("/catalog/{catalog_id}", response_model=ApiResponse[PestResponse])
async def get_catalog_detail(catalog_id: int, db: AsyncSession = Depends(get_db)):
    result = await KnowledgeService.get_catalog_detail(db, catalog_id)
    return ApiResponse.success(data=result)
