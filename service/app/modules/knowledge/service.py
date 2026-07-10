from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.business.models import Pest, KnowledgeDoc
from app.modules.knowledge.schemas import PestResponse, KnowledgeDocResponse, SearchResult
from app.common.response import PaginatedData
from app.common.exceptions import NotFoundException


class KnowledgeService:

    @staticmethod
    async def search(db: AsyncSession, keyword: str) -> SearchResult:
        like_pattern = f"%{keyword}%"

        pest_q = select(Pest).where(
            or_(Pest.name.like(like_pattern), Pest.symptoms.like(like_pattern))
        ).limit(20)
        pest_rows = (await db.execute(pest_q)).scalars().all()

        doc_q = select(KnowledgeDoc).where(
            or_(KnowledgeDoc.title.like(like_pattern), KnowledgeDoc.content.like(like_pattern))
        ).limit(20)
        doc_rows = (await db.execute(doc_q)).scalars().all()

        pest_matches = [PestResponse.model_validate(r) for r in pest_rows]
        doc_matches = [KnowledgeDocResponse.model_validate(r) for r in doc_rows]

        return SearchResult(
            keyword=keyword,
            pest_matches=pest_matches,
            doc_matches=doc_matches,
            total=len(pest_matches) + len(doc_matches),
        )

    @staticmethod
    async def get_catalog_list(
        db: AsyncSession, pest_type: str | None = None, page: int = 1, page_size: int = 20
    ) -> PaginatedData:
        base = select(Pest)
        count_q = select(func.count()).select_from(Pest)
        if pest_type:
            base = base.where(Pest.type == pest_type)
            count_q = count_q.where(Pest.type == pest_type)
        total = (await db.execute(count_q)).scalar()
        rows = (
            await db.execute(
                base.order_by(Pest.id.desc()).offset((page - 1) * page_size).limit(page_size)
            )
        ).scalars().all()
        return PaginatedData(
            items=[PestResponse.model_validate(r) for r in rows],
            total=total,
            page=page,
            page_size=page_size,
        )

    @staticmethod
    async def get_catalog_detail(db: AsyncSession, catalog_id: int) -> PestResponse:
        pest = await db.get(Pest, catalog_id)
        if not pest:
            raise NotFoundException(message="Pest catalog entry not found")
        return PestResponse.model_validate(pest)
