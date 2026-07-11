from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.common.exceptions import register_exception_handlers
from app.modules.auth.router import router as auth_router
from app.modules.disease.router import router as disease_router
from app.modules.knowledge.router import router as knowledge_router
from app.modules.agent.router import router as agent_router
from app.modules.business.router import router as business_router

app = FastAPI(title=settings.APP_NAME, version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
register_exception_handlers(app)

# Routers
app.include_router(auth_router)
app.include_router(disease_router)
app.include_router(knowledge_router)
app.include_router(agent_router)
app.include_router(business_router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}
