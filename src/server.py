"""FastAPI 服务器"""
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import Any

from .config import settings
from .database import db
from .embeddings import embedding_service
from .memory import memory_service
from .models import (
    SaveMemoryRequest,
    SearchMemoryResponse,
    MemoryResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 启动
    logger.info("启动服务器...")
    settings.ensure_db_dir()
    db.connect()
    embedding_service.load_model()
    logger.info("服务器启动完成！")
    
    yield
    
    # 关闭
    logger.info("关闭服务器...")
    db.close()

app = FastAPI(
    title="Memory System",
    description="长期记忆存储系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> dict[str, str]:
    """根路径"""
    return {"name": "Memory System", "version": "1.0.0"}

@app.post("/memory/save")
def save_memory(request: SaveMemoryRequest) -> dict[str, Any]:
    """保存记忆"""
    try:
        memory_id, is_duplicate, reason = memory_service.save_memory(request)
        return {
            "id": memory_id,
            "saved": True,
            "duplicate": is_duplicate,
            "reason": reason,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"保存记忆失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/memory/search", response_model=SearchMemoryResponse)
def search_memory(
    q: str,
    project: str | None = None,
    limit: int = 5,
    threshold: float = 0.7,
) -> SearchMemoryResponse:
    """搜索记忆"""
    try:
        results = memory_service.search_memory(
            query=q,
            project=project,
            limit=limit,
            threshold=threshold,
        )
        return SearchMemoryResponse(
            query=q,
            results=results,
            total=len(results),
        )
    except Exception as e:
        logger.error(f"搜索记忆失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/health")
def health_check() -> dict[str, str]:
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)