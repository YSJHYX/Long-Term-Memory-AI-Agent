"""记忆管理核心业务逻辑"""
import logging
import uuid
from typing import Any

from .config import settings
from .database import db
from .embeddings import embedding_service
from .models import Memory, MemoryResult, SaveMemoryRequest
from .utils import format_timestamp, generate_text_hash, get_timestamp

logger = logging.getLogger(__name__)

class MemoryService:
    """记忆服务"""
    
    def save_memory(self, request: SaveMemoryRequest) -> tuple[str, bool, str]:
        """保存记忆"""
        # 验证长度
        if len(request.text) > settings.max_text_length:
            raise ValueError(f"文本超过最大长度 {settings.max_text_length}")
        
        # 检查重复
        text_hash = generate_text_hash(request.text)
        existing = db.get_memory_by_hash(text_hash)
        if existing:
            logger.info(f"发现重复记忆: {existing.id}")
            return existing.id, True, "duplicate"
        
        # 生成嵌入向量
        embedding = embedding_service.encode(request.text)
        
        # 创建记忆对象
        memory = Memory(
            id=str(uuid.uuid4()),
            text=request.text,
            text_hash=text_hash,
            embedding=embedding,
            project=request.project,
            tags=request.tags,
            created_at=get_timestamp(),
            updated_at=get_timestamp(),
        )
        
        # 保存到数据库
        db.save_memory(memory)
        logger.info(f"记忆已保存: {memory.id}")
        
        return memory.id, False, "created"
    
    def search_memory(
        self,
        query: str,
        project: str | None = None,
        limit: int = 5,
        threshold: float = 0.7,
    ) -> list[MemoryResult]:
        """搜索记忆"""
        # 生成查询向量
        query_embedding = embedding_service.encode(query)
        
        # 获取记忆
        memories = db.get_all_memories(project=project)
        
        # 计算相似度
        results = []
        for memory in memories:
            if memory.embedding is None:
                continue
            
            score = embedding_service.cosine_similarity(
                query_embedding,
                memory.embedding
            )
            
            if score >= threshold:
                results.append((memory, score))
        
        # 按分数排序并取前 N 个
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]
        
        # 转为响应格式
        return [
            MemoryResult(
                id=memory.id,
                text=memory.text,
                score=round(score, 4),
                project=memory.project,
                tags=memory.tags,
                created_at=format_timestamp(memory.created_at),
            )
            for memory, score in results
        ]

memory_service = MemoryService()