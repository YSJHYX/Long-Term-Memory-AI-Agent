"""数据库操作"""
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .config import settings
from .models import Memory

logger = logging.getLogger(__name__)

class Database:
    """SQLite 数据库管理器"""
    
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.conn: sqlite3.Connection | None = None
    
    def connect(self) -> None:
        """连接数据库并初始化表结构"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"数据库连接成功: {self.db_path}")
        
        self._init_schema()
    
    def _init_schema(self) -> None:
        """创建数据表"""
        if self.conn is None:
            raise RuntimeError("数据库未连接")
        
        cursor = self.conn.cursor()
        
        # 创建记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                text_hash TEXT,
                embedding BLOB,
                project TEXT,
                tags TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                archived INTEGER DEFAULT 0
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_project ON memories(project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON memories(text_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")
        
        self.conn.commit()
        logger.info("数据库表结构初始化完成")
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
    
    def save_memory(self, memory: Memory) -> None:
        """保存记忆"""
        if self.conn is None:
            raise RuntimeError("数据库未连接")
        
        # 将嵌入向量转为 JSON 字符串
        embedding_bytes = None
        if memory.embedding:
            embedding_bytes = json.dumps(memory.embedding).encode("utf-8")
        
        tags_str = json.dumps(memory.tags)
        
        self.conn.execute("""
            INSERT INTO memories 
            (id, text, text_hash, embedding, project, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.text,
            memory.text_hash,
            embedding_bytes,
            memory.project,
            tags_str,
            memory.created_at,
            memory.updated_at,
        ))
        self.conn.commit()
    
    def get_memory_by_hash(self, text_hash: str) -> Memory | None:
        """根据哈希值查找记忆（去重用）"""
        if self.conn is None:
            raise RuntimeError("数据库未连接")
        
        cursor = self.conn.execute(
            "SELECT * FROM memories WHERE text_hash = ? AND archived = 0",
            (text_hash,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_memory(row)
    
    def get_all_memories(
        self,
        project: str | None = None,
        tags: list[str] | None = None,
    ) -> list[Memory]:
        """获取所有记忆（支持过滤）"""
        if self.conn is None:
            raise RuntimeError("数据库未连接")
        
        query = "SELECT * FROM memories WHERE archived = 0"
        params: list[Any] = []
        
        if project:
            query += " AND project = ?"
            params.append(project)
        
        query += " ORDER BY created_at DESC"
        
        cursor = self.conn.execute(query, tuple(params))
        rows = cursor.fetchall()
        
        return [self._row_to_memory(row) for row in rows]
    
    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """将数据库行转为 Memory 对象"""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"].decode("utf-8"))
        
        tags = json.loads(row["tags"]) if row["tags"] else []
        
        return Memory(
            id=row["id"],
            text=row["text"],
            text_hash=row["text_hash"],
            embedding=embedding,
            project=row["project"],
            tags=tags,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

db = Database()