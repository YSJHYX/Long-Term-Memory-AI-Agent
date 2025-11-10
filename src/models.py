from pydantic import BaseModel, Field

# ===== 请求模型 =====
class SaveMemoryRequest(BaseModel):
    """保存记忆请求"""
    text: str = Field(..., max_length=10000, description="记忆内容")
    project: str | None = Field(None, description="项目名称")
    tags: list[str] = Field(default_factory=list, description="标签列表")

# ===== 响应模型 =====
class MemoryResult(BaseModel):
    """记忆搜索结果"""
    id: str
    text: str
    score: float | None = None
    project: str | None = None
    tags: list[str]
    created_at: str

class SearchMemoryResponse(BaseModel):
    """搜索响应"""
    query: str
    results: list[MemoryResult]
    total: int

# ===== 内部模型 =====
class Memory(BaseModel):
    """内部记忆表示"""
    id: str
    text: str
    text_hash: str
    embedding: list[float] | None = None
    project: str | None = None
    tags: list[str]
    created_at: int
    updated_at: int

