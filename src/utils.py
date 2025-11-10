"""工具函数"""
import hashlib
from datetime import datetime

def generate_text_hash(text: str) -> str:
    """生成文本哈希（用于去重）"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_timestamp() -> int:
    """获取当前 Unix 时间戳"""
    return int(datetime.now().timestamp())

def format_timestamp(timestamp: int) -> str:
    """格式化时间戳为 ISO 8601"""
    return datetime.fromtimestamp(timestamp).isoformat() + "Z"