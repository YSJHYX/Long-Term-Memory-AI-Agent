"""向量嵌入生成服务"""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """嵌入向量服务"""
    
    def __init__(self) -> None:
        self.model_name = settings.embed_model
        self.device = settings.embed_device
        self.model: SentenceTransformer | None = None
    
    def load_model(self) -> None:
        """加载模型"""
        if self.model is None:
            logger.info(f"加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("模型加载完成")
    
    def encode(self, text: str) -> list[float]:
        """生成文本嵌入向量"""
        if self.model is None:
            self.load_model()
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))

embedding_service = EmbeddingService()