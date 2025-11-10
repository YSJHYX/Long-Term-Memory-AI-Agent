from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 数据库配置
    db_path: str = "./data/memory.db"
    
    # 嵌入模型配置
    embed_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embed_device: str = "cpu"
    
    # API 配置
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_key: str | None = None
    
    # 业务配置
    max_text_length: int = 10000
    similarity_threshold: float = 0.7
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    def ensure_db_dir(self) -> None:
        """确保数据库目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

settings = Settings()