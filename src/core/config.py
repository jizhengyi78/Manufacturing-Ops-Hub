"""
config.py — 全局配置管理
=======================
角色：整个系统的唯一配置入口，所有模块从这里获取配置。
数据来源优先级: 环境变量 > .env 文件 > 代码默认值
使用方式: settings = get_settings()  (单例, 全局缓存)

关键设计:
- 所有配置项集中管理，不散落在各模块中硬编码
- .env 文件不进入 Git (见 .gitignore)
- 生产环境通过 K8s ConfigMap/Secret 注入环境变量
- 敏感值 (API Key, Secret) 必须通过环境变量注入，不允许在 .env.example 有真实值

使用示例:
    from src.core.config import get_settings
    settings = get_settings()
    model = settings.default_model  # "deepseek-chat"

注意事项:
- 新增配置项时同步更新 .env.example
- 不要在代码里 os.getenv()，统一走这里
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略 .env 中未定义的字段
    )

    # ── LLM ──────────────────────────────────────
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model: str = "deepseek-chat"
    fallback_model: str = "deepseek-chat"
    simple_task_model: str = "deepseek-chat"
    complex_task_model: str = "deepseek-chat"

    # ── Milvus ───────────────────────────────────
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""

    # ── Elasticsearch ────────────────────────────
    es_host: str = "http://localhost:9200"
    es_user: str = ""
    es_password: str = ""

    # ── PostgreSQL ───────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "manufacturing"
    postgres_password: str = "manufacturing"
    postgres_db: str = "manufacturing_agent"

    # ── Redis ────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # ── Embedding ────────────────────────────────
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_dim: int = 1024
    hf_endpoint: str = ""  # HuggingFace 镜像

    # ── Reranker ─────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"

    # ── JWT / Auth ───────────────────────────────
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # ── A2A ──────────────────────────────────────
    a2a_secret_key: str = ""
    a2a_signature_algorithm: str = "HS256"
    a2a_key_rotation_days: int = 30

    # ── Rate Limit ───────────────────────────────
    rate_limit_per_minute: int = 30
    a2a_max_concurrent: int = 20
    max_parallel_agents: int = 3

    # ── Token Budget ─────────────────────────────
    token_budget_warn_ratio: float = 0.8
    token_budget_degrade_ratio: float = 0.9
    token_budget_limit_ratio: float = 1.0

    # ── Circuit Breaker ──────────────────────────
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: int = 30
    circuit_max_cooldown_seconds: int = 300

    # ── Checkpoint ───────────────────────────────
    checkpoint_ttl_seconds: int = 3600

    # ── Retrieval ────────────────────────────────
    retrieval_top_k: int = 20
    hybrid_bm25_weight_exact: float = 0.7
    hybrid_bm25_weight_mixed: float = 0.5
    hybrid_bm25_weight_semantic: float = 0.3

    # ── Compression ──────────────────────────────
    context_window_ratio_threshold: float = 0.7
    compressor_model: str = "qwen-turbo"

    # ── Offline ──────────────────────────────────
    offline_enabled: bool = True
    offline_llm_model: str = "qwen-2.5-14b-int4"

    # ── Logging ──────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Derived Properties ───────────────────────
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        base = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
        if self.redis_password:
            base = f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return base


@lru_cache
def get_settings() -> Settings:
    return Settings()
