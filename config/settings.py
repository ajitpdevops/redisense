import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_USERNAME: str = os.getenv("REDIS_USERNAME", "")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # AI/ML Settings
    ANOMALY_THRESHOLD: float = float(os.getenv("ANOMALY_THRESHOLD", "2.0"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L6-v2")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "384"))

    # Data Generation Settings
    DEVICE_COUNT: int = int(os.getenv("DEVICE_COUNT", "5"))
    DATA_GENERATION_INTERVAL: int = int(os.getenv("DATA_GENERATION_INTERVAL", "60"))
    ENERGY_MIN: float = float(os.getenv("ENERGY_MIN", "0.5"))
    ENERGY_MAX: float = float(os.getenv("ENERGY_MAX", "10.0"))
    ANOMALY_SPIKE_MULTIPLIER: float = float(os.getenv("ANOMALY_SPIKE_MULTIPLIER", "3.0"))

settings = Settings()
