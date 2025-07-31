import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Redis Configuration Toggle
    USE_REDIS_CLOUD: bool = os.getenv("USE_REDIS_CLOUD", "false").lower() == "true"

    # Redis Cloud Configuration
    REDIS_CLOUD_HOST: str = os.getenv("REDIS_CLOUD_HOST", "")
    REDIS_CLOUD_PORT: int = int(os.getenv("REDIS_CLOUD_PORT", "6380"))
    REDIS_CLOUD_USERNAME: str = os.getenv("REDIS_CLOUD_USERNAME", "")
    REDIS_CLOUD_PASSWORD: str = os.getenv("REDIS_CLOUD_PASSWORD", "")

    # Local Redis Configuration
    REDIS_LOCAL_HOST: str = os.getenv("REDIS_LOCAL_HOST", "redis")  # Use 'redis' for Docker, 'localhost' for local
    REDIS_LOCAL_PORT: int = int(os.getenv("REDIS_LOCAL_PORT", "6379"))
    REDIS_LOCAL_PASSWORD: str = os.getenv("REDIS_LOCAL_PASSWORD", "")

    # Dynamic Redis Settings (based on toggle)
    @property
    def REDIS_HOST(self) -> str:
        return self.REDIS_CLOUD_HOST if self.USE_REDIS_CLOUD else self.REDIS_LOCAL_HOST

    @property
    def REDIS_PORT(self) -> int:
        return self.REDIS_CLOUD_PORT if self.USE_REDIS_CLOUD else self.REDIS_LOCAL_PORT

    @property
    def REDIS_USERNAME(self) -> str:
        return self.REDIS_CLOUD_USERNAME if self.USE_REDIS_CLOUD else ""

    @property
    def REDIS_PASSWORD(self) -> str:
        return self.REDIS_CLOUD_PASSWORD if self.USE_REDIS_CLOUD else self.REDIS_LOCAL_PASSWORD

    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Web Application Settings
    WEB_HOST: str = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))

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
