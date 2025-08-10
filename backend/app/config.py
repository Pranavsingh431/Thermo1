from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Database - standardized on PostgreSQL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/thermal_inspection")
    
    # JWT
    SECRET_KEY: str = "thermal-inspection-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_TYPES: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    UPLOAD_DIR: str = "static/thermal_images"
    PROCESSED_DIR: str = "static/processed_images"
    
    # Thermal Analysis
    AMBIENT_TEMPERATURE: float = 34.0
    POTENTIAL_HOTSPOT_THRESHOLD: float = 20.0  # +20°C above ambient
    CRITICAL_HOTSPOT_THRESHOLD: float = 40.0   # +40°C above ambient
    EMISSIVITY_DEFAULT: float = 0.95
    REFLECTED_TEMP_DEFAULT: float = 20.0
    ATMOSPHERIC_TEMP_DEFAULT: float = 20.0
    OBJECT_DISTANCE_DEFAULT: float = 1.0
    RELATIVE_HUMIDITY_DEFAULT: float = 50.0
    RADIOMETRIC_TOOL: str = os.getenv("RADIOMETRIC_TOOL", "auto")  # auto|flirpy|exiftool|none
    EXIFTOOL_PATH: str = os.getenv("EXIFTOOL_PATH", "exiftool")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", os.path.join("models", "cache"))

    # Object Storage
    USE_OBJECT_STORAGE: bool = os.getenv("USE_OBJECT_STORAGE", "false").lower() == "true"
    OBJECT_STORAGE_PROVIDER: str = os.getenv("OBJECT_STORAGE_PROVIDER", "none")  # none|s3|gcs
    OBJECT_STORAGE_BUCKET: str = os.getenv("OBJECT_STORAGE_BUCKET", "")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    GCS_SERVICE_ACCOUNT_JSON: str = os.getenv("GCS_SERVICE_ACCOUNT_JSON", "")  # path or JSON string

    # Observability
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    
    # Email Configuration (for Gmail notifications)
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = "singhpranav431@gmail.com"  # User's Gmail
    SMTP_PASSWORD: str = ""  # Will be set via env var - use app password
    
    # Notification settings
    CHIEF_ENGINEER_EMAIL: str = "singhpranav431@gmail.com"  # Using user's email for testing
    
    # OpenRouter / LLM
    OPEN_ROUTER_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODELS: List[str] = [
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1-0528:free",
        "deepseek/deepseek-chat-v3-0324:free",
    ]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Production Settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Mumbai Salsette Substation Coordinates (for testing)
    SALSETTE_CAMP_LAT: float = 19.1262
    SALSETTE_CAMP_LON: float = 72.8897
    
    class Config:
        env_file = ".env"

settings = Settings()    