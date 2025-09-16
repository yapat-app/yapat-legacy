# src/utils/settings.py
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvironmentError(Exception):
    """Custom exception for environment configuration errors."""
    pass

def _to_bool(val: Optional[str], default: bool = False) -> bool:
    """Convert string environment variable to boolean."""
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def _to_int(val: Optional[str], default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Convert string environment variable to integer with range validation."""
    if val is None:
        return default
    try:
        result = int(val)
        if min_val is not None and result < min_val:
            raise EnvironmentError(f"Value {result} is below minimum {min_val}")
        if max_val is not None and result > max_val:
            raise EnvironmentError(f"Value {result} is above maximum {max_val}")
        return result
    except ValueError:
        raise EnvironmentError(f"Invalid integer value: {val}")

def _get_env_file() -> Path:
    """Locate and validate the environment file."""
    env_file = os.getenv("ENVIRONMENT_FILE", ".env")
    candidates = [
        Path(env_file),                                   # absolute or relative to CWD
        Path("/app/instance") / env_file,                 # mounted instance dir
        Path(__file__).resolve().parents[2] / env_file,   # project root
    ]

    for p in candidates:
        if p.is_file():
            logger.info(f"Using environment file: {p}")
            return p
    
    # If .env is required but not found, show helpful message
    if env_file != ".env.example":
        example_file = Path(__file__).resolve().parents[2] / ".env.example"
        if example_file.exists():
            logger.error(f"No {env_file} file found. Copy .env.example to {env_file} and adjust values.")
        raise EnvironmentError(f"Environment file not found. Searched in: {[str(p) for p in candidates]}")
    
    return Path(".env")  # Default for fallback

# Load environment variables
try:
    env_path = _get_env_file()
    load_dotenv(dotenv_path=env_path, override=True)
except EnvironmentError as e:
    logger.warning(f"Environment file loading failed: {e}")
    if not os.getenv("IGNORE_ENV_ERRORS"):
        sys.exit(1)

# Server Configuration
APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = _to_int(os.getenv("PORT"), 1050, min_val=1, max_val=65535)
APP_DEBUG = _to_bool(os.getenv("DEBUG"), False)
DEV_TOOLS_PROPS_CHECK = _to_bool(os.getenv("DEV_TOOLS_PROPS_CHECK"), False)

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY and APP_DEBUG:
    logger.warning("No SECRET_KEY set. Using an insecure default for development.")
    SECRET_KEY = "dev-secret-key"
elif not SECRET_KEY:
    raise EnvironmentError("SECRET_KEY must be set in production")

API_KEY = os.getenv("API_KEY")
if not API_KEY and not APP_DEBUG:
    raise EnvironmentError("API_KEY must be set in production")

# Database Configuration
INSTANCE_DIR = Path("/app/instance") if os.path.exists("/app") else Path(__file__).resolve().parents[2] / "instance"
os.makedirs(INSTANCE_DIR, exist_ok=True)

USER_DB_PATH = os.getenv("USER_DB_PATH", "user.db")
PIPELINE_DB_PATH = os.getenv("PIPELINE_DB_PATH", "pipeline.db")

SQLALCHEMY_DATABASE_PATHS = {
    'user_db': str(INSTANCE_DIR / USER_DB_PATH),
    'pipeline_db': str(INSTANCE_DIR / PIPELINE_DB_PATH),
}

# Storage Configuration
AUDIO_STORAGE_PATH = os.getenv("AUDIO_STORAGE_PATH")
EMBEDDINGS_STORAGE_PATH = os.getenv("EMBEDDINGS_STORAGE_PATH")

# Ensure critical paths exist
for path in [INSTANCE_DIR, AUDIO_STORAGE_PATH, EMBEDDINGS_STORAGE_PATH]:
    if path:
        os.makedirs(Path(path), exist_ok=True)
