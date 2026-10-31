"""Configuration management for AgentDrone.

Handles environment variable loading and dataset path resolution.
Supports both relative paths from project root and absolute paths.
"""

import os
from pathlib import Path

# Try to load python-dotenv if installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Minimal fallback parser if dotenv is not installed
    env_file = Path(__file__).resolve().parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "dataset" / "uploaded"


def find_dataset_root(custom_path: str | Path | None = None) -> Path:
    """Find and return the root directory of the drone dataset.

    By default, points to `dataset/uploaded` which starts empty so the user
    uploads their own cadastral plots via the web UI.
    """
    if custom_path:
        p = Path(custom_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    env_dir = os.getenv("DATASET_DIR") or os.getenv("DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        target = p if p.is_absolute() else PROJECT_ROOT / p
        target.mkdir(parents=True, exist_ok=True)
        return target

    DEFAULT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_UPLOAD_DIR


DATASET_ROOT = find_dataset_root()
YANDEX_DISK_PUBLIC_URL = os.getenv(
    "YANDEX_DISK_PUBLIC_URL",
    "https://disk.yandex.ru/d/_R6xYg1frCGW0w"
)
