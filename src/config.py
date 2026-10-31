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

DEFAULT_SEARCH_PATHS = [
    PROJECT_ROOT / "dataset" / "24.09.20206",
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "data" / "24.09.20206",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "24.09.20206",
]

def find_dataset_root(custom_path: str | Path | None = None) -> Path:
    """Find and return the root directory of the drone dataset.
    
    Resolution order:
    1. custom_path argument if provided.
    2. DATASET_DIR or DATA_DIR environment variable from .env or OS.
    3. Standard candidate paths relative to project root.
    
    If the resolved directory contains a single subfolder containing the cadastral
    directories (such as dataset/24.09.20206), that subfolder is automatically resolved.
    """
    candidates = []
    
    if custom_path:
        p = Path(custom_path)
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
        
    env_dir = os.getenv("DATASET_DIR") or os.getenv("DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
        
    candidates.extend(DEFAULT_SEARCH_PATHS)
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if this candidate directly has cadastral dirs (numeric folders)
            subdirs = [d for d in candidate.iterdir() if d.is_dir() and not d.name.startswith('.')]
            has_cadastral = any(d.name.isdigit() and len(d.name) >= 10 for d in subdirs)
            if has_cadastral:
                return candidate
            
            # Check if candidate has a subfolder with date name (e.g. 24.09.20206) that has cadastral dirs
            for sub in subdirs:
                sub_children = [d for d in sub.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if any(d.name.isdigit() and len(d.name) >= 10 for d in sub_children):
                    return sub

    # If nothing matched, return the best candidate or default
    return candidates[0] if candidates else PROJECT_ROOT / "dataset"

DATASET_ROOT = find_dataset_root()
YANDEX_DISK_PUBLIC_URL = os.getenv(
    "YANDEX_DISK_PUBLIC_URL",
    "https://disk.yandex.ru/d/_R6xYg1frCGW0w"
)
