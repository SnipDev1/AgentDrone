import subprocess
import sys

mods = [
    "folium",
    "streamlit_folium",
    "pandas",
    "PIL",
    "streamlit",
    "src.cv.pipeline",
    "src.dataset",
    "src.engine.decision_engine",
    "src.ingestion.cadastral_registry",
    "src.ingestion.credit_registry",
    "src.models.schemas",
]

for m in mods:
    r = subprocess.run([sys.executable, "-c", f"import {m}"])
    print(f"MODULE: {m} -> EXIT: {r.returncode}")
