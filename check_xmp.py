import re
from src.ingestion.telemetry_parser import TelemetryParser

xmp = TelemetryParser.extract_image_xmp("data/samples/DJI_20260807124907_0002_D.JPG")
for k, v in re.findall(r'drone-dji:([A-Za-z0-9]+)="([^"]*)"', xmp):
    print(f"{k}: {v}")
