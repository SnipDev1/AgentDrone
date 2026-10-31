import cv2
import numpy as np
from src.dataset import DroneDataset
from src.ingestion.cadastral_registry import CadastralRegistry
from src.ingestion.telemetry_parser import TelemetryParser

ds = DroneDataset()

for pid in ds.list_plots():
    plot = ds.get_plot(pid)
    parcel = CadastralRegistry.get_parcel(pid)
    
    # Find nadir image
    nadir_img = None
    nadir_meta = None
    for img_p in plot.rgb_images:
        meta = TelemetryParser.parse_image_metadata(img_p)
        if meta.gimbal_pitch_deg is not None and meta.gimbal_pitch_deg <= -75.0 and (meta.altitude_rel_m or 0) > 30.0:
            nadir_img = img_p
            nadir_meta = meta
            break
            
    if not nadir_img:
        continue
        
    # Read thumbnail to find dominant line angles
    img = cv2.imread(str(nadir_img))
    if img is None:
        continue
    h, w = img.shape[:2]
    # Resize to 1000px for speed
    scale = 1000.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            coords = line.flatten()
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle % 180)
            
    # Histogram of angles
    hist, bin_edges = np.histogram(angles, bins=18, range=(0, 180))
    dominant_img_angle = bin_edges[np.argmax(hist)]
    
    yaw = nadir_meta.gimbal_yaw_deg or 0.0
    roll = nadir_meta.gimbal_roll_deg or 0.0
    effective_yaw = yaw if roll < 90 else (yaw - 180) % 360
    
    print(f"Plot {pid}: drone_yaw={yaw:.1f}, roll={roll:.1f}, eff_yaw={effective_yaw:.1f}, img_line_angle={dominant_img_angle:.1f} deg")
