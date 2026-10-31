import re
import cv2
import numpy as np
from src.ingestion.telemetry_parser import TelemetryParser
from test_rotation import get_rotation_matrix

img_path = "data/samples/DJI_20260807124907_0002_D.JPG"
meta = TelemetryParser.parse_image_metadata(img_path)
xmp = TelemetryParser.extract_image_xmp(img_path)

# Extract calibrated camera intrinsics from XMP
focal_m = re.search(r'drone-dji:CalibratedFocalLength="([^"]+)"', xmp)
focal = float(focal_m.group(1)) if focal_m else 3725.15

cx_m = re.search(r'drone-dji:CalibratedOpticalCenterX="([^"]+)"', xmp)
cx = float(cx_m.group(1)) if cx_m else 2640.0

cy_m = re.search(r'drone-dji:CalibratedOpticalCenterY="([^"]+)"', xmp)
cy = float(cy_m.group(1)) if cy_m else 1978.0

dewarp_m = re.search(r'drone-dji:DewarpData="([^"]+)"', xmp)
if dewarp_m:
    # DewarpData: date;fx,fy,cx,cy,k1,k2,p1,p2,k3
    parts = dewarp_m.group(1).split(";")[-1].split(",")
    # k1, k2, p1, p2, k3
    k1 = float(parts[4])
    k2 = float(parts[5])
    p1 = float(parts[6])
    p2 = float(parts[7])
    k3 = float(parts[8])
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=float)
else:
    dist_coeffs = np.zeros(5, dtype=float)

K = np.array([
    [focal, 0, cx],
    [0, focal, cy],
    [0, 0, 1]
], dtype=float)

# Rotation matrix from NED to Cam
R = get_rotation_matrix(meta.gimbal_yaw_deg, meta.gimbal_pitch_deg, meta.gimbal_roll_deg)
rvec, _ = cv2.Rodrigues(R)
tvec = np.zeros((3, 1), dtype=float)

print("Camera Matrix K:\n", K)
print("Distortion Coeffs:", dist_coeffs)
print("Drone Lat/Lon/Alt:", meta.latitude, meta.longitude, meta.altitude_rel_m)

# Centroid of plot 5407055001178: 54.815520, 83.240680
c_lat = 54.815520
c_lon = 83.240680

# Convert to NED relative to drone
dx_north = (c_lat - meta.latitude) * 111139.0
dy_east = (c_lon - meta.longitude) * 111139.0 * np.cos(np.radians(meta.latitude))
dz_down = meta.altitude_rel_m

pt_ned = np.array([[dx_north, dy_east, dz_down]], dtype=np.float32)
print("Vector NED to Centroid:", pt_ned)

# Project
pts_2d, _ = cv2.projectPoints(pt_ned, rvec, tvec, K, dist_coeffs)
u, v = pts_2d[0][0]
print(f"Centroid projected pixel: u={u:.1f}, v={v:.1f} (Image size: 5280 x 3956)")
