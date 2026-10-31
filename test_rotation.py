import math
import numpy as np

def get_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Compute rotation matrix from NED to OpenCV camera frame.
    
    yaw: compass heading in degrees (-180 to 180 or 0 to 360)
    pitch: tilt angle (-90 is nadir / straight down, 0 is horizontal)
    roll: roll angle
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    
    # R_z(yaw)
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # R_y(pitch)
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # R_x(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # R_ned_to_body = (Rz @ Ry @ Rx).T
    # Gimbal to OpenCV camera: X_cam = Y_body, Y_cam = Z_body, Z_cam = X_body
    R_body_to_cam = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=float)
    
    R_ned_to_body = (Rz @ Ry @ Rx).T
    R_ned_to_cam = R_body_to_cam @ R_ned_to_body
    return R_ned_to_cam

# Test 1: Nadir looking straight down (pitch = -90, yaw = 0, roll = 0)
R = get_rotation_matrix(0, -90, 0)
# Point 50m straight down (North=0, East=0, Down=50)
pt_down = np.array([0, 0, 50.0])
cam_pt = R @ pt_down
print("Nadir (yaw=0, pitch=-90, roll=0) on pt_down:", cam_pt)
# Test 2: Point 10m North of drone on ground (North=10, East=0, Down=50)
pt_north = np.array([10.0, 0, 50.0])
cam_pt_north = R @ pt_north
print("Pt 10m North:", cam_pt_north)
# Test 3: Point 10m East of drone on ground (North=0, East=10, Down=50)
pt_east = np.array([0, 10.0, 50.0])
cam_pt_east = R @ pt_east
print("Pt 10m East:", cam_pt_east)
