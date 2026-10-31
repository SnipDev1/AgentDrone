from src.dataset import DroneDataset
from src.ingestion.cadastral_registry import CadastralRegistry
from src.ingestion.telemetry_parser import TelemetryParser
import numpy as np

ds = DroneDataset()
print(f"Total plots: {len(ds.list_plots())}")

for pid in ds.list_plots():
    plot = ds.get_plot(pid)
    parcel = CadastralRegistry.get_parcel(pid)
    # Find high nadir shot
    high_nadir = None
    for img_p in plot.rgb_images:
        meta = TelemetryParser.parse_image_metadata(img_p)
        if meta.gimbal_pitch_deg is not None and meta.gimbal_pitch_deg <= -75.0 and (meta.altitude_rel_m or 0) > 30.0:
            high_nadir = (img_p, meta)
            break
    if not high_nadir and plot.rgb_images:
        high_nadir = (plot.rgb_images[0], TelemetryParser.parse_image_metadata(plot.rgb_images[0]))
        
    print(f"Plot {pid}:")
    print(f"  Parcel: centroid=({parcel.centroid_lat}, {parcel.centroid_lon}), area={parcel.area_sqm} m2")
    if high_nadir:
        img_p, meta = high_nadir
        print(f"  High nadir: {img_p.name}, drone=({meta.latitude}, {meta.longitude}), alt={meta.altitude_rel_m}, pitch={meta.gimbal_pitch_deg}, yaw={meta.gimbal_yaw_deg}, roll={meta.gimbal_roll_deg}")
