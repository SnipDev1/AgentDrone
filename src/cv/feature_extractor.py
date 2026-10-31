"""Computer vision feature extraction for drone aerial imagery.

Extracts geometric, structural, color, texture, and glazing features to classify
building construction stages, wall materials, and defect anomalies.
Supports cadastral parcel polygon masking for strict parcel-bounded analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ImageFeatures:
    """Quantitative visual features extracted from a single drone shot."""
    image_name: str
    width: int
    height: int
    vegetation_ratio: float
    edge_density: float
    horizontal_lines_count: int
    vertical_lines_count: int
    structural_density: float
    dominant_hue: float
    mean_saturation: float
    color_shares: dict[str, float]
    has_large_structure: bool
    structure_area_ratio: float
    has_roof_contour: bool
    window_openings_count: int
    # Advanced stage & glazing verification fields
    glazed_windows_count: int = 0
    empty_openings_count: int = 0
    has_finished_facade: bool = False
    has_finished_roof: bool = False
    is_parcel_empty: bool = False


class FeatureExtractor:
    """Extracts CV features from RGB drone images with parcel-boundary support."""

    TARGET_WIDTH = 960
    TARGET_HEIGHT = 720

    @classmethod
    def extract_from_file(
        cls,
        img_path: str | Path,
        parcel_polygon: list[tuple[float, float]] | None = None,
    ) -> ImageFeatures | None:
        """Load image from disk and compute feature representation."""
        path = Path(img_path)
        if not path.exists():
            return None

        # Load image via OpenCV
        img = cv2.imread(str(path))
        if img is None:
            return None

        return cls.extract_from_image(img, image_name=path.name, parcel_polygon=parcel_polygon)

    @classmethod
    def extract_from_image(
        cls,
        img: np.ndarray,
        image_name: str = "image.jpg",
        parcel_polygon: list[tuple[float, float]] | None = None,
    ) -> ImageFeatures:
        """Process raw BGR image array and compute visual metrics.

        If parcel_polygon is provided (in original image coordinates), metrics
        are computed strictly within the parcel boundary mask, ignoring neighbor objects.
        """
        orig_h, orig_w = img.shape[:2]

        # Resize to fixed processing resolution for invariant thresholds and fast inference
        resized = cv2.resize(img, (cls.TARGET_WIDTH, cls.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]

        # 0. Build parcel mask if parcel_polygon is provided
        parcel_mask = None
        if parcel_polygon and len(parcel_polygon) >= 3:
            pts_scaled = []
            valid_polygon = True
            for pt in parcel_polygon:
                if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                    valid_polygon = False
                    break
                sx = float(pt[0]) * (cls.TARGET_WIDTH / float(orig_w))
                sy = float(pt[1]) * (cls.TARGET_HEIGHT / float(orig_h))
                # Protect against overflow if points are astronomically far offscreen
                if abs(sx) > 100000.0 or abs(sy) > 100000.0:
                    valid_polygon = False
                    break
                pts_scaled.append([sx, sy])

            if valid_polygon and len(pts_scaled) >= 3:
                pts_arr = np.array(pts_scaled, dtype=np.int32)
                p_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(p_mask, [pts_arr], 255)
                # Use mask if it covers a significant portion of the scene (>0.5%)
                if np.count_nonzero(p_mask) > (h * w * 0.005):
                    parcel_mask = p_mask

        total_pixels = float(np.count_nonzero(parcel_mask)) if parcel_mask is not None else float(h * w)

        # 1. Color space transformations
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 2. Vegetation analysis (Green HSV mask: H in 32..85, S > 35, V > 35)
        raw_green_mask = cv2.inRange(hsv, np.array([32, 35, 35]), np.array([85, 255, 255]))
        if parcel_mask is not None:
            green_mask = cv2.bitwise_and(raw_green_mask, raw_green_mask, mask=parcel_mask)
        else:
            green_mask = raw_green_mask

        veg_pixels = float(np.count_nonzero(green_mask))
        vegetation_ratio = veg_pixels / total_pixels

        # 3. Non-vegetation structural area (candidate building/ground region)
        if parcel_mask is not None:
            non_veg_mask = cv2.bitwise_and(cv2.bitwise_not(raw_green_mask), parcel_mask)
        else:
            non_veg_mask = cv2.bitwise_not(green_mask)

        # Remove small foliage noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_non_veg = cv2.morphologyEx(non_veg_mask, cv2.MORPH_OPEN, kernel)
        clean_non_veg = cv2.morphologyEx(clean_non_veg, cv2.MORPH_CLOSE, kernel)

        # 4. Edge and line detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Mask edges to only non-vegetation structural areas inside parcel
        structural_edges = cv2.bitwise_and(edges, edges, mask=clean_non_veg)
        edge_pixels = float(np.count_nonzero(structural_edges))
        edge_density = edge_pixels / total_pixels

        # Hough lines for architectural vertical and horizontal elements
        lines = cv2.HoughLinesP(
            structural_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10,
        )

        h_lines = 0
        v_lines = 0
        if lines is not None:
            for item in lines:
                is_nested = hasattr(item, "__len__") and len(item) == 1 and hasattr(item[0], "__len__")
                coords = item[0] if is_nested else item
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                dx = abs(int(x2) - int(x1))
                dy = abs(int(y2) - int(y1))
                if dx == 0 and dy == 0:
                    continue
                angle_deg = abs(math.degrees(math.atan2(dy, dx)))
                # Horizontal lines: <= 20°
                if angle_deg <= 20.0:
                    h_lines += 1
                # Vertical lines: >= 70°
                elif angle_deg >= 70.0:
                    v_lines += 1

        structural_density = (h_lines + v_lines) / 100.0

        # 5. Connected component & contour analysis for building body
        contours, _ = cv2.findContours(clean_non_veg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_large_structure = False
        structure_area_ratio = 0.0
        has_roof_contour = False
        has_finished_roof = False
        has_finished_facade = False
        window_openings = 0
        glazed_windows = 0
        empty_openings = 0

        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = sorted_contours[0]
            largest_area = cv2.contourArea(largest_contour)
            structure_area_ratio = largest_area / total_pixels

            # Building footprint criteria: occupies at least 5% of parcel (or frame)
            if structure_area_ratio >= 0.05:
                has_large_structure = True

                # Check if top portion has roof-like triangular or trapezoidal geometry
                peri = cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
                if 3 <= len(approx) <= 8 and largest_area > (total_pixels * 0.08):
                    has_roof_contour = True

                # Bounding box for building region
                bx, by, bw, bh = cv2.boundingRect(largest_contour)
                roi_gray = gray[by:by + bh, bx:bx + bw]
                roi_area = float(bw * bh)

                # Roof finish analysis: check upper 40% of building bounding box
                roof_h = int(bh * 0.40)
                if roof_h > 20 and bw > 30:
                    roof_patch = roi_gray[0:roof_h, :]
                    roof_edges = cv2.Canny(roof_patch, 50, 150)
                    roof_edge_ratio = np.count_nonzero(roof_edges) / float(roof_patch.size)
                    # Finished roof has continuous sheeting/tile texture with moderate edge density
                    if 0.02 < roof_edge_ratio < 0.25:
                        has_finished_roof = True

                # Facade wall analysis: lower 60% of building
                wall_y = roof_h
                if (bh - wall_y) > 30 and bw > 30:
                    wall_patch = roi_gray[wall_y:, :]
                    wall_edges = cv2.Canny(wall_patch, 40, 120)
                    wall_edge_density = np.count_nonzero(wall_edges) / float(wall_patch.size)
                    # Neat finished wall (siding, plaster, finished brick) has controlled micro-texture
                    if wall_edge_density < 0.18:
                        has_finished_facade = True

                # Window openings & glazing detector: find architectural cutouts in building wall
                if roi_gray.shape[0] > 40 and roi_gray.shape[1] > 40:
                    thresh = cv2.adaptiveThreshold(
                        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
                    )
                    win_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for wc in win_contours:
                        w_area = cv2.contourArea(wc)
                        # Realistic architectural window opening: 1.0% to 15% of building area
                        if (roi_area * 0.010) < w_area < (roi_area * 0.15):
                            wx, wy, ww, wh = cv2.boundingRect(wc)
                            aspect = float(ww) / float(wh) if wh > 0 else 0
                            if 0.45 <= aspect <= 2.2:
                                window_openings += 1

                                # Inspect internal window patch
                                w_patch_gray = roi_gray[wy:wy + wh, wx:wx + ww]
                                mean_val = float(np.mean(w_patch_gray))
                                std_val = float(np.std(w_patch_gray))

                                # Inner region for frame / mullion edge analysis
                                if wh > 8 and ww > 8:
                                    inner = w_patch_gray[2:-2, 2:-2]
                                    inner_edges = cv2.Canny(inner, 40, 120)
                                    inner_edge_ratio = np.count_nonzero(inner_edges) / float(inner.size)
                                else:
                                    inner_edge_ratio = 0.0

                                # Glazed window has contrasting frame, mullion crossbars, or specular reflections
                                if mean_val < 45 and std_val < 22 and inner_edge_ratio < 0.03:
                                    empty_openings += 1
                                elif inner_edge_ratio >= 0.04 or std_val >= 28:
                                    glazed_windows += 1
                                else:
                                    empty_openings += 1

        # 6. Wall & building color shares (focusing on non-green structural pixels)
        struct_pixels = resized[clean_non_veg > 0]
        color_shares = {
            "red_brick": 0.0,
            "gray_block": 0.0,
            "wood_timber": 0.0,
            "yellow_osb": 0.0,
            "metal_panel": 0.0,
        }

        dominant_h = 0.0
        mean_s = 0.0

        if len(struct_pixels) > 100:
            struct_hsv = hsv[clean_non_veg > 0]
            h_vals = struct_hsv[:, 0]
            s_vals = struct_hsv[:, 1]
            v_vals = struct_hsv[:, 2]

            dominant_h = float(np.median(h_vals))
            mean_s = float(np.mean(s_vals))

            n_pts = float(len(struct_pixels))

            # Red brick: H in [0..15] or [165..180], S > 45, V > 40
            brick_mask = ((h_vals <= 15) | (h_vals >= 165)) & (s_vals > 45) & (v_vals > 40)
            color_shares["red_brick"] = float(np.count_nonzero(brick_mask)) / n_pts

            # Wood timber: H in [16..32], S > 40, V in [40..210]
            wood_mask = (h_vals >= 16) & (h_vals <= 32) & (s_vals > 40) & (v_vals >= 40) & (v_vals <= 210)
            color_shares["wood_timber"] = float(np.count_nonzero(wood_mask)) / n_pts

            # Yellow/beige OSB or shield panel: H in [18..35], S in [25..90], V > 120
            osb_mask = (h_vals >= 18) & (h_vals <= 35) & (s_vals >= 25) & (s_vals <= 90) & (v_vals > 120)
            color_shares["yellow_osb"] = float(np.count_nonzero(osb_mask)) / n_pts

            # Gray concrete/gas block: S < 35, V in [60..210]
            gray_mask = (s_vals < 35) & (v_vals >= 60) & (v_vals <= 210)
            color_shares["gray_block"] = float(np.count_nonzero(gray_mask)) / n_pts

            # Metal/commercial panel: S < 30, V in [40..170] with high line regularity
            panel_mask = (s_vals < 30) & (v_vals >= 40) & (v_vals <= 170)
            color_shares["metal_panel"] = float(np.count_nonzero(panel_mask)) / n_pts

        is_parcel_empty = (not has_large_structure and structure_area_ratio < 0.035 and edge_density < 0.06)

        return ImageFeatures(
            image_name=image_name,
            width=orig_w,
            height=orig_h,
            vegetation_ratio=vegetation_ratio,
            edge_density=edge_density,
            horizontal_lines_count=h_lines,
            vertical_lines_count=v_lines,
            structural_density=structural_density,
            dominant_hue=dominant_h,
            mean_saturation=mean_s,
            color_shares=color_shares,
            has_large_structure=has_large_structure,
            structure_area_ratio=structure_area_ratio,
            has_roof_contour=has_roof_contour,
            window_openings_count=window_openings,
            glazed_windows_count=glazed_windows,
            empty_openings_count=empty_openings,
            has_finished_facade=has_finished_facade,
            has_finished_roof=has_finished_roof,
            is_parcel_empty=is_parcel_empty,
        )
