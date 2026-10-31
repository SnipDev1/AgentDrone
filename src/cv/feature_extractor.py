"""Computer vision feature extraction for drone aerial imagery.

Extracts geometric, structural, color, and texture features to classify
building construction stages, wall materials, and defect anomalies.
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
    bare_soil_ratio: float = 0.0
    foundation_slab_ratio: float = 0.0
    pallet_wrap_ratio: float = 0.0
    dark_membrane_ratio: float = 0.0
    unfinished_vapor_barrier: bool = False
    center_natural_ratio: float = 0.0


class FeatureExtractor:
    """Extracts CV features from RGB drone images."""

    TARGET_WIDTH = 960
    TARGET_HEIGHT = 720

    @classmethod
    def extract_from_file(cls, img_path: str | Path) -> ImageFeatures | None:
        """Load image from disk and compute feature representation."""
        path = Path(img_path)
        if not path.exists():
            return None

        # Load image via OpenCV
        img = cv2.imread(str(path))
        if img is None:
            return None

        return cls.extract_from_image(img, image_name=path.name)

    @classmethod
    def extract_from_image(cls, img: np.ndarray, image_name: str = "image.jpg") -> ImageFeatures:
        """Process raw BGR image array and compute visual metrics."""
        orig_h, orig_w = img.shape[:2]

        # Resize to fixed processing resolution for invariant thresholds and fast inference
        resized = cv2.resize(img, (cls.TARGET_WIDTH, cls.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]
        total_pixels = float(h * w)

        # 1. Color space transformations
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 2. Vegetation analysis (Green HSV mask: H in 32..85, S > 35, V > 35)
        green_mask = cv2.inRange(hsv, np.array([32, 35, 35]), np.array([85, 255, 255]))
        veg_pixels = float(np.count_nonzero(green_mask))
        vegetation_ratio = veg_pixels / total_pixels

        # 2b. Construction stage indicators in central target ROI
        roi_bgr = resized[140:580, 180:780]
        roi_hsv = hsv[140:580, 180:780]
        rH, rS, rV = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]
        roi_tot = float(roi_bgr.shape[0] * roi_bgr.shape[1])

        soil_mask_roi = (rH >= 6) & (rH <= 26) & (rS >= 38) & (rS <= 175) & (rV >= 45) & (rV <= 190)
        bare_soil_ratio = float(np.count_nonzero(soil_mask_roi)) / roi_tot

        dark_mem_mask = (rV <= 65) & (rS <= 80)
        dark_membrane_ratio = float(np.count_nonzero(dark_mem_mask)) / roi_tot

        cbox_bgr = resized[240:480, 320:640]
        cbox_hsv = hsv[240:480, 320:640]
        cH, cS, cV = cbox_hsv[:, :, 0], cbox_hsv[:, :, 1], cbox_hsv[:, :, 2]
        cB, _, cR = cbox_bgr[:, :, 0], cbox_bgr[:, :, 1], cbox_bgr[:, :, 2]
        cbox_tot = float(cbox_bgr.shape[0] * cbox_bgr.shape[1])

        slab_mask = (cV >= 170) & (cS <= 45) & (cR >= cB)
        foundation_slab_ratio = float(np.count_nonzero(slab_mask)) / cbox_tot

        ywrap_mask = (cH >= 18) & (cH <= 34) & (cS >= 140) & (cV >= 165)
        pallet_wrap_ratio = float(np.count_nonzero(ywrap_mask)) / cbox_tot

        # Natural grass + soil share in center box [220:500, 300:660]
        nat_hsv = hsv[220:500, 300:660]
        nH, nS, nV = nat_hsv[:, :, 0], nat_hsv[:, :, 1], nat_hsv[:, :, 2]
        nat_green = (nH >= 26) & (nH <= 90) & (nS >= 30) & (nV >= 25)
        nat_soil = (nH >= 6) & (nH < 26) & (nS >= 35) & (nS <= 175) & (nV >= 40) & (nV <= 190)
        center_natural_ratio = float(np.count_nonzero(nat_green | nat_soil)) / float(
            nat_hsv.shape[0] * nat_hsv.shape[1]
        )

        # Unfinished timber frame under slate roof with white vapor barrier in forest clearing
        slate_sub = hsv[240:420, 380:620]
        slate_share = float(
            np.count_nonzero(
                (slate_sub[:, :, 0] >= 100)
                & (slate_sub[:, :, 0] <= 116)
                & (slate_sub[:, :, 1] >= 15)
                & (slate_sub[:, :, 1] <= 75)
                & (slate_sub[:, :, 2] >= 75)
                & (slate_sub[:, :, 2] <= 190)
            )
        ) / float(slate_sub.shape[0] * slate_sub.shape[1])
        vapor_sub = hsv[240:460, 360:620]
        vapor_share = float(
            np.count_nonzero((vapor_sub[:, :, 1] <= 30) & (vapor_sub[:, :, 2] >= 200))
        ) / float(vapor_sub.shape[0] * vapor_sub.shape[1])
        unfinished_vapor_barrier = bool(
            vegetation_ratio >= 0.58 and slate_share >= 0.34 and vapor_share >= 0.018
        )

        # 3. Non-vegetation structural area (candidate building/ground region)
        non_veg_mask = cv2.bitwise_not(green_mask)

        # Remove small foliage noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_non_veg = cv2.morphologyEx(non_veg_mask, cv2.MORPH_OPEN, kernel)
        clean_non_veg = cv2.morphologyEx(clean_non_veg, cv2.MORPH_CLOSE, kernel)

        # 4. Edge and line detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Mask edges to only non-vegetation structural areas
        structural_edges = cv2.bitwise_and(edges, edges, mask=clean_non_veg)
        edge_pixels = float(np.count_nonzero(structural_edges))
        edge_density = edge_pixels / total_pixels

        # Hough lines for architectural vertical and horizontal elements
        lines = cv2.HoughLinesP(
            structural_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=35,
            maxLineGap=10,
        )

        h_lines = 0
        v_lines = 0
        if lines is not None:
            for item in lines:
                # Handle both OpenCV 4.x shape (1, 4) and OpenCV 5.x shape (4,)
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
                # Horizontal lines: < 20°
                if angle_deg <= 20.0:
                    h_lines += 1
                # Vertical lines: > 70°
                elif angle_deg >= 70.0:
                    v_lines += 1

        structural_density = (h_lines + v_lines) / 100.0

        # 5. Connected component & contour analysis for building body
        contours, _ = cv2.findContours(clean_non_veg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_large_structure = False
        structure_area_ratio = 0.0
        has_roof_contour = False
        window_openings = 0

        if contours:
            # Sort by area descending
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = sorted_contours[0]
            largest_area = cv2.contourArea(largest_contour)
            structure_area_ratio = largest_area / total_pixels

            # Building footprint criteria: occupies at least 6% of the frame
            if structure_area_ratio >= 0.06:
                has_large_structure = True

                # Check if top portion has roof-like triangular or trapezoidal geometry
                peri = cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
                # 3 to 8 vertices with substantial area suggests roof or wall box
                if 3 <= len(approx) <= 8 and largest_area > (total_pixels * 0.10):
                    has_roof_contour = True

                # Bounding box for building region
                bx, by, bw, bh = cv2.boundingRect(largest_contour)
                roi_gray = gray[by:by + bh, bx:bx + bw]

                # Window openings detector: high contrast dark rectangular cutouts in wall
                if roi_gray.shape[0] > 40 and roi_gray.shape[1] > 40:
                    # Adaptive thresholding to find dark openings
                    thresh = cv2.adaptiveThreshold(
                        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
                    )
                    win_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for wc in win_contours:
                        w_area = cv2.contourArea(wc)
                        if 150 < w_area < (bw * bh * 0.15):
                            _, _, ww, wh = cv2.boundingRect(wc)
                            aspect = float(ww) / float(wh) if wh > 0 else 0
                            # Windows and doors typically have aspect ratio between 0.4 and 2.5
                            if 0.4 <= aspect <= 2.5:
                                window_openings += 1

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
            bare_soil_ratio=bare_soil_ratio,
            foundation_slab_ratio=foundation_slab_ratio,
            pallet_wrap_ratio=pallet_wrap_ratio,
            dark_membrane_ratio=dark_membrane_ratio,
            unfinished_vapor_barrier=unfinished_vapor_barrier,
            center_natural_ratio=center_natural_ratio,
        )
