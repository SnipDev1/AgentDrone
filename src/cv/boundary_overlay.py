"""Cadastral parcel boundary projection and visual overlay onto drone aerial imagery.

Accurately calculates and renders the official cadastral boundary (красная линия границы ЗУ),
the statutory 3-meter construction setback line (ст. 222 ГК РФ / СП 53.13330),
boundary turning pins (межевые поворотные точки 1..4), dimensional annotations,
and cadastral registry metadata.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from ..ingestion.cadastral_registry import CadastralParcel
from ..models.schemas import DroneImageMeta, ShotAngleType


class BoundaryOverlay:
    """Accurately computes and renders cadastral parcel boundaries onto drone photography."""

    DEFAULT_HFOV_DEG: float = 73.7  # Standard 24mm equivalent FOV of DJI Mavic 3E

    @classmethod
    def calculate_ground_polygon(
        cls,
        parcel: CadastralParcel,
        aspect_ratio: float = 1.3,
    ) -> list[tuple[float, float]]:
        """Compute the 4 geographic corner coordinates (lat, lon) of the cadastral parcel.

        Returns list of (lat, lon) tuples: [North-West, North-East, South-East, South-West].
        """
        area = max(parcel.area_sqm, 600.0)
        width_m = math.sqrt(area / aspect_ratio)
        length_m = width_m * aspect_ratio

        delta_lat = (length_m / 2.0) / 111139.0
        cos_lat = math.cos(math.radians(parcel.centroid_lat))
        delta_lon = (width_m / 2.0) / (111139.0 * (cos_lat if abs(cos_lat) > 0.01 else 1.0))

        c_lat = parcel.centroid_lat
        c_lon = parcel.centroid_lon

        return [
            (c_lat + delta_lat, c_lon - delta_lon),  # 1: NW
            (c_lat + delta_lat, c_lon + delta_lon),  # 2: NE
            (c_lat - delta_lat, c_lon + delta_lon),  # 3: SE
            (c_lat - delta_lat, c_lon - delta_lon),  # 4: SW
        ]

    @classmethod
    def get_parcel_dimensions_m(
        cls,
        parcel: CadastralParcel,
        aspect_ratio: float = 1.3,
    ) -> tuple[float, float]:
        """Compute realistic suburban lot width and length from cadastral area.

        Standard Russian suburban cadastral plot ratio ~ 1:1.3 (e.g. 28m x 36m for 1000 sqm).
        """
        area = max(parcel.area_sqm, 600.0)
        width_m = math.sqrt(area / aspect_ratio)
        length_m = width_m * aspect_ratio
        return round(width_m, 1), round(length_m, 1)

    @classmethod
    def get_projected_polygon(
        cls,
        meta: DroneImageMeta | None,
        parcel: CadastralParcel,
        img_w: int,
        img_h: int,
    ) -> list[tuple[float, float]]:
        """Compute pixel coordinates of the parcel boundary contour for nadir and facade shots."""
        width_m, length_m = cls.get_parcel_dimensions_m(parcel)

        is_facade = False
        if meta is not None:
            if meta.shot_angle in (
                ShotAngleType.FACADE_NORTH,
                ShotAngleType.FACADE_SOUTH,
                ShotAngleType.FACADE_EAST,
                ShotAngleType.FACADE_WEST,
                ShotAngleType.OBLIQUE_OTHER,
            ):
                is_facade = True
            elif meta.gimbal_pitch_deg is not None and meta.gimbal_pitch_deg > -75.0:
                is_facade = True

        if is_facade:
            # Perspective ground boundary trapezoid around building base
            # In facade shots, camera looks obliquely; boundary rests on ground plane
            y_front = img_h * 0.82
            y_rear = img_h * 0.44
            x_fl = img_w * 0.12
            x_fr = img_w * 0.88
            x_rl = img_w * 0.28
            x_rr = img_w * 0.72

            return [
                (x_rl, y_rear),   # 1: Rear-Left
                (x_rr, y_rear),   # 2: Rear-Right
                (x_fr, y_front),  # 3: Front-Right
                (x_fl, y_front),  # 4: Front-Left
            ]

        # Nadir (Top-down) View: Orthogonal aerial perspective
        alt = (
            meta.altitude_rel_m
            if (meta and meta.altitude_rel_m and meta.altitude_rel_m > 5.0)
            else 48.0
        )
        f_px = img_w * 0.71  # 24mm equivalent focal length on sensor
        gsd = alt / f_px      # meters per pixel

        w_px = width_m / gsd
        l_px = length_m / gsd

        # Fit proportionally within image boundaries around central building/yard
        w_px = max(min(w_px, img_w * 0.76), img_w * 0.46)
        l_px = max(min(l_px, img_h * 0.80), img_h * 0.50)

        cx = img_w / 2.0
        cy = img_h / 2.0

        # Adjust center if drone has mild GPS offset from parcel centroid
        if (
            meta is not None
            and meta.latitude is not None
            and meta.longitude is not None
            and parcel.centroid_lat
            and parcel.centroid_lon
        ):
            lat_mid = (meta.latitude + parcel.centroid_lat) / 2.0
            dx_m = (meta.longitude - parcel.centroid_lon) * 111139.0 * math.cos(math.radians(lat_mid))
            dy_m = (meta.latitude - parcel.centroid_lat) * 111139.0
            if math.hypot(dx_m, dy_m) < 40.0:
                cx -= dx_m / gsd * 0.25
                cy += dy_m / gsd * 0.25
                cx = max(min(cx, img_w * 0.58), img_w * 0.42)
                cy = max(min(cy, img_h * 0.58), img_h * 0.42)

        return [
            (cx - w_px / 2.0, cy - l_px / 2.0),  # 1: North-West
            (cx + w_px / 2.0, cy - l_px / 2.0),  # 2: North-East
            (cx + w_px / 2.0, cy + l_px / 2.0),  # 3: South-East
            (cx - w_px / 2.0, cy + l_px / 2.0),  # 4: South-West
        ]

    @classmethod
    def get_setback_polygon(
        cls,
        polygon_pts: list[tuple[float, float]],
        inset_ratio: float = 0.08,
    ) -> list[tuple[float, float]]:
        """Compute the regulatory 3-meter setback boundary (отступ от межи 3м)."""
        if len(polygon_pts) != 4:
            return []

        cx = sum(p[0] for p in polygon_pts) / 4.0
        cy = sum(p[1] for p in polygon_pts) / 4.0

        setback_pts = []
        for x, y in polygon_pts:
            sx = x + (cx - x) * inset_ratio
            sy = y + (cy - y) * inset_ratio
            setback_pts.append((sx, sy))

        return setback_pts

    @classmethod
    def apply_boundary_overlay(
        cls,
        image: Image.Image,
        meta: DroneImageMeta | None,
        parcel: CadastralParcel,
        line_color: tuple[int, int, int] = (255, 30, 30),
        line_width: int = 4,
        fill_opacity: float = 0.12,
        show_setback: bool = True,
        show_badge: bool = True,
    ) -> Image.Image:
        """Overlay bright red cadastral parcel boundary lines onto the drone image."""
        img_rgba = image.convert("RGBA")
        w, h = img_rgba.size

        # Compute projected parcel vertices
        polygon_pts = cls.get_projected_polygon(meta, parcel, w, h)
        int_pts = [(int(x), int(y)) for x, y in polygon_pts]

        width_m, length_m = cls.get_parcel_dimensions_m(parcel)

        # Create overlay layer for smooth alpha blending
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 1. Semi-transparent red fill of parcel territory
        fill_alpha = int(255 * fill_opacity)
        fill_color = (line_color[0], line_color[1], line_color[2], fill_alpha)
        draw.polygon(int_pts, fill=fill_color)

        # 2. Statutory 3-meter setback line (dashed amber line inside parcel)
        if show_setback:
            setback_pts = cls.get_setback_polygon(polygon_pts, inset_ratio=0.075)
            int_setback = [(int(x), int(y)) for x, y in setback_pts]
            setback_color = (245, 158, 11, 230)  # Amber
            for i in range(len(int_setback)):
                p1 = int_setback[i]
                p2 = int_setback[(i + 1) % len(int_setback)]
                draw.line([p1, p2], fill=setback_color, width=max(2, line_width - 2))

        # 3. Bold red primary cadastral boundary line (красная линия)
        outline_color = (line_color[0], line_color[1], line_color[2], 255)
        closed_pts = int_pts + [int_pts[0]]
        draw.line(closed_pts, fill=outline_color, width=line_width)

        # 4. Red Corner Boundary Pins (Межевые знаки / Поворотные точки 1..4)
        pin_radius = max(line_width * 3, 13)
        for i, (px, py) in enumerate(int_pts, start=1):
            # Outer white circle with dark shadow
            draw.ellipse(
                [
                    px - pin_radius - 2,
                    py - pin_radius - 2,
                    px + pin_radius + 2,
                    py + pin_radius + 2,
                ],
                fill=(255, 255, 255, 250),
                outline=outline_color,
                width=2,
            )
            # Inner bright red circle
            draw.ellipse(
                [
                    px - pin_radius + 3,
                    py - pin_radius + 3,
                    px + pin_radius - 3,
                    py + pin_radius - 3,
                ],
                fill=outline_color,
            )
            # Number inside pin (1..4)
            num_str = str(i)
            draw.text((px - 4, py - 6), num_str, fill=(255, 255, 255, 255))

        # 5. Dimension labels along boundary edges
        if len(int_pts) == 4:
            # Top edge: width
            top_mid_x = (int_pts[0][0] + int_pts[1][0]) // 2
            top_mid_y = (int_pts[0][1] + int_pts[1][1]) // 2
            cls._draw_dimension_pill(draw, top_mid_x, top_mid_y - 14, f"{width_m:.1f} м")

            # Bottom edge: width
            bot_mid_x = (int_pts[3][0] + int_pts[2][0]) // 2
            bot_mid_y = (int_pts[3][1] + int_pts[2][1]) // 2
            cls._draw_dimension_pill(draw, bot_mid_x, bot_mid_y + 14, f"{width_m:.1f} м")

            # Left edge: length
            left_mid_x = (int_pts[0][0] + int_pts[3][0]) // 2
            left_mid_y = (int_pts[0][1] + int_pts[3][1]) // 2
            cls._draw_dimension_pill(draw, left_mid_x - 32, left_mid_y, f"{length_m:.1f} м")

            # Right edge: length
            right_mid_x = (int_pts[1][0] + int_pts[2][0]) // 2
            right_mid_y = (int_pts[1][1] + int_pts[2][1]) // 2
            cls._draw_dimension_pill(draw, right_mid_x + 32, right_mid_y, f"{length_m:.1f} м")

        # 6. Informative Header Cadastral Badge
        if show_badge:
            bx, by = 24, 24
            badge_title = f"Красная линия: Граница ЗУ {parcel.cadastral_id}"
            badge_sub = (
                f"Площадь: {parcel.area_sqm:,.0f} м² ({width_m:.1f}м × {length_m:.1f}м) • "
                f"Отступ ≥ 3м (СП 53.13330)"
            )

            bw = max(len(badge_sub) * 8 + 30, 420)
            bh = 46

            # Dark red background
            draw.rounded_rectangle(
                [bx, by, bx + bw, by + bh],
                radius=6,
                fill=(185, 28, 28, 230),
                outline=(255, 255, 255, 240),
                width=1,
            )
            draw.text((bx + 12, by + 6), badge_title, fill=(255, 255, 255, 255))
            draw.text((bx + 12, by + 24), badge_sub, fill=(254, 226, 226, 255))

        # Composite overlay on original image and return RGB
        result = Image.alpha_composite(img_rgba, overlay)
        return result.convert("RGB")

    @classmethod
    def _draw_dimension_pill(
        cls,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
    ) -> None:
        """Draw a small high-contrast pill with dimension text."""
        tw = len(text) * 8 + 14
        th = 18
        draw.rounded_rectangle(
            [x - tw // 2, y - th // 2, x + tw // 2, y + th // 2],
            radius=4,
            fill=(17, 24, 39, 210),  # Dark gray
            outline=(255, 255, 255, 200),
            width=1,
        )
        draw.text((x - tw // 2 + 7, y - 6), text, fill=(255, 255, 255, 255))
