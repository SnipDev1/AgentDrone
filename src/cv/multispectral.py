"""Multispectral imaging and vegetation index (NDVI / GNDVI) analyzer for DJI Mavic 3M.

Analyzes 4-band calibrated multispectral imagery (NIR 860nm, Red 650nm, Green 560nm, RedEdge 730nm)
to evaluate construction site dynamics, earthwork footprint, and vegetation encroachment / idle status.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..models.schemas import MultispectralAnalysis


class MultispectralAnalyzer:
    """Computes spectral indices and land surface classification from drone TIFFs."""

    @classmethod
    def analyze_plot(
        cls,
        plot_dir: str | Path,
        generate_heatmap: bool = True,
    ) -> MultispectralAnalysis:
        """Process multispectral TIFF bands in the plot directory."""
        dir_path = Path(plot_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            return MultispectralAnalysis(has_multispectral=False)

        nir_files = sorted(dir_path.glob("*MS_NIR.TIF"))
        r_files = sorted(dir_path.glob("*MS_R.TIF"))
        g_files = sorted(dir_path.glob("*MS_G.TIF"))
        re_files = sorted(dir_path.glob("*MS_RE.TIF"))

        if not nir_files or not r_files:
            return MultispectralAnalysis(has_multispectral=False)

        nir_path = nir_files[0]
        r_path = r_files[0]

        try:
            with Image.open(nir_path) as im_nir, Image.open(r_path) as im_r:
                # Downsample 4x for fast processing and minimal memory footprint
                w, h = im_nir.size
                proc_w, proc_h = max(w // 4, 160), max(h // 4, 120)
                nir_arr = np.array(im_nir.resize((proc_w, proc_h), Image.Resampling.BILINEAR), dtype=np.float32)
                r_arr = np.array(im_r.resize((proc_w, proc_h), Image.Resampling.BILINEAR), dtype=np.float32)

            denom = nir_arr + r_arr
            denom[denom == 0] = 1e-6
            ndvi = (nir_arr - r_arr) / denom
            ndvi = np.clip(ndvi, -1.0, 1.0)

            mean_ndvi = float(np.mean(ndvi))
            # Bare soil, foundation concrete, roofing: NDVI <= 0.25
            construction_soil_pct = float(np.mean(ndvi <= 0.25) * 100.0)
            # Active green vegetation / weeds: NDVI > 0.40
            vegetation_pct = float(np.mean(ndvi > 0.40) * 100.0)

            if construction_soil_pct >= 40.0:
                activity_status = "Активная застройка (земляные работы / бетонный контур / кровля)"
                weed_risk = "Низкий (территория расчищена под строительство)"
            elif vegetation_pct >= 60.0:
                activity_status = "Признаки заброшенности стройки (густое зарастание сорной растительностью)"
                weed_risk = "ВЫСОКИЙ (Зарастание участка >60%, риск пожароопасности СП 4.13130)"
            else:
                activity_status = "Смешанное покрытие (строительная площадка + частичное озеленение)"
                weed_risk = "Умеренный"

            bands = [
                "NIR: Ближний инфракрасный (860 нм)",
                "Red: Красный спектр (650 нм)",
            ]
            if g_files:
                bands.append("Green: Зеленый спектр (560 нм)")
            if re_files:
                bands.append("RedEdge: Красный край (730 нм)")

            heatmap_path = None
            if generate_heatmap:
                heatmap_path = cls._render_ndvi_colormap(ndvi, dir_path)

            return MultispectralAnalysis(
                has_multispectral=True,
                mean_ndvi=round(mean_ndvi, 3),
                construction_soil_pct=round(construction_soil_pct, 1),
                vegetation_pct=round(vegetation_pct, 1),
                site_activity_status=activity_status,
                weed_overgrowth_risk=weed_risk,
                bands_available=bands,
                heatmap_rel_path=heatmap_path,
            )
        except Exception:
            return MultispectralAnalysis(has_multispectral=False)

    @classmethod
    def _render_ndvi_colormap(cls, ndvi: np.ndarray, out_dir: Path) -> str:
        """Render a color-coded NDVI heatmap image (Red/Brown -> Yellow -> Bright Green)."""
        h, w = ndvi.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Normalize NDVI (-0.2..0.8) to (0..1)
        norm = np.clip((ndvi + 0.2) / 1.0, 0.0, 1.0)

        # 1. Bare soil/concrete/foundation (norm < 0.4): Brown/Red -> Amber
        mask_low = norm < 0.4
        t_low = norm[mask_low] / 0.4
        rgb[mask_low, 0] = (180 + t_low * 40).astype(np.uint8)   # Red
        rgb[mask_low, 1] = (60 + t_low * 120).astype(np.uint8)   # Green
        rgb[mask_low, 2] = 40                                    # Blue

        # 2. Transition / sparse grass (0.4 <= norm < 0.7): Amber -> Lime
        mask_mid = (norm >= 0.4) & (norm < 0.7)
        t_mid = (norm[mask_mid] - 0.4) / 0.3
        rgb[mask_mid, 0] = (220 - t_mid * 120).astype(np.uint8)
        rgb[mask_mid, 1] = (180 + t_mid * 50).astype(np.uint8)
        rgb[mask_mid, 2] = 50

        # 3. Dense vegetation / trees (norm >= 0.7): Rich Green -> Forest Green
        mask_high = norm >= 0.7
        t_high = (norm[mask_high] - 0.7) / 0.3
        rgb[mask_high, 0] = (100 - t_high * 70).astype(np.uint8)
        rgb[mask_high, 1] = (230 - t_high * 60).astype(np.uint8)
        rgb[mask_high, 2] = (50 + t_high * 40).astype(np.uint8)

        out_file = out_dir / "ndvi_heatmap_preview.png"
        try:
            im = Image.fromarray(rgb)
            im.save(out_file, format="PNG")
            return str(out_file)
        except Exception:
            # Fallback to temp path if plot_dir is read-only
            tmp_file = Path(os.getenv("TEMP", "/tmp")) / "ndvi_heatmap_preview.png"
            im = Image.fromarray(rgb)
            im.save(tmp_file, format="PNG")
            return str(tmp_file)
