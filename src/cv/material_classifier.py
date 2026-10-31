"""Wall material classifier for residential and commercial structures.

Classifies wall construction materials: Brick, Gas Concrete/Blocks, Timber/Logs,
Frame-panel/SIP, and Commercial sandwich panels.
"""

from __future__ import annotations

from ..models.schemas import WallMaterial
from .feature_extractor import ImageFeatures


class MaterialClassifier:
    """Classifies construction material based on visual color and texture features."""

    @classmethod
    def classify(cls, features_list: list[ImageFeatures]) -> tuple[WallMaterial, float]:
        """Aggregate facade features and predict dominant wall material."""
        if not features_list:
            return WallMaterial.UNKNOWN, 0.0

        # Accumulate color shares
        agg_shares: dict[str, float] = {
            "red_brick": 0.0,
            "gray_block": 0.0,
            "wood_timber": 0.0,
            "yellow_osb": 0.0,
            "metal_panel": 0.0,
        }

        count = float(len(features_list))
        for f in features_list:
            for k in agg_shares:
                agg_shares[k] += f.color_shares.get(k, 0.0)

        for k in agg_shares:
            agg_shares[k] /= count

        avg_veg = sum(f.vegetation_ratio for f in features_list) / count
        max_struct_ratio = max(f.structure_area_ratio for f in features_list)
        big_metal_shots = sum(
            1
            for f in features_list
            if f.structure_area_ratio >= 0.48 and f.color_shares.get("metal_panel", 0.0) >= 0.32
        )

        # Check for commercial metal sandwich panel / supermarket roof & facade
        # Supports both open commercial lots and commercial buildings surrounded by dense trees
        if (
            (avg_veg < 0.25 and max_struct_ratio > 0.30 and agg_shares["metal_panel"] > 0.20)
            or (big_metal_shots >= 3 and max_struct_ratio >= 0.70 and agg_shares["metal_panel"] >= 0.32)
        ):
            return WallMaterial.SANDWICH_PANEL, 0.92

        # Compare residential materials
        res_scores = {
            WallMaterial.GAS_CONCRETE: agg_shares["gray_block"] * 1.35,
            WallMaterial.BRICK: agg_shares["red_brick"] * 1.25,
            WallMaterial.TIMBER: agg_shares["wood_timber"] * 1.30,
            WallMaterial.FRAME_PANEL: agg_shares["yellow_osb"] * 1.30,
        }

        # Check if no building features exist
        best_mat, best_score = max(res_scores.items(), key=lambda item: item[1])

        if best_score < 0.05:
            # Fallback to gas concrete or unknown if neutral grey walls
            if agg_shares["gray_block"] > 0.15:
                return WallMaterial.GAS_CONCRETE, 0.50
            return WallMaterial.UNKNOWN, 0.0

        # Normalize confidence to [0.55 .. 0.95]
        confidence = min(0.95, max(0.55, best_score * 2.5))
        return best_mat, round(confidence, 2)
