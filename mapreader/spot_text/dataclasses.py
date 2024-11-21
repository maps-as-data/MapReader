from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

# Detection only


@dataclass
class Prediction:
    geometry: Polygon
    score: float
    text: str = None
    patch_id: str | None = None
    crs: str | None = None
