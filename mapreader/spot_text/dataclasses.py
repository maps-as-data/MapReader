from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon


@dataclass(frozen=True)
class PatchPrediction:
    pixel_geometry: Polygon
    score: float
    text: str = None


@dataclass(frozen=True)
class ParentPrediction:
    pixel_geometry: Polygon
    score: float
    patch_id: str
    text: str = None


@dataclass(frozen=True)
class GeoPrediction:
    pixel_geometry: Polygon
    score: float
    patch_id: str
    geometry: Polygon
    crs: str
    text: str = None
