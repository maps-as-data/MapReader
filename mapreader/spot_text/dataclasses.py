from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, Polygon


@dataclass(frozen=True)
class PatchPrediction:
    pixel_geometry: Polygon
    pixel_line: LineString
    score: float
    text: str = None


@dataclass(frozen=True)
class ParentPrediction:
    pixel_geometry: Polygon
    pixel_line: LineString
    score: float
    patch_id: str
    text: str = None


@dataclass(frozen=True)
class GeoPrediction:
    pixel_geometry: Polygon
    pixel_line: LineString
    score: float
    patch_id: str
    geometry: Polygon
    line: LineString
    crs: str
    text: str = None
