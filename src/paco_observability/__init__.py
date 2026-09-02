"""PACO observability analysis and validation toolkit."""

from .geometry import GroundPlane, OrientedBox, RayHit, ray_obb_intersection
from .occlusion import (
    center_coverage_flags,
    center_top_visibility,
    multi_ray_visibility,
    projected_visible_fraction,
)

__all__ = [
    "GroundPlane",
    "OrientedBox",
    "RayHit",
    "center_coverage_flags",
    "center_top_visibility",
    "multi_ray_visibility",
    "projected_visible_fraction",
    "ray_obb_intersection",
]

__version__ = "1.1.0.dev0"
