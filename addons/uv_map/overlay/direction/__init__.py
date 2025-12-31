"""
UV Direction overlay generators.

Generates curved lines showing U and V directions on projection surfaces.
The lines start at UV (0, 0) and extend to (1, 0) for U and (0, 1) for V,
following the curvature of each projection type.

Also generates dashed "projected" lines that complete the parallelogram
showing how the UV space maps to the projection surface.
"""

from __future__ import annotations

from .generators import generate_uv_direction_vertices
from .inverse import (
    inverse_uv_cylindrical,
    inverse_uv_planar,
    inverse_uv_shrink_wrap,
    inverse_uv_spherical,
)
from .utils import compute_adaptive_segments, generate_uv_direction_line

__all__ = [
    "compute_adaptive_segments",
    "generate_uv_direction_line",
    "generate_uv_direction_vertices",
    "inverse_uv_cylindrical",
    "inverse_uv_planar",
    "inverse_uv_shrink_wrap",
    "inverse_uv_spherical",
]
