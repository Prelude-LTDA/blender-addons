"""
Projection wireframe generators for the UV Map overlay.

Contains functions to generate vertices for different UV mapping shape overlays:
- Planar (plane)
- Box (cube wireframe)
- Cylindrical (cylinder)
- Spherical (sphere)
- Shrink Wrap (azimuthal equidistant projection on sphere)

Also includes "normal-based" variants that show rotation/scale only.
"""

from __future__ import annotations

from .box import generate_box_vertices
from .cylinder import (
    generate_cylinder_capped_normal_vertices,
    generate_cylinder_capped_vertices,
    generate_cylinder_normal_vertices,
    generate_cylinder_vertices,
)
from .plane import generate_plane_vertices
from .shrink_wrap import (
    generate_shrink_wrap_normal_vertices,
    generate_shrink_wrap_vertices,
)
from .sphere import generate_sphere_normal_vertices, generate_sphere_vertices

__all__ = [
    "generate_box_vertices",
    "generate_cylinder_capped_normal_vertices",
    "generate_cylinder_capped_vertices",
    "generate_cylinder_normal_vertices",
    "generate_cylinder_vertices",
    "generate_plane_vertices",
    "generate_shrink_wrap_normal_vertices",
    "generate_shrink_wrap_vertices",
    "generate_sphere_normal_vertices",
    "generate_sphere_vertices",
]
