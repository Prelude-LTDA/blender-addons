"""
UV Direction overlay generators.

Generates curved lines showing U and V directions on projection surfaces.
The lines start at UV (0, 0) and extend to (1, 0) for U and (0, 1) for V,
following the curvature of each projection type.

Also generates dashed "projected" lines that complete the parallelogram
showing how the UV space maps to the projection surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mathutils import Euler, Matrix, Vector

from ...shared.uv_map.constants import (
    MAPPING_BOX,
    MAPPING_CYLINDRICAL,
    MAPPING_CYLINDRICAL_CAPPED,
    MAPPING_CYLINDRICAL_NORMAL,
    MAPPING_CYLINDRICAL_NORMAL_CAPPED,
    MAPPING_PLANAR,
    MAPPING_SHRINK_WRAP,
    MAPPING_SHRINK_WRAP_NORMAL,
    MAPPING_SPHERICAL,
    MAPPING_SPHERICAL_NORMAL,
)
from .box import generate_box_direction
from .cylinder import (
    generate_cylindrical_capped_direction,
    generate_cylindrical_direction,
    generate_cylindrical_normal_capped_direction,
    generate_cylindrical_normal_direction,
)
from .inverse import (
    inverse_uv_cylindrical,
    inverse_uv_planar,
    inverse_uv_shrink_wrap,
    inverse_uv_spherical,
)
from .plane import add_planar_face_direction, generate_planar_direction
from .shrink_wrap import (
    generate_shrink_wrap_direction,
    generate_shrink_wrap_normal_direction,
)
from .sphere import generate_spherical_direction, generate_spherical_normal_direction
from .utils import compute_adaptive_segments, generate_uv_direction_line

if TYPE_CHECKING:
    from collections.abc import Callable

    # Type alias for direction generator functions
    DirectionGenerator = Callable[
        [
            int,  # segments
            float,  # u_tile
            float,  # v_tile
            float,  # u_offset
            float,  # v_offset
            float,  # uv_rotation
            bool,  # u_flip
            bool,  # v_flip
            Matrix,  # transform
            list[tuple[float, float, float]],  # u_vertices
            list[tuple[float, float, float]],  # v_vertices
            list[tuple[float, float, float]],  # u_proj_vertices
            list[tuple[float, float, float]],  # v_proj_vertices
            list[tuple[float, float, float]],  # u_labels
            list[tuple[float, float, float]],  # v_labels
        ],
        None,
    ]

# Mapping from mapping type to generator function
_DIRECTION_GENERATORS: dict[str, DirectionGenerator] = {
    MAPPING_PLANAR: generate_planar_direction,
    MAPPING_CYLINDRICAL: generate_cylindrical_direction,
    MAPPING_CYLINDRICAL_CAPPED: generate_cylindrical_capped_direction,
    MAPPING_SPHERICAL: generate_spherical_direction,
    MAPPING_SHRINK_WRAP: generate_shrink_wrap_direction,
    MAPPING_BOX: generate_box_direction,
    MAPPING_CYLINDRICAL_NORMAL: generate_cylindrical_normal_direction,
    MAPPING_CYLINDRICAL_NORMAL_CAPPED: generate_cylindrical_normal_capped_direction,
    MAPPING_SPHERICAL_NORMAL: generate_spherical_normal_direction,
    MAPPING_SHRINK_WRAP_NORMAL: generate_shrink_wrap_normal_direction,
}


def generate_uv_direction_vertices(
    mapping_type: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
]:
    """Generate UV direction indicator vertices for a given mapping type.

    Returns six lists:
    - U direction line vertices (for LINES primitive)
    - V direction line vertices (for LINES primitive)
    - U projected line vertices (dashed, from V endpoint in U direction)
    - V projected line vertices (dashed, from U endpoint in V direction)
    - U label positions (endpoint of each U line)
    - V label positions (endpoint of each V line)

    For planar/cylindrical/spherical/shrink_wrap: returns one U and one V line
    For cylindrical capped: returns lines for side + top/bottom caps
    For box: returns lines for each face projection
    """
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build TRS transform matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = (
        Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix
    )

    # Adaptive segments based on tiling (more segments for longer lines)
    base_segments = 16
    segments = compute_adaptive_segments(u_tile, v_tile, base_segments)

    u_vertices: list[tuple[float, float, float]] = []
    v_vertices: list[tuple[float, float, float]] = []
    u_proj_vertices: list[tuple[float, float, float]] = []
    v_proj_vertices: list[tuple[float, float, float]] = []
    u_labels: list[tuple[float, float, float]] = []
    v_labels: list[tuple[float, float, float]] = []

    # Look up and call the appropriate generator function
    generator = _DIRECTION_GENERATORS.get(mapping_type)
    if generator is not None:
        generator(
            segments,
            u_tile,
            v_tile,
            u_offset,
            v_offset,
            uv_rotation,
            u_flip,
            v_flip,
            transform,
            u_vertices,
            v_vertices,
            u_proj_vertices,
            v_proj_vertices,
            u_labels,
            v_labels,
        )

    return u_vertices, v_vertices, u_proj_vertices, v_proj_vertices, u_labels, v_labels


__all__ = [
    # Utilities
    "add_planar_face_direction",
    "compute_adaptive_segments",
    # Individual generators
    "generate_box_direction",
    "generate_cylindrical_capped_direction",
    "generate_cylindrical_direction",
    "generate_cylindrical_normal_capped_direction",
    "generate_cylindrical_normal_direction",
    "generate_planar_direction",
    "generate_shrink_wrap_direction",
    "generate_shrink_wrap_normal_direction",
    "generate_spherical_direction",
    "generate_spherical_normal_direction",
    "generate_uv_direction_line",
    # Main entry point
    "generate_uv_direction_vertices",
    # Inverse UV functions
    "inverse_uv_cylindrical",
    "inverse_uv_planar",
    "inverse_uv_shrink_wrap",
    "inverse_uv_spherical",
]
