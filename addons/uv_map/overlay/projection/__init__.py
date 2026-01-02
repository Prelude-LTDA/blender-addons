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

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from mathutils import Euler, Vector

    # All projection generators have uniform signature: (position, rotation, size) -> vertices
    ProjectionGenerator = Callable[
        [
            Vector,
            Euler,
            Vector,
        ],
        list[Vector],
    ]

# Mapping type transformations based on normal_based and cap flags
_EFFECTIVE_MAPPING_TYPES: dict[tuple[str, bool, bool], str] = {
    # (base_type, normal_based, cap) -> effective_type
    # Cylindrical variants
    (MAPPING_CYLINDRICAL, False, False): MAPPING_CYLINDRICAL,
    (MAPPING_CYLINDRICAL, False, True): MAPPING_CYLINDRICAL_CAPPED,
    (MAPPING_CYLINDRICAL, True, False): MAPPING_CYLINDRICAL_NORMAL,
    (MAPPING_CYLINDRICAL, True, True): MAPPING_CYLINDRICAL_NORMAL_CAPPED,
    # Spherical variants
    (MAPPING_SPHERICAL, False, False): MAPPING_SPHERICAL,
    (MAPPING_SPHERICAL, True, False): MAPPING_SPHERICAL_NORMAL,
    # Shrink wrap variants
    (MAPPING_SHRINK_WRAP, False, False): MAPPING_SHRINK_WRAP,
    (MAPPING_SHRINK_WRAP, True, False): MAPPING_SHRINK_WRAP_NORMAL,
}

# Projection generators: effective_type -> generator_fn
_PROJECTION_GENERATORS: dict[str, ProjectionGenerator] = {
    MAPPING_PLANAR: generate_plane_vertices,
    MAPPING_BOX: generate_box_vertices,
    MAPPING_CYLINDRICAL: generate_cylinder_vertices,
    MAPPING_CYLINDRICAL_CAPPED: generate_cylinder_capped_vertices,
    MAPPING_CYLINDRICAL_NORMAL: generate_cylinder_normal_vertices,
    MAPPING_CYLINDRICAL_NORMAL_CAPPED: generate_cylinder_capped_normal_vertices,
    MAPPING_SPHERICAL: generate_sphere_vertices,
    MAPPING_SPHERICAL_NORMAL: generate_sphere_normal_vertices,
    MAPPING_SHRINK_WRAP: generate_shrink_wrap_vertices,
    MAPPING_SHRINK_WRAP_NORMAL: generate_shrink_wrap_normal_vertices,
}


def _get_effective_mapping_type(
    mapping_type: str, normal_based: bool, cap: bool
) -> str:
    """Get the effective mapping type based on normal_based and cap flags."""
    return _EFFECTIVE_MAPPING_TYPES.get(
        (mapping_type, normal_based, cap),
        mapping_type,  # Default to original if no transformation needed
    )


def generate_projection_vertices(
    mapping_type: str,
    position: Vector,
    rotation: Euler,
    size: Vector,
    normal_based: bool = False,
    cap: bool = False,
) -> tuple[str, list[Vector]]:
    """Generate projection wireframe vertices for a given mapping type.

    Args:
        mapping_type: Base mapping type (MAPPING_PLANAR, MAPPING_CYLINDRICAL, etc.)
        position: World position of the projection
        rotation: Rotation as Euler angles
        size: Scale as Vector
        normal_based: Whether to use normal-based mapping variant
        cap: Whether to use capped variant (cylindrical only)

    Returns:
        Tuple of (effective_mapping_type, vertices) where vertices is a list of
        Vectors for LINES primitive drawing.
    """
    # Determine the effective mapping type based on normal_based and cap flags
    effective_mapping_type = _get_effective_mapping_type(
        mapping_type, normal_based, cap
    )

    # Look up the appropriate projection generator
    generator_fn = _PROJECTION_GENERATORS.get(effective_mapping_type)
    if generator_fn is None:
        return effective_mapping_type, []

    vertices = generator_fn(position, rotation, size)

    return effective_mapping_type, vertices


__all__ = [
    "generate_box_vertices",
    "generate_cylinder_capped_normal_vertices",
    "generate_cylinder_capped_vertices",
    "generate_cylinder_normal_vertices",
    "generate_cylinder_vertices",
    "generate_plane_vertices",
    "generate_projection_vertices",
    "generate_shrink_wrap_normal_vertices",
    "generate_shrink_wrap_vertices",
    "generate_sphere_normal_vertices",
    "generate_sphere_vertices",
]
