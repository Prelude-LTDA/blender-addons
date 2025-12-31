"""UV direction vertex generators for each mapping type."""

from __future__ import annotations

import math

from mathutils import Euler, Matrix, Vector

from ...shared.uv_map.constants import (
    MAPPING_BOX,
    MAPPING_CYLINDRICAL,
    MAPPING_CYLINDRICAL_NORMAL,
    MAPPING_PLANAR,
    MAPPING_SHRINK_WRAP,
    MAPPING_SHRINK_WRAP_NORMAL,
    MAPPING_SPHERICAL,
    MAPPING_SPHERICAL_NORMAL,
)
from .inverse import (
    inverse_uv_cylindrical,
    inverse_uv_planar,
    inverse_uv_shrink_wrap,
    inverse_uv_spherical,
)
from .utils import compute_adaptive_segments, generate_uv_direction_line


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
    cap: bool = False,
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

    if mapping_type == MAPPING_PLANAR:
        _generate_planar_direction(
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

    elif mapping_type == MAPPING_CYLINDRICAL:
        _generate_cylindrical_direction(
            segments,
            u_tile,
            v_tile,
            u_offset,
            v_offset,
            uv_rotation,
            u_flip,
            v_flip,
            transform,
            cap,
            u_vertices,
            v_vertices,
            u_proj_vertices,
            v_proj_vertices,
            u_labels,
            v_labels,
        )

    elif mapping_type == MAPPING_SPHERICAL:
        _generate_spherical_direction(
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

    elif mapping_type == MAPPING_SHRINK_WRAP:
        _generate_shrink_wrap_direction(
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

    elif mapping_type == MAPPING_BOX:
        _generate_box_direction(
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

    elif mapping_type == MAPPING_CYLINDRICAL_NORMAL:
        _generate_cylindrical_normal_direction(
            segments,
            u_tile,
            v_tile,
            u_offset,
            v_offset,
            uv_rotation,
            u_flip,
            v_flip,
            transform,
            cap,
            u_vertices,
            v_vertices,
            u_proj_vertices,
            v_proj_vertices,
            u_labels,
            v_labels,
        )

    elif mapping_type == MAPPING_SPHERICAL_NORMAL:
        _generate_spherical_normal_direction(
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

    elif mapping_type == MAPPING_SHRINK_WRAP_NORMAL:
        _generate_shrink_wrap_normal_direction(
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


def _generate_planar_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate planar UV direction indicators."""
    # U line from (0,0) to (1,0)
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_planar,
        0.0,
        0.0,
        1.0,
        0.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    # V line from (0,0) to (0,1)
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_planar,
        0.0,
        0.0,
        0.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    # U projection (dashed) from (0,1) to (1,1)
    verts, _ = generate_uv_direction_line(
        inverse_uv_planar,
        0.0,
        1.0,
        1.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    # V projection (dashed) from (1,0) to (1,1)
    verts, _ = generate_uv_direction_line(
        inverse_uv_planar,
        1.0,
        0.0,
        1.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)


def _generate_cylindrical_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    cap: bool,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate cylindrical UV direction indicators."""
    # U line (around circumference)
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_cylindrical,
        0.0,
        0.0,
        1.0,
        0.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    # V line (along height)
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_cylindrical,
        0.0,
        0.0,
        0.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    # U projection (dashed)
    verts, _ = generate_uv_direction_line(
        inverse_uv_cylindrical,
        0.0,
        1.0,
        1.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    # V projection (dashed)
    verts, _ = generate_uv_direction_line(
        inverse_uv_cylindrical,
        1.0,
        0.0,
        1.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)

    # Cap indicators if capped
    if cap:
        _add_cap_direction(
            segments,
            u_tile,
            v_tile,
            u_offset,
            v_offset,
            uv_rotation,
            u_flip,
            v_flip,
            transform,
            1.0,  # cap_z
            u_vertices,
            v_vertices,
            u_proj_vertices,
            v_proj_vertices,
            u_labels,
            v_labels,
        )


def _generate_spherical_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate spherical UV direction indicators."""
    v_equator = 0.0
    v_north = 1.0

    # U line at equator
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_spherical,
        0.0,
        v_equator,
        1.0,
        v_equator,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    # V line from equator toward north
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_spherical,
        0.0,
        v_equator,
        0.0,
        v_north,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    # Projected lines (dashed)
    verts, _ = generate_uv_direction_line(
        inverse_uv_spherical,
        0.0,
        v_north,
        1.0,
        v_north,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    verts, _ = generate_uv_direction_line(
        inverse_uv_spherical,
        1.0,
        v_equator,
        1.0,
        v_north,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)


def _generate_shrink_wrap_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate shrink wrap UV direction indicators."""
    # U line
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        0.0,
        0.0,
        1.0,
        0.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    # V line
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        0.0,
        0.0,
        0.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    # Projected lines (dashed)
    verts, _ = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        0.0,
        1.0,
        1.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    verts, _ = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        1.0,
        0.0,
        1.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)


def _generate_box_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate box (tri-planar) UV direction indicators."""
    # +Z face (top)
    z_face_offset = Matrix.Translation(Vector((0, 0, 1)))
    z_transform = transform @ z_face_offset
    _add_planar_face_direction(
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        z_transform,
        u_vertices,
        v_vertices,
        u_proj_vertices,
        v_proj_vertices,
        u_labels,
        v_labels,
    )

    # +X face
    x_rot = Euler((math.pi / 2, 0, math.pi / 2), "XYZ").to_matrix().to_4x4()
    x_face_offset = Matrix.Translation(Vector((1, 0, 0)))
    x_transform = transform @ x_face_offset @ x_rot
    _add_planar_face_direction(
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        x_transform,
        u_vertices,
        v_vertices,
        u_proj_vertices,
        v_proj_vertices,
        u_labels,
        v_labels,
    )

    # +Y face
    y_rot = Euler((math.pi / 2, 0, 0), "XYZ").to_matrix().to_4x4()
    y_face_offset = Matrix.Translation(Vector((0, 1, 0)))
    y_transform = transform @ y_face_offset @ y_rot
    _add_planar_face_direction(
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        y_transform,
        u_vertices,
        v_vertices,
        u_proj_vertices,
        v_proj_vertices,
        u_labels,
        v_labels,
    )


def _generate_cylindrical_normal_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    cap: bool,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate cylindrical normal-based UV direction indicators."""
    # Same as regular cylindrical
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_cylindrical,
        0.0,
        0.0,
        1.0,
        0.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    verts, endpoint = generate_uv_direction_line(
        inverse_uv_cylindrical,
        0.0,
        0.0,
        0.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    verts, _ = generate_uv_direction_line(
        inverse_uv_cylindrical,
        0.0,
        1.0,
        1.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    verts, _ = generate_uv_direction_line(
        inverse_uv_cylindrical,
        1.0,
        0.0,
        1.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)

    if cap:
        _add_cap_direction(
            segments,
            u_tile,
            v_tile,
            u_offset,
            v_offset,
            uv_rotation,
            u_flip,
            v_flip,
            transform,
            1.0,
            u_vertices,
            v_vertices,
            u_proj_vertices,
            v_proj_vertices,
            u_labels,
            v_labels,
        )


def _generate_spherical_normal_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate spherical normal-based UV direction indicators."""
    v_equator = 0.0
    v_north = 1.0

    verts, endpoint = generate_uv_direction_line(
        inverse_uv_spherical,
        0.0,
        v_equator,
        1.0,
        v_equator,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    verts, endpoint = generate_uv_direction_line(
        inverse_uv_spherical,
        0.0,
        v_equator,
        0.0,
        v_north,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    verts, _ = generate_uv_direction_line(
        inverse_uv_spherical,
        0.0,
        v_north,
        1.0,
        v_north,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    verts, _ = generate_uv_direction_line(
        inverse_uv_spherical,
        1.0,
        v_equator,
        1.0,
        v_north,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)


def _generate_shrink_wrap_normal_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Generate shrink wrap normal-based UV direction indicators."""
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        0.0,
        0.0,
        1.0,
        0.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    verts, endpoint = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        0.0,
        0.0,
        0.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    verts, _ = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        0.0,
        1.0,
        1.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    verts, _ = generate_uv_direction_line(
        inverse_uv_shrink_wrap,
        1.0,
        0.0,
        1.0,
        1.0,
        segments * 2,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)


def _add_planar_face_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Add planar face direction indicators (for box mapping)."""
    verts, endpoint = generate_uv_direction_line(
        inverse_uv_planar,
        0.0,
        0.0,
        1.0,
        0.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    u_vertices.extend(verts)
    u_labels.append(endpoint)

    verts, endpoint = generate_uv_direction_line(
        inverse_uv_planar,
        0.0,
        0.0,
        0.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
    )
    v_vertices.extend(verts)
    v_labels.append(endpoint)

    verts, _ = generate_uv_direction_line(
        inverse_uv_planar,
        0.0,
        1.0,
        1.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    u_proj_vertices.extend(verts)

    verts, _ = generate_uv_direction_line(
        inverse_uv_planar,
        1.0,
        0.0,
        1.0,
        1.0,
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        transform,
        dashed=True,
    )
    v_proj_vertices.extend(verts)


def _add_cap_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    cap_z: float,
    u_vertices: list[tuple[float, float, float]],
    v_vertices: list[tuple[float, float, float]],
    u_proj_vertices: list[tuple[float, float, float]],
    v_proj_vertices: list[tuple[float, float, float]],
    u_labels: list[tuple[float, float, float]],
    v_labels: list[tuple[float, float, float]],
) -> None:
    """Add cap direction indicators (for capped cylinder)."""
    cap_offset = Matrix.Translation(Vector((0, 0, cap_z)))
    cap_scale = Matrix.Diagonal(Vector((1.0, 1.0, 0.0, 1.0)))
    cap_xform = transform @ cap_offset @ cap_scale

    _add_planar_face_direction(
        segments,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rotation,
        u_flip,
        v_flip,
        cap_xform,
        u_vertices,
        v_vertices,
        u_proj_vertices,
        v_proj_vertices,
        u_labels,
        v_labels,
    )
