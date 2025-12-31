"""Cylindrical UV direction generators."""

from __future__ import annotations

from mathutils import Matrix, Vector

from .inverse import inverse_uv_cylindrical
from .plane import add_planar_face_direction
from .utils import generate_uv_direction_line


def generate_cylindrical_direction(
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

    add_planar_face_direction(
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


def generate_cylindrical_capped_direction(
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
    """Generate cylindrical capped UV direction indicators."""
    # Generate the base cylindrical directions
    generate_cylindrical_direction(
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

    # Add cap indicators
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


def generate_cylindrical_normal_direction(
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


def generate_cylindrical_normal_capped_direction(
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
    """Generate cylindrical normal-based capped UV direction indicators."""
    # Generate the base cylindrical normal directions
    generate_cylindrical_normal_direction(
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

    # Add cap indicators
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
