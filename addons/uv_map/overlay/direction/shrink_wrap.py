"""Shrink wrap (azimuthal) UV direction generators."""

from __future__ import annotations

from mathutils import Matrix

from .inverse import inverse_uv_shrink_wrap
from .utils import generate_uv_direction_line


def generate_shrink_wrap_direction(
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


def generate_shrink_wrap_normal_direction(
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
