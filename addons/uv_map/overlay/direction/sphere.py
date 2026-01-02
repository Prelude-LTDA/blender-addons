"""Spherical UV direction generators."""

from __future__ import annotations

from mathutils import Matrix, Vector

from .inverse import inverse_uv_spherical
from .utils import generate_uv_direction_line


def generate_spherical_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[Vector],
    v_vertices: list[Vector],
    u_proj_vertices: list[Vector],
    v_proj_vertices: list[Vector],
    u_labels: list[Vector],
    v_labels: list[Vector],
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


def generate_spherical_normal_direction(
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    u_vertices: list[Vector],
    v_vertices: list[Vector],
    u_proj_vertices: list[Vector],
    v_proj_vertices: list[Vector],
    u_labels: list[Vector],
    v_labels: list[Vector],
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
