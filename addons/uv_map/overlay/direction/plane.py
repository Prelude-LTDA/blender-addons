"""Planar UV direction generators."""

from __future__ import annotations

from mathutils import Matrix, Vector

from .inverse import inverse_uv_planar
from .utils import generate_uv_direction_line


def generate_planar_direction(
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


def add_planar_face_direction(
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
