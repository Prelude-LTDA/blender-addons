"""Box (tri-planar) UV direction generators."""

from __future__ import annotations

import math

from mathutils import Euler, Matrix, Vector

from .plane import add_planar_face_direction


def generate_box_direction(
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
    """Generate box (tri-planar) UV direction indicators."""
    # +Z face (top)
    z_face_offset = Matrix.Translation(Vector((0, 0, 1)))
    z_transform = transform @ z_face_offset
    add_planar_face_direction(
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
    add_planar_face_direction(
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
    add_planar_face_direction(
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
