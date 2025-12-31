"""Utility functions for UV direction generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from mathutils import Matrix, Vector


def compute_adaptive_segments(u_tile: float, v_tile: float, base_segments: int) -> int:
    """Compute number of segments based on UV tiling.

    Lower tiling values mean longer lines that may wrap around the shape
    multiple times, requiring more segments to look smooth.
    """
    # More segments for lower tile values (longer lines)
    min_tile = min(u_tile, v_tile)
    min_tile = max(min_tile, 0.001)

    # Scale segments inversely with tile value
    # At tile=1, use base segments
    # At tile=0.1, use 10x base segments
    # At tile=0.01, use 100x base segments (capped)
    multiplier = max(1.0, 1.0 / min_tile)
    segments = int(base_segments * min(multiplier, 50.0))  # Cap at 50x

    return max(base_segments, min(segments, base_segments * 50))


def generate_uv_direction_line(
    inverse_func: Callable[
        [float, float, float, float, float, float, float, bool, bool], Vector
    ],
    u_start: float,
    v_start: float,
    u_end: float,
    v_end: float,
    segments: int,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
    transform: Matrix,
    dashed: bool = False,
) -> tuple[list[tuple[float, float, float]], tuple[float, float, float]]:
    """Generate a curved line in 3D space by sampling UV coordinates.

    Args:
        inverse_func: Function that converts UV to 3D position
        u_start, v_start: Starting UV coordinates
        u_end, v_end: Ending UV coordinates
        segments: Number of line segments
        transform: World transform matrix to apply
        dashed: If True, generate dashed line (every other segment)
        ... UV processing parameters

    Returns:
        Tuple of (vertex list for LINES primitive, endpoint position for label)
    """
    vertices: list[tuple[float, float, float]] = []
    endpoint: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # For dashed lines, use more segments to create finer dashes
    if dashed:
        segments = segments * 2  # 2x more segments for finer dashes

    for i in range(segments):
        # For dashed lines, skip every other segment
        if dashed and i % 2 == 1:
            # Still need to track the endpoint
            if i == segments - 1:
                t2 = (i + 1) / segments
                u2 = u_start + (u_end - u_start) * t2
                v2 = v_start + (v_end - v_start) * t2
                p2 = inverse_func(
                    u2,
                    v2,
                    u_tile,
                    v_tile,
                    u_offset,
                    v_offset,
                    uv_rotation,
                    u_flip,
                    v_flip,
                )
                t2_vec = transform @ p2
                endpoint = (t2_vec.x, t2_vec.y, t2_vec.z)
            continue

        t1 = i / segments
        t2 = (i + 1) / segments

        u1 = u_start + (u_end - u_start) * t1
        v1 = v_start + (v_end - v_start) * t1
        u2 = u_start + (u_end - u_start) * t2
        v2 = v_start + (v_end - v_start) * t2

        p1 = inverse_func(
            u1, v1, u_tile, v_tile, u_offset, v_offset, uv_rotation, u_flip, v_flip
        )
        p2 = inverse_func(
            u2, v2, u_tile, v_tile, u_offset, v_offset, uv_rotation, u_flip, v_flip
        )

        t1_vec = transform @ p1
        t2_vec = transform @ p2

        vertices.append((t1_vec.x, t1_vec.y, t1_vec.z))
        vertices.append((t2_vec.x, t2_vec.y, t2_vec.z))

        # Store the last endpoint for label positioning
        if i == segments - 1:
            endpoint = (t2_vec.x, t2_vec.y, t2_vec.z)

    return vertices, endpoint
