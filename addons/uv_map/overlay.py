"""
Overlay module for UV Map addon.

Renders wireframe visualization of the UV map shape (plane, cylinder, sphere, box)
in the 3D viewport when a UV Map modifier is active.

Also renders UV direction indicators showing U and V axis directions as curved
lines that follow the projection surface.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Matrix, Vector

if TYPE_CHECKING:
    from collections.abc import Callable

from .constants import (
    MAPPING_BOX,
    MAPPING_CYLINDRICAL,
    MAPPING_CYLINDRICAL_NORMAL,
    MAPPING_PLANAR,
    MAPPING_SHRINK_WRAP,
    MAPPING_SHRINK_WRAP_NORMAL,
    MAPPING_SPHERICAL,
    MAPPING_SPHERICAL_NORMAL,
    OVERLAY_COLOR,
    OVERLAY_LINE_WIDTH,
    OVERLAY_U_DIRECTION_COLOR,
    OVERLAY_UV_DIRECTION_LINE_WIDTH,
    OVERLAY_V_DIRECTION_COLOR,
)
from .nodes import is_uv_map_node_group
from .operators import (
    get_uv_map_modifier_params,
    get_uv_map_node_group_defaults,
    get_uv_map_node_instance_params,
)

# Draw handler references
_draw_handler: object | None = None
_text_draw_handler: object | None = None

# Text label settings
OVERLAY_LABEL_FONT_SIZE = 14
OVERLAY_LABEL_OFFSET_X = 5  # Offset from point in screen pixels
OVERLAY_LABEL_OFFSET_Y = 5

# Cached label positions for text drawing (populated by 3D overlay, used by 2D text overlay)
_cached_u_labels: list[tuple[float, float, float]] = []
_cached_v_labels: list[tuple[float, float, float]] = []

# Cached shader
_shader: gpu.types.GPUShader | None = None


def _get_shader() -> gpu.types.GPUShader:
    """Get or create the polyline shader for smooth anti-aliased lines."""
    global _shader  # noqa: PLW0603
    if _shader is None:
        _shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    return _shader


def _generate_plane_vertices(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Generate vertices for a plane outline."""
    # Create transform matrix: Translation * Rotation * Scale (TRS)
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = (
        Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix
    )

    # Plane corners (in XY plane, centered at origin)
    corners = [
        Vector((-1.0, -1.0, 0.0)),
        Vector((1.0, -1.0, 0.0)),
        Vector((1.0, 1.0, 0.0)),
        Vector((-1.0, 1.0, 0.0)),
    ]

    # Transform vertices
    vertices: list[tuple[float, float, float]] = []
    for i, corner in enumerate(corners):
        transformed = transform @ corner
        vertices.append((transformed.x, transformed.y, transformed.z))
        # Add next corner for line
        next_corner = corners[(i + 1) % 4]
        transformed_next = transform @ next_corner
        vertices.append((transformed_next.x, transformed_next.y, transformed_next.z))

    return vertices


def _generate_box_vertices(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Generate vertices for a box wireframe."""
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = (
        Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix
    )

    # Box corners
    corners = [
        Vector((-1.0, -1.0, -1.0)),
        Vector((1.0, -1.0, -1.0)),
        Vector((1.0, 1.0, -1.0)),
        Vector((-1.0, 1.0, -1.0)),
        Vector((-1.0, -1.0, 1.0)),
        Vector((1.0, -1.0, 1.0)),
        Vector((1.0, 1.0, 1.0)),
        Vector((-1.0, 1.0, 1.0)),
    ]

    # Edges (pairs of corner indices)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]

    vertices: list[tuple[float, float, float]] = []
    for i1, i2 in edges:
        for idx in (i1, i2):
            corner = corners[idx]
            transformed = transform @ corner
            vertices.append((transformed.x, transformed.y, transformed.z))

    return vertices


def _generate_cylinder_vertices(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 32,
) -> list[tuple[float, float, float]]:
    """Generate vertices for a cylinder wireframe."""
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = (
        Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[tuple[float, float, float]] = []

    # Generate circle at bottom (z = -1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -1.0))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate circle at top (z = 1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, 1.0))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate vertical lines (4 evenly spaced)
    for i in range(4):
        angle = (i / 4) * 2.0 * math.pi
        x, y = 1.0 * math.cos(angle), 1.0 * math.sin(angle)

        # Bottom point
        point_bottom = Vector((x, y, -1.0))
        transformed_bottom = transform @ point_bottom
        vertices.append(
            (transformed_bottom.x, transformed_bottom.y, transformed_bottom.z)
        )

        # Top point
        point_top = Vector((x, y, 1.0))
        transformed_top = transform @ point_top
        vertices.append((transformed_top.x, transformed_top.y, transformed_top.z))

    return vertices


def _generate_cylinder_capped_vertices(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 32,
) -> list[tuple[float, float, float]]:
    """Generate vertices for a capped cylinder wireframe.

    Same as cylinder but with X marks on the top and bottom caps.
    """
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = (
        Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[tuple[float, float, float]] = []

    # Generate circle at bottom (z = -1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -1.0))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate circle at top (z = 1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, 1.0))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate vertical lines (4 evenly spaced)
    for i in range(4):
        angle = (i / 4) * 2.0 * math.pi
        x, y = 1.0 * math.cos(angle), 1.0 * math.sin(angle)

        # Bottom point
        point_bottom = Vector((x, y, -1.0))
        transformed_bottom = transform @ point_bottom
        vertices.append(
            (transformed_bottom.x, transformed_bottom.y, transformed_bottom.z)
        )

        # Top point
        point_top = Vector((x, y, 1.0))
        transformed_top = transform @ point_top
        vertices.append((transformed_top.x, transformed_top.y, transformed_top.z))

    # Add X marks on caps to indicate planar mapping
    # Points are on the unit circle at 45° angles (inscribed in the circle)
    sqrt2_inv = 1.0 / math.sqrt(2.0)  # ≈ 0.707

    # Bottom cap X (inscribed in circle) - solid lines
    for p1, p2 in [
        ((-sqrt2_inv, -sqrt2_inv), (sqrt2_inv, sqrt2_inv)),
        ((-sqrt2_inv, sqrt2_inv), (sqrt2_inv, -sqrt2_inv)),
    ]:
        point1 = Vector((p1[0], p1[1], -1.0))
        point2 = Vector((p2[0], p2[1], -1.0))
        t1 = transform @ point1
        t2 = transform @ point2
        vertices.append((t1.x, t1.y, t1.z))
        vertices.append((t2.x, t2.y, t2.z))

    # Top cap X (inscribed in circle) - solid lines
    for p1, p2 in [
        ((-sqrt2_inv, -sqrt2_inv), (sqrt2_inv, sqrt2_inv)),
        ((-sqrt2_inv, sqrt2_inv), (sqrt2_inv, -sqrt2_inv)),
    ]:
        point1 = Vector((p1[0], p1[1], 1.0))
        point2 = Vector((p2[0], p2[1], 1.0))
        t1 = transform @ point1
        t2 = transform @ point2
        vertices.append((t1.x, t1.y, t1.z))
        vertices.append((t2.x, t2.y, t2.z))

    return vertices


def _generate_sphere_vertices(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 32,
    drawn_rings: int = 4,
    smooth_rings: int = 32,
) -> list[tuple[float, float, float]]:
    """Generate vertices for a sphere wireframe.

    Args:
        segments: Number of segments for each circle
        drawn_rings: Number of latitude circles to actually draw
        smooth_rings: Number of steps for longitude lines (higher = smoother)
    """
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = (
        Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[tuple[float, float, float]] = []

    # Generate latitude circles (only every Nth ring, based on ratio)
    ring_step = max(1, smooth_rings // drawn_rings)
    for ring in range(ring_step, smooth_rings, ring_step):
        phi = (ring / smooth_rings) * math.pi  # 0 to pi
        z = 1.0 * math.cos(phi)
        radius = 1.0 * math.sin(phi)

        for i in range(segments):
            angle1 = (i / segments) * 2.0 * math.pi
            angle2 = ((i + 1) / segments) * 2.0 * math.pi

            x1, y1 = radius * math.cos(angle1), radius * math.sin(angle1)
            x2, y2 = radius * math.cos(angle2), radius * math.sin(angle2)

            for x, y in [(x1, y1), (x2, y2)]:
                point = Vector((x, y, z))
                transformed = transform @ point
                vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate longitude lines (4 evenly spaced) with smooth_rings steps
    for i in range(4):
        theta = (i / 4) * 2.0 * math.pi

        for ring in range(smooth_rings):
            phi1 = (ring / smooth_rings) * math.pi
            phi2 = ((ring + 1) / smooth_rings) * math.pi

            for phi in [phi1, phi2]:
                x = 1.0 * math.sin(phi) * math.cos(theta)
                y = 1.0 * math.sin(phi) * math.sin(theta)
                z = 1.0 * math.cos(phi)

                point = Vector((x, y, z))
                transformed = transform @ point
                vertices.append((transformed.x, transformed.y, transformed.z))

    return vertices


def _generate_shrink_wrap_vertices(  # noqa: PLR0915
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    grid_lines: int = 4,
    segments_per_line: int = 32,
) -> list[tuple[float, float, float]]:
    """Generate vertices for shrink wrap (azimuthal equidistant) wireframe.

    Shows a UV grid projected onto the sphere using the inverse azimuthal
    equidistant projection. This visualizes what a regular checkerboard
    pattern looks like when mapped - lines that are straight in UV space
    become curves that all meet at the -Z pole.

    Inverse projection formula:
      dx = u - 0.5, dy = v - 0.5
      r = sqrt(dx² + dy²)
      phi = atan2(dy, dx)
      theta = r * π
      x = sin(theta) * cos(phi)
      y = sin(theta) * sin(phi)
      z = cos(theta)

    Lines are extended beyond [0,1] UV range to reach r=1 (the -Z pole).
    """
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = (
        Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[tuple[float, float, float]] = []

    def uv_to_sphere(u: float, v: float) -> Vector:
        """Inverse azimuthal equidistant projection from UV to sphere."""
        dx = u - 0.5
        dy = v - 0.5
        r = math.sqrt(dx * dx + dy * dy)

        if r < 1e-6:
            # At the center (top pole)
            return Vector((0.0, 0.0, 1.0))

        # Clamp r to 1.0 (the bottom pole)
        if r > 1.0:
            r = 1.0
            scale = 1.0 / math.sqrt(dx * dx + dy * dy)
            dx *= scale
            dy *= scale

        phi = math.atan2(dy, dx)
        theta = r * math.pi  # r goes from 0 to 1, theta from 0 to π

        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)

        return Vector((x, y, z))

    # Generate vertical lines (constant U) in UV space
    # Extend v range to reach r=1 (bottom pole)
    for i in range(grid_lines + 1):
        u = i / grid_lines  # 0 to 1
        dx = u - 0.5

        # Calculate v range to reach r=1
        # r² = dx² + dy² = 1, so dy² = 1 - dx²
        if abs(dx) < 1.0:
            dy_max = math.sqrt(1.0 - dx * dx)
            v_min = 0.5 - dy_max
            v_max = 0.5 + dy_max
        else:
            # dx >= 1, line doesn't exist in valid projection
            continue

        for j in range(segments_per_line):
            t1 = j / segments_per_line
            t2 = (j + 1) / segments_per_line
            v1 = v_min + t1 * (v_max - v_min)
            v2 = v_min + t2 * (v_max - v_min)

            p1 = uv_to_sphere(u, v1)
            p2 = uv_to_sphere(u, v2)

            t1_vec = transform @ p1
            t2_vec = transform @ p2

            vertices.append((t1_vec.x, t1_vec.y, t1_vec.z))
            vertices.append((t2_vec.x, t2_vec.y, t2_vec.z))

    # Generate horizontal lines (constant V) in UV space
    # Extend u range to reach r=1 (bottom pole)
    for j in range(grid_lines + 1):
        v = j / grid_lines  # 0 to 1
        dy = v - 0.5

        # Calculate u range to reach r=1
        # r² = dx² + dy² = 1, so dx² = 1 - dy²
        if abs(dy) < 1.0:
            dx_max = math.sqrt(1.0 - dy * dy)
            u_min = 0.5 - dx_max
            u_max = 0.5 + dx_max
        else:
            # dy >= 1, line doesn't exist in valid projection
            continue

        for i in range(segments_per_line):
            t1 = i / segments_per_line
            t2 = (i + 1) / segments_per_line
            u1 = u_min + t1 * (u_max - u_min)
            u2 = u_min + t2 * (u_max - u_min)

            p1 = uv_to_sphere(u1, v)
            p2 = uv_to_sphere(u2, v)

            t1_vec = transform @ p1
            t2_vec = transform @ p2

            vertices.append((t1_vec.x, t1_vec.y, t1_vec.z))
            vertices.append((t2_vec.x, t2_vec.y, t2_vec.z))

    return vertices


def _generate_cylinder_normal_vertices(
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 64,
) -> list[tuple[float, float, float]]:
    """Generate vertices for a normal-based cylinder wireframe indicator.

    Shows rotation and scale orientation. Position is irrelevant for normal-based mapping.
    Dashed circles indicate that this represents normal direction, not position.
    """
    rot_euler = Euler(rotation, "XYZ")
    transform = rot_euler.to_matrix().to_4x4()

    vertices: list[tuple[float, float, float]] = []

    # Use size to scale the cylinder (affects normal transformation)
    radius_x = size[0] * 0.5
    radius_y = size[1] * 0.5
    height = size[2] * 0.5

    # Generate ellipse at bottom (z = -height) with dashed pattern (every other segment)
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -height))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate ellipse at top (z = height) with dashed pattern
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, height))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate vertical lines (4 evenly spaced, dashed with subdivisions)
    num_v_segments = 16  # Number of segments for vertical lines
    for i in range(4):
        angle = (i / 4) * 2.0 * math.pi
        x, y = radius_x * math.cos(angle), radius_y * math.sin(angle)

        # Draw dashed vertical line with subdivisions (every other segment)
        for seg in range(0, num_v_segments, 2):
            z1 = -height + (seg / num_v_segments) * (2.0 * height)
            z2 = -height + ((seg + 1) / num_v_segments) * (2.0 * height)

            point1 = Vector((x, y, z1))
            point2 = Vector((x, y, z2))
            t1 = transform @ point1
            t2 = transform @ point2
            vertices.append((t1.x, t1.y, t1.z))
            vertices.append((t2.x, t2.y, t2.z))

    # Add axis indicator arrow pointing along Z (shows the cylinder's main axis)
    # Arrow shaft - scale with height but cap the arrow size for visibility
    avg_radius = (radius_x + radius_y) * 0.5
    shaft_start = Vector((0.0, 0.0, 0.0))
    shaft_end = Vector((0.0, 0.0, height * 1.5))
    t_start = transform @ shaft_start
    t_end = transform @ shaft_end
    vertices.append((t_start.x, t_start.y, t_start.z))
    vertices.append((t_end.x, t_end.y, t_end.z))

    # Arrowhead - scale with average radius
    arrow_size = min(0.15, avg_radius * 0.3)
    for offset in [
        (arrow_size, 0),
        (-arrow_size, 0),
        (0, arrow_size),
        (0, -arrow_size),
    ]:
        arrow_base = Vector((offset[0], offset[1], height * 1.5 - arrow_size))
        t_base = transform @ arrow_base
        vertices.append((t_end.x, t_end.y, t_end.z))
        vertices.append((t_base.x, t_base.y, t_base.z))

    return vertices


def _generate_cylinder_capped_normal_vertices(  # noqa: PLR0915
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 64,
) -> list[tuple[float, float, float]]:
    """Generate vertices for a capped normal-based cylinder wireframe indicator.

    Same as cylinder normal but with X marks on the top and bottom caps
    to indicate planar normal-based mapping on caps.
    """
    rot_euler = Euler(rotation, "XYZ")
    transform = rot_euler.to_matrix().to_4x4()

    vertices: list[tuple[float, float, float]] = []

    # Use size to scale the cylinder (affects normal transformation)
    radius_x = size[0] * 0.5
    radius_y = size[1] * 0.5
    height = size[2] * 0.5

    # Generate ellipse at bottom (z = -height) with dashed pattern (every other segment)
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -height))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate ellipse at top (z = height) with dashed pattern
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, height))
            transformed = transform @ point
            vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate vertical lines (4 evenly spaced, dashed with subdivisions)
    num_v_segments = 16  # Number of segments for vertical lines
    for i in range(4):
        angle = (i / 4) * 2.0 * math.pi
        x, y = radius_x * math.cos(angle), radius_y * math.sin(angle)

        # Draw dashed vertical line with subdivisions (every other segment)
        for seg in range(0, num_v_segments, 2):
            z1 = -height + (seg / num_v_segments) * (2.0 * height)
            z2 = -height + ((seg + 1) / num_v_segments) * (2.0 * height)

            point1 = Vector((x, y, z1))
            point2 = Vector((x, y, z2))
            t1 = transform @ point1
            t2 = transform @ point2
            vertices.append((t1.x, t1.y, t1.z))
            vertices.append((t2.x, t2.y, t2.z))

    # Add axis indicator arrow pointing along Z (shows the cylinder's main axis)
    avg_radius = (radius_x + radius_y) * 0.5
    shaft_start = Vector((0.0, 0.0, 0.0))
    shaft_end = Vector((0.0, 0.0, height * 1.5))
    t_start = transform @ shaft_start
    t_end = transform @ shaft_end
    vertices.append((t_start.x, t_start.y, t_start.z))
    vertices.append((t_end.x, t_end.y, t_end.z))

    # Arrowhead - scale with average radius
    arrow_size = min(0.15, avg_radius * 0.3)
    for offset in [
        (arrow_size, 0),
        (-arrow_size, 0),
        (0, arrow_size),
        (0, -arrow_size),
    ]:
        arrow_base = Vector((offset[0], offset[1], height * 1.5 - arrow_size))
        t_base = transform @ arrow_base
        vertices.append((t_end.x, t_end.y, t_end.z))
        vertices.append((t_base.x, t_base.y, t_base.z))

    # Add X marks on caps to indicate planar normal-based mapping
    # Scale the X marks to fit within the ellipse
    sqrt2_inv = 1.0 / math.sqrt(2.0)  # ≈ 0.707

    # Number of segments for dashed X lines
    x_segments = 16  # Even number to create dashes (8 dashes per line)

    # Bottom cap X (inscribed in ellipse) - dashed
    for p1, p2 in [
        (
            (-sqrt2_inv * radius_x, -sqrt2_inv * radius_y),
            (sqrt2_inv * radius_x, sqrt2_inv * radius_y),
        ),
        (
            (-sqrt2_inv * radius_x, sqrt2_inv * radius_y),
            (sqrt2_inv * radius_x, -sqrt2_inv * radius_y),
        ),
    ]:
        # Draw dashed line (every other segment)
        for seg in range(0, x_segments, 2):
            t = seg / x_segments
            t_next = (seg + 1) / x_segments
            x1 = p1[0] + (p2[0] - p1[0]) * t
            y1 = p1[1] + (p2[1] - p1[1]) * t
            x2 = p1[0] + (p2[0] - p1[0]) * t_next
            y2 = p1[1] + (p2[1] - p1[1]) * t_next
            point1 = Vector((x1, y1, -height))
            point2 = Vector((x2, y2, -height))
            t1 = transform @ point1
            t2 = transform @ point2
            vertices.append((t1.x, t1.y, t1.z))
            vertices.append((t2.x, t2.y, t2.z))

    # Top cap X (inscribed in ellipse) - dashed
    for p1, p2 in [
        (
            (-sqrt2_inv * radius_x, -sqrt2_inv * radius_y),
            (sqrt2_inv * radius_x, sqrt2_inv * radius_y),
        ),
        (
            (-sqrt2_inv * radius_x, sqrt2_inv * radius_y),
            (sqrt2_inv * radius_x, -sqrt2_inv * radius_y),
        ),
    ]:
        # Draw dashed line (every other segment)
        for seg in range(0, x_segments, 2):
            t = seg / x_segments
            t_next = (seg + 1) / x_segments
            x1 = p1[0] + (p2[0] - p1[0]) * t
            y1 = p1[1] + (p2[1] - p1[1]) * t
            x2 = p1[0] + (p2[0] - p1[0]) * t_next
            y2 = p1[1] + (p2[1] - p1[1]) * t_next
            point1 = Vector((x1, y1, height))
            point2 = Vector((x2, y2, height))
            t1 = transform @ point1
            t2 = transform @ point2
            vertices.append((t1.x, t1.y, t1.z))
            vertices.append((t2.x, t2.y, t2.z))

    return vertices


def _generate_sphere_normal_vertices(
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 64,
    drawn_rings: int = 4,
    smooth_rings: int = 32,
) -> list[tuple[float, float, float]]:
    """Generate vertices for a normal-based sphere/ellipsoid wireframe indicator.

    Shows rotation and scale orientation. Position is irrelevant for normal-based mapping.
    Dashed latitude lines indicate that this represents normal direction, not position.
    """
    rot_euler = Euler(rotation, "XYZ")
    transform = rot_euler.to_matrix().to_4x4()

    vertices: list[tuple[float, float, float]] = []

    # Use size to scale the ellipsoid (affects normal transformation)
    radius_x = size[0] * 0.5
    radius_y = size[1] * 0.5
    radius_z = size[2] * 0.5

    # Generate latitude circles (dashed - every other segment)
    ring_step = max(1, smooth_rings // drawn_rings)
    for ring in range(ring_step, smooth_rings, ring_step):
        phi = (ring / smooth_rings) * math.pi
        z = radius_z * math.cos(phi)
        # Scale ring radii based on latitude position and axis scales
        ring_scale = math.sin(phi)
        ring_radius_x = radius_x * ring_scale
        ring_radius_y = radius_y * ring_scale

        for i in range(0, segments, 2):  # Dashed pattern
            angle1 = (i / segments) * 2.0 * math.pi
            angle2 = ((i + 1) / segments) * 2.0 * math.pi

            x1, y1 = ring_radius_x * math.cos(angle1), ring_radius_y * math.sin(angle1)
            x2, y2 = ring_radius_x * math.cos(angle2), ring_radius_y * math.sin(angle2)

            for x, y in [(x1, y1), (x2, y2)]:
                point = Vector((x, y, z))
                transformed = transform @ point
                vertices.append((transformed.x, transformed.y, transformed.z))

    # Generate longitude lines (4 evenly spaced, dashed)
    for i in range(4):
        theta = (i / 4) * 2.0 * math.pi

        # Draw dashed longitude line (every other segment)
        for ring in range(0, smooth_rings, 2):
            phi1 = (ring / smooth_rings) * math.pi
            phi2 = ((ring + 1) / smooth_rings) * math.pi

            x1 = radius_x * math.sin(phi1) * math.cos(theta)
            y1 = radius_y * math.sin(phi1) * math.sin(theta)
            z1 = radius_z * math.cos(phi1)

            x2 = radius_x * math.sin(phi2) * math.cos(theta)
            y2 = radius_y * math.sin(phi2) * math.sin(theta)
            z2 = radius_z * math.cos(phi2)

            point1 = Vector((x1, y1, z1))
            point2 = Vector((x2, y2, z2))
            t1 = transform @ point1
            t2 = transform @ point2
            vertices.append((t1.x, t1.y, t1.z))
            vertices.append((t2.x, t2.y, t2.z))

    # Add axis indicator arrows showing orientation
    # Z axis (primary) - scale with radius_z
    avg_radius = (radius_x + radius_y + radius_z) / 3.0
    shaft_start = Vector((0.0, 0.0, 0.0))
    shaft_end = Vector((0.0, 0.0, radius_z * 1.5))
    t_start = transform @ shaft_start
    t_end = transform @ shaft_end
    vertices.append((t_start.x, t_start.y, t_start.z))
    vertices.append((t_end.x, t_end.y, t_end.z))

    # Arrowhead for Z - scale with average radius
    arrow_size = min(0.1, avg_radius * 0.2)
    for offset in [
        (arrow_size, 0),
        (-arrow_size, 0),
        (0, arrow_size),
        (0, -arrow_size),
    ]:
        arrow_base = Vector((offset[0], offset[1], radius_z * 1.5 - arrow_size))
        t_base = transform @ arrow_base
        vertices.append((t_end.x, t_end.y, t_end.z))
        vertices.append((t_base.x, t_base.y, t_base.z))

    return vertices


def _generate_shrink_wrap_normal_vertices(  # noqa: PLR0915
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    grid_lines: int = 4,
    segments_per_line: int = 64,
) -> list[tuple[float, float, float]]:
    """Generate vertices for shrink wrap (azimuthal) normal-based wireframe indicator.

    Shows rotation and scale orientation. Position is irrelevant for normal-based mapping.
    Uses dashed lines to indicate that this represents normal direction, not position.

    Uses inverse azimuthal equidistant projection but with dashed pattern.
    """
    rot_euler = Euler(rotation, "XYZ")

    # Scale affects normal transformation - use as ellipsoid radii
    scale_vec = Vector(size)
    scale_matrix = Matrix.Diagonal((scale_vec * 0.5).to_4d())
    transform = rot_euler.to_matrix().to_4x4() @ scale_matrix

    vertices: list[tuple[float, float, float]] = []

    def uv_to_sphere(u: float, v: float) -> Vector:
        """Inverse azimuthal equidistant projection from UV to sphere."""
        dx = u - 0.5
        dy = v - 0.5
        r = math.sqrt(dx * dx + dy * dy)

        if r < 1e-6:
            # At the center (top pole)
            return Vector((0.0, 0.0, 1.0))

        # Clamp r to 1.0 (the bottom pole)
        if r > 1.0:
            r = 1.0
            scale = 1.0 / math.sqrt(dx * dx + dy * dy)
            dx *= scale
            dy *= scale

        phi = math.atan2(dy, dx)
        theta = r * math.pi

        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)

        return Vector((x, y, z))

    # Generate vertical lines (constant U) in UV space - dashed pattern
    for i in range(grid_lines + 1):
        u = i / grid_lines
        dx = u - 0.5

        if abs(dx) < 1.0:
            dy_max = math.sqrt(1.0 - dx * dx)
            v_min = 0.5 - dy_max
            v_max = 0.5 + dy_max
        else:
            continue

        # Dashed: every other pair of segments
        for j in range(0, segments_per_line, 2):
            t1 = j / segments_per_line
            t2 = (j + 1) / segments_per_line
            v1 = v_min + t1 * (v_max - v_min)
            v2 = v_min + t2 * (v_max - v_min)

            p1 = uv_to_sphere(u, v1)
            p2 = uv_to_sphere(u, v2)

            t1_vec = transform @ p1
            t2_vec = transform @ p2

            vertices.append((t1_vec.x, t1_vec.y, t1_vec.z))
            vertices.append((t2_vec.x, t2_vec.y, t2_vec.z))

    # Generate horizontal lines (constant V) in UV space - dashed pattern
    for j in range(grid_lines + 1):
        v = j / grid_lines
        dy = v - 0.5

        if abs(dy) < 1.0:
            dx_max = math.sqrt(1.0 - dy * dy)
            u_min = 0.5 - dx_max
            u_max = 0.5 + dx_max
        else:
            continue

        # Dashed: every other pair of segments
        for i in range(0, segments_per_line, 2):
            t1 = i / segments_per_line
            t2 = (i + 1) / segments_per_line
            u1 = u_min + t1 * (u_max - u_min)
            u2 = u_min + t2 * (u_max - u_min)

            p1 = uv_to_sphere(u1, v)
            p2 = uv_to_sphere(u2, v)

            t1_vec = transform @ p1
            t2_vec = transform @ p2

            vertices.append((t1_vec.x, t1_vec.y, t1_vec.z))
            vertices.append((t2_vec.x, t2_vec.y, t2_vec.z))

    # Add axis indicator arrow showing +Z orientation (center of projection)
    avg_radius = sum(size) / 6.0  # Average of half-sizes
    shaft_start = Vector((0.0, 0.0, 0.0))
    shaft_end = Vector((0.0, 0.0, size[2] * 0.75))
    rot_mat = rot_euler.to_matrix().to_4x4()
    t_start = rot_mat @ shaft_start
    t_end = rot_mat @ shaft_end
    vertices.append((t_start.x, t_start.y, t_start.z))
    vertices.append((t_end.x, t_end.y, t_end.z))

    # Arrowhead for Z
    arrow_size = min(0.1, avg_radius * 0.2)
    for offset in [
        (arrow_size, 0),
        (-arrow_size, 0),
        (0, arrow_size),
        (0, -arrow_size),
    ]:
        arrow_base = Vector((offset[0], offset[1], size[2] * 0.75 - arrow_size))
        t_base = rot_mat @ arrow_base
        vertices.append((t_end.x, t_end.y, t_end.z))
        vertices.append((t_base.x, t_base.y, t_base.z))

    return vertices


# =============================================================================
# UV Direction Overlay Functions
# =============================================================================
# These functions generate curved lines showing U and V directions on the
# projection surface. The lines start at UV (0, 0) and extend to (1, 0) for U
# and (0, 1) for V, following the curvature of each projection type.
#
# The inverse functions take final UV coordinates (after all processing) and
# convert them back to 3D positions on the projection surface.


def _compute_adaptive_segments(u_tile: float, v_tile: float, base_segments: int) -> int:
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


def _inverse_uv_planar(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for planar mapping.

    Forward chain: raw_uv -> tile -> rotate -> flip -> offset -> final_uv
    Reverse chain: final_uv -> un-offset -> un-flip -> un-rotate -> un-tile -> raw_uv
    Then: raw_uv -> 3D position

    Planar formula: U = x * 0.5, V = y * 0.5
    Inverse: x = U * 2, y = V * 2, z = 0
    """
    # 1. Remove offset
    u1 = u - u_offset
    v1 = v - v_offset

    # 2. Reverse flip (flip formula is: tile - value)
    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    # 3. Reverse rotation (apply negative rotation)
    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    # 4. Reverse tiling
    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # 5. Reverse mapping formula (Planar: U = x * 0.5, V = y * 0.5)
    x = u_raw * 2.0
    y = v_raw * 2.0
    z = 0.0

    return Vector((x, y, z))


def _inverse_uv_cylindrical(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for cylindrical mapping.

    The cylindrical mapping uses:
    - U: angle around Z axis (one full rotation per U unit, before tiling)
    - V: height along Z axis

    Internal scale: -4x on U (so U=0.25 = one rotation), 0.75x on V
    """
    # Reverse UV processing chain
    u1 = u - u_offset
    v1 = v - v_offset

    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # Convert to rotation fraction (0-1 = full rotation)
    # Internal scale is -4, so U_raw=1 means -0.25 rotations = +0.75 rotations
    u_frac = -u_raw / 4.0

    # Cylindrical coordinates
    # U fraction -> angle theta (0 to 2π for full rotation)
    theta = u_frac * 2.0 * math.pi

    x = math.sin(theta)
    y = math.cos(theta)
    # V has internal scale of 0.75, so reverse it
    z = v_raw / 0.75

    return Vector((x, y, z))


def _inverse_uv_spherical(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for spherical mapping.

    The spherical mapping in nodes.py uses:
    - U: atan2(x,y) / 2π, then scaled by -4
    - V: (acos(z/length) / π - 0.5), then scaled by -2

    The -0.5 offset centers V=0 at the equator:
    - V=0 → equator (z=0)
    - V>0 → northern hemisphere
    - V<0 → southern hemisphere

    Inverse:
        atan2(x,y) = U_output * π / (-2) = -U_output * π / 2
        acos(z_norm) = (V_output / (-2) + 0.5) * π = -V_output * π / 2 + π/2
        z_norm = cos(-V_output * π / 2 + π/2) = sin(V_output * π / 2)
    """
    # Reverse UV processing chain
    u1 = u - u_offset
    v1 = v - v_offset

    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # Reverse the spherical mapping
    # atan2(x,y) = -u_raw * π / 2 (after accounting for -4x scale)
    phi = -u_raw * math.pi / 2.0

    # With equator offset: V_raw = (-2) * (acos(z_norm)/π - 0.5)
    # Solving: acos(z_norm) = π * (0.5 - V_raw/2)
    # z_norm = cos(π * (0.5 - V_raw/2)) = cos(π/2 - V_raw*π/2) = sin(V_raw*π/2)
    # theta (polar angle from +Z) = acos(z_norm) = π/2 - V_raw*π/2
    theta = math.pi / 2.0 - v_raw * math.pi / 2.0
    theta = max(0.0, min(math.pi, theta))  # Clamp to valid range

    # The theta here is the polar angle from +Z axis
    # sin(theta) gives the xy-plane distance, cos(theta) gives z
    z = math.cos(theta)
    xy_radius = math.sin(theta)

    # phi is atan2(x, y), so x = r*sin(phi), y = r*cos(phi)
    x = xy_radius * math.sin(phi)
    y = xy_radius * math.cos(phi)

    return Vector((x, y, z))


def _inverse_uv_shrink_wrap(
    u: float,
    v: float,
    u_tile: float,
    v_tile: float,
    u_offset: float,
    v_offset: float,
    uv_rotation: float,
    u_flip: bool,
    v_flip: bool,
) -> Vector:
    """Convert UV coordinates back to 3D position for shrink wrap mapping.

    Shrink wrap (azimuthal equidistant) formula:
    theta = acos(z / length)
    phi = atan2(y, x)
    r = theta / π
    U = r * cos(phi), V = r * sin(phi)

    Inverse:
    r = sqrt(U² + V²)
    phi = atan2(V, U)
    theta = r * π
    x = sin(theta) * cos(phi), y = sin(theta) * sin(phi), z = cos(theta)
    """
    # Reverse UV processing chain
    u1 = u - u_offset
    v1 = v - v_offset

    if u_flip:
        u1 = u_tile - u1
    if v_flip:
        v1 = v_tile - v1

    cos_r = math.cos(-uv_rotation)
    sin_r = math.sin(-uv_rotation)
    u2 = u1 * cos_r - v1 * sin_r
    v2 = u1 * sin_r + v1 * cos_r

    u_raw = u2 / u_tile if u_tile != 0 else u2
    v_raw = v2 / v_tile if v_tile != 0 else v2

    # Reverse shrink wrap mapping
    r = math.sqrt(u_raw * u_raw + v_raw * v_raw)

    if r < 1e-6:
        return Vector((0.0, 0.0, 1.0))

    phi = math.atan2(v_raw, u_raw)
    theta = r * math.pi

    # Clamp theta
    theta = min(theta, math.pi)

    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    return Vector((x, y, z))


def _generate_uv_direction_line(
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


def _generate_uv_direction_vertices(  # noqa: PLR0915
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
    segments = _compute_adaptive_segments(u_tile, v_tile, base_segments)

    u_vertices: list[tuple[float, float, float]] = []
    v_vertices: list[tuple[float, float, float]] = []
    u_proj_vertices: list[tuple[float, float, float]] = []
    v_proj_vertices: list[tuple[float, float, float]] = []
    u_labels: list[tuple[float, float, float]] = []
    v_labels: list[tuple[float, float, float]] = []

    if mapping_type == MAPPING_PLANAR:
        # Planar: simple straight lines from (0,0) to (1,0) and (0,0) to (0,1)
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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

        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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

        # Projected lines (dashed) to complete the parallelogram
        # U projection: from V endpoint (0,1) in U direction to (1,1)
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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

        # V projection: from U endpoint (1,0) in V direction to (1,1)
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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

    elif mapping_type == MAPPING_CYLINDRICAL:
        if cap:
            # Capped cylinder: side + top + bottom projections
            # Side projection (cylindrical)
            verts, endpoint = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

            verts, endpoint = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

            # Projected lines (dashed) for capped cylinder side
            # U projection: from V endpoint (0,1) in U direction to (1,1)
            verts, _ = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

            # V projection: from U endpoint (1,0) in V direction to (1,1)
            verts, _ = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

            # Cap indicator at z=+1 (top only) using planar mapping
            for cap_z in [1.0]:
                cap_offset = Matrix.Translation(Vector((0, 0, cap_z)))
                cap_scale = Matrix.Diagonal(Vector((1.0, 1.0, 0.0, 1.0)))
                cap_xform = transform @ cap_offset @ cap_scale

                verts, endpoint = _generate_uv_direction_line(
                    _inverse_uv_planar,
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
                    cap_xform,
                )
                u_vertices.extend(verts)
                u_labels.append(endpoint)

                verts, endpoint = _generate_uv_direction_line(
                    _inverse_uv_planar,
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
                    cap_xform,
                )
                v_vertices.extend(verts)
                v_labels.append(endpoint)

                # Cap projected lines
                verts, _ = _generate_uv_direction_line(
                    _inverse_uv_planar,
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
                    cap_xform,
                    dashed=True,
                )
                u_proj_vertices.extend(verts)
                verts, _ = _generate_uv_direction_line(
                    _inverse_uv_planar,
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
                    cap_xform,
                    dashed=True,
                )
                v_proj_vertices.extend(verts)
        else:
            # Regular cylinder: curved U line around circumference
            verts, endpoint = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

            verts, endpoint = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

            # Projected lines (dashed) for regular cylinder
            # U projection: from V endpoint (0,1) in U direction to (1,1)
            verts, _ = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

            # V projection: from U endpoint (1,0) in V direction to (1,1)
            verts, _ = _generate_uv_direction_line(
                _inverse_uv_cylindrical,
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

    elif mapping_type == MAPPING_SPHERICAL:
        # Spherical: curved lines at the equator to avoid pole singularity
        # With equator-centered coordinates: V=0 is equator in output space
        # V>0 is northern hemisphere, V<0 is southern hemisphere
        # Pass raw UV coordinates (0-1 range), tiling is handled by inverse function
        v_equator = 0.0
        v_north = 1.0  # One unit toward north pole

        # U line at equator going around
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

        # V line at U=0 going from equator toward north pole
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

        # Projected lines (dashed) for spherical
        # U projection: from V endpoint in U direction
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

        # V projection: from U endpoint in V direction
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

    elif mapping_type == MAPPING_SHRINK_WRAP:
        # Shrink wrap: radial projection from center
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

        # Projected lines (dashed) for shrink wrap
        # U projection: from V endpoint (0,1) in U direction to (1,1)
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

        # V projection: from U endpoint (1,0) in V direction to (1,1)
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

    elif mapping_type == MAPPING_BOX:
        # Box: show UV directions on each of the 3 positive faces
        # Each face is at +1 on its axis, using planar projection

        # +Z face (top): at z=+1, projects XY
        z_face_offset = Matrix.Translation(Vector((0, 0, 1)))
        z_transform = transform @ z_face_offset
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            z_transform,
        )
        u_vertices.extend(verts)
        u_labels.append(endpoint)

        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            z_transform,
        )
        v_vertices.extend(verts)
        v_labels.append(endpoint)

        # Z face projected lines
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            z_transform,
            dashed=True,
        )
        u_proj_vertices.extend(verts)
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            z_transform,
            dashed=True,
        )
        v_proj_vertices.extend(verts)

        # +X face: at x=+1, projects YZ (U=Y, V=Z in nodes)
        # Rotate +90° around Z to map X->Y, then -90° around Y to face +X direction
        # Combined: we need planar's U(X)->Y and V(Y)->Z
        # Use rotation that swaps appropriately
        x_rot = Euler((math.pi / 2, 0, math.pi / 2), "XYZ").to_matrix().to_4x4()
        x_face_offset = Matrix.Translation(Vector((1, 0, 0)))
        x_transform = transform @ x_face_offset @ x_rot
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            x_transform,
        )
        u_vertices.extend(verts)
        u_labels.append(endpoint)

        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            x_transform,
        )
        v_vertices.extend(verts)
        v_labels.append(endpoint)

        # X face projected lines
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            x_transform,
            dashed=True,
        )
        u_proj_vertices.extend(verts)
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            x_transform,
            dashed=True,
        )
        v_proj_vertices.extend(verts)

        # +Y face: at y=+1, projects XZ (rotate +90° around X to align Z->Y)
        y_rot = Euler((math.pi / 2, 0, 0), "XYZ").to_matrix().to_4x4()
        y_face_offset = Matrix.Translation(Vector((0, 1, 0)))
        y_transform = transform @ y_face_offset @ y_rot
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            y_transform,
        )
        u_vertices.extend(verts)
        u_labels.append(endpoint)

        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            y_transform,
        )
        v_vertices.extend(verts)
        v_labels.append(endpoint)

        # Y face projected lines
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            y_transform,
            dashed=True,
        )
        u_proj_vertices.extend(verts)
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_planar,
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
            y_transform,
            dashed=True,
        )
        v_proj_vertices.extend(verts)

    elif mapping_type == MAPPING_CYLINDRICAL_NORMAL:
        # Cylindrical Normal: uses normal direction instead of position
        # Show on a unit sphere to visualize the normal-space mapping
        # U line at equator (z=0) going around
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_cylindrical,
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

        # V line going up along z axis
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_cylindrical,
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

        # Projected lines
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_cylindrical,
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

        verts, _ = _generate_uv_direction_line(
            _inverse_uv_cylindrical,
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

        # Cap UV indicator for normal-based capped cylinder
        if cap:
            # For normal-based, the cap is at z=+1 on the unit sphere
            # Use planar mapping on the cap (normal = +Z maps to center)
            cap_offset = Matrix.Translation(Vector((0, 0, 1.0)))
            cap_scale = Matrix.Diagonal(Vector((1.0, 1.0, 0.0, 1.0)))
            cap_xform = transform @ cap_offset @ cap_scale

            verts, endpoint = _generate_uv_direction_line(
                _inverse_uv_planar,
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
                cap_xform,
            )
            u_vertices.extend(verts)
            u_labels.append(endpoint)

            verts, endpoint = _generate_uv_direction_line(
                _inverse_uv_planar,
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
                cap_xform,
            )
            v_vertices.extend(verts)
            v_labels.append(endpoint)

            # Cap projected lines
            verts, _ = _generate_uv_direction_line(
                _inverse_uv_planar,
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
                cap_xform,
                dashed=True,
            )
            u_proj_vertices.extend(verts)
            verts, _ = _generate_uv_direction_line(
                _inverse_uv_planar,
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
                cap_xform,
                dashed=True,
            )
            v_proj_vertices.extend(verts)

    elif mapping_type == MAPPING_SPHERICAL_NORMAL:
        # Spherical Normal: uses normal direction instead of position
        # Same UV mapping as spherical, visualized on a unit sphere
        v_equator = 0.0
        v_north = 1.0

        # U line at equator going around
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

        # V line from equator toward north pole
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

        # Projected lines
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

        verts, _ = _generate_uv_direction_line(
            _inverse_uv_spherical,
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

    elif mapping_type == MAPPING_SHRINK_WRAP_NORMAL:
        # Shrink Wrap Normal: uses normal direction instead of position
        # Same UV mapping as shrink wrap, visualized on a unit sphere
        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

        verts, endpoint = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

        # Projected lines
        verts, _ = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

        verts, _ = _generate_uv_direction_line(
            _inverse_uv_shrink_wrap,
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

    return u_vertices, v_vertices, u_proj_vertices, v_proj_vertices, u_labels, v_labels


def _draw_overlay() -> None:  # noqa: PLR0912, PLR0915
    """Draw the UV map shape overlay in the 3D viewport."""
    global _cached_u_labels, _cached_v_labels  # noqa: PLW0603
    context = bpy.context

    # Clear cached labels at the start - they'll be repopulated if we draw
    _cached_u_labels = []
    _cached_v_labels = []

    # Check if we're in the 3D viewport
    if context.area is None or context.area.type != "VIEW_3D":
        return

    # Try to get parameters from multiple sources:
    # 1. Active modifier on active object
    # 2. UV map node group being edited in node editor

    params: dict[str, object] | None = None
    obj_matrix: Matrix | None = None

    # First, try active modifier
    obj = context.active_object
    if obj is not None:
        modifier = obj.modifiers.active
        if modifier is not None and modifier.type == "NODES":
            node_tree = getattr(modifier, "node_group", None)
            if node_tree is not None and is_uv_map_node_group(node_tree):
                params = get_uv_map_modifier_params(obj, modifier)
                obj_matrix = obj.matrix_world

    # If no modifier, check for UV map node group in node editor
    if params is None and context.screen is not None:
        # Look for a node editor with a UV map node group
        for area in context.screen.areas:
            if area.type != "NODE_EDITOR":
                continue
            space = area.spaces.active
            if space is None or space.type != "NODE_EDITOR":
                continue
            # Check if it's editing a geometry node tree
            if getattr(space, "tree_type", None) != "GeometryNodeTree":
                continue
            edit_tree = getattr(space, "edit_tree", None)
            if edit_tree is None:
                continue

            # Use active object's matrix if available, otherwise identity
            fallback_matrix = (
                obj.matrix_world if obj is not None else Matrix.Identity(4)
            )

            # Check if the edit_tree itself is a UV map node group
            if is_uv_map_node_group(edit_tree):
                # Get defaults from the node group interface
                params = get_uv_map_node_group_defaults(edit_tree)
                obj_matrix = fallback_matrix
                break

            # Also check if there's a selected node that is a UV Map node group
            for node in edit_tree.nodes:
                if not node.select:
                    continue
                if node.type != "GROUP":
                    continue
                node_group = getattr(node, "node_tree", None)
                if node_group is not None and is_uv_map_node_group(node_group):
                    # Get values from the node instance's inputs
                    params = get_uv_map_node_instance_params(node)
                    obj_matrix = fallback_matrix
                    break
            if params is not None:
                break

    if params is None or obj_matrix is None:
        return

    # Extract parameters with defaults
    mapping_type = str(params.get("mapping_type", MAPPING_PLANAR))
    position = params.get("position", (0.0, 0.0, 0.0))
    rotation = params.get("rotation", (0.0, 0.0, 0.0))
    size = params.get("size", (1.0, 1.0, 1.0))

    # UV processing parameters
    u_tile = float(params.get("u_tile", 1.0))  # type: ignore[arg-type]
    v_tile = float(params.get("v_tile", 1.0))  # type: ignore[arg-type]
    u_offset = float(params.get("u_offset", 0.0))  # type: ignore[arg-type]
    v_offset = float(params.get("v_offset", 0.0))  # type: ignore[arg-type]
    uv_rot = float(params.get("uv_rotation", 0.0))  # type: ignore[arg-type]
    u_flip = bool(params.get("u_flip", False))
    v_flip = bool(params.get("v_flip", False))

    # Ensure tuples
    if not isinstance(position, tuple):
        position = tuple(position) if hasattr(position, "__iter__") else (0.0, 0.0, 0.0)  # type: ignore[arg-type]
    if not isinstance(rotation, tuple):
        rotation = tuple(rotation) if hasattr(rotation, "__iter__") else (0.0, 0.0, 0.0)  # type: ignore[arg-type]
    if not isinstance(size, tuple):
        size = tuple(size) if hasattr(size, "__iter__") else (1.0, 1.0, 1.0)  # type: ignore[arg-type]

    # Ensure 3-element tuples for rotation (may be 4 for quaternion)
    if len(rotation) > 3:
        rotation = rotation[:3]  # type: ignore[assignment]

    # Transform position, rotation, size to world space (using obj_matrix set above)
    world_position = obj_matrix @ Vector(position)  # type: ignore[arg-type]

    # For rotation, we need to combine object rotation with UV map rotation
    obj_rotation = obj_matrix.to_euler()
    uv_rotation = Euler(rotation, "XYZ")  # type: ignore[arg-type]
    combined_rotation = (
        obj_rotation.x + uv_rotation.x,
        obj_rotation.y + uv_rotation.y,
        obj_rotation.z + uv_rotation.z,
    )

    # For size, scale by object scale
    obj_scale = obj_matrix.to_scale()
    world_size = (
        size[0] * obj_scale.x,  # type: ignore[index]
        size[1] * obj_scale.y,  # type: ignore[index]
        size[2] * obj_scale.z,  # type: ignore[index]
    )

    # Generate vertices based on mapping type
    # Check if normal-based mapping is enabled (for cylindrical, spherical, shrink wrap)
    normal_based = bool(params.get("normal_based", False))
    cap = bool(params.get("cap", False))

    if mapping_type == MAPPING_PLANAR:
        vertices = _generate_plane_vertices(
            (world_position.x, world_position.y, world_position.z),
            combined_rotation,
            world_size,
        )
    elif mapping_type == MAPPING_BOX:
        vertices = _generate_box_vertices(
            (world_position.x, world_position.y, world_position.z),
            combined_rotation,
            world_size,
        )
    elif mapping_type == MAPPING_CYLINDRICAL:
        if normal_based:
            # Normal-based mapping: position is irrelevant, but scale affects normals
            # Center at object origin to make it clear these don't depend on position
            obj_origin = obj_matrix @ Vector((0.0, 0.0, 0.0))
            if cap:
                vertices = _generate_cylinder_capped_normal_vertices(
                    combined_rotation, world_size
                )
            else:
                vertices = _generate_cylinder_normal_vertices(
                    combined_rotation, world_size
                )
            # Offset all vertices to object origin
            vertices = [
                (v[0] + obj_origin.x, v[1] + obj_origin.y, v[2] + obj_origin.z)
                for v in vertices
            ]
        # Position-based mapping
        elif cap:
            vertices = _generate_cylinder_capped_vertices(
                (world_position.x, world_position.y, world_position.z),
                combined_rotation,
                world_size,
            )
        else:
            vertices = _generate_cylinder_vertices(
                (world_position.x, world_position.y, world_position.z),
                combined_rotation,
                world_size,
            )
    elif mapping_type == MAPPING_SPHERICAL:
        if normal_based:
            # Normal-based mapping: position is irrelevant, but scale affects normals
            # Center at object origin to make it clear these don't depend on position
            obj_origin = obj_matrix @ Vector((0.0, 0.0, 0.0))
            vertices = _generate_sphere_normal_vertices(combined_rotation, world_size)
            # Offset all vertices to object origin
            vertices = [
                (v[0] + obj_origin.x, v[1] + obj_origin.y, v[2] + obj_origin.z)
                for v in vertices
            ]
        else:
            vertices = _generate_sphere_vertices(
                (world_position.x, world_position.y, world_position.z),
                combined_rotation,
                world_size,
            )
    elif mapping_type == MAPPING_SHRINK_WRAP:
        if normal_based:
            # Normal-based mapping: position is irrelevant, but scale affects normals
            # Center at object origin to make it clear these don't depend on position
            obj_origin = obj_matrix @ Vector((0.0, 0.0, 0.0))
            vertices = _generate_shrink_wrap_normal_vertices(
                combined_rotation, world_size
            )
            # Offset all vertices to object origin
            vertices = [
                (v[0] + obj_origin.x, v[1] + obj_origin.y, v[2] + obj_origin.z)
                for v in vertices
            ]
        else:
            # Shrink wrap uses azimuthal grid overlay to show single-pole projection
            vertices = _generate_shrink_wrap_vertices(
                (world_position.x, world_position.y, world_position.z),
                combined_rotation,
                world_size,
            )
    else:
        return

    if not vertices:
        return

    # Get viewport dimensions for the polyline shader
    region = context.region
    viewport_size = (region.width, region.height) if region else (1920, 1080)

    # Draw wireframe using POLYLINE shader for smooth anti-aliased lines
    shader = _get_shader()
    batch = batch_for_shader(shader, "LINES", {"pos": vertices})

    # Enable blending and disable depth test so overlay is always visible
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)

    shader.bind()
    shader.uniform_float("lineWidth", OVERLAY_LINE_WIDTH)
    shader.uniform_float("viewportSize", viewport_size)
    shader.uniform_float("color", OVERLAY_COLOR)
    batch.draw(shader)

    # Draw UV direction indicators (yellow lines showing U and V axes)
    # For normal-based mappings, use the normal mapping type and center at object origin
    if normal_based:
        # Determine the actual normal-based mapping type
        if mapping_type == MAPPING_CYLINDRICAL:
            effective_mapping_type = MAPPING_CYLINDRICAL_NORMAL
        elif mapping_type == MAPPING_SPHERICAL:
            effective_mapping_type = MAPPING_SPHERICAL_NORMAL
        elif mapping_type == MAPPING_SHRINK_WRAP:
            effective_mapping_type = MAPPING_SHRINK_WRAP_NORMAL
        else:
            effective_mapping_type = mapping_type

        # Use object origin for position since normal-based mappings don't use position
        obj_origin = obj_matrix @ Vector((0.0, 0.0, 0.0))
        direction_position = (obj_origin.x, obj_origin.y, obj_origin.z)
        # Normal-based overlays use half scale (same as shape wireframe)
        direction_size = (world_size[0] * 0.5, world_size[1] * 0.5, world_size[2] * 0.5)
    else:
        effective_mapping_type = mapping_type
        direction_position = (world_position.x, world_position.y, world_position.z)
        direction_size = world_size

    (
        u_dir_vertices,
        v_dir_vertices,
        u_proj_vertices,
        v_proj_vertices,
        u_labels,
        v_labels,
    ) = _generate_uv_direction_vertices(
        effective_mapping_type,
        direction_position,
        combined_rotation,
        direction_size,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rot,
        u_flip,
        v_flip,
        cap,
    )

    # Draw U direction line (yellow)
    if u_dir_vertices:
        u_batch = batch_for_shader(shader, "LINES", {"pos": u_dir_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH)
        shader.uniform_float("color", OVERLAY_U_DIRECTION_COLOR)
        u_batch.draw(shader)

    # Draw V direction line (yellow-green)
    if v_dir_vertices:
        v_batch = batch_for_shader(shader, "LINES", {"pos": v_dir_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH)
        shader.uniform_float("color", OVERLAY_V_DIRECTION_COLOR)
        v_batch.draw(shader)

    # Draw projected U line (dashed, same color as U but thinner)
    if u_proj_vertices:
        u_proj_batch = batch_for_shader(shader, "LINES", {"pos": u_proj_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH * 0.7)
        shader.uniform_float("color", OVERLAY_U_DIRECTION_COLOR)
        u_proj_batch.draw(shader)

    # Draw projected V line (dashed, same color as V but thinner)
    if v_proj_vertices:
        v_proj_batch = batch_for_shader(shader, "LINES", {"pos": v_proj_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH * 0.7)
        shader.uniform_float("color", OVERLAY_V_DIRECTION_COLOR)
        v_proj_batch.draw(shader)

    # Cache label positions for the text drawing handler
    _cached_u_labels = u_labels
    _cached_v_labels = v_labels

    # Restore state
    gpu.state.blend_set("NONE")
    gpu.state.depth_mask_set(True)


def _draw_text_overlay() -> None:
    """Draw U and V text labels in screen space (POST_PIXEL handler)."""
    context = bpy.context

    # Check if we're in the 3D viewport
    if context.area is None or context.area.type != "VIEW_3D":
        return

    # Get region data for 3D to 2D conversion
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    # Check if there are any labels to draw
    if not _cached_u_labels and not _cached_v_labels:
        return

    # Set up blf font
    font_id = 0
    blf.size(font_id, OVERLAY_LABEL_FONT_SIZE)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.7)  # Dark shadow for readability
    blf.shadow_offset(font_id, 1, -1)

    # Draw U labels
    for pos_3d in _cached_u_labels:
        pos_2d = location_3d_to_region_2d(region, rv3d, Vector(pos_3d))
        if pos_2d is not None:
            blf.position(
                font_id,
                pos_2d.x + OVERLAY_LABEL_OFFSET_X,
                pos_2d.y + OVERLAY_LABEL_OFFSET_Y,
                0,
            )
            blf.color(font_id, *OVERLAY_U_DIRECTION_COLOR)
            blf.draw(font_id, "U")

    # Draw V labels
    for pos_3d in _cached_v_labels:
        pos_2d = location_3d_to_region_2d(region, rv3d, Vector(pos_3d))
        if pos_2d is not None:
            blf.position(
                font_id,
                pos_2d.x + OVERLAY_LABEL_OFFSET_X,
                pos_2d.y + OVERLAY_LABEL_OFFSET_Y,
                0,
            )
            blf.color(font_id, *OVERLAY_V_DIRECTION_COLOR)
            blf.draw(font_id, "V")

    blf.disable(font_id, blf.SHADOW)


def register_draw_handler() -> None:
    """Register the overlay draw handlers."""
    global _draw_handler, _text_draw_handler  # noqa: PLW0603
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay, (), "WINDOW", "POST_VIEW"
        )
    if _text_draw_handler is None:
        _text_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_text_overlay, (), "WINDOW", "POST_PIXEL"
        )


def unregister_draw_handler() -> None:
    """Unregister the overlay draw handlers."""
    global _draw_handler, _text_draw_handler  # noqa: PLW0603
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    if _text_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_text_draw_handler, "WINDOW")
        _text_draw_handler = None


# Classes to register (none for this module)
classes: list[type] = []
