"""
Overlay module for UV Map addon.

Renders wireframe visualization of the UV map shape (plane, cylinder, sphere, box)
in the 3D viewport when a UV Map modifier is active.
"""

from __future__ import annotations

import math

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Matrix, Vector

from .constants import (
    MAPPING_BOX,
    MAPPING_CYLINDRICAL,
    MAPPING_PLANAR,
    MAPPING_SHRINK_WRAP,
    MAPPING_SPHERICAL,
    OVERLAY_COLOR,
    OVERLAY_LINE_WIDTH,
)
from .nodes import is_uv_map_node_group
from .operators import get_uv_map_modifier_params

# Draw handler reference
_draw_handler: object | None = None

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


def _generate_shrink_wrap_vertices(
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
    segments: int = 32,
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
    num_v_segments = 8  # Number of segments for vertical lines
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


def _generate_cylinder_capped_normal_vertices(
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 32,
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
    num_v_segments = 8  # Number of segments for vertical lines
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
    x_segments = 8  # Even number to create dashes (4 dashes per line)

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
    segments: int = 32,
    drawn_rings: int = 4,
    smooth_rings: int = 16,
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
    segments_per_line: int = 32,
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


def _draw_overlay() -> None:  # noqa: PLR0911, PLR0912, PLR0915
    """Draw the UV map shape overlay in the 3D viewport."""
    context = bpy.context

    # Check if we're in the 3D viewport
    if context.area is None or context.area.type != "VIEW_3D":
        return

    # Get active object
    obj = context.active_object
    if obj is None:
        return

    # Get active modifier
    modifier = obj.modifiers.active
    if modifier is None:
        return

    # Check if it's a UV Map modifier
    if modifier.type != "NODES":
        return

    node_tree = getattr(modifier, "node_group", None)
    if node_tree is None or not is_uv_map_node_group(node_tree):
        return

    # Get parameters from modifier
    params = get_uv_map_modifier_params(obj, modifier)
    if params is None:
        return

    # Extract parameters with defaults
    mapping_type = str(params.get("mapping_type", MAPPING_PLANAR))
    position = params.get("position", (0.0, 0.0, 0.0))
    rotation = params.get("rotation", (0.0, 0.0, 0.0))
    size = params.get("size", (1.0, 1.0, 1.0))

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

    # Apply object transform to get world-space coordinates
    obj_matrix = obj.matrix_world

    # Transform position, rotation, size to world space
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
        else:
            # Position-based mapping
            if cap:
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

    # Restore state
    gpu.state.blend_set("NONE")
    gpu.state.depth_mask_set(True)


def register_draw_handler() -> None:
    """Register the overlay draw handler."""
    global _draw_handler  # noqa: PLW0603
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay, (), "WINDOW", "POST_VIEW"
        )


def unregister_draw_handler() -> None:
    """Unregister the overlay draw handler."""
    global _draw_handler  # noqa: PLW0603
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None


# Classes to register (none for this module)
classes: list[type] = []
