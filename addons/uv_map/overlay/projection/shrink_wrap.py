"""Shrink wrap (azimuthal equidistant) projection wireframe generators."""

from __future__ import annotations

import math

from mathutils import Euler, Matrix, Vector


def generate_shrink_wrap_vertices(  # noqa: PLR0915
    position: Vector,
    rotation: Euler,
    size: Vector,
    grid_lines: int = 4,
    segments_per_line: int = 32,
) -> list[Vector]:
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
    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(size.to_4d())
    transform = (
        Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[Vector] = []

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

            vertices.append(Vector((t1_vec.x, t1_vec.y, t1_vec.z)))
            vertices.append(Vector((t2_vec.x, t2_vec.y, t2_vec.z)))

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

            vertices.append(Vector((t1_vec.x, t1_vec.y, t1_vec.z)))
            vertices.append(Vector((t2_vec.x, t2_vec.y, t2_vec.z)))

    return vertices


def generate_shrink_wrap_normal_vertices(  # noqa: PLR0915
    position: Vector,
    rotation: Euler,
    size: Vector,
    grid_lines: int = 4,
    segments_per_line: int = 64,
) -> list[Vector]:
    """Generate vertices for shrink wrap (azimuthal) normal-based wireframe indicator.

    Shows rotation and scale orientation. Position is typically set to object origin
    since normal-based mapping doesn't depend on projection position.
    Uses dashed lines to indicate that this represents normal direction, not position.

    Uses inverse azimuthal equidistant projection but with dashed pattern.
    """
    # Scale affects normal transformation - use as ellipsoid radii
    scale_matrix = Matrix.Diagonal((size * 0.5).to_4d())
    transform = (
        Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[Vector] = []

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

            vertices.append(Vector((t1_vec.x, t1_vec.y, t1_vec.z)))
            vertices.append(Vector((t2_vec.x, t2_vec.y, t2_vec.z)))

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

            vertices.append(Vector((t1_vec.x, t1_vec.y, t1_vec.z)))
            vertices.append(Vector((t2_vec.x, t2_vec.y, t2_vec.z)))

    # Add axis indicator arrow showing +Z orientation (center of projection)
    avg_radius = (size.x + size.y + size.z) / 6.0  # Average of half-sizes
    shaft_start = Vector((0.0, 0.0, 0.0))
    shaft_end = Vector((0.0, 0.0, size.z * 0.75))
    rot_mat = rotation.to_matrix().to_4x4()
    t_start = rot_mat @ shaft_start
    t_end = rot_mat @ shaft_end
    vertices.append(Vector((t_start.x, t_start.y, t_start.z)))
    vertices.append(Vector((t_end.x, t_end.y, t_end.z)))

    # Arrowhead for Z
    arrow_size = min(0.1, avg_radius * 0.2)
    for offset in [
        (arrow_size, 0),
        (-arrow_size, 0),
        (0, arrow_size),
        (0, -arrow_size),
    ]:
        arrow_base = Vector((offset[0], offset[1], size.z * 0.75 - arrow_size))
        t_base = rot_mat @ arrow_base
        vertices.append(Vector((t_end.x, t_end.y, t_end.z)))
        vertices.append(Vector((t_base.x, t_base.y, t_base.z)))

    return vertices
