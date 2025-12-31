"""Sphere projection wireframe generators."""

from __future__ import annotations

import math

from mathutils import Euler, Matrix, Vector


def generate_sphere_vertices(
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


def generate_sphere_normal_vertices(
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
