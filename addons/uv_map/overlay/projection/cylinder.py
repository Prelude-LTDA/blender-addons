"""Cylinder projection wireframe generators."""

from __future__ import annotations

import math

from mathutils import Euler, Matrix, Vector


def generate_cylinder_vertices(
    position: Vector,
    rotation: Euler,
    size: Vector,
    segments: int = 32,
) -> list[Vector]:
    """Generate vertices for a cylinder wireframe."""
    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(size.to_4d())
    transform = (
        Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[Vector] = []

    # Generate circle at bottom (z = -1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -1.0))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

    # Generate circle at top (z = 1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, 1.0))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

    # Generate vertical lines (4 evenly spaced)
    for i in range(4):
        angle = (i / 4) * 2.0 * math.pi
        x, y = 1.0 * math.cos(angle), 1.0 * math.sin(angle)

        # Bottom point
        point_bottom = Vector((x, y, -1.0))
        transformed_bottom = transform @ point_bottom
        vertices.append(
            Vector((transformed_bottom.x, transformed_bottom.y, transformed_bottom.z))
        )

        # Top point
        point_top = Vector((x, y, 1.0))
        transformed_top = transform @ point_top
        vertices.append(Vector((transformed_top.x, transformed_top.y, transformed_top.z)))

    return vertices


def generate_cylinder_capped_vertices(
    position: Vector,
    rotation: Euler,
    size: Vector,
    segments: int = 32,
) -> list[Vector]:
    """Generate vertices for a capped cylinder wireframe.

    Same as cylinder but with X marks on the top and bottom caps.
    """
    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(size.to_4d())
    transform = (
        Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ scale_matrix
    )

    vertices: list[Vector] = []

    # Generate circle at bottom (z = -1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -1.0))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

    # Generate circle at top (z = 1.0)
    for i in range(segments):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = 1.0 * math.cos(angle1), 1.0 * math.sin(angle1)
        x2, y2 = 1.0 * math.cos(angle2), 1.0 * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, 1.0))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

    # Generate vertical lines (4 evenly spaced)
    for i in range(4):
        angle = (i / 4) * 2.0 * math.pi
        x, y = 1.0 * math.cos(angle), 1.0 * math.sin(angle)

        # Bottom point
        point_bottom = Vector((x, y, -1.0))
        transformed_bottom = transform @ point_bottom
        vertices.append(
            Vector((transformed_bottom.x, transformed_bottom.y, transformed_bottom.z))
        )

        # Top point
        point_top = Vector((x, y, 1.0))
        transformed_top = transform @ point_top
        vertices.append(Vector((transformed_top.x, transformed_top.y, transformed_top.z)))

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
        vertices.append(Vector((t1.x, t1.y, t1.z)))
        vertices.append(Vector((t2.x, t2.y, t2.z)))

    # Top cap X (inscribed in circle) - solid lines
    for p1, p2 in [
        ((-sqrt2_inv, -sqrt2_inv), (sqrt2_inv, sqrt2_inv)),
        ((-sqrt2_inv, sqrt2_inv), (sqrt2_inv, -sqrt2_inv)),
    ]:
        point1 = Vector((p1[0], p1[1], 1.0))
        point2 = Vector((p2[0], p2[1], 1.0))
        t1 = transform @ point1
        t2 = transform @ point2
        vertices.append(Vector((t1.x, t1.y, t1.z)))
        vertices.append(Vector((t2.x, t2.y, t2.z)))

    return vertices


def generate_cylinder_normal_vertices(
    position: Vector,
    rotation: Euler,
    size: Vector,
    segments: int = 64,
) -> list[Vector]:
    """Generate vertices for a normal-based cylinder wireframe indicator.

    Shows rotation and scale orientation. Position is typically set to object origin
    since normal-based mapping doesn't depend on projection position.
    Dashed circles indicate that this represents normal direction, not position.
    """
    transform = Matrix.Translation(position) @ rotation.to_matrix().to_4x4()

    vertices: list[Vector] = []

    # Use size to scale the cylinder (affects normal transformation)
    radius_x = size.x * 0.5
    radius_y = size.y * 0.5
    height = size.z * 0.5

    # Generate ellipse at bottom (z = -height) with dashed pattern (every other segment)
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -height))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

    # Generate ellipse at top (z = height) with dashed pattern
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, height))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

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
            vertices.append(Vector((t1.x, t1.y, t1.z)))
            vertices.append(Vector((t2.x, t2.y, t2.z)))

    # Add axis indicator arrow pointing along Z (shows the cylinder's main axis)
    # Arrow shaft - scale with height but cap the arrow size for visibility
    avg_radius = (radius_x + radius_y) * 0.5
    shaft_start = Vector((0.0, 0.0, 0.0))
    shaft_end = Vector((0.0, 0.0, height * 1.5))
    t_start = transform @ shaft_start
    t_end = transform @ shaft_end
    vertices.append(Vector((t_start.x, t_start.y, t_start.z)))
    vertices.append(Vector((t_end.x, t_end.y, t_end.z)))

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
        vertices.append(Vector((t_end.x, t_end.y, t_end.z)))
        vertices.append(Vector((t_base.x, t_base.y, t_base.z)))

    return vertices


def generate_cylinder_capped_normal_vertices(  # noqa: PLR0915
    position: Vector,
    rotation: Euler,
    size: Vector,
    segments: int = 64,
) -> list[Vector]:
    """Generate vertices for a capped normal-based cylinder wireframe indicator.

    Same as cylinder normal but with X marks on the top and bottom caps
    to indicate planar normal-based mapping on caps.
    Position is typically set to object origin since normal-based mapping
    doesn't depend on projection position.
    """
    transform = Matrix.Translation(position) @ rotation.to_matrix().to_4x4()

    vertices: list[Vector] = []

    # Use size to scale the cylinder (affects normal transformation)
    radius_x = size.x * 0.5
    radius_y = size.y * 0.5
    height = size.z * 0.5

    # Generate ellipse at bottom (z = -height) with dashed pattern (every other segment)
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, -height))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

    # Generate ellipse at top (z = height) with dashed pattern
    for i in range(0, segments, 2):
        angle1 = (i / segments) * 2.0 * math.pi
        angle2 = ((i + 1) / segments) * 2.0 * math.pi

        x1, y1 = radius_x * math.cos(angle1), radius_y * math.sin(angle1)
        x2, y2 = radius_x * math.cos(angle2), radius_y * math.sin(angle2)

        for x, y in [(x1, y1), (x2, y2)]:
            point = Vector((x, y, height))
            transformed = transform @ point
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

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
            vertices.append(Vector((t1.x, t1.y, t1.z)))
            vertices.append(Vector((t2.x, t2.y, t2.z)))

    # Add axis indicator arrow pointing along Z (shows the cylinder's main axis)
    avg_radius = (radius_x + radius_y) * 0.5
    shaft_start = Vector((0.0, 0.0, 0.0))
    shaft_end = Vector((0.0, 0.0, height * 1.5))
    t_start = transform @ shaft_start
    t_end = transform @ shaft_end
    vertices.append(Vector((t_start.x, t_start.y, t_start.z)))
    vertices.append(Vector((t_end.x, t_end.y, t_end.z)))

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
        vertices.append(Vector((t_end.x, t_end.y, t_end.z)))
        vertices.append(Vector((t_base.x, t_base.y, t_base.z)))

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
            vertices.append(Vector((t1.x, t1.y, t1.z)))
            vertices.append(Vector((t2.x, t2.y, t2.z)))

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
            vertices.append(Vector((t1.x, t1.y, t1.z)))
            vertices.append(Vector((t2.x, t2.y, t2.z)))

    return vertices
