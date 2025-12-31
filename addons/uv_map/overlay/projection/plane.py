"""Plane projection wireframe generator."""

from __future__ import annotations

from mathutils import Euler, Matrix, Vector


def generate_plane_vertices(
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
