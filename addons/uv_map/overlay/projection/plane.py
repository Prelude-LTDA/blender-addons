"""Plane projection wireframe generator."""

from __future__ import annotations

from mathutils import Euler, Matrix, Vector


def generate_plane_vertices(
    position: Vector,
    rotation: Euler,
    size: Vector,
) -> list[Vector]:
    """Generate vertices for a plane outline."""
    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(size.to_4d())
    transform = (
        Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ scale_matrix
    )

    # Plane corners (in XY plane, centered at origin)
    corners = [
        Vector((-1.0, -1.0, 0.0)),
        Vector((1.0, -1.0, 0.0)),
        Vector((1.0, 1.0, 0.0)),
        Vector((-1.0, 1.0, 0.0)),
    ]

    # Transform vertices
    vertices: list[Vector] = []
    for i, corner in enumerate(corners):
        transformed = transform @ corner
        vertices.append(Vector((transformed.x, transformed.y, transformed.z)))
        # Add next corner for line
        next_corner = corners[(i + 1) % 4]
        transformed_next = transform @ next_corner
        vertices.append(
            Vector((transformed_next.x, transformed_next.y, transformed_next.z))
        )

    return vertices
