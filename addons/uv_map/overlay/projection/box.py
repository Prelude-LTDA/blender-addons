"""Box projection wireframe generator."""

from __future__ import annotations

from mathutils import Euler, Matrix, Vector


def generate_box_vertices(
    position: Vector,
    rotation: Euler,
    size: Vector,
) -> list[Vector]:
    """Generate vertices for a box wireframe."""
    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(size.to_4d())
    transform = (
        Matrix.Translation(position) @ rotation.to_matrix().to_4x4() @ scale_matrix
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

    vertices: list[Vector] = []
    for i1, i2 in edges:
        for idx in (i1, i2):
            corner = corners[idx]
            transformed = transform @ corner
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))

    return vertices
