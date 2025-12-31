"""Box projection wireframe generator."""

from __future__ import annotations

from mathutils import Euler, Matrix, Vector


def generate_box_vertices(
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
