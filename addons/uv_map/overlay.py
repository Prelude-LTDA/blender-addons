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
    """Get or create the wireframe shader."""
    global _shader  # noqa: PLW0603
    if _shader is None:
        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
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
    transform = Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix

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
    transform = Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix

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
    transform = Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix

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


def _generate_sphere_vertices(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    size: tuple[float, float, float],
    segments: int = 32,
    rings: int = 8,
) -> list[tuple[float, float, float]]:
    """Generate vertices for a sphere wireframe."""
    pos_vec = Vector(position)
    rot_euler = Euler(rotation, "XYZ")
    scale_vec = Vector(size)

    # Build full TRS matrix
    scale_matrix = Matrix.Diagonal(scale_vec.to_4d())
    transform = Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix

    vertices: list[tuple[float, float, float]] = []

    # Generate latitude circles
    for ring in range(1, rings):
        phi = (ring / rings) * math.pi  # 0 to pi
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

    # Generate longitude lines (4 evenly spaced)
    for i in range(4):
        theta = (i / 4) * 2.0 * math.pi

        for ring in range(rings):
            phi1 = (ring / rings) * math.pi
            phi2 = ((ring + 1) / rings) * math.pi

            for phi in [phi1, phi2]:
                x = 1.0 * math.sin(phi) * math.cos(theta)
                y = 1.0 * math.sin(phi) * math.sin(theta)
                z = 1.0 * math.cos(phi)

                point = Vector((x, y, z))
                transformed = transform @ point
                vertices.append((transformed.x, transformed.y, transformed.z))

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
        vertices = _generate_cylinder_vertices(
            (world_position.x, world_position.y, world_position.z),
            combined_rotation,
            world_size,
        )
    elif mapping_type == MAPPING_SPHERICAL:
        vertices = _generate_sphere_vertices(
            (world_position.x, world_position.y, world_position.z),
            combined_rotation,
            world_size,
        )
    else:
        return

    if not vertices:
        return

    # Draw wireframe
    shader = _get_shader()
    batch = batch_for_shader(shader, "LINES", {"pos": vertices})

    # Enable line smoothing and set width
    # Disable depth test so overlay is always visible regardless of clipping planes
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(OVERLAY_LINE_WIDTH)
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)

    shader.bind()
    shader.uniform_float("color", OVERLAY_COLOR)
    batch.draw(shader)

    # Restore state
    gpu.state.blend_set("NONE")
    gpu.state.line_width_set(1.0)
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
