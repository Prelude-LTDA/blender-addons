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
    MAPPING_CYLINDRICAL_CAPPED,
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

    # Add X marks on caps to indicate planar mapping
    # Points are on the unit circle at 45° angles (inscribed in the circle)
    sqrt2_inv = 1.0 / math.sqrt(2.0)  # ≈ 0.707

    # Bottom cap X (inscribed in circle)
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

    # Top cap X (inscribed in circle)
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
    drawn_rings: int = 8,
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
    transform = Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix

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
    grid_lines: int = 8,
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
    transform = Matrix.Translation(pos_vec) @ rot_euler.to_matrix().to_4x4() @ scale_matrix

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
    elif mapping_type == MAPPING_CYLINDRICAL_CAPPED:
        vertices = _generate_cylinder_capped_vertices(
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
    elif mapping_type == MAPPING_SHRINK_WRAP:
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
