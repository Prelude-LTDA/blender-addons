"""Draw handlers for UV Map overlay visualization."""

from __future__ import annotations

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Matrix, Vector

from ..constants import (
    OVERLAY_COLOR,
    OVERLAY_LABEL_FONT_SIZE,
    OVERLAY_LABEL_OFFSET_X,
    OVERLAY_LABEL_OFFSET_Y,
    OVERLAY_LINE_WIDTH,
    OVERLAY_U_DIRECTION_COLOR,
    OVERLAY_UV_DIRECTION_LINE_WIDTH,
    OVERLAY_V_DIRECTION_COLOR,
)
from ..nodes import is_uv_map_node_group
from ..operators import (
    get_uv_map_modifier_params,
    get_uv_map_node_group_defaults,
    get_uv_map_node_instance_params,
)
from ..shared.uv_map.constants import (
    MAPPING_BOX,
    MAPPING_CYLINDRICAL,
    MAPPING_CYLINDRICAL_NORMAL,
    MAPPING_PLANAR,
    MAPPING_SHRINK_WRAP,
    MAPPING_SHRINK_WRAP_NORMAL,
    MAPPING_SPHERICAL,
    MAPPING_SPHERICAL_NORMAL,
)
from .direction import generate_uv_direction_vertices
from .projection import (
    generate_box_vertices,
    generate_cylinder_capped_normal_vertices,
    generate_cylinder_capped_vertices,
    generate_cylinder_normal_vertices,
    generate_cylinder_vertices,
    generate_plane_vertices,
    generate_shrink_wrap_normal_vertices,
    generate_shrink_wrap_vertices,
    generate_sphere_normal_vertices,
    generate_sphere_vertices,
)

# Draw handler references
_draw_handler: object | None = None
_text_draw_handler: object | None = None

# Cached label positions for text drawing (populated by 3D overlay, used by 2D text overlay)
_cached_u_labels: list[tuple[float, float, float]] = []
_cached_v_labels: list[tuple[float, float, float]] = []

# Cached shader
_shader: gpu.types.GPUShader | None = None


def _get_shader() -> gpu.types.GPUShader:
    """Get or create the polyline shader for smooth anti-aliased lines."""
    global _shader  # noqa: PLW0603
    if _shader is None:
        _shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    return _shader


def _draw_overlay() -> None:  # noqa: PLR0912, PLR0915
    """Draw the UV map shape overlay in the 3D viewport."""
    global _cached_u_labels, _cached_v_labels  # noqa: PLW0603
    context = bpy.context

    # Clear cached labels at the start - they'll be repopulated if we draw
    _cached_u_labels = []
    _cached_v_labels = []

    # Check if we're in the 3D viewport
    if context.area is None or context.area.type != "VIEW_3D":
        return

    # Try to get parameters from multiple sources:
    # 1. Active modifier on active object
    # 2. UV map node group being edited in node editor

    params: dict[str, object] | None = None
    obj_matrix: Matrix | None = None

    # First, try active modifier
    obj = context.active_object
    if obj is not None:
        modifier = obj.modifiers.active
        if modifier is not None and modifier.type == "NODES":
            node_tree = getattr(modifier, "node_group", None)
            if node_tree is not None and is_uv_map_node_group(node_tree):
                params = get_uv_map_modifier_params(obj, modifier)
                obj_matrix = obj.matrix_world

    # If no modifier, check for UV map node group in node editor
    if params is None and context.screen is not None:
        # Look for a node editor with a UV map node group
        for area in context.screen.areas:
            if area.type != "NODE_EDITOR":
                continue
            space = area.spaces.active
            if space is None or space.type != "NODE_EDITOR":
                continue
            # Check if it's editing a geometry node tree
            if getattr(space, "tree_type", None) != "GeometryNodeTree":
                continue
            edit_tree = getattr(space, "edit_tree", None)
            if edit_tree is None:
                continue

            # Use active object's matrix if available, otherwise identity
            fallback_matrix = (
                obj.matrix_world if obj is not None else Matrix.Identity(4)
            )

            # Check if the edit_tree itself is a UV map node group
            if is_uv_map_node_group(edit_tree):
                # Get defaults from the node group interface
                params = get_uv_map_node_group_defaults(edit_tree)
                obj_matrix = fallback_matrix
                break

            # Also check if there's a selected node that is a UV Map node group
            for node in edit_tree.nodes:
                if not node.select:
                    continue
                if node.type != "GROUP":
                    continue
                node_group = getattr(node, "node_tree", None)
                if node_group is not None and is_uv_map_node_group(node_group):
                    # Get values from the node instance's inputs
                    params = get_uv_map_node_instance_params(node)
                    obj_matrix = fallback_matrix
                    break
            if params is not None:
                break

    if params is None or obj_matrix is None:
        return

    # Extract parameters with defaults
    mapping_type = str(params.get("mapping_type", MAPPING_PLANAR))
    position = params.get("position", (0.0, 0.0, 0.0))
    rotation = params.get("rotation", (0.0, 0.0, 0.0))
    size = params.get("size", (1.0, 1.0, 1.0))

    # UV processing parameters
    u_tile = float(params.get("u_tile", 1.0))  # type: ignore[arg-type]
    v_tile = float(params.get("v_tile", 1.0))  # type: ignore[arg-type]
    u_offset = float(params.get("u_offset", 0.0))  # type: ignore[arg-type]
    v_offset = float(params.get("v_offset", 0.0))  # type: ignore[arg-type]
    uv_rot = float(params.get("uv_rotation", 0.0))  # type: ignore[arg-type]
    u_flip = bool(params.get("u_flip", False))
    v_flip = bool(params.get("v_flip", False))

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

    # Transform position, rotation, size to world space (using obj_matrix set above)
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
        vertices = generate_plane_vertices(
            (world_position.x, world_position.y, world_position.z),
            combined_rotation,
            world_size,
        )
    elif mapping_type == MAPPING_BOX:
        vertices = generate_box_vertices(
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
                vertices = generate_cylinder_capped_normal_vertices(
                    combined_rotation, world_size
                )
            else:
                vertices = generate_cylinder_normal_vertices(
                    combined_rotation, world_size
                )
            # Offset all vertices to object origin
            vertices = [
                (v[0] + obj_origin.x, v[1] + obj_origin.y, v[2] + obj_origin.z)
                for v in vertices
            ]
        # Position-based mapping
        elif cap:
            vertices = generate_cylinder_capped_vertices(
                (world_position.x, world_position.y, world_position.z),
                combined_rotation,
                world_size,
            )
        else:
            vertices = generate_cylinder_vertices(
                (world_position.x, world_position.y, world_position.z),
                combined_rotation,
                world_size,
            )
    elif mapping_type == MAPPING_SPHERICAL:
        if normal_based:
            # Normal-based mapping: position is irrelevant, but scale affects normals
            # Center at object origin to make it clear these don't depend on position
            obj_origin = obj_matrix @ Vector((0.0, 0.0, 0.0))
            vertices = generate_sphere_normal_vertices(combined_rotation, world_size)
            # Offset all vertices to object origin
            vertices = [
                (v[0] + obj_origin.x, v[1] + obj_origin.y, v[2] + obj_origin.z)
                for v in vertices
            ]
        else:
            vertices = generate_sphere_vertices(
                (world_position.x, world_position.y, world_position.z),
                combined_rotation,
                world_size,
            )
    elif mapping_type == MAPPING_SHRINK_WRAP:
        if normal_based:
            # Normal-based mapping: position is irrelevant, but scale affects normals
            # Center at object origin to make it clear these don't depend on position
            obj_origin = obj_matrix @ Vector((0.0, 0.0, 0.0))
            vertices = generate_shrink_wrap_normal_vertices(
                combined_rotation, world_size
            )
            # Offset all vertices to object origin
            vertices = [
                (v[0] + obj_origin.x, v[1] + obj_origin.y, v[2] + obj_origin.z)
                for v in vertices
            ]
        else:
            # Shrink wrap uses azimuthal grid overlay to show single-pole projection
            vertices = generate_shrink_wrap_vertices(
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

    # Draw UV direction indicators (yellow lines showing U and V axes)
    # For normal-based mappings, use the normal mapping type and center at object origin
    if normal_based:
        # Determine the actual normal-based mapping type
        if mapping_type == MAPPING_CYLINDRICAL:
            effective_mapping_type = MAPPING_CYLINDRICAL_NORMAL
        elif mapping_type == MAPPING_SPHERICAL:
            effective_mapping_type = MAPPING_SPHERICAL_NORMAL
        elif mapping_type == MAPPING_SHRINK_WRAP:
            effective_mapping_type = MAPPING_SHRINK_WRAP_NORMAL
        else:
            effective_mapping_type = mapping_type

        # Use object origin for position since normal-based mappings don't use position
        obj_origin = obj_matrix @ Vector((0.0, 0.0, 0.0))
        direction_position = (obj_origin.x, obj_origin.y, obj_origin.z)
        # Normal-based overlays use half scale (same as shape wireframe)
        direction_size = (world_size[0] * 0.5, world_size[1] * 0.5, world_size[2] * 0.5)
    else:
        effective_mapping_type = mapping_type
        direction_position = (world_position.x, world_position.y, world_position.z)
        direction_size = world_size

    (
        u_dir_vertices,
        v_dir_vertices,
        u_proj_vertices,
        v_proj_vertices,
        u_labels,
        v_labels,
    ) = generate_uv_direction_vertices(
        effective_mapping_type,
        direction_position,
        combined_rotation,
        direction_size,
        u_tile,
        v_tile,
        u_offset,
        v_offset,
        uv_rot,
        u_flip,
        v_flip,
        cap,
    )

    # Draw U direction line (yellow)
    if u_dir_vertices:
        u_batch = batch_for_shader(shader, "LINES", {"pos": u_dir_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH)
        shader.uniform_float("color", OVERLAY_U_DIRECTION_COLOR)
        u_batch.draw(shader)

    # Draw V direction line (yellow-green)
    if v_dir_vertices:
        v_batch = batch_for_shader(shader, "LINES", {"pos": v_dir_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH)
        shader.uniform_float("color", OVERLAY_V_DIRECTION_COLOR)
        v_batch.draw(shader)

    # Draw projected U line (dashed, same color as U but thinner)
    if u_proj_vertices:
        u_proj_batch = batch_for_shader(shader, "LINES", {"pos": u_proj_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH * 0.7)
        shader.uniform_float("color", OVERLAY_U_DIRECTION_COLOR)
        u_proj_batch.draw(shader)

    # Draw projected V line (dashed, same color as V but thinner)
    if v_proj_vertices:
        v_proj_batch = batch_for_shader(shader, "LINES", {"pos": v_proj_vertices})
        shader.uniform_float("lineWidth", OVERLAY_UV_DIRECTION_LINE_WIDTH * 0.7)
        shader.uniform_float("color", OVERLAY_V_DIRECTION_COLOR)
        v_proj_batch.draw(shader)

    # Cache label positions for the text drawing handler
    _cached_u_labels = u_labels
    _cached_v_labels = v_labels

    # Restore state
    gpu.state.blend_set("NONE")
    gpu.state.depth_mask_set(True)


def _draw_text_overlay() -> None:
    """Draw U and V text labels in screen space (POST_PIXEL handler)."""
    context = bpy.context

    # Check if we're in the 3D viewport
    if context.area is None or context.area.type != "VIEW_3D":
        return

    # Get region data for 3D to 2D conversion
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    # Check if there are any labels to draw
    if not _cached_u_labels and not _cached_v_labels:
        return

    # Set up blf font
    font_id = 0
    blf.size(font_id, OVERLAY_LABEL_FONT_SIZE)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.7)  # Dark shadow for readability
    blf.shadow_offset(font_id, 1, -1)

    # Draw U labels
    for pos_3d in _cached_u_labels:
        pos_2d = location_3d_to_region_2d(region, rv3d, Vector(pos_3d))
        if pos_2d is not None:
            blf.position(
                font_id,
                pos_2d.x + OVERLAY_LABEL_OFFSET_X,
                pos_2d.y + OVERLAY_LABEL_OFFSET_Y,
                0,
            )
            blf.color(font_id, *OVERLAY_U_DIRECTION_COLOR)
            blf.draw(font_id, "U")

    # Draw V labels
    for pos_3d in _cached_v_labels:
        pos_2d = location_3d_to_region_2d(region, rv3d, Vector(pos_3d))
        if pos_2d is not None:
            blf.position(
                font_id,
                pos_2d.x + OVERLAY_LABEL_OFFSET_X,
                pos_2d.y + OVERLAY_LABEL_OFFSET_Y,
                0,
            )
            blf.color(font_id, *OVERLAY_V_DIRECTION_COLOR)
            blf.draw(font_id, "V")

    blf.disable(font_id, blf.SHADOW)


def register_draw_handler() -> None:
    """Register the overlay draw handlers."""
    global _draw_handler, _text_draw_handler  # noqa: PLW0603
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay, (), "WINDOW", "POST_VIEW"
        )
    if _text_draw_handler is None:
        _text_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_text_overlay, (), "WINDOW", "POST_PIXEL"
        )


def unregister_draw_handler() -> None:
    """Unregister the overlay draw handlers."""
    global _draw_handler, _text_draw_handler  # noqa: PLW0603
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    if _text_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_text_draw_handler, "WINDOW")
        _text_draw_handler = None
