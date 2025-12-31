"""
Nodes module for UV Map addon.

Procedurally generates geometry node groups for UV mapping.
Supports Planar, Cylindrical, Spherical, and Box (tri-planar) mapping.
"""

from __future__ import annotations

import math

import bpy

from .constants import (
    DEFAULT_POSITION,
    DEFAULT_SIZE,
    DEFAULT_TILE,
    GIZMO_TYPES,
    MAPPING_TYPES,
    SOCKET_CAP,
    SOCKET_GEOMETRY,
    SOCKET_MAPPING_TYPE,
    SOCKET_NORMAL_BASED,
    SOCKET_POSITION,
    SOCKET_ROTATION,
    SOCKET_SELECTION,
    SOCKET_SHOW_GIZMO,
    SOCKET_SIZE,
    SOCKET_SMOOTH_NORMALS,
    SOCKET_U_FLIP,
    SOCKET_U_OFFSET,
    SOCKET_U_TILE,
    SOCKET_UV_MAP,
    SOCKET_UV_ROTATION,
    SOCKET_V_FLIP,
    SOCKET_V_OFFSET,
    SOCKET_V_TILE,
    UV_MAP_NODE_GROUP_PREFIX,
    UV_MAP_NODE_GROUP_TAG,
)
from .shared.node_group_compare import node_groups_match
from .shared.node_layout import layout_nodes_pcb_style

# Sub-group suffixes (used for cleanup during regeneration)
_SUB_GROUP_SUFFIXES = [
    " - Planar",
    " - Cylindrical",
    " - Cylindrical Capped",
    " - Spherical",
    " - Shrink Wrap",
    " - Box",
    " - Cylindrical Normal",
    " - Cylindrical Capped Normal",
    " - Spherical Normal",
    " - Shrink Wrap Normal",
]

# Flag to force creation of new sub-groups (used during comparison)
_force_new_subgroups = False


def get_or_create_uv_map_node_group() -> bpy.types.NodeTree:
    """Get an existing UV Map node group or create a new one.

    First checks if a structurally matching "UV Map" node group already exists.
    If found, returns that. Otherwise creates a new one.

    This prevents accumulation of duplicate node groups (UV Map.001, .002, etc.)
    when the existing one hasn't been modified by the user.

    Returns:
        Either an existing matching node group, or a newly created one.
    """
    global _force_new_subgroups

    base_name = UV_MAP_NODE_GROUP_PREFIX

    # Check if "UV Map" already exists BEFORE creating a reference
    existing: bpy.types.NodeTree | None = None
    if base_name in bpy.data.node_groups:
        existing = bpy.data.node_groups[base_name]

    # Create a fresh node group for comparison
    # Force new sub-groups to get accurate comparison
    _force_new_subgroups = True
    try:
        reference_group = create_uv_map_node_group()
    finally:
        _force_new_subgroups = False

    # If an existing group was found, check if it matches structurally
    if existing is not None:
        if node_groups_match(existing, reference_group):
            # Remove the reference group and its sub-groups, use existing
            _cleanup_reference_groups(reference_group)
            return existing

    # No match found - clean up the temporary sub-groups and create permanent ones
    # First clean up the force-created groups
    _cleanup_reference_groups(reference_group)
    # Now create the actual group with reusable sub-groups
    return create_uv_map_node_group()


def _create_main_interface(node_tree: bpy.types.NodeTree) -> None:
    """Create the main interface sockets for the UV Map node group."""
    interface = node_tree.interface
    assert interface is not None

    # Output socket
    interface.new_socket(
        name=SOCKET_GEOMETRY,
        socket_type="NodeSocketGeometry",
        in_out="OUTPUT",
    )

    # Input sockets
    interface.new_socket(
        name=SOCKET_GEOMETRY,
        socket_type="NodeSocketGeometry",
        in_out="INPUT",
    )

    # Selection (optional - defaults to all geometry when not connected)
    selection_socket = interface.new_socket(
        name=SOCKET_SELECTION,
        socket_type="NodeSocketBool",
        in_out="INPUT",
    )
    selection_socket.default_value = True  # type: ignore[attr-defined]
    selection_socket.hide_value = True  # type: ignore[attr-defined]
    selection_socket.hide_in_modifier = True  # type: ignore[attr-defined]

    # UV Map attribute name (first for easy access)
    uv_map_socket = interface.new_socket(
        name=SOCKET_UV_MAP,
        socket_type="NodeSocketString",
        in_out="INPUT",
    )
    uv_map_socket.default_value = "UVMap"  # type: ignore[attr-defined]

    # Mapping type menu
    # Note: The default_value for NodeSocketMenu must be set after the Menu Switch
    # node is created and its enum items are defined. See _set_mapping_type_default().
    interface.new_socket(
        name=SOCKET_MAPPING_TYPE,
        socket_type="NodeSocketMenu",
        in_out="INPUT",
    )

    # Cap option (for cylindrical mapping)
    # When enabled, uses planar caps at top and bottom of cylinder
    interface.new_socket(
        name=SOCKET_CAP,
        socket_type="NodeSocketBool",
        in_out="INPUT",
    )

    # Normal-based option (for cylindrical, spherical, and shrink wrap mappings)
    # When enabled, uses normal direction instead of position for UV mapping
    interface.new_socket(
        name=SOCKET_NORMAL_BASED,
        socket_type="NodeSocketBool",
        in_out="INPUT",
    )

    # Smooth Normals option (for normal-based mappings)
    # When enabled, computes smooth vertex normals even on flat-shaded geometry
    interface.new_socket(
        name=SOCKET_SMOOTH_NORMALS,
        socket_type="NodeSocketBool",
        in_out="INPUT",
    )

    # Gizmo selection menu
    # Note: The default_value for NodeSocketMenu must be set after the Menu Switch
    # node is created and its enum items are defined. See _set_gizmo_default().
    interface.new_socket(
        name=SOCKET_SHOW_GIZMO,
        socket_type="NodeSocketMenu",
        in_out="INPUT",
    )

    # Transform inputs
    pos_socket = interface.new_socket(
        name=SOCKET_POSITION,
        socket_type="NodeSocketVector",
        in_out="INPUT",
    )
    pos_socket.default_value = DEFAULT_POSITION  # type: ignore[attr-defined]
    pos_socket.subtype = "TRANSLATION"  # type: ignore[attr-defined]

    _rot_socket = interface.new_socket(
        name=SOCKET_ROTATION,
        socket_type="NodeSocketRotation",
        in_out="INPUT",
    )
    # Rotation defaults to identity (0, 0, 0)

    size_socket = interface.new_socket(
        name=SOCKET_SIZE,
        socket_type="NodeSocketVector",
        in_out="INPUT",
    )
    size_socket.default_value = DEFAULT_SIZE  # type: ignore[attr-defined]
    size_socket.subtype = "XYZ"  # type: ignore[attr-defined]
    size_socket.min_value = 0.001  # type: ignore[attr-defined]

    # Tiling inputs
    u_tile_socket = interface.new_socket(
        name=SOCKET_U_TILE,
        socket_type="NodeSocketFloat",
        in_out="INPUT",
    )
    u_tile_socket.default_value = DEFAULT_TILE[0]  # type: ignore[attr-defined]
    u_tile_socket.min_value = 0.001  # type: ignore[attr-defined]

    v_tile_socket = interface.new_socket(
        name=SOCKET_V_TILE,
        socket_type="NodeSocketFloat",
        in_out="INPUT",
    )
    v_tile_socket.default_value = DEFAULT_TILE[1]  # type: ignore[attr-defined]
    v_tile_socket.min_value = 0.001  # type: ignore[attr-defined]

    # Offset inputs
    interface.new_socket(
        name=SOCKET_U_OFFSET,
        socket_type="NodeSocketFloat",
        in_out="INPUT",
    )

    interface.new_socket(
        name=SOCKET_V_OFFSET,
        socket_type="NodeSocketFloat",
        in_out="INPUT",
    )

    # UV Rotation input (in radians, applied after tile)
    uv_rotation_socket = interface.new_socket(
        name=SOCKET_UV_ROTATION,
        socket_type="NodeSocketFloat",
        in_out="INPUT",
    )
    uv_rotation_socket.subtype = "ANGLE"  # type: ignore[attr-defined]

    # Flip inputs (at the end, as boolean toggles)
    interface.new_socket(
        name=SOCKET_U_FLIP,
        socket_type="NodeSocketBool",
        in_out="INPUT",
    )

    interface.new_socket(
        name=SOCKET_V_FLIP,
        socket_type="NodeSocketBool",
        in_out="INPUT",
    )


def _set_mapping_type_default(node_tree: bpy.types.NodeTree) -> None:
    """Set the default value for the Mapping Type interface socket.

    This must be called after the Menu Switch node is created and its enum items
    are defined, as the interface socket needs to reference a valid enum identifier.
    """
    interface = node_tree.interface
    assert interface is not None

    # Find the Mapping Type socket in the interface
    for item in interface.items_tree:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "in_out", None) != "INPUT":
            continue
        if getattr(item, "name", None) == SOCKET_MAPPING_TYPE:
            # Set default to first mapping type (Planar)
            # Note: The enum uses display names, not identifiers
            item.default_value = MAPPING_TYPES[0][1]  # type: ignore[attr-defined]
            break


def _set_gizmo_default(node_tree: bpy.types.NodeTree) -> None:
    """Set the default value for the Gizmo interface socket.

    This must be called after the Gizmo Menu Switch node is created and its enum items
    are defined, as the interface socket needs to reference a valid enum identifier.
    """
    interface = node_tree.interface
    assert interface is not None

    # Find the Gizmo socket in the interface
    for item in interface.items_tree:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "in_out", None) != "INPUT":
            continue
        if getattr(item, "name", None) == SOCKET_SHOW_GIZMO:
            # Set default to first gizmo type (None)
            # Note: The enum uses display names, not identifiers
            item.default_value = GIZMO_TYPES[0][1]  # type: ignore[attr-defined]
            break


def _create_planar_mapping_group(
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for planar UV mapping."""
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Planar"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    # Create interface
    interface = group.interface
    assert interface is not None

    # Inputs
    interface.new_socket(
        name="Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")

    # Output
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    # Create nodes
    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # Planar projection: use X and Y components of position as U and V
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Position"

    # Scale by 0.5 for proper UV range
    scale_x = group.nodes.new("ShaderNodeMath")
    scale_x.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_x.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_x.label = "Scale X (0.5x)"

    scale_y = group.nodes.new("ShaderNodeMath")
    scale_y.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_y.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_y.label = "Scale Y (0.5x)"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # Connect
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])
    group.links.new(separate_xyz.outputs["X"], scale_x.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], scale_y.inputs[0])
    group.links.new(scale_x.outputs["Value"], combine_xyz.inputs["X"])
    group.links.new(scale_y.outputs["Value"], combine_xyz.inputs["Y"])
    # Z stays 0 for UV
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_cylindrical_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for cylindrical UV mapping with seam and axis correction.

    Includes pole-like blending near the cylinder axis (r → 0) to prevent
    UV discontinuities where all angles converge to a single point.
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Cylindrical"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(
        name="Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(
        name="Face Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # Cylindrical: U = atan2(x, y) / (2*pi) + 0.5, V = z
    # With seam correction using face center position
    # And axis blending (like spherical poles) when r is small

    # === Corner position processing ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Position"

    # Corner radius r = sqrt(x² + y²) for axis blending
    corner_x_sq = group.nodes.new("ShaderNodeMath")
    corner_x_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_x_sq.label = "X²"

    corner_y_sq = group.nodes.new("ShaderNodeMath")
    corner_y_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_y_sq.label = "Y²"

    corner_r_sq = group.nodes.new("ShaderNodeMath")
    corner_r_sq.operation = "ADD"  # type: ignore[attr-defined]
    corner_r_sq.label = "X² + Y²"

    corner_r = group.nodes.new("ShaderNodeMath")
    corner_r.operation = "SQRT"  # type: ignore[attr-defined]
    corner_r.label = "Corner R"

    # Corner U = atan2(x, y) / (2*pi)
    # No offset so that U=0 is at +Y direction, matching overlay
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(X, Y)"

    divide_node = group.nodes.new("ShaderNodeMath")
    divide_node.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_node.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_node.label = "/ 2π (Corner U)"

    # === Face center processing ===
    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Position"

    # Face U
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2 Face"

    divide_face = group.nodes.new("ShaderNodeMath")
    divide_face.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_face.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_face.label = "/ 2π (Face U)"

    # === Axis blending (like pole blending in spherical) ===
    # Blend towards face U when corner is near the axis
    axis_threshold = group.nodes.new("ShaderNodeValue")
    axis_threshold.outputs[0].default_value = 0.1  # type: ignore[index]
    axis_threshold.label = "Axis Threshold"

    axis_blend_raw = group.nodes.new("ShaderNodeMath")
    axis_blend_raw.operation = "DIVIDE"  # type: ignore[attr-defined]
    axis_blend_raw.label = "Corner R / Threshold"

    axis_blend = group.nodes.new("ShaderNodeClamp")
    axis_blend.inputs["Min"].default_value = 0.0  # type: ignore[index]
    axis_blend.inputs["Max"].default_value = 1.0  # type: ignore[index]
    axis_blend.label = "Axis Blend"

    # === Seam correction (applied BEFORE axis blending) ===
    delta_u = group.nodes.new("ShaderNodeMath")
    delta_u.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_u.label = "U_corner - U_face"

    delta_gt_half = group.nodes.new("ShaderNodeMath")
    delta_gt_half.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_half.inputs[1].default_value = 0.5  # type: ignore[index]
    delta_gt_half.label = "delta > 0.5"

    delta_lt_neg_half = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_half.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_half.inputs[1].default_value = -0.5  # type: ignore[index]
    delta_lt_neg_half.label = "delta < -0.5"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -1.0  # type: ignore[index]
    neg_offset.label = "-1 * gt"

    total_offset = group.nodes.new("ShaderNodeMath")
    total_offset.operation = "ADD"  # type: ignore[attr-defined]
    total_offset.label = "Offset"

    corrected_corner_u = group.nodes.new("ShaderNodeMath")
    corrected_corner_u.operation = "ADD"  # type: ignore[attr-defined]
    corrected_corner_u.label = "Corrected Corner U"

    # === Blend between face U and corrected corner U ===
    # blended_u = face_u * (1 - blend) + corrected_corner_u * blend
    one_minus_blend = group.nodes.new("ShaderNodeMath")
    one_minus_blend.operation = "SUBTRACT"  # type: ignore[attr-defined]
    one_minus_blend.inputs[0].default_value = 1.0  # type: ignore[index]
    one_minus_blend.label = "1 - Blend"

    face_u_weighted = group.nodes.new("ShaderNodeMath")
    face_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_u_weighted.label = "Face U * (1-blend)"

    corner_u_weighted = group.nodes.new("ShaderNodeMath")
    corner_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_u_weighted.label = "Corrected Corner U * blend"

    blended_u = group.nodes.new("ShaderNodeMath")
    blended_u.operation = "ADD"  # type: ignore[attr-defined]
    blended_u.label = "Final U"

    # Internal scale factor for U (-4x to get sensible UV range with correct orientation)
    scale_u = group.nodes.new("ShaderNodeMath")
    scale_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_u.inputs[1].default_value = -4.0  # type: ignore[index]
    scale_u.label = "Scale U (-4x)"

    # Internal scale factor for V (0.75x for better aspect ratio)
    scale_v = group.nodes.new("ShaderNodeMath")
    scale_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_v.inputs[1].default_value = 0.75  # type: ignore[index]
    scale_v.label = "Scale V (0.75x)"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # === Connect corner position processing ===
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])

    # Corner radius
    group.links.new(separate_xyz.outputs["X"], corner_x_sq.inputs[0])
    group.links.new(separate_xyz.outputs["X"], corner_x_sq.inputs[1])
    group.links.new(separate_xyz.outputs["Y"], corner_y_sq.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], corner_y_sq.inputs[1])
    group.links.new(corner_x_sq.outputs["Value"], corner_r_sq.inputs[0])
    group.links.new(corner_y_sq.outputs["Value"], corner_r_sq.inputs[1])
    group.links.new(corner_r_sq.outputs["Value"], corner_r.inputs[0])

    # Corner U
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_node.inputs[0])

    # === Connect face processing ===
    group.links.new(input_node.outputs["Face Position"], separate_face.inputs["Vector"])
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[1])
    group.links.new(atan2_face.outputs["Value"], divide_face.inputs[0])

    # === Connect axis blending ===
    group.links.new(corner_r.outputs["Value"], axis_blend_raw.inputs[0])
    group.links.new(axis_threshold.outputs[0], axis_blend_raw.inputs[1])
    group.links.new(axis_blend_raw.outputs["Value"], axis_blend.inputs["Value"])

    # === Connect seam correction (BEFORE blending) ===
    # Now using divide_node directly (no +0.5 offset)
    group.links.new(divide_node.outputs["Value"], delta_u.inputs[0])
    group.links.new(divide_face.outputs["Value"], delta_u.inputs[1])

    group.links.new(delta_u.outputs["Value"], delta_gt_half.inputs[0])
    group.links.new(delta_u.outputs["Value"], delta_lt_neg_half.inputs[0])

    group.links.new(delta_gt_half.outputs["Value"], neg_offset.inputs[0])
    group.links.new(neg_offset.outputs["Value"], total_offset.inputs[0])
    group.links.new(delta_lt_neg_half.outputs["Value"], total_offset.inputs[1])

    group.links.new(divide_node.outputs["Value"], corrected_corner_u.inputs[0])
    group.links.new(total_offset.outputs["Value"], corrected_corner_u.inputs[1])

    # === Connect axis blending (AFTER seam correction) ===
    group.links.new(axis_blend.outputs["Result"], one_minus_blend.inputs[1])

    group.links.new(divide_face.outputs["Value"], face_u_weighted.inputs[0])
    group.links.new(one_minus_blend.outputs["Value"], face_u_weighted.inputs[1])

    group.links.new(corrected_corner_u.outputs["Value"], corner_u_weighted.inputs[0])
    group.links.new(axis_blend.outputs["Result"], corner_u_weighted.inputs[1])

    group.links.new(face_u_weighted.outputs["Value"], blended_u.inputs[0])
    group.links.new(corner_u_weighted.outputs["Value"], blended_u.inputs[1])

    # === Apply internal U scale ===
    group.links.new(blended_u.outputs["Value"], scale_u.inputs[0])

    # === Apply internal V scale ===
    group.links.new(separate_xyz.outputs["Z"], scale_v.inputs[0])

    # === Final UV output ===
    group.links.new(scale_u.outputs["Value"], combine_xyz.inputs["X"])  # U (scaled)
    group.links.new(scale_v.outputs["Value"], combine_xyz.inputs["Y"])  # V (scaled)
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_cylindrical_capped_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for cylindrical UV mapping with planar caps.

    Uses cylindrical mapping for sides and planar mapping for caps.
    Cap detection is based on face position: faces near Z = ±1 are caps.
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Cylindrical Capped"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(
        name="Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(
        name="Face Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # === Separate inputs ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Position"

    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Position"

    # === Position-based cap detection ===
    # A face is a cap when its center is INSIDE the cylinder (radial distance < 1)
    # Side faces are ON the cylinder surface (radial distance ≈ 1)
    # is_cap = sqrt(face_x² + face_y²) < threshold
    face_x_sq = group.nodes.new("ShaderNodeMath")
    face_x_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_x_sq.label = "Face X²"

    face_y_sq = group.nodes.new("ShaderNodeMath")
    face_y_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_y_sq.label = "Face Y²"

    face_xy_sum = group.nodes.new("ShaderNodeMath")
    face_xy_sum.operation = "ADD"  # type: ignore[attr-defined]
    face_xy_sum.label = "X² + Y²"

    face_radius = group.nodes.new("ShaderNodeMath")
    face_radius.operation = "SQRT"  # type: ignore[attr-defined]
    face_radius.label = "Radial Distance"

    is_cap = group.nodes.new("ShaderNodeMath")
    is_cap.operation = "LESS_THAN"  # type: ignore[attr-defined]
    is_cap.inputs[1].default_value = 0.99  # type: ignore[index]
    is_cap.label = "Is Cap (r < 0.99)"

    # === Cylindrical UV calculation (for sides) ===
    # No +0.5 offset so U=0 is at +Y direction, matching overlay
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(X, Y)"

    divide_2pi = group.nodes.new("ShaderNodeMath")
    divide_2pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi.label = "/ 2π (Corner U)"

    # === Face center U calculation (for seam correction) ===
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2 Face"

    divide_2pi_face = group.nodes.new("ShaderNodeMath")
    divide_2pi_face.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi_face.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi_face.label = "/ 2π (Face U)"

    # === Seam correction ===
    delta_u = group.nodes.new("ShaderNodeMath")
    delta_u.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_u.label = "U_corner - U_face"

    delta_gt_half = group.nodes.new("ShaderNodeMath")
    delta_gt_half.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_half.inputs[1].default_value = 0.5  # type: ignore[index]
    delta_gt_half.label = "delta > 0.5"

    delta_lt_neg_half = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_half.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_half.inputs[1].default_value = -0.5  # type: ignore[index]
    delta_lt_neg_half.label = "delta < -0.5"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -1.0  # type: ignore[index]
    neg_offset.label = "-1 * gt"

    total_offset = group.nodes.new("ShaderNodeMath")
    total_offset.operation = "ADD"  # type: ignore[attr-defined]
    total_offset.label = "Offset"

    corrected_cyl_u = group.nodes.new("ShaderNodeMath")
    corrected_cyl_u.operation = "ADD"  # type: ignore[attr-defined]
    corrected_cyl_u.label = "Corrected Cyl U"

    # Internal scale factor for cylindrical U (-4x to get sensible UV range with correct orientation)
    scale_cyl_u = group.nodes.new("ShaderNodeMath")
    scale_cyl_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cyl_u.inputs[1].default_value = -4.0  # type: ignore[index]
    scale_cyl_u.label = "Scale Cyl U (-4x)"

    # Internal scale factor for cylindrical V (0.75x for better aspect ratio)
    scale_cyl_v = group.nodes.new("ShaderNodeMath")
    scale_cyl_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cyl_v.inputs[1].default_value = 0.75  # type: ignore[index]
    scale_cyl_v.label = "Scale Cyl V (0.75x)"

    # Cylindrical UV (U = corrected angle, V = Z * 0.75)
    combine_cyl_uv = group.nodes.new("ShaderNodeCombineXYZ")
    combine_cyl_uv.label = "Cylindrical UV"

    # === Planar UV calculation (for caps) ===
    # Scale X, Y by 0.5 for proper UV range
    scale_cap_x = group.nodes.new("ShaderNodeMath")
    scale_cap_x.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cap_x.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_cap_x.label = "Scale Cap X (0.5x)"

    scale_cap_y = group.nodes.new("ShaderNodeMath")
    scale_cap_y.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cap_y.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_cap_y.label = "Scale Cap Y (0.5x)"

    combine_planar_uv = group.nodes.new("ShaderNodeCombineXYZ")
    combine_planar_uv.label = "Planar UV"

    # === Mix between cylindrical and planar based on is_cap ===
    uv_switch = group.nodes.new("GeometryNodeSwitch")
    uv_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    uv_switch.label = "Cap/Side Switch"

    # === Connect position ===
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])

    # === Connect face position and cap detection ===
    group.links.new(input_node.outputs["Face Position"], separate_face.inputs["Vector"])
    group.links.new(separate_face.outputs["X"], face_x_sq.inputs[0])
    group.links.new(separate_face.outputs["X"], face_x_sq.inputs[1])
    group.links.new(separate_face.outputs["Y"], face_y_sq.inputs[0])
    group.links.new(separate_face.outputs["Y"], face_y_sq.inputs[1])
    group.links.new(face_x_sq.outputs["Value"], face_xy_sum.inputs[0])
    group.links.new(face_y_sq.outputs["Value"], face_xy_sum.inputs[1])
    group.links.new(face_xy_sum.outputs["Value"], face_radius.inputs[0])
    group.links.new(face_radius.outputs["Value"], is_cap.inputs[0])

    # === Connect cylindrical U calculation ===
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_2pi.inputs[0])

    # === Connect face U calculation ===
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[1])
    group.links.new(atan2_face.outputs["Value"], divide_2pi_face.inputs[0])

    # === Connect seam correction ===
    # Now using divide_2pi directly (no +0.5 offset)
    group.links.new(divide_2pi.outputs["Value"], delta_u.inputs[0])
    group.links.new(divide_2pi_face.outputs["Value"], delta_u.inputs[1])
    group.links.new(delta_u.outputs["Value"], delta_gt_half.inputs[0])
    group.links.new(delta_u.outputs["Value"], delta_lt_neg_half.inputs[0])
    group.links.new(delta_gt_half.outputs["Value"], neg_offset.inputs[0])
    group.links.new(neg_offset.outputs["Value"], total_offset.inputs[0])
    group.links.new(delta_lt_neg_half.outputs["Value"], total_offset.inputs[1])
    group.links.new(divide_2pi.outputs["Value"], corrected_cyl_u.inputs[0])
    group.links.new(total_offset.outputs["Value"], corrected_cyl_u.inputs[1])

    # === Apply internal U scale to cylindrical ===
    group.links.new(corrected_cyl_u.outputs["Value"], scale_cyl_u.inputs[0])

    # === Apply internal V scale to cylindrical ===
    group.links.new(separate_xyz.outputs["Z"], scale_cyl_v.inputs[0])

    # === Combine cylindrical UV ===
    group.links.new(scale_cyl_u.outputs["Value"], combine_cyl_uv.inputs["X"])
    group.links.new(scale_cyl_v.outputs["Value"], combine_cyl_uv.inputs["Y"])

    # === Combine planar UV (X * 0.5, Y * 0.5) ===
    group.links.new(separate_xyz.outputs["X"], scale_cap_x.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], scale_cap_y.inputs[0])
    group.links.new(scale_cap_x.outputs["Value"], combine_planar_uv.inputs["X"])
    group.links.new(scale_cap_y.outputs["Value"], combine_planar_uv.inputs["Y"])

    # === Switch based on is_cap ===
    group.links.new(is_cap.outputs["Value"], uv_switch.inputs["Switch"])
    group.links.new(combine_cyl_uv.outputs["Vector"], uv_switch.inputs["False"])
    group.links.new(combine_planar_uv.outputs["Vector"], uv_switch.inputs["True"])

    # === Output ===
    group.links.new(uv_switch.outputs["Output"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_spherical_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for spherical UV mapping with seam and pole correction."""
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Spherical"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(
        name="Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(
        name="Face Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # Spherical: convert to spherical coordinates with seam and pole correction
    # U = atan2(x, y) / (2*pi) + 0.5
    # V = acos(z / length) / pi

    # === Corner position processing ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Position"

    length_node = group.nodes.new("ShaderNodeVectorMath")
    length_node.operation = "LENGTH"  # type: ignore[attr-defined]
    length_node.label = "Length"

    # Corner horizontal radius r = sqrt(x² + y²) for pole detection
    corner_x_sq = group.nodes.new("ShaderNodeMath")
    corner_x_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_x_sq.label = "X²"

    corner_y_sq = group.nodes.new("ShaderNodeMath")
    corner_y_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_y_sq.label = "Y²"

    corner_r_sq = group.nodes.new("ShaderNodeMath")
    corner_r_sq.operation = "ADD"  # type: ignore[attr-defined]
    corner_r_sq.label = "X² + Y²"

    corner_r = group.nodes.new("ShaderNodeMath")
    corner_r.operation = "SQRT"  # type: ignore[attr-defined]
    corner_r.label = "Corner R"

    # Corner U calculation
    # No +0.5 offset so U=0 is at +Y direction, matching overlay
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(X, Y)"

    divide_2pi = group.nodes.new("ShaderNodeMath")
    divide_2pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi.label = "/ 2π (Corner U)"

    # Corner V calculation
    divide_z_len = group.nodes.new("ShaderNodeMath")
    divide_z_len.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_z_len.label = "Z / Length"

    clamp_node = group.nodes.new("ShaderNodeClamp")
    clamp_node.inputs["Min"].default_value = -1.0  # type: ignore[index]
    clamp_node.inputs["Max"].default_value = 1.0  # type: ignore[index]
    clamp_node.label = "Clamp"

    acos_node = group.nodes.new("ShaderNodeMath")
    acos_node.operation = "ARCCOSINE"  # type: ignore[attr-defined]
    acos_node.label = "acos"

    divide_pi = group.nodes.new("ShaderNodeMath")
    divide_pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_pi.inputs[1].default_value = math.pi  # type: ignore[index]
    divide_pi.label = "/ π"

    # === Face center processing ===
    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Position"

    # Face U calculation (needed for seam correction reference)
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2 Face"

    divide_2pi_face = group.nodes.new("ShaderNodeMath")
    divide_2pi_face.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi_face.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi_face.label = "/ 2π (Face U)"

    # === Pole blending ===
    # Blend factor based ONLY on corner position (not face) for continuity
    # Use smoothstep-like transition: blend = clamp(corner_r / threshold, 0, 1)
    # threshold is a small value where U becomes stable
    # At pole (corner_r ≈ 0): blend ≈ 0 → use face U
    # Away from pole (corner_r > threshold): blend = 1 → use corner U
    pole_threshold = group.nodes.new("ShaderNodeValue")
    pole_threshold.outputs[0].default_value = 0.01  # type: ignore[index]
    pole_threshold.label = "Pole Threshold"

    pole_blend_raw = group.nodes.new("ShaderNodeMath")
    pole_blend_raw.operation = "DIVIDE"  # type: ignore[attr-defined]
    pole_blend_raw.label = "Corner R / Threshold"

    pole_blend = group.nodes.new("ShaderNodeClamp")
    pole_blend.inputs["Min"].default_value = 0.0  # type: ignore[index]
    pole_blend.inputs["Max"].default_value = 1.0  # type: ignore[index]
    pole_blend.label = "Pole Blend"

    # Blend between face U and corner U based on pole proximity
    # blended_u = face_u * (1 - blend) + corner_u_corrected * blend
    # NOTE: We must apply seam correction BEFORE blending, otherwise
    # blending values on opposite sides of the seam produces garbage.
    one_minus_blend = group.nodes.new("ShaderNodeMath")
    one_minus_blend.operation = "SUBTRACT"  # type: ignore[attr-defined]
    one_minus_blend.inputs[0].default_value = 1.0  # type: ignore[index]
    one_minus_blend.label = "1 - Blend"

    face_u_weighted = group.nodes.new("ShaderNodeMath")
    face_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_u_weighted.label = "Face U * (1-blend)"

    corner_u_weighted = group.nodes.new("ShaderNodeMath")
    corner_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_u_weighted.label = "Corrected Corner U * blend"

    blended_u = group.nodes.new("ShaderNodeMath")
    blended_u.operation = "ADD"  # type: ignore[attr-defined]
    blended_u.label = "Final U"

    # Internal scale factors for U and V (to get sensible UV range with correct orientation)
    scale_u = group.nodes.new("ShaderNodeMath")
    scale_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_u.inputs[1].default_value = -4.0  # type: ignore[index]
    scale_u.label = "Scale U (-4x)"

    # Offset V so that equator (acos(0)/π = 0.5) becomes 0 in UV space
    # This anchors UV origin at the equator, matching the overlay behavior
    offset_v = group.nodes.new("ShaderNodeMath")
    offset_v.operation = "SUBTRACT"  # type: ignore[attr-defined]
    offset_v.inputs[1].default_value = 0.5  # type: ignore[index]
    offset_v.label = "V - 0.5 (Equator Offset)"

    scale_v = group.nodes.new("ShaderNodeMath")
    scale_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_v.inputs[1].default_value = -2.0  # type: ignore[index]
    scale_v.label = "Scale V (-2x)"

    # === Seam correction (applied BEFORE pole blending) ===
    # Correct corner U to be on same side of seam as face U
    delta_u = group.nodes.new("ShaderNodeMath")
    delta_u.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_u.label = "U_corner - U_face"

    delta_gt_half = group.nodes.new("ShaderNodeMath")
    delta_gt_half.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_half.inputs[1].default_value = 0.5  # type: ignore[index]
    delta_gt_half.label = "delta > 0.5"

    delta_lt_neg_half = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_half.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_half.inputs[1].default_value = -0.5  # type: ignore[index]
    delta_lt_neg_half.label = "delta < -0.5"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -1.0  # type: ignore[index]
    neg_offset.label = "-1 * gt"

    total_offset = group.nodes.new("ShaderNodeMath")
    total_offset.operation = "ADD"  # type: ignore[attr-defined]
    total_offset.label = "Offset"

    corrected_corner_u = group.nodes.new("ShaderNodeMath")
    corrected_corner_u.operation = "ADD"  # type: ignore[attr-defined]
    corrected_corner_u.label = "Corrected Corner U"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # === Connect corner position processing ===
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])
    group.links.new(input_node.outputs["Position"], length_node.inputs[0])

    # Corner R calculation
    group.links.new(separate_xyz.outputs["X"], corner_x_sq.inputs[0])
    group.links.new(separate_xyz.outputs["X"], corner_x_sq.inputs[1])
    group.links.new(separate_xyz.outputs["Y"], corner_y_sq.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], corner_y_sq.inputs[1])
    group.links.new(corner_x_sq.outputs["Value"], corner_r_sq.inputs[0])
    group.links.new(corner_y_sq.outputs["Value"], corner_r_sq.inputs[1])
    group.links.new(corner_r_sq.outputs["Value"], corner_r.inputs[0])

    # Corner U
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_2pi.inputs[0])

    # Corner V
    group.links.new(separate_xyz.outputs["Z"], divide_z_len.inputs[0])
    group.links.new(length_node.outputs["Value"], divide_z_len.inputs[1])
    group.links.new(divide_z_len.outputs["Value"], clamp_node.inputs["Value"])
    group.links.new(clamp_node.outputs["Result"], acos_node.inputs[0])
    group.links.new(acos_node.outputs["Value"], divide_pi.inputs[0])

    # === Connect face processing ===
    group.links.new(input_node.outputs["Face Position"], separate_face.inputs["Vector"])

    # Face U
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[1])
    group.links.new(atan2_face.outputs["Value"], divide_2pi_face.inputs[0])

    # === Connect pole blending ===
    group.links.new(corner_r.outputs["Value"], pole_blend_raw.inputs[0])
    group.links.new(pole_threshold.outputs[0], pole_blend_raw.inputs[1])
    group.links.new(pole_blend_raw.outputs["Value"], pole_blend.inputs["Value"])

    # === Connect seam correction (BEFORE blending) ===
    # First correct corner U to be on same side of seam as face U
    # Now using divide_2pi directly (no +0.5 offset)
    group.links.new(divide_2pi.outputs["Value"], delta_u.inputs[0])
    group.links.new(divide_2pi_face.outputs["Value"], delta_u.inputs[1])

    group.links.new(delta_u.outputs["Value"], delta_gt_half.inputs[0])
    group.links.new(delta_u.outputs["Value"], delta_lt_neg_half.inputs[0])

    group.links.new(delta_gt_half.outputs["Value"], neg_offset.inputs[0])
    group.links.new(neg_offset.outputs["Value"], total_offset.inputs[0])
    group.links.new(delta_lt_neg_half.outputs["Value"], total_offset.inputs[1])

    group.links.new(divide_2pi.outputs["Value"], corrected_corner_u.inputs[0])
    group.links.new(total_offset.outputs["Value"], corrected_corner_u.inputs[1])

    # === Connect pole blending (AFTER seam correction) ===
    # Now blend the seam-corrected corner U with face U
    group.links.new(pole_blend.outputs["Result"], one_minus_blend.inputs[1])
    group.links.new(divide_2pi_face.outputs["Value"], face_u_weighted.inputs[0])
    group.links.new(one_minus_blend.outputs["Value"], face_u_weighted.inputs[1])
    group.links.new(corrected_corner_u.outputs["Value"], corner_u_weighted.inputs[0])
    group.links.new(pole_blend.outputs["Result"], corner_u_weighted.inputs[1])
    group.links.new(face_u_weighted.outputs["Value"], blended_u.inputs[0])
    group.links.new(corner_u_weighted.outputs["Value"], blended_u.inputs[1])

    # === Apply internal UV scale ===
    group.links.new(blended_u.outputs["Value"], scale_u.inputs[0])
    group.links.new(divide_pi.outputs["Value"], offset_v.inputs[0])
    group.links.new(offset_v.outputs["Value"], scale_v.inputs[0])

    # === Final UV output ===
    group.links.new(
        scale_u.outputs["Value"], combine_xyz.inputs["X"]
    )  # Final U (scaled)
    group.links.new(scale_v.outputs["Value"], combine_xyz.inputs["Y"])  # V (scaled)
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_shrink_wrap_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for shrink wrap (azimuthal equidistant) UV mapping.

    Uses azimuthal equidistant projection - maps entire sphere to a circle
    with a single pole at the center. The +Z axis is at center (0, 0)
    and -Z spreads around the edge of the circular UV region.

    Formula:
      theta = acos(z / length)  # angle from +Z axis
      phi = atan2(y, x)         # azimuthal angle
      r = theta / pi            # radial distance (0 at +Z, 1 at -Z)
      u = r * cos(phi)
      v = r * sin(phi)

    Near the pole (small r), seam correction is disabled because phi is
    numerically unstable there, but it doesn't matter since r→0 makes U,V→0.
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Shrink Wrap"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(
        name="Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(
        name="Face Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # === Corner position processing ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Position"

    length_node = group.nodes.new("ShaderNodeVectorMath")
    length_node.operation = "LENGTH"  # type: ignore[attr-defined]
    length_node.label = "Length"

    # theta = acos(z / length)
    divide_z_len = group.nodes.new("ShaderNodeMath")
    divide_z_len.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_z_len.label = "Z / Length"

    clamp_node = group.nodes.new("ShaderNodeClamp")
    clamp_node.inputs["Min"].default_value = -1.0  # type: ignore[index]
    clamp_node.inputs["Max"].default_value = 1.0  # type: ignore[index]
    clamp_node.label = "Clamp [-1, 1]"

    acos_node = group.nodes.new("ShaderNodeMath")
    acos_node.operation = "ARCCOSINE"  # type: ignore[attr-defined]
    acos_node.label = "acos (theta)"

    # r = theta / pi
    divide_pi = group.nodes.new("ShaderNodeMath")
    divide_pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_pi.inputs[1].default_value = math.pi  # type: ignore[index]
    divide_pi.label = "/ π (r)"

    # phi = atan2(y, x)
    atan2_corner = group.nodes.new("ShaderNodeMath")
    atan2_corner.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_corner.label = "atan2(Y, X) corner"

    # === Face position processing ===
    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Position"

    # Face phi for seam correction reference
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2(Y, X) face"

    # === Seam correction for phi ===
    # Only apply when NOT near the pole (r > threshold)
    # delta = phi_corner - phi_face
    delta_phi = group.nodes.new("ShaderNodeMath")
    delta_phi.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_phi.label = "phi_corner - phi_face"

    delta_gt_pi = group.nodes.new("ShaderNodeMath")
    delta_gt_pi.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_pi.inputs[1].default_value = math.pi  # type: ignore[index]
    delta_gt_pi.label = "delta > π"

    delta_lt_neg_pi = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_pi.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_pi.inputs[1].default_value = -math.pi  # type: ignore[index]
    delta_lt_neg_pi.label = "delta < -π"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -2.0 * math.pi  # type: ignore[index]
    neg_offset.label = "-2π * gt"

    pos_offset = group.nodes.new("ShaderNodeMath")
    pos_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    pos_offset.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    pos_offset.label = "+2π * lt"

    raw_offset = group.nodes.new("ShaderNodeMath")
    raw_offset.operation = "ADD"  # type: ignore[attr-defined]
    raw_offset.label = "Raw Offset"

    # Disable seam correction near pole: multiply offset by step(r - threshold)
    # This is a hard cutoff: r < 0.15 → no correction, r >= 0.15 → full correction
    pole_cutoff = group.nodes.new("ShaderNodeMath")
    pole_cutoff.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    pole_cutoff.inputs[1].default_value = 0.15  # type: ignore[index]
    pole_cutoff.label = "r > 0.15"

    scaled_offset = group.nodes.new("ShaderNodeMath")
    scaled_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scaled_offset.label = "Scaled Offset"

    corrected_phi = group.nodes.new("ShaderNodeMath")
    corrected_phi.operation = "ADD"  # type: ignore[attr-defined]
    corrected_phi.label = "Corrected Phi"

    # === Final UV calculation (corner) ===
    cos_phi = group.nodes.new("ShaderNodeMath")
    cos_phi.operation = "COSINE"  # type: ignore[attr-defined]
    cos_phi.label = "cos(phi)"

    sin_phi = group.nodes.new("ShaderNodeMath")
    sin_phi.operation = "SINE"  # type: ignore[attr-defined]
    sin_phi.label = "sin(phi)"

    # U = r * cos(phi), V = r * sin(phi)
    corner_u = group.nodes.new("ShaderNodeMath")
    corner_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_u.label = "Corner U"

    corner_v = group.nodes.new("ShaderNodeMath")
    corner_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_v.label = "Corner V"

    # === Face UV calculation (for -Z pole blending) ===
    face_length = group.nodes.new("ShaderNodeVectorMath")
    face_length.operation = "LENGTH"  # type: ignore[attr-defined]
    face_length.label = "Face Length"

    face_z_div_len = group.nodes.new("ShaderNodeMath")
    face_z_div_len.operation = "DIVIDE"  # type: ignore[attr-defined]
    face_z_div_len.label = "Face Z / Len"

    face_clamp = group.nodes.new("ShaderNodeClamp")
    face_clamp.inputs["Min"].default_value = -1.0  # type: ignore[index]
    face_clamp.inputs["Max"].default_value = 1.0  # type: ignore[index]
    face_clamp.label = "Clamp Face"

    face_acos = group.nodes.new("ShaderNodeMath")
    face_acos.operation = "ARCCOSINE"  # type: ignore[attr-defined]
    face_acos.label = "Face acos"

    face_r = group.nodes.new("ShaderNodeMath")
    face_r.operation = "DIVIDE"  # type: ignore[attr-defined]
    face_r.inputs[1].default_value = math.pi  # type: ignore[index]
    face_r.label = "Face R"

    face_cos_phi = group.nodes.new("ShaderNodeMath")
    face_cos_phi.operation = "COSINE"  # type: ignore[attr-defined]
    face_cos_phi.label = "cos(face phi)"

    face_sin_phi = group.nodes.new("ShaderNodeMath")
    face_sin_phi.operation = "SINE"  # type: ignore[attr-defined]
    face_sin_phi.label = "sin(face phi)"

    face_u = group.nodes.new("ShaderNodeMath")
    face_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_u.label = "Face U"

    face_v = group.nodes.new("ShaderNodeMath")
    face_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_v.label = "Face V"

    # === -Z pole blending (blend toward face UV when r > 0.85) ===
    # At -Z (r=1), phi is undefined. Blend toward face UV which is more stable.
    pole_blend_threshold = group.nodes.new("ShaderNodeMath")
    pole_blend_threshold.operation = "SUBTRACT"  # type: ignore[attr-defined]
    pole_blend_threshold.inputs[1].default_value = 0.85  # type: ignore[index]
    pole_blend_threshold.label = "r - 0.85"

    pole_blend_scale = group.nodes.new("ShaderNodeMath")
    pole_blend_scale.operation = "MULTIPLY"  # type: ignore[attr-defined]
    pole_blend_scale.inputs[1].default_value = 1.0 / 0.15  # type: ignore[index]
    pole_blend_scale.label = "Scale to 0-1"

    pole_blend = group.nodes.new("ShaderNodeClamp")
    pole_blend.inputs["Min"].default_value = 0.0  # type: ignore[index]
    pole_blend.inputs["Max"].default_value = 1.0  # type: ignore[index]
    pole_blend.label = "-Z Pole Blend"

    # blend=0: use corner UV, blend=1: use face UV
    one_minus_blend = group.nodes.new("ShaderNodeMath")
    one_minus_blend.operation = "SUBTRACT"  # type: ignore[attr-defined]
    one_minus_blend.inputs[0].default_value = 1.0  # type: ignore[index]
    one_minus_blend.label = "1 - blend"

    corner_u_weighted = group.nodes.new("ShaderNodeMath")
    corner_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_u_weighted.label = "Corner U * (1-b)"

    face_u_weighted = group.nodes.new("ShaderNodeMath")
    face_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_u_weighted.label = "Face U * b"

    final_u = group.nodes.new("ShaderNodeMath")
    final_u.operation = "ADD"  # type: ignore[attr-defined]
    final_u.label = "Final U"

    corner_v_weighted = group.nodes.new("ShaderNodeMath")
    corner_v_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_v_weighted.label = "Corner V * (1-b)"

    face_v_weighted = group.nodes.new("ShaderNodeMath")
    face_v_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_v_weighted.label = "Face V * b"

    final_v = group.nodes.new("ShaderNodeMath")
    final_v.operation = "ADD"  # type: ignore[attr-defined]
    final_v.label = "Final V"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # === Connect corner position ===
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])
    group.links.new(input_node.outputs["Position"], length_node.inputs[0])

    # Theta and r
    group.links.new(separate_xyz.outputs["Z"], divide_z_len.inputs[0])
    group.links.new(length_node.outputs["Value"], divide_z_len.inputs[1])
    group.links.new(divide_z_len.outputs["Value"], clamp_node.inputs["Value"])
    group.links.new(clamp_node.outputs["Result"], acos_node.inputs[0])
    group.links.new(acos_node.outputs["Value"], divide_pi.inputs[0])

    # Corner phi
    group.links.new(separate_xyz.outputs["Y"], atan2_corner.inputs[0])
    group.links.new(separate_xyz.outputs["X"], atan2_corner.inputs[1])

    # === Connect face position ===
    group.links.new(input_node.outputs["Face Position"], separate_face.inputs["Vector"])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[1])

    # === Connect seam correction ===
    group.links.new(atan2_corner.outputs["Value"], delta_phi.inputs[0])
    group.links.new(atan2_face.outputs["Value"], delta_phi.inputs[1])

    group.links.new(delta_phi.outputs["Value"], delta_gt_pi.inputs[0])
    group.links.new(delta_phi.outputs["Value"], delta_lt_neg_pi.inputs[0])

    group.links.new(delta_gt_pi.outputs["Value"], neg_offset.inputs[0])
    group.links.new(delta_lt_neg_pi.outputs["Value"], pos_offset.inputs[0])

    group.links.new(neg_offset.outputs["Value"], raw_offset.inputs[0])
    group.links.new(pos_offset.outputs["Value"], raw_offset.inputs[1])

    # Disable seam correction near pole (hard cutoff at r = 0.15)
    group.links.new(divide_pi.outputs["Value"], pole_cutoff.inputs[0])
    group.links.new(raw_offset.outputs["Value"], scaled_offset.inputs[0])
    group.links.new(pole_cutoff.outputs["Value"], scaled_offset.inputs[1])

    group.links.new(atan2_corner.outputs["Value"], corrected_phi.inputs[0])
    group.links.new(scaled_offset.outputs["Value"], corrected_phi.inputs[1])

    # === Connect final UV ===
    group.links.new(corrected_phi.outputs["Value"], cos_phi.inputs[0])
    group.links.new(corrected_phi.outputs["Value"], sin_phi.inputs[0])

    group.links.new(divide_pi.outputs["Value"], corner_u.inputs[0])
    group.links.new(cos_phi.outputs["Value"], corner_u.inputs[1])

    group.links.new(divide_pi.outputs["Value"], corner_v.inputs[0])
    group.links.new(sin_phi.outputs["Value"], corner_v.inputs[1])

    # === Connect face UV calculation ===
    group.links.new(input_node.outputs["Face Position"], face_length.inputs[0])
    group.links.new(separate_face.outputs["Z"], face_z_div_len.inputs[0])
    group.links.new(face_length.outputs["Value"], face_z_div_len.inputs[1])
    group.links.new(face_z_div_len.outputs["Value"], face_clamp.inputs["Value"])
    group.links.new(face_clamp.outputs["Result"], face_acos.inputs[0])
    group.links.new(face_acos.outputs["Value"], face_r.inputs[0])

    group.links.new(atan2_face.outputs["Value"], face_cos_phi.inputs[0])
    group.links.new(atan2_face.outputs["Value"], face_sin_phi.inputs[0])

    group.links.new(face_r.outputs["Value"], face_u.inputs[0])
    group.links.new(face_cos_phi.outputs["Value"], face_u.inputs[1])

    group.links.new(face_r.outputs["Value"], face_v.inputs[0])
    group.links.new(face_sin_phi.outputs["Value"], face_v.inputs[1])

    # === Connect -Z pole blending ===
    # blend = clamp((r - 0.85) / 0.15, 0, 1)
    # At r < 0.85: blend=0 (use corner UV)
    # At r > 1.0: blend=1 (use face UV)
    group.links.new(divide_pi.outputs["Value"], pole_blend_threshold.inputs[0])
    group.links.new(pole_blend_threshold.outputs["Value"], pole_blend_scale.inputs[0])
    group.links.new(pole_blend_scale.outputs["Value"], pole_blend.inputs["Value"])

    group.links.new(pole_blend.outputs["Result"], one_minus_blend.inputs[1])

    # Blend U
    group.links.new(corner_u.outputs["Value"], corner_u_weighted.inputs[0])
    group.links.new(one_minus_blend.outputs["Value"], corner_u_weighted.inputs[1])

    group.links.new(face_u.outputs["Value"], face_u_weighted.inputs[0])
    group.links.new(pole_blend.outputs["Result"], face_u_weighted.inputs[1])

    group.links.new(corner_u_weighted.outputs["Value"], final_u.inputs[0])
    group.links.new(face_u_weighted.outputs["Value"], final_u.inputs[1])

    # Blend V
    group.links.new(corner_v.outputs["Value"], corner_v_weighted.inputs[0])
    group.links.new(one_minus_blend.outputs["Value"], corner_v_weighted.inputs[1])

    group.links.new(face_v.outputs["Value"], face_v_weighted.inputs[0])
    group.links.new(pole_blend.outputs["Result"], face_v_weighted.inputs[1])

    group.links.new(corner_v_weighted.outputs["Value"], final_v.inputs[0])
    group.links.new(face_v_weighted.outputs["Value"], final_v.inputs[1])

    group.links.new(final_u.outputs["Value"], combine_xyz.inputs["X"])
    group.links.new(final_v.outputs["Value"], combine_xyz.inputs["Y"])
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_box_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for box (tri-planar) UV mapping.

    Uses the dominant axis of the normal to determine which two axes
    to use for UV coordinates. Projects onto the plane perpendicular
    to the dominant normal axis.
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Box"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(
        name="Position", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # Triplanar mapping: choose projection based on dominant normal axis
    sep_pos = group.nodes.new("ShaderNodeSeparateXYZ")
    sep_pos.label = "Separate Position"

    sep_norm = group.nodes.new("ShaderNodeSeparateXYZ")
    sep_norm.label = "Separate Normal"

    # Absolute values of normal components
    abs_nx = group.nodes.new("ShaderNodeMath")
    abs_nx.operation = "ABSOLUTE"  # type: ignore[attr-defined]
    abs_nx.label = "|Nx|"

    abs_ny = group.nodes.new("ShaderNodeMath")
    abs_ny.operation = "ABSOLUTE"  # type: ignore[attr-defined]
    abs_ny.label = "|Ny|"

    abs_nz = group.nodes.new("ShaderNodeMath")
    abs_nz.operation = "ABSOLUTE"  # type: ignore[attr-defined]
    abs_nz.label = "|Nz|"

    # Find dominant axis by comparing absolute values
    # X dominant: |Nx| >= |Ny| AND |Nx| >= |Nz|
    # Y dominant: |Ny| > |Nx| AND |Ny| >= |Nz|
    # Z dominant: otherwise (default)
    #
    # Note: COMPARE checks equality (|A-B| < epsilon), not >= comparison!
    # For A >= B, we use: NOT (B > A), implemented as: 1 - GREATER_THAN(B, A)
    # Or equivalently: LESS_THAN(A, B) inverted

    # |Ny| > |Nx| (used to check |Nx| >= |Ny| via inversion)
    ny_gt_nx = group.nodes.new("ShaderNodeMath")
    ny_gt_nx.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    ny_gt_nx.label = "|Ny| > |Nx|"

    # |Nx| >= |Ny| = 1 - (|Ny| > |Nx|)
    nx_ge_ny = group.nodes.new("ShaderNodeMath")
    nx_ge_ny.operation = "SUBTRACT"  # type: ignore[attr-defined]
    nx_ge_ny.inputs[0].default_value = 1.0  # type: ignore[index]
    nx_ge_ny.label = "|Nx| >= |Ny|"

    # |Nz| > |Nx| (used to check |Nx| >= |Nz| via inversion)
    nz_gt_nx = group.nodes.new("ShaderNodeMath")
    nz_gt_nx.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    nz_gt_nx.label = "|Nz| > |Nx|"

    # |Nx| >= |Nz| = 1 - (|Nz| > |Nx|)
    nx_ge_nz = group.nodes.new("ShaderNodeMath")
    nx_ge_nz.operation = "SUBTRACT"  # type: ignore[attr-defined]
    nx_ge_nz.inputs[0].default_value = 1.0  # type: ignore[index]
    nx_ge_nz.label = "|Nx| >= |Nz|"

    # |Nz| > |Ny| (used to check |Ny| >= |Nz| via inversion)
    nz_gt_ny = group.nodes.new("ShaderNodeMath")
    nz_gt_ny.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    nz_gt_ny.label = "|Nz| > |Ny|"

    # |Ny| >= |Nz| = 1 - (|Nz| > |Ny|)
    ny_ge_nz = group.nodes.new("ShaderNodeMath")
    ny_ge_nz.operation = "SUBTRACT"  # type: ignore[attr-defined]
    ny_ge_nz.inputs[0].default_value = 1.0  # type: ignore[index]
    ny_ge_nz.label = "|Ny| >= |Nz|"

    # Combine conditions for X dominant: |Nx| >= |Ny| AND |Nx| >= |Nz|
    x_dominant = group.nodes.new("ShaderNodeMath")
    x_dominant.operation = "MULTIPLY"  # type: ignore[attr-defined]
    x_dominant.label = "X Dominant"

    # Combine conditions for Y dominant: |Ny| > |Nx| AND |Ny| >= |Nz|
    y_dominant = group.nodes.new("ShaderNodeMath")
    y_dominant.operation = "MULTIPLY"  # type: ignore[attr-defined]
    y_dominant.label = "Y Dominant"

    # Scale factors for 0.5x UV range
    scale_x = group.nodes.new("ShaderNodeMath")
    scale_x.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_x.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_x.label = "Scale X (0.5x)"

    scale_y = group.nodes.new("ShaderNodeMath")
    scale_y.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_y.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_y.label = "Scale Y (0.5x)"

    scale_z = group.nodes.new("ShaderNodeMath")
    scale_z.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_z.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_z.label = "Scale Z (0.5x)"

    # UV projections
    # X-dominant: U=Y, V=Z (project onto YZ plane)
    uv_x = group.nodes.new("ShaderNodeCombineXYZ")
    uv_x.label = "UV (YZ plane)"

    # Y-dominant: U=X, V=Z (project onto XZ plane)
    uv_y = group.nodes.new("ShaderNodeCombineXYZ")
    uv_y.label = "UV (XZ plane)"

    # Z-dominant (default): U=X, V=Y (project onto XY plane)
    uv_z = group.nodes.new("ShaderNodeCombineXYZ")
    uv_z.label = "UV (XY plane)"

    # Switch nodes for selection
    # First switch: Y dominant? Use XZ projection, else use XY (Z dominant default)
    switch_yz = group.nodes.new("GeometryNodeSwitch")
    switch_yz.input_type = "VECTOR"  # type: ignore[attr-defined]
    switch_yz.label = "Y vs Z"

    # Second switch: X dominant? Use YZ projection, else use result from first switch
    switch_x = group.nodes.new("GeometryNodeSwitch")
    switch_x.input_type = "VECTOR"  # type: ignore[attr-defined]
    switch_x.label = "X vs YZ"

    # Connect position and normal separation
    group.links.new(input_node.outputs["Position"], sep_pos.inputs["Vector"])
    group.links.new(input_node.outputs["Normal"], sep_norm.inputs["Vector"])

    # Absolute normal values
    group.links.new(sep_norm.outputs["X"], abs_nx.inputs[0])
    group.links.new(sep_norm.outputs["Y"], abs_ny.inputs[0])
    group.links.new(sep_norm.outputs["Z"], abs_nz.inputs[0])

    # Comparisons for X dominant: |Nx| >= |Ny| AND |Nx| >= |Nz|
    # |Ny| > |Nx|
    group.links.new(abs_ny.outputs["Value"], ny_gt_nx.inputs[0])
    group.links.new(abs_nx.outputs["Value"], ny_gt_nx.inputs[1])
    # |Nx| >= |Ny| = 1 - (|Ny| > |Nx|)
    group.links.new(ny_gt_nx.outputs["Value"], nx_ge_ny.inputs[1])

    # |Nz| > |Nx|
    group.links.new(abs_nz.outputs["Value"], nz_gt_nx.inputs[0])
    group.links.new(abs_nx.outputs["Value"], nz_gt_nx.inputs[1])
    # |Nx| >= |Nz| = 1 - (|Nz| > |Nx|)
    group.links.new(nz_gt_nx.outputs["Value"], nx_ge_nz.inputs[1])

    # Comparisons for Y dominant: |Ny| > |Nx| AND |Ny| >= |Nz|
    # |Ny| > |Nx| already computed above

    # |Nz| > |Ny|
    group.links.new(abs_nz.outputs["Value"], nz_gt_ny.inputs[0])
    group.links.new(abs_ny.outputs["Value"], nz_gt_ny.inputs[1])
    # |Ny| >= |Nz| = 1 - (|Nz| > |Ny|)
    group.links.new(nz_gt_ny.outputs["Value"], ny_ge_nz.inputs[1])

    # X dominant = |Nx| >= |Ny| AND |Nx| >= |Nz|
    group.links.new(nx_ge_ny.outputs["Value"], x_dominant.inputs[0])
    group.links.new(nx_ge_nz.outputs["Value"], x_dominant.inputs[1])

    # Y dominant = |Ny| > |Nx| AND |Ny| >= |Nz|
    group.links.new(ny_gt_nx.outputs["Value"], y_dominant.inputs[0])
    group.links.new(ny_ge_nz.outputs["Value"], y_dominant.inputs[1])

    # Scale position components by 0.5
    group.links.new(sep_pos.outputs["X"], scale_x.inputs[0])
    group.links.new(sep_pos.outputs["Y"], scale_y.inputs[0])
    group.links.new(sep_pos.outputs["Z"], scale_z.inputs[0])

    # UV projections (using scaled values)
    # X-dominant: U=Y, V=Z
    group.links.new(scale_y.outputs["Value"], uv_x.inputs["X"])
    group.links.new(scale_z.outputs["Value"], uv_x.inputs["Y"])

    # Y-dominant: U=X, V=Z
    group.links.new(scale_x.outputs["Value"], uv_y.inputs["X"])
    group.links.new(scale_z.outputs["Value"], uv_y.inputs["Y"])

    # Z-dominant: U=X, V=Y
    group.links.new(scale_x.outputs["Value"], uv_z.inputs["X"])
    group.links.new(scale_y.outputs["Value"], uv_z.inputs["Y"])

    # Switch chain: start with Z default, switch to Y if Y dominant, then to X if X dominant
    # First switch: if Y dominant, use uv_y (XZ plane), else use uv_z (XY plane)
    group.links.new(y_dominant.outputs["Value"], switch_yz.inputs["Switch"])
    group.links.new(uv_z.outputs["Vector"], switch_yz.inputs["False"])
    group.links.new(uv_y.outputs["Vector"], switch_yz.inputs["True"])

    # Second switch: if X dominant, use uv_x (YZ plane), else use result from first switch
    group.links.new(x_dominant.outputs["Value"], switch_x.inputs["Switch"])
    group.links.new(switch_yz.outputs["Output"], switch_x.inputs["False"])
    group.links.new(uv_x.outputs["Vector"], switch_x.inputs["True"])

    group.links.new(switch_x.outputs["Output"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_cylindrical_normal_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for cylindrical UV mapping based on normal direction.

    Unlike regular cylindrical mapping which uses position, this maps UVs
    based on the surface normal direction. This is useful for objects
    where the surface orientation matters more than its position.

    U is derived from the normal's azimuthal angle around the Z axis.
    V is derived from the normal's Z component (elevation).
    Scale, origin, tiling, and pole placement match the position-based version.
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Cylindrical Normal"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(
        name="Face Normal", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # Cylindrical normal mapping:
    # U = atan2(nx, ny) / (2*pi) * -4  (azimuthal angle, same scale as position-based)
    # V = nz * 0.75  (elevation, same scale as position-based)
    # With seam correction using face normal

    # === Corner normal processing ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Normal"

    # Corner U = atan2(nx, ny) / (2*pi)
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(Nx, Ny)"

    divide_node = group.nodes.new("ShaderNodeMath")
    divide_node.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_node.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_node.label = "/ 2π (Corner U)"

    # V = nz * 0.75 (same scale as position-based cylindrical)
    scale_v = group.nodes.new("ShaderNodeMath")
    scale_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_v.inputs[1].default_value = 0.75  # type: ignore[index]
    scale_v.label = "Scale V (0.75x)"

    # === Face normal processing ===
    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Normal"

    # Face U (no +0.5 offset to match overlay)
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2 Face"

    divide_face = group.nodes.new("ShaderNodeMath")
    divide_face.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_face.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_face.label = "/ 2π (Face U)"

    # === Seam correction ===
    delta_u = group.nodes.new("ShaderNodeMath")
    delta_u.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_u.label = "U_corner - U_face"

    delta_gt_half = group.nodes.new("ShaderNodeMath")
    delta_gt_half.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_half.inputs[1].default_value = 0.5  # type: ignore[index]
    delta_gt_half.label = "delta > 0.5"

    delta_lt_neg_half = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_half.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_half.inputs[1].default_value = -0.5  # type: ignore[index]
    delta_lt_neg_half.label = "delta < -0.5"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -1.0  # type: ignore[index]
    neg_offset.label = "-1 * gt"

    total_offset = group.nodes.new("ShaderNodeMath")
    total_offset.operation = "ADD"  # type: ignore[attr-defined]
    total_offset.label = "Offset"

    corrected_u = group.nodes.new("ShaderNodeMath")
    corrected_u.operation = "ADD"  # type: ignore[attr-defined]
    corrected_u.label = "Corrected U"

    # Internal scale factor for U (-4x to get sensible UV range with correct orientation)
    scale_u = group.nodes.new("ShaderNodeMath")
    scale_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_u.inputs[1].default_value = -4.0  # type: ignore[index]
    scale_u.label = "Scale U (-4x)"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # === Connect corner normal processing ===
    group.links.new(input_node.outputs["Normal"], separate_xyz.inputs["Vector"])

    # Corner U (from normal)
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_node.inputs[0])

    # V = nz * 0.75 (directly from normal Z, same scale as position-based)
    group.links.new(separate_xyz.outputs["Z"], scale_v.inputs[0])

    # === Connect face processing ===
    group.links.new(input_node.outputs["Face Normal"], separate_face.inputs["Vector"])
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[1])
    group.links.new(atan2_face.outputs["Value"], divide_face.inputs[0])

    # === Connect seam correction ===
    # Now using divide_node directly (no +0.5 offset)
    group.links.new(divide_node.outputs["Value"], delta_u.inputs[0])
    group.links.new(divide_face.outputs["Value"], delta_u.inputs[1])

    group.links.new(delta_u.outputs["Value"], delta_gt_half.inputs[0])
    group.links.new(delta_u.outputs["Value"], delta_lt_neg_half.inputs[0])

    group.links.new(delta_gt_half.outputs["Value"], neg_offset.inputs[0])
    group.links.new(neg_offset.outputs["Value"], total_offset.inputs[0])
    group.links.new(delta_lt_neg_half.outputs["Value"], total_offset.inputs[1])

    group.links.new(divide_node.outputs["Value"], corrected_u.inputs[0])
    group.links.new(total_offset.outputs["Value"], corrected_u.inputs[1])

    # === Apply internal U scale ===
    group.links.new(corrected_u.outputs["Value"], scale_u.inputs[0])

    # === Final UV output ===
    group.links.new(scale_u.outputs["Value"], combine_xyz.inputs["X"])  # U (scaled)
    group.links.new(scale_v.outputs["Value"], combine_xyz.inputs["Y"])  # V (scaled)
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_cylindrical_capped_normal_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for cylindrical normal UV mapping with planar caps.

    Uses cylindrical normal mapping for sides and planar normal mapping for caps.
    Cap detection is based on face normal direction: |face_nz| > threshold means cap.
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Cylindrical Capped Normal"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(
        name="Face Normal", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # === Separate inputs ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Normal"

    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Normal"

    # === Normal-based cap detection ===
    # A face is a cap when its normal points mostly up or down (|nz| > threshold)
    # Side faces have normals pointing outward (|nz| small)
    abs_face_nz = group.nodes.new("ShaderNodeMath")
    abs_face_nz.operation = "ABSOLUTE"  # type: ignore[attr-defined]
    abs_face_nz.label = "|Face Nz|"

    is_cap = group.nodes.new("ShaderNodeMath")
    is_cap.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    is_cap.inputs[1].default_value = 0.7  # type: ignore[index]
    is_cap.label = "Is Cap (|Nz| > 0.7)"

    # === Cylindrical UV calculation (for sides) ===
    # U = atan2(nx, ny) / (2*pi) + 0.5
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(Nx, Ny)"

    divide_2pi = group.nodes.new("ShaderNodeMath")
    divide_2pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi.label = "/ 2π (Corner U)"

    # V = nz * 0.75 (same scale as position-based cylindrical)
    scale_cyl_v = group.nodes.new("ShaderNodeMath")
    scale_cyl_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cyl_v.inputs[1].default_value = 0.75  # type: ignore[index]
    scale_cyl_v.label = "Scale Cyl V (0.75x)"

    # === Face center U calculation (for seam correction) ===
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2 Face"

    divide_2pi_face = group.nodes.new("ShaderNodeMath")
    divide_2pi_face.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi_face.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi_face.label = "/ 2π (Face U)"

    # === Seam correction ===
    delta_u = group.nodes.new("ShaderNodeMath")
    delta_u.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_u.label = "U_corner - U_face"

    delta_gt_half = group.nodes.new("ShaderNodeMath")
    delta_gt_half.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_half.inputs[1].default_value = 0.5  # type: ignore[index]
    delta_gt_half.label = "delta > 0.5"

    delta_lt_neg_half = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_half.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_half.inputs[1].default_value = -0.5  # type: ignore[index]
    delta_lt_neg_half.label = "delta < -0.5"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -1.0  # type: ignore[index]
    neg_offset.label = "-1 * gt"

    total_offset = group.nodes.new("ShaderNodeMath")
    total_offset.operation = "ADD"  # type: ignore[attr-defined]
    total_offset.label = "Offset"

    corrected_cyl_u = group.nodes.new("ShaderNodeMath")
    corrected_cyl_u.operation = "ADD"  # type: ignore[attr-defined]
    corrected_cyl_u.label = "Corrected Cyl U"

    # Internal scale factor for cylindrical U (-4x to get sensible UV range)
    scale_cyl_u = group.nodes.new("ShaderNodeMath")
    scale_cyl_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cyl_u.inputs[1].default_value = -4.0  # type: ignore[index]
    scale_cyl_u.label = "Scale Cyl U (-4x)"

    # Cylindrical UV
    combine_cyl_uv = group.nodes.new("ShaderNodeCombineXYZ")
    combine_cyl_uv.label = "Cylindrical UV"

    # === Planar UV calculation (for caps) ===
    # Use Nx, Ny as UV (normal points up/down, so XY plane is the cap surface)
    scale_cap_x = group.nodes.new("ShaderNodeMath")
    scale_cap_x.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cap_x.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_cap_x.label = "Scale Cap X (0.5x)"

    scale_cap_y = group.nodes.new("ShaderNodeMath")
    scale_cap_y.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_cap_y.inputs[1].default_value = 0.5  # type: ignore[index]
    scale_cap_y.label = "Scale Cap Y (0.5x)"

    combine_planar_uv = group.nodes.new("ShaderNodeCombineXYZ")
    combine_planar_uv.label = "Planar UV"

    # === Mix between cylindrical and planar based on is_cap ===
    uv_switch = group.nodes.new("GeometryNodeSwitch")
    uv_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    uv_switch.label = "Cap/Side Switch"

    # === Connect normal ===
    group.links.new(input_node.outputs["Normal"], separate_xyz.inputs["Vector"])

    # === Connect face normal and cap detection ===
    group.links.new(input_node.outputs["Face Normal"], separate_face.inputs["Vector"])
    group.links.new(separate_face.outputs["Z"], abs_face_nz.inputs[0])
    group.links.new(abs_face_nz.outputs["Value"], is_cap.inputs[0])

    # === Connect cylindrical U calculation ===
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_2pi.inputs[0])

    # === Connect V calculation (V = nz * 0.75) ===
    group.links.new(separate_xyz.outputs["Z"], scale_cyl_v.inputs[0])

    # === Connect face U calculation ===
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[1])
    group.links.new(atan2_face.outputs["Value"], divide_2pi_face.inputs[0])

    # === Connect seam correction ===
    # Now using divide_2pi directly (no +0.5 offset)
    group.links.new(divide_2pi.outputs["Value"], delta_u.inputs[0])
    group.links.new(divide_2pi_face.outputs["Value"], delta_u.inputs[1])
    group.links.new(delta_u.outputs["Value"], delta_gt_half.inputs[0])
    group.links.new(delta_u.outputs["Value"], delta_lt_neg_half.inputs[0])
    group.links.new(delta_gt_half.outputs["Value"], neg_offset.inputs[0])
    group.links.new(neg_offset.outputs["Value"], total_offset.inputs[0])
    group.links.new(delta_lt_neg_half.outputs["Value"], total_offset.inputs[1])
    group.links.new(divide_2pi.outputs["Value"], corrected_cyl_u.inputs[0])
    group.links.new(total_offset.outputs["Value"], corrected_cyl_u.inputs[1])

    # === Apply internal U scale to cylindrical ===
    group.links.new(corrected_cyl_u.outputs["Value"], scale_cyl_u.inputs[0])

    # === Combine cylindrical UV ===
    group.links.new(scale_cyl_u.outputs["Value"], combine_cyl_uv.inputs["X"])
    group.links.new(scale_cyl_v.outputs["Value"], combine_cyl_uv.inputs["Y"])

    # === Combine planar UV (Nx * 0.5, Ny * 0.5) ===
    group.links.new(separate_xyz.outputs["X"], scale_cap_x.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], scale_cap_y.inputs[0])
    group.links.new(scale_cap_x.outputs["Value"], combine_planar_uv.inputs["X"])
    group.links.new(scale_cap_y.outputs["Value"], combine_planar_uv.inputs["Y"])

    # === Switch based on is_cap ===
    group.links.new(is_cap.outputs["Value"], uv_switch.inputs["Switch"])
    group.links.new(combine_cyl_uv.outputs["Vector"], uv_switch.inputs["False"])
    group.links.new(combine_planar_uv.outputs["Vector"], uv_switch.inputs["True"])

    # === Output ===
    group.links.new(uv_switch.outputs["Output"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_spherical_normal_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for spherical UV mapping based on normal direction.

    Unlike regular spherical mapping which uses position, this maps UVs
    based on the surface normal direction. This is useful for objects
    where the surface orientation matters more than its position.

    Position and scale inputs are ignored - only rotation affects the result.
    Uses the normal direction as if it were a point on a unit sphere.
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Spherical Normal"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(
        name="Face Normal", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # Spherical normal mapping:
    # Normal is already normalized (unit length), so we treat it as a point on unit sphere
    # U = atan2(nx, ny) / (2*pi) + 0.5
    # V = acos(nz) / pi
    # With seam and pole correction using face normal

    # === Corner normal processing ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Normal"

    # Corner horizontal radius r = sqrt(nx² + ny²) for pole detection
    corner_x_sq = group.nodes.new("ShaderNodeMath")
    corner_x_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_x_sq.label = "Nx²"

    corner_y_sq = group.nodes.new("ShaderNodeMath")
    corner_y_sq.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_y_sq.label = "Ny²"

    corner_r_sq = group.nodes.new("ShaderNodeMath")
    corner_r_sq.operation = "ADD"  # type: ignore[attr-defined]
    corner_r_sq.label = "Nx² + Ny²"

    corner_r = group.nodes.new("ShaderNodeMath")
    corner_r.operation = "SQRT"  # type: ignore[attr-defined]
    corner_r.label = "Corner R"

    # Corner U calculation
    # No +0.5 offset so U=0 is at +Y direction, matching overlay
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(Nx, Ny)"

    divide_2pi = group.nodes.new("ShaderNodeMath")
    divide_2pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi.label = "/ 2π (Corner U)"

    # Corner V calculation (normal is already unit length)
    clamp_node = group.nodes.new("ShaderNodeClamp")
    clamp_node.inputs["Min"].default_value = -1.0  # type: ignore[index]
    clamp_node.inputs["Max"].default_value = 1.0  # type: ignore[index]
    clamp_node.label = "Clamp Nz"

    acos_node = group.nodes.new("ShaderNodeMath")
    acos_node.operation = "ARCCOSINE"  # type: ignore[attr-defined]
    acos_node.label = "acos"

    divide_pi = group.nodes.new("ShaderNodeMath")
    divide_pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_pi.inputs[1].default_value = math.pi  # type: ignore[index]
    divide_pi.label = "/ π"

    # === Face normal processing ===
    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Normal"

    # Face U calculation (needed for seam correction reference)
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2 Face"

    divide_2pi_face = group.nodes.new("ShaderNodeMath")
    divide_2pi_face.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi_face.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi_face.label = "/ 2π (Face U)"

    # === Pole blending ===
    pole_threshold = group.nodes.new("ShaderNodeValue")
    pole_threshold.outputs[0].default_value = 0.01  # type: ignore[index]
    pole_threshold.label = "Pole Threshold"

    pole_blend_raw = group.nodes.new("ShaderNodeMath")
    pole_blend_raw.operation = "DIVIDE"  # type: ignore[attr-defined]
    pole_blend_raw.label = "Corner R / Threshold"

    pole_blend = group.nodes.new("ShaderNodeClamp")
    pole_blend.inputs["Min"].default_value = 0.0  # type: ignore[index]
    pole_blend.inputs["Max"].default_value = 1.0  # type: ignore[index]
    pole_blend.label = "Pole Blend"

    one_minus_blend = group.nodes.new("ShaderNodeMath")
    one_minus_blend.operation = "SUBTRACT"  # type: ignore[attr-defined]
    one_minus_blend.inputs[0].default_value = 1.0  # type: ignore[index]
    one_minus_blend.label = "1 - Blend"

    face_u_weighted = group.nodes.new("ShaderNodeMath")
    face_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_u_weighted.label = "Face U * (1-blend)"

    corner_u_weighted = group.nodes.new("ShaderNodeMath")
    corner_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_u_weighted.label = "Corrected Corner U * blend"

    blended_u = group.nodes.new("ShaderNodeMath")
    blended_u.operation = "ADD"  # type: ignore[attr-defined]
    blended_u.label = "Final U"

    # Internal scale factors
    scale_u = group.nodes.new("ShaderNodeMath")
    scale_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_u.inputs[1].default_value = -4.0  # type: ignore[index]
    scale_u.label = "Scale U (-4x)"

    # Offset V so that equator (acos(0)/π = 0.5) becomes 0 in UV space
    # This anchors UV origin at the equator, matching the overlay behavior
    offset_v = group.nodes.new("ShaderNodeMath")
    offset_v.operation = "SUBTRACT"  # type: ignore[attr-defined]
    offset_v.inputs[1].default_value = 0.5  # type: ignore[index]
    offset_v.label = "V - 0.5 (Equator Offset)"

    scale_v = group.nodes.new("ShaderNodeMath")
    scale_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scale_v.inputs[1].default_value = -2.0  # type: ignore[index]
    scale_v.label = "Scale V (-2x)"

    # === Seam correction ===
    delta_u = group.nodes.new("ShaderNodeMath")
    delta_u.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_u.label = "U_corner - U_face"

    delta_gt_half = group.nodes.new("ShaderNodeMath")
    delta_gt_half.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_half.inputs[1].default_value = 0.5  # type: ignore[index]
    delta_gt_half.label = "delta > 0.5"

    delta_lt_neg_half = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_half.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_half.inputs[1].default_value = -0.5  # type: ignore[index]
    delta_lt_neg_half.label = "delta < -0.5"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -1.0  # type: ignore[index]
    neg_offset.label = "-1 * gt"

    total_offset = group.nodes.new("ShaderNodeMath")
    total_offset.operation = "ADD"  # type: ignore[attr-defined]
    total_offset.label = "Offset"

    corrected_corner_u = group.nodes.new("ShaderNodeMath")
    corrected_corner_u.operation = "ADD"  # type: ignore[attr-defined]
    corrected_corner_u.label = "Corrected Corner U"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # === Connect corner normal processing ===
    group.links.new(input_node.outputs["Normal"], separate_xyz.inputs["Vector"])

    # Corner R calculation
    group.links.new(separate_xyz.outputs["X"], corner_x_sq.inputs[0])
    group.links.new(separate_xyz.outputs["X"], corner_x_sq.inputs[1])
    group.links.new(separate_xyz.outputs["Y"], corner_y_sq.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], corner_y_sq.inputs[1])
    group.links.new(corner_x_sq.outputs["Value"], corner_r_sq.inputs[0])
    group.links.new(corner_y_sq.outputs["Value"], corner_r_sq.inputs[1])
    group.links.new(corner_r_sq.outputs["Value"], corner_r.inputs[0])

    # Corner U
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_2pi.inputs[0])

    # Corner V (nz clamped then acos)
    group.links.new(separate_xyz.outputs["Z"], clamp_node.inputs["Value"])
    group.links.new(clamp_node.outputs["Result"], acos_node.inputs[0])
    group.links.new(acos_node.outputs["Value"], divide_pi.inputs[0])

    # === Connect face processing ===
    group.links.new(input_node.outputs["Face Normal"], separate_face.inputs["Vector"])

    # Face U
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[1])
    group.links.new(atan2_face.outputs["Value"], divide_2pi_face.inputs[0])

    # === Connect pole blending ===
    group.links.new(corner_r.outputs["Value"], pole_blend_raw.inputs[0])
    group.links.new(pole_threshold.outputs[0], pole_blend_raw.inputs[1])
    group.links.new(pole_blend_raw.outputs["Value"], pole_blend.inputs["Value"])

    # === Connect seam correction (BEFORE blending) ===
    # Now using divide_2pi directly (no +0.5 offset)
    group.links.new(divide_2pi.outputs["Value"], delta_u.inputs[0])
    group.links.new(divide_2pi_face.outputs["Value"], delta_u.inputs[1])

    group.links.new(delta_u.outputs["Value"], delta_gt_half.inputs[0])
    group.links.new(delta_u.outputs["Value"], delta_lt_neg_half.inputs[0])

    group.links.new(delta_gt_half.outputs["Value"], neg_offset.inputs[0])
    group.links.new(neg_offset.outputs["Value"], total_offset.inputs[0])
    group.links.new(delta_lt_neg_half.outputs["Value"], total_offset.inputs[1])

    group.links.new(divide_2pi.outputs["Value"], corrected_corner_u.inputs[0])
    group.links.new(total_offset.outputs["Value"], corrected_corner_u.inputs[1])

    # === Connect pole blending (AFTER seam correction) ===
    group.links.new(pole_blend.outputs["Result"], one_minus_blend.inputs[1])
    group.links.new(divide_2pi_face.outputs["Value"], face_u_weighted.inputs[0])
    group.links.new(one_minus_blend.outputs["Value"], face_u_weighted.inputs[1])
    group.links.new(corrected_corner_u.outputs["Value"], corner_u_weighted.inputs[0])
    group.links.new(pole_blend.outputs["Result"], corner_u_weighted.inputs[1])
    group.links.new(face_u_weighted.outputs["Value"], blended_u.inputs[0])
    group.links.new(corner_u_weighted.outputs["Value"], blended_u.inputs[1])

    # === Apply internal UV scale ===
    group.links.new(blended_u.outputs["Value"], scale_u.inputs[0])
    group.links.new(divide_pi.outputs["Value"], offset_v.inputs[0])
    group.links.new(offset_v.outputs["Value"], scale_v.inputs[0])

    # === Final UV output ===
    group.links.new(scale_u.outputs["Value"], combine_xyz.inputs["X"])  # Final U
    group.links.new(scale_v.outputs["Value"], combine_xyz.inputs["Y"])  # V
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_shrink_wrap_normal_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for shrink wrap UV mapping based on normal direction.

    Unlike regular shrink wrap mapping which uses position, this maps UVs
    based on the surface normal direction using azimuthal equidistant projection.
    This is useful for objects where the surface orientation matters more than position.

    Position and scale inputs are ignored - only rotation affects the result.

    Formula (same as position-based, but using normal):
      theta = acos(nz)           # angle from +Z axis (normal is unit length)
      phi = atan2(ny, nx)        # azimuthal angle
      r = theta / pi             # radial distance (0 at +Z, 1 at -Z)
      u = r * cos(phi)
      v = r * sin(phi)
    """
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Shrink Wrap Normal"

    # Check if group already exists (unless forcing new)
    if not _force_new_subgroups and group_name in bpy.data.node_groups:
        return bpy.data.node_groups[group_name]

    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.use_fake_user = True

    interface = group.interface
    assert interface is not None
    interface.new_socket(name="Normal", socket_type="NodeSocketVector", in_out="INPUT")
    interface.new_socket(
        name="Face Normal", socket_type="NodeSocketVector", in_out="INPUT"
    )
    interface.new_socket(name="UV", socket_type="NodeSocketVector", in_out="OUTPUT")

    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")

    # === Corner normal processing ===
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Normal"

    # theta = acos(nz) - normal is already unit length, no need to divide by length
    clamp_node = group.nodes.new("ShaderNodeClamp")
    clamp_node.inputs["Min"].default_value = -1.0  # type: ignore[index]
    clamp_node.inputs["Max"].default_value = 1.0  # type: ignore[index]
    clamp_node.label = "Clamp Nz"

    acos_node = group.nodes.new("ShaderNodeMath")
    acos_node.operation = "ARCCOSINE"  # type: ignore[attr-defined]
    acos_node.label = "acos (theta)"

    # r = theta / pi
    divide_pi = group.nodes.new("ShaderNodeMath")
    divide_pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_pi.inputs[1].default_value = math.pi  # type: ignore[index]
    divide_pi.label = "/ π (r)"

    # phi = atan2(ny, nx)
    atan2_corner = group.nodes.new("ShaderNodeMath")
    atan2_corner.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_corner.label = "atan2(Ny, Nx) corner"

    # === Face normal processing ===
    separate_face = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_face.label = "Separate Face Normal"

    # Face phi for seam correction reference
    atan2_face = group.nodes.new("ShaderNodeMath")
    atan2_face.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_face.label = "atan2(Ny, Nx) face"

    # === Seam correction for phi ===
    # delta = phi_corner - phi_face
    delta_phi = group.nodes.new("ShaderNodeMath")
    delta_phi.operation = "SUBTRACT"  # type: ignore[attr-defined]
    delta_phi.label = "phi_corner - phi_face"

    delta_gt_pi = group.nodes.new("ShaderNodeMath")
    delta_gt_pi.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    delta_gt_pi.inputs[1].default_value = math.pi  # type: ignore[index]
    delta_gt_pi.label = "delta > π"

    delta_lt_neg_pi = group.nodes.new("ShaderNodeMath")
    delta_lt_neg_pi.operation = "LESS_THAN"  # type: ignore[attr-defined]
    delta_lt_neg_pi.inputs[1].default_value = -math.pi  # type: ignore[index]
    delta_lt_neg_pi.label = "delta < -π"

    neg_offset = group.nodes.new("ShaderNodeMath")
    neg_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    neg_offset.inputs[1].default_value = -2.0 * math.pi  # type: ignore[index]
    neg_offset.label = "-2π * gt"

    pos_offset = group.nodes.new("ShaderNodeMath")
    pos_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    pos_offset.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    pos_offset.label = "+2π * lt"

    raw_offset = group.nodes.new("ShaderNodeMath")
    raw_offset.operation = "ADD"  # type: ignore[attr-defined]
    raw_offset.label = "Raw Offset"

    # Disable seam correction near pole: multiply offset by step(r - threshold)
    pole_cutoff = group.nodes.new("ShaderNodeMath")
    pole_cutoff.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    pole_cutoff.inputs[1].default_value = 0.15  # type: ignore[index]
    pole_cutoff.label = "r > 0.15"

    scaled_offset = group.nodes.new("ShaderNodeMath")
    scaled_offset.operation = "MULTIPLY"  # type: ignore[attr-defined]
    scaled_offset.label = "Scaled Offset"

    corrected_phi = group.nodes.new("ShaderNodeMath")
    corrected_phi.operation = "ADD"  # type: ignore[attr-defined]
    corrected_phi.label = "Corrected Phi"

    # === Final UV calculation (corner) ===
    cos_phi = group.nodes.new("ShaderNodeMath")
    cos_phi.operation = "COSINE"  # type: ignore[attr-defined]
    cos_phi.label = "cos(phi)"

    sin_phi = group.nodes.new("ShaderNodeMath")
    sin_phi.operation = "SINE"  # type: ignore[attr-defined]
    sin_phi.label = "sin(phi)"

    # U = r * cos(phi), V = r * sin(phi)
    corner_u = group.nodes.new("ShaderNodeMath")
    corner_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_u.label = "Corner U"

    corner_v = group.nodes.new("ShaderNodeMath")
    corner_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_v.label = "Corner V"

    # === Face UV calculation (for -Z pole blending) ===
    face_clamp = group.nodes.new("ShaderNodeClamp")
    face_clamp.inputs["Min"].default_value = -1.0  # type: ignore[index]
    face_clamp.inputs["Max"].default_value = 1.0  # type: ignore[index]
    face_clamp.label = "Clamp Face Nz"

    face_acos = group.nodes.new("ShaderNodeMath")
    face_acos.operation = "ARCCOSINE"  # type: ignore[attr-defined]
    face_acos.label = "Face acos"

    face_r = group.nodes.new("ShaderNodeMath")
    face_r.operation = "DIVIDE"  # type: ignore[attr-defined]
    face_r.inputs[1].default_value = math.pi  # type: ignore[index]
    face_r.label = "Face R"

    face_cos_phi = group.nodes.new("ShaderNodeMath")
    face_cos_phi.operation = "COSINE"  # type: ignore[attr-defined]
    face_cos_phi.label = "cos(face phi)"

    face_sin_phi = group.nodes.new("ShaderNodeMath")
    face_sin_phi.operation = "SINE"  # type: ignore[attr-defined]
    face_sin_phi.label = "sin(face phi)"

    face_u = group.nodes.new("ShaderNodeMath")
    face_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_u.label = "Face U"

    face_v = group.nodes.new("ShaderNodeMath")
    face_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_v.label = "Face V"

    # === -Z pole blending (blend toward face UV when r > 0.85) ===
    pole_blend_threshold = group.nodes.new("ShaderNodeMath")
    pole_blend_threshold.operation = "SUBTRACT"  # type: ignore[attr-defined]
    pole_blend_threshold.inputs[1].default_value = 0.85  # type: ignore[index]
    pole_blend_threshold.label = "r - 0.85"

    pole_blend_scale = group.nodes.new("ShaderNodeMath")
    pole_blend_scale.operation = "MULTIPLY"  # type: ignore[attr-defined]
    pole_blend_scale.inputs[1].default_value = 1.0 / 0.15  # type: ignore[index]
    pole_blend_scale.label = "Scale to 0-1"

    pole_blend = group.nodes.new("ShaderNodeClamp")
    pole_blend.inputs["Min"].default_value = 0.0  # type: ignore[index]
    pole_blend.inputs["Max"].default_value = 1.0  # type: ignore[index]
    pole_blend.label = "-Z Pole Blend"

    one_minus_blend = group.nodes.new("ShaderNodeMath")
    one_minus_blend.operation = "SUBTRACT"  # type: ignore[attr-defined]
    one_minus_blend.inputs[0].default_value = 1.0  # type: ignore[index]
    one_minus_blend.label = "1 - blend"

    corner_u_weighted = group.nodes.new("ShaderNodeMath")
    corner_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_u_weighted.label = "Corner U * (1-b)"

    face_u_weighted = group.nodes.new("ShaderNodeMath")
    face_u_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_u_weighted.label = "Face U * b"

    final_u = group.nodes.new("ShaderNodeMath")
    final_u.operation = "ADD"  # type: ignore[attr-defined]
    final_u.label = "Final U"

    corner_v_weighted = group.nodes.new("ShaderNodeMath")
    corner_v_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    corner_v_weighted.label = "Corner V * (1-b)"

    face_v_weighted = group.nodes.new("ShaderNodeMath")
    face_v_weighted.operation = "MULTIPLY"  # type: ignore[attr-defined]
    face_v_weighted.label = "Face V * b"

    final_v = group.nodes.new("ShaderNodeMath")
    final_v.operation = "ADD"  # type: ignore[attr-defined]
    final_v.label = "Final V"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # === Connect corner normal ===
    group.links.new(input_node.outputs["Normal"], separate_xyz.inputs["Vector"])

    # Theta and r (no length division needed - normal is unit)
    group.links.new(separate_xyz.outputs["Z"], clamp_node.inputs["Value"])
    group.links.new(clamp_node.outputs["Result"], acos_node.inputs[0])
    group.links.new(acos_node.outputs["Value"], divide_pi.inputs[0])

    # Corner phi
    group.links.new(separate_xyz.outputs["Y"], atan2_corner.inputs[0])
    group.links.new(separate_xyz.outputs["X"], atan2_corner.inputs[1])

    # === Connect face normal ===
    group.links.new(input_node.outputs["Face Normal"], separate_face.inputs["Vector"])
    group.links.new(separate_face.outputs["Y"], atan2_face.inputs[0])
    group.links.new(separate_face.outputs["X"], atan2_face.inputs[1])

    # === Connect seam correction ===
    group.links.new(atan2_corner.outputs["Value"], delta_phi.inputs[0])
    group.links.new(atan2_face.outputs["Value"], delta_phi.inputs[1])

    group.links.new(delta_phi.outputs["Value"], delta_gt_pi.inputs[0])
    group.links.new(delta_phi.outputs["Value"], delta_lt_neg_pi.inputs[0])

    group.links.new(delta_gt_pi.outputs["Value"], neg_offset.inputs[0])
    group.links.new(delta_lt_neg_pi.outputs["Value"], pos_offset.inputs[0])

    group.links.new(neg_offset.outputs["Value"], raw_offset.inputs[0])
    group.links.new(pos_offset.outputs["Value"], raw_offset.inputs[1])

    # Disable seam correction near pole
    group.links.new(divide_pi.outputs["Value"], pole_cutoff.inputs[0])
    group.links.new(raw_offset.outputs["Value"], scaled_offset.inputs[0])
    group.links.new(pole_cutoff.outputs["Value"], scaled_offset.inputs[1])

    group.links.new(atan2_corner.outputs["Value"], corrected_phi.inputs[0])
    group.links.new(scaled_offset.outputs["Value"], corrected_phi.inputs[1])

    # === Connect final UV ===
    group.links.new(corrected_phi.outputs["Value"], cos_phi.inputs[0])
    group.links.new(corrected_phi.outputs["Value"], sin_phi.inputs[0])

    group.links.new(divide_pi.outputs["Value"], corner_u.inputs[0])
    group.links.new(cos_phi.outputs["Value"], corner_u.inputs[1])

    group.links.new(divide_pi.outputs["Value"], corner_v.inputs[0])
    group.links.new(sin_phi.outputs["Value"], corner_v.inputs[1])

    # === Connect face UV calculation ===
    group.links.new(separate_face.outputs["Z"], face_clamp.inputs["Value"])
    group.links.new(face_clamp.outputs["Result"], face_acos.inputs[0])
    group.links.new(face_acos.outputs["Value"], face_r.inputs[0])

    group.links.new(atan2_face.outputs["Value"], face_cos_phi.inputs[0])
    group.links.new(atan2_face.outputs["Value"], face_sin_phi.inputs[0])

    group.links.new(face_r.outputs["Value"], face_u.inputs[0])
    group.links.new(face_cos_phi.outputs["Value"], face_u.inputs[1])

    group.links.new(face_r.outputs["Value"], face_v.inputs[0])
    group.links.new(face_sin_phi.outputs["Value"], face_v.inputs[1])

    # === Connect -Z pole blending ===
    group.links.new(divide_pi.outputs["Value"], pole_blend_threshold.inputs[0])
    group.links.new(pole_blend_threshold.outputs["Value"], pole_blend_scale.inputs[0])
    group.links.new(pole_blend_scale.outputs["Value"], pole_blend.inputs["Value"])

    group.links.new(pole_blend.outputs["Result"], one_minus_blend.inputs[1])

    # Blend U
    group.links.new(corner_u.outputs["Value"], corner_u_weighted.inputs[0])
    group.links.new(one_minus_blend.outputs["Value"], corner_u_weighted.inputs[1])

    group.links.new(face_u.outputs["Value"], face_u_weighted.inputs[0])
    group.links.new(pole_blend.outputs["Result"], face_u_weighted.inputs[1])

    group.links.new(corner_u_weighted.outputs["Value"], final_u.inputs[0])
    group.links.new(face_u_weighted.outputs["Value"], final_u.inputs[1])

    # Blend V
    group.links.new(corner_v.outputs["Value"], corner_v_weighted.inputs[0])
    group.links.new(one_minus_blend.outputs["Value"], corner_v_weighted.inputs[1])

    group.links.new(face_v.outputs["Value"], face_v_weighted.inputs[0])
    group.links.new(pole_blend.outputs["Result"], face_v_weighted.inputs[1])

    group.links.new(corner_v_weighted.outputs["Value"], final_v.inputs[0])
    group.links.new(face_v_weighted.outputs["Value"], final_v.inputs[1])

    group.links.new(final_u.outputs["Value"], combine_xyz.inputs["X"])
    group.links.new(final_v.outputs["Value"], combine_xyz.inputs["Y"])
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _populate_uv_map_node_group(node_tree: bpy.types.NodeTree) -> None:  # noqa: PLR0915
    """Populate a UV Map node group with all mapping nodes and connections.

    This is the internal implementation that creates all nodes and links.
    Used by both create_uv_map_node_group() and regenerate_uv_map_node_group().
    """
    # Create nodes
    input_node = node_tree.nodes.new("NodeGroupInput")
    input_node.select = False

    output_node = node_tree.nodes.new("NodeGroupOutput")
    output_node.select = False

    # Get position and normal
    position_node = node_tree.nodes.new("GeometryNodeInputPosition")
    position_node.select = False

    normal_node = node_tree.nodes.new("GeometryNodeInputNormal")
    normal_node.select = False

    # Get face normal for box mapping (constant per face, avoids interpolation artifacts)
    # We evaluate the normal field at Face domain, indexed by the face of each corner
    face_normal_node = node_tree.nodes.new("GeometryNodeInputNormal")
    face_normal_node.label = "Normal (for Face lookup)"
    face_normal_node.select = False

    # Evaluate normal at face domain to get per-corner face normal
    # This node evaluates the input field at the specified domain using the given index
    evaluate_face_normal = node_tree.nodes.new("GeometryNodeFieldAtIndex")
    evaluate_face_normal.data_type = "FLOAT_VECTOR"  # type: ignore[attr-defined]
    evaluate_face_normal.domain = "FACE"  # type: ignore[attr-defined]
    evaluate_face_normal.label = "Evaluate Normal at Face"
    evaluate_face_normal.select = False

    # Get face position for seam correction in cylindrical/spherical mapping
    face_position_node = node_tree.nodes.new("GeometryNodeInputPosition")
    face_position_node.label = "Position (for Face lookup)"
    face_position_node.select = False

    # Evaluate position at face domain to get face center position
    evaluate_face_position = node_tree.nodes.new("GeometryNodeFieldAtIndex")
    evaluate_face_position.data_type = "FLOAT_VECTOR"  # type: ignore[attr-defined]
    evaluate_face_position.domain = "FACE"  # type: ignore[attr-defined]
    evaluate_face_position.label = "Evaluate Position at Face"
    evaluate_face_position.select = False

    # Get the face index for each corner
    face_of_corner = node_tree.nodes.new("GeometryNodeFaceOfCorner")
    face_of_corner.label = "Face of Corner"
    face_of_corner.select = False

    # Get corner index (current evaluation context)
    corner_index_node = node_tree.nodes.new("GeometryNodeInputIndex")
    corner_index_node.label = "Corner Index"
    corner_index_node.select = False

    # === Smooth Normals Computation ===
    # Compute smooth vertex normals by averaging face normals at each vertex
    # This is used for normal-based mappings when "Smooth Normals" is enabled

    # Get vertex index for each corner (used as group ID for accumulation)
    vertex_of_corner = node_tree.nodes.new("GeometryNodeVertexOfCorner")
    vertex_of_corner.label = "Vertex of Corner"
    vertex_of_corner.select = False

    # Get face normal at each corner's face
    face_normal_at_corner = node_tree.nodes.new("GeometryNodeFieldAtIndex")
    face_normal_at_corner.data_type = "FLOAT_VECTOR"  # type: ignore[attr-defined]
    face_normal_at_corner.domain = "FACE"  # type: ignore[attr-defined]
    face_normal_at_corner.label = "Face Normal at Corner"
    face_normal_at_corner.select = False

    # Accumulate face normals at vertices (sum all face normals sharing a vertex)
    # Group by vertex index so all corners of the same vertex accumulate together
    accumulate_normals = node_tree.nodes.new("GeometryNodeAccumulateField")
    accumulate_normals.data_type = "FLOAT_VECTOR"  # type: ignore[attr-defined]
    accumulate_normals.domain = "CORNER"  # type: ignore[attr-defined]
    accumulate_normals.label = "Accumulate Face Normals"
    accumulate_normals.select = False

    # Normalize the accumulated normal to get smooth vertex normal
    normalize_smooth_normal = node_tree.nodes.new("ShaderNodeVectorMath")
    normalize_smooth_normal.operation = "NORMALIZE"  # type: ignore[attr-defined]
    normalize_smooth_normal.label = "Normalize Smooth Normal"
    normalize_smooth_normal.select = False

    # Switch between raw normal and smooth normal based on toggle
    # NOTE: Only the corner normal is smoothed. Face normal is NEVER smoothed
    # because it's used as a stable per-face reference for seam correction.
    smooth_normal_switch = node_tree.nodes.new("GeometryNodeSwitch")
    smooth_normal_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    smooth_normal_switch.label = "Smooth Normal Switch"
    smooth_normal_switch.select = False

    # Transform position relative to UV map origin
    # First subtract position, then apply inverse rotation, then divide by size
    subtract_pos = node_tree.nodes.new("ShaderNodeVectorMath")
    subtract_pos.operation = "SUBTRACT"  # type: ignore[attr-defined]
    subtract_pos.label = "Offset Position"
    subtract_pos.select = False

    # Invert rotation for applying inverse transform
    invert_rot = node_tree.nodes.new("FunctionNodeInvertRotation")
    invert_rot.label = "Invert Rotation"
    invert_rot.select = False

    # Rotate position by inverse rotation
    rotate_pos = node_tree.nodes.new("FunctionNodeRotateVector")
    rotate_pos.label = "Rotate Position"
    rotate_pos.select = False

    # Divide by size
    divide_size = node_tree.nodes.new("ShaderNodeVectorMath")
    divide_size.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_size.label = "Scale by Size"
    divide_size.select = False

    # Rotate normal by inverse rotation
    rotate_normal = node_tree.nodes.new("FunctionNodeRotateVector")
    rotate_normal.label = "Rotate Normal"
    rotate_normal.select = False

    # Rotate face normal by inverse rotation (for box mapping)
    rotate_face_normal = node_tree.nodes.new("FunctionNodeRotateVector")
    rotate_face_normal.label = "Rotate Face Normal"
    rotate_face_normal.select = False

    # For normal-based mapping: transform normals by inverse scale and renormalize
    # This handles non-uniform scaling correctly (inverse transpose of scale matrix)
    # Step 1: Divide rotated normal by size
    scale_normal = node_tree.nodes.new("ShaderNodeVectorMath")
    scale_normal.operation = "DIVIDE"  # type: ignore[attr-defined]
    scale_normal.label = "Scale Normal (1/Size)"
    scale_normal.select = False

    # Step 2: Normalize the result
    normalize_normal = node_tree.nodes.new("ShaderNodeVectorMath")
    normalize_normal.operation = "NORMALIZE"  # type: ignore[attr-defined]
    normalize_normal.label = "Normalize Scaled Normal"
    normalize_normal.select = False

    # Same for face normal
    scale_face_normal = node_tree.nodes.new("ShaderNodeVectorMath")
    scale_face_normal.operation = "DIVIDE"  # type: ignore[attr-defined]
    scale_face_normal.label = "Scale Face Normal (1/Size)"
    scale_face_normal.select = False

    normalize_face_normal = node_tree.nodes.new("ShaderNodeVectorMath")
    normalize_face_normal.operation = "NORMALIZE"  # type: ignore[attr-defined]
    normalize_face_normal.label = "Normalize Scaled Face Normal"
    normalize_face_normal.select = False

    # Transform face position for seam correction (cylindrical/spherical)
    # Face position needs the same transform as vertex position
    subtract_face_pos = node_tree.nodes.new("ShaderNodeVectorMath")
    subtract_face_pos.operation = "SUBTRACT"  # type: ignore[attr-defined]
    subtract_face_pos.label = "Offset Face Position"
    subtract_face_pos.select = False

    rotate_face_pos = node_tree.nodes.new("FunctionNodeRotateVector")
    rotate_face_pos.label = "Rotate Face Position"
    rotate_face_pos.select = False

    divide_face_size = node_tree.nodes.new("ShaderNodeVectorMath")
    divide_face_size.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_face_size.label = "Scale Face by Size"
    divide_face_size.select = False

    # Create mapping sub-groups
    planar_group = _create_planar_mapping_group(node_tree)
    cylindrical_group = _create_cylindrical_mapping_group(node_tree)
    cylindrical_capped_group = _create_cylindrical_capped_mapping_group(node_tree)
    spherical_group = _create_spherical_mapping_group(node_tree)
    shrink_wrap_group = _create_shrink_wrap_mapping_group(node_tree)
    box_group = _create_box_mapping_group(node_tree)
    cylindrical_normal_group = _create_cylindrical_normal_mapping_group(node_tree)
    cylindrical_capped_normal_group = _create_cylindrical_capped_normal_mapping_group(
        node_tree
    )
    spherical_normal_group = _create_spherical_normal_mapping_group(node_tree)
    shrink_wrap_normal_group = _create_shrink_wrap_normal_mapping_group(node_tree)

    # Create group nodes for each mapping type
    planar_node = node_tree.nodes.new("GeometryNodeGroup")
    planar_node.node_tree = planar_group  # type: ignore[attr-defined]
    planar_node.label = "Planar"
    planar_node.select = False

    cylindrical_node = node_tree.nodes.new("GeometryNodeGroup")
    cylindrical_node.node_tree = cylindrical_group  # type: ignore[attr-defined]
    cylindrical_node.label = "Cylindrical"
    cylindrical_node.select = False

    cylindrical_capped_node = node_tree.nodes.new("GeometryNodeGroup")
    cylindrical_capped_node.node_tree = cylindrical_capped_group  # type: ignore[attr-defined]
    cylindrical_capped_node.label = "Cylindrical Capped"
    cylindrical_capped_node.select = False

    spherical_node = node_tree.nodes.new("GeometryNodeGroup")
    spherical_node.node_tree = spherical_group  # type: ignore[attr-defined]
    spherical_node.label = "Spherical"
    spherical_node.select = False

    shrink_wrap_node = node_tree.nodes.new("GeometryNodeGroup")
    shrink_wrap_node.node_tree = shrink_wrap_group  # type: ignore[attr-defined]
    shrink_wrap_node.label = "Shrink Wrap"
    shrink_wrap_node.select = False

    box_node = node_tree.nodes.new("GeometryNodeGroup")
    box_node.node_tree = box_group  # type: ignore[attr-defined]
    box_node.label = "Box"
    box_node.select = False

    cylindrical_normal_node = node_tree.nodes.new("GeometryNodeGroup")
    cylindrical_normal_node.node_tree = cylindrical_normal_group  # type: ignore[attr-defined]
    cylindrical_normal_node.label = "Cylindrical (Normal)"
    cylindrical_normal_node.select = False

    cylindrical_capped_normal_node = node_tree.nodes.new("GeometryNodeGroup")
    cylindrical_capped_normal_node.node_tree = cylindrical_capped_normal_group  # type: ignore[attr-defined]
    cylindrical_capped_normal_node.label = "Cylindrical Capped (Normal)"
    cylindrical_capped_normal_node.select = False

    spherical_normal_node = node_tree.nodes.new("GeometryNodeGroup")
    spherical_normal_node.node_tree = spherical_normal_group  # type: ignore[attr-defined]
    spherical_normal_node.label = "Spherical (Normal)"
    spherical_normal_node.select = False

    shrink_wrap_normal_node = node_tree.nodes.new("GeometryNodeGroup")
    shrink_wrap_normal_node.node_tree = shrink_wrap_normal_group  # type: ignore[attr-defined]
    shrink_wrap_normal_node.label = "Shrink Wrap (Normal)"
    shrink_wrap_normal_node.select = False

    # Switch between cylindrical and cylindrical capped based on Cap toggle
    cylindrical_cap_switch = node_tree.nodes.new("GeometryNodeSwitch")
    cylindrical_cap_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    cylindrical_cap_switch.label = "Cylindrical Cap Switch"
    cylindrical_cap_switch.select = False

    # Switch between cylindrical normal and cylindrical capped normal based on Cap toggle
    cylindrical_normal_cap_switch = node_tree.nodes.new("GeometryNodeSwitch")
    cylindrical_normal_cap_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    cylindrical_normal_cap_switch.label = "Cylindrical Normal Cap Switch"
    cylindrical_normal_cap_switch.select = False

    # Switch between position-based and normal-based cylindrical based on Normal-based toggle
    cylindrical_normal_switch = node_tree.nodes.new("GeometryNodeSwitch")
    cylindrical_normal_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    cylindrical_normal_switch.label = "Cylindrical Normal-based Switch"
    cylindrical_normal_switch.select = False

    # Switch between position-based and normal-based spherical based on Normal-based toggle
    spherical_normal_switch = node_tree.nodes.new("GeometryNodeSwitch")
    spherical_normal_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    spherical_normal_switch.label = "Spherical Normal-based Switch"
    spherical_normal_switch.select = False

    # Switch between position-based and normal-based shrink wrap based on Normal-based toggle
    shrink_wrap_normal_switch = node_tree.nodes.new("GeometryNodeSwitch")
    shrink_wrap_normal_switch.input_type = "VECTOR"  # type: ignore[attr-defined]
    shrink_wrap_normal_switch.label = "Shrink Wrap Normal-based Switch"
    shrink_wrap_normal_switch.select = False

    # Menu switch for UV output selection
    uv_switch = node_tree.nodes.new("GeometryNodeMenuSwitch")
    uv_switch.label = "Select UV"
    uv_switch.data_type = "VECTOR"  # type: ignore[attr-defined]
    uv_switch.select = False

    # Configure UV switch menu items to match mapping types
    # Use identifier as both name and display - this ensures consistent ordering
    uv_switch.enum_definition.enum_items.clear()  # type: ignore[attr-defined]
    for identifier, display_name, description in MAPPING_TYPES:
        item = uv_switch.enum_definition.enum_items.new(identifier)  # type: ignore[attr-defined]
        item.name = display_name
        item.description = description

    # Set default to first item (Planar)
    # Note: Blender's enum_items are 2-indexed internally, so 2 = first item
    uv_switch.active_index = 2  # type: ignore[attr-defined]

    # Apply tiling and flip to UVs
    # Separate UV components
    separate_uv = node_tree.nodes.new("ShaderNodeSeparateXYZ")
    separate_uv.label = "Separate UV"
    separate_uv.select = False

    # Multiply by tile amounts
    multiply_u = node_tree.nodes.new("ShaderNodeMath")
    multiply_u.operation = "MULTIPLY"  # type: ignore[attr-defined]
    multiply_u.label = "U x Tile"
    multiply_u.select = False

    multiply_v = node_tree.nodes.new("ShaderNodeMath")
    multiply_v.operation = "MULTIPLY"  # type: ignore[attr-defined]
    multiply_v.label = "V x Tile"
    multiply_v.select = False

    # UV Rotation: rotate around origin
    # u' = u * cos(θ) - v * sin(θ)
    # v' = u * sin(θ) + v * cos(θ)
    rot_cos = node_tree.nodes.new("ShaderNodeMath")
    rot_cos.operation = "COSINE"  # type: ignore[attr-defined]
    rot_cos.label = "cos(θ)"
    rot_cos.select = False

    rot_sin = node_tree.nodes.new("ShaderNodeMath")
    rot_sin.operation = "SINE"  # type: ignore[attr-defined]
    rot_sin.label = "sin(θ)"
    rot_sin.select = False

    # u' = u*cos - v*sin
    rot_u_cos = node_tree.nodes.new("ShaderNodeMath")
    rot_u_cos.operation = "MULTIPLY"  # type: ignore[attr-defined]
    rot_u_cos.label = "U * cos"
    rot_u_cos.select = False

    rot_v_sin = node_tree.nodes.new("ShaderNodeMath")
    rot_v_sin.operation = "MULTIPLY"  # type: ignore[attr-defined]
    rot_v_sin.label = "V * sin"
    rot_v_sin.select = False

    rot_u_final = node_tree.nodes.new("ShaderNodeMath")
    rot_u_final.operation = "SUBTRACT"  # type: ignore[attr-defined]
    rot_u_final.label = "U' = U*cos - V*sin"
    rot_u_final.select = False

    # v' = u*sin + v*cos
    rot_u_sin = node_tree.nodes.new("ShaderNodeMath")
    rot_u_sin.operation = "MULTIPLY"  # type: ignore[attr-defined]
    rot_u_sin.label = "U * sin"
    rot_u_sin.select = False

    rot_v_cos = node_tree.nodes.new("ShaderNodeMath")
    rot_v_cos.operation = "MULTIPLY"  # type: ignore[attr-defined]
    rot_v_cos.label = "V * cos"
    rot_v_cos.select = False

    rot_v_final = node_tree.nodes.new("ShaderNodeMath")
    rot_v_final.operation = "ADD"  # type: ignore[attr-defined]
    rot_v_final.label = "V' = U*sin + V*cos"
    rot_v_final.select = False

    # Flip U: tile * (1 - u) = tile - tile*u
    # We compute: tile - (u * tile) when flip is enabled
    flip_u_sub = node_tree.nodes.new("ShaderNodeMath")
    flip_u_sub.operation = "SUBTRACT"  # type: ignore[attr-defined]
    flip_u_sub.label = "Tile - U*Tile"
    flip_u_sub.select = False

    flip_u_switch = node_tree.nodes.new("GeometryNodeSwitch")
    flip_u_switch.input_type = "FLOAT"  # type: ignore[attr-defined]
    flip_u_switch.label = "Flip U Switch"
    flip_u_switch.select = False

    # Flip V: tile * (1 - v) = tile - tile*v
    flip_v_sub = node_tree.nodes.new("ShaderNodeMath")
    flip_v_sub.operation = "SUBTRACT"  # type: ignore[attr-defined]
    flip_v_sub.label = "Tile - V*Tile"
    flip_v_sub.select = False

    flip_v_switch = node_tree.nodes.new("GeometryNodeSwitch")
    flip_v_switch.input_type = "FLOAT"  # type: ignore[attr-defined]
    flip_v_switch.label = "Flip V Switch"
    flip_v_switch.select = False

    # Offset U (applied after flip)
    offset_u = node_tree.nodes.new("ShaderNodeMath")
    offset_u.operation = "ADD"  # type: ignore[attr-defined]
    offset_u.label = "U + Offset"
    offset_u.select = False

    # Offset V (applied after flip)
    offset_v = node_tree.nodes.new("ShaderNodeMath")
    offset_v.operation = "ADD"  # type: ignore[attr-defined]
    offset_v.label = "V + Offset"
    offset_v.select = False

    # Combine back to UV
    combine_uv = node_tree.nodes.new("ShaderNodeCombineXYZ")
    combine_uv.label = "Combine UV"
    combine_uv.select = False

    # Store Named Attribute for UV
    store_uv = node_tree.nodes.new("GeometryNodeStoreNamedAttribute")
    store_uv.data_type = "FLOAT2"  # type: ignore[attr-defined]
    store_uv.domain = "CORNER"  # type: ignore[attr-defined]
    store_uv.label = "Store UV"
    store_uv.select = False

    # Connect nodes

    # Position transformation chain
    node_tree.links.new(position_node.outputs["Position"], subtract_pos.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_POSITION], subtract_pos.inputs[1])

    node_tree.links.new(
        input_node.outputs[SOCKET_ROTATION], invert_rot.inputs["Rotation"]
    )
    node_tree.links.new(subtract_pos.outputs["Vector"], rotate_pos.inputs["Vector"])
    node_tree.links.new(invert_rot.outputs["Rotation"], rotate_pos.inputs["Rotation"])

    node_tree.links.new(rotate_pos.outputs["Vector"], divide_size.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_SIZE], divide_size.inputs[1])

    # Normal transformation (uses smooth normal if enabled)
    node_tree.links.new(
        smooth_normal_switch.outputs["Output"], rotate_normal.inputs["Vector"]
    )
    node_tree.links.new(
        invert_rot.outputs["Rotation"], rotate_normal.inputs["Rotation"]
    )

    # Face normal evaluation for box mapping
    # Get the face normal by sampling at face domain, then looking it up by face index
    # This gives a constant normal per face, avoiding interpolation artifacts at boundaries
    node_tree.links.new(
        face_normal_node.outputs["Normal"], evaluate_face_normal.inputs["Value"]
    )
    node_tree.links.new(
        face_of_corner.outputs["Face Index"], evaluate_face_normal.inputs["Index"]
    )
    node_tree.links.new(
        corner_index_node.outputs["Index"], face_of_corner.inputs["Corner Index"]
    )

    # === Smooth Normals Computation Connections ===
    # Compute smooth vertex normals by averaging face normals at each vertex
    node_tree.links.new(
        corner_index_node.outputs["Index"], vertex_of_corner.inputs["Corner Index"]
    )
    # Get face normal at each corner
    node_tree.links.new(
        face_normal_node.outputs["Normal"], face_normal_at_corner.inputs["Value"]
    )
    node_tree.links.new(
        face_of_corner.outputs["Face Index"], face_normal_at_corner.inputs["Index"]
    )
    # Accumulate face normals at each vertex (group by vertex index)
    node_tree.links.new(
        face_normal_at_corner.outputs["Value"], accumulate_normals.inputs["Value"]
    )
    node_tree.links.new(
        vertex_of_corner.outputs["Vertex Index"],
        accumulate_normals.inputs["Group ID"],
    )
    # Normalize to get smooth vertex normal
    node_tree.links.new(
        accumulate_normals.outputs["Total"], normalize_smooth_normal.inputs[0]
    )
    # Switch: False = raw normal, True = smooth normal
    node_tree.links.new(
        input_node.outputs[SOCKET_SMOOTH_NORMALS],
        smooth_normal_switch.inputs["Switch"],
    )
    node_tree.links.new(
        normal_node.outputs["Normal"], smooth_normal_switch.inputs["False"]
    )
    node_tree.links.new(
        normalize_smooth_normal.outputs["Vector"], smooth_normal_switch.inputs["True"]
    )

    # Rotate the face normal (NEVER smoothed - used for seam correction)
    # Face normal must be constant per face to provide stable reference for detecting seams
    node_tree.links.new(
        evaluate_face_normal.outputs["Value"], rotate_face_normal.inputs["Vector"]
    )
    node_tree.links.new(
        invert_rot.outputs["Rotation"], rotate_face_normal.inputs["Rotation"]
    )

    # Scale-adjusted normals for normal-based mapping (handles non-uniform scale)
    # Normal transformation for non-uniform scale: n' = normalize(n / scale)
    node_tree.links.new(rotate_normal.outputs["Vector"], scale_normal.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_SIZE], scale_normal.inputs[1])
    node_tree.links.new(scale_normal.outputs["Vector"], normalize_normal.inputs[0])

    # Same for face normal
    node_tree.links.new(
        rotate_face_normal.outputs["Vector"], scale_face_normal.inputs[0]
    )
    node_tree.links.new(input_node.outputs[SOCKET_SIZE], scale_face_normal.inputs[1])
    node_tree.links.new(
        scale_face_normal.outputs["Vector"], normalize_face_normal.inputs[0]
    )

    # Face position evaluation for seam correction (cylindrical/spherical)
    # Same index lookup as face normal
    node_tree.links.new(
        face_position_node.outputs["Position"], evaluate_face_position.inputs["Value"]
    )
    node_tree.links.new(
        face_of_corner.outputs["Face Index"], evaluate_face_position.inputs["Index"]
    )
    # Transform face position same as vertex position
    node_tree.links.new(
        evaluate_face_position.outputs["Value"], subtract_face_pos.inputs[0]
    )
    node_tree.links.new(
        input_node.outputs[SOCKET_POSITION], subtract_face_pos.inputs[1]
    )
    node_tree.links.new(
        subtract_face_pos.outputs["Vector"], rotate_face_pos.inputs["Vector"]
    )
    node_tree.links.new(
        invert_rot.outputs["Rotation"], rotate_face_pos.inputs["Rotation"]
    )
    node_tree.links.new(rotate_face_pos.outputs["Vector"], divide_face_size.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_SIZE], divide_face_size.inputs[1])

    # Connect transformed position/normal to mapping nodes
    node_tree.links.new(divide_size.outputs["Vector"], planar_node.inputs["Position"])
    node_tree.links.new(rotate_normal.outputs["Vector"], planar_node.inputs["Normal"])

    # Cylindrical mapping with face position for seam correction
    node_tree.links.new(
        divide_size.outputs["Vector"], cylindrical_node.inputs["Position"]
    )
    node_tree.links.new(
        rotate_normal.outputs["Vector"], cylindrical_node.inputs["Normal"]
    )
    node_tree.links.new(
        divide_face_size.outputs["Vector"], cylindrical_node.inputs["Face Position"]
    )

    # Cylindrical capped mapping (uses face position Z to detect caps)
    node_tree.links.new(
        divide_size.outputs["Vector"], cylindrical_capped_node.inputs["Position"]
    )
    node_tree.links.new(
        divide_face_size.outputs["Vector"],
        cylindrical_capped_node.inputs["Face Position"],
    )

    # Spherical mapping with face position for seam correction
    node_tree.links.new(
        divide_size.outputs["Vector"], spherical_node.inputs["Position"]
    )
    node_tree.links.new(
        rotate_normal.outputs["Vector"], spherical_node.inputs["Normal"]
    )
    node_tree.links.new(
        divide_face_size.outputs["Vector"], spherical_node.inputs["Face Position"]
    )

    # Shrink wrap mapping with face position for seam correction
    node_tree.links.new(
        divide_size.outputs["Vector"], shrink_wrap_node.inputs["Position"]
    )
    node_tree.links.new(
        divide_face_size.outputs["Vector"], shrink_wrap_node.inputs["Face Position"]
    )

    # Box mapping uses face normal for projection selection (avoids interpolation issues)
    node_tree.links.new(divide_size.outputs["Vector"], box_node.inputs["Position"])
    node_tree.links.new(rotate_face_normal.outputs["Vector"], box_node.inputs["Normal"])

    # Cylindrical (Normal) mapping - uses scale-adjusted normals
    # For non-uniform scale, normal = normalize(rotated_normal / scale)
    node_tree.links.new(
        normalize_normal.outputs["Vector"], cylindrical_normal_node.inputs["Normal"]
    )
    node_tree.links.new(
        normalize_face_normal.outputs["Vector"],
        cylindrical_normal_node.inputs["Face Normal"],
    )

    # Cylindrical Capped (Normal) mapping - uses scale-adjusted normals
    node_tree.links.new(
        normalize_normal.outputs["Vector"],
        cylindrical_capped_normal_node.inputs["Normal"],
    )
    node_tree.links.new(
        normalize_face_normal.outputs["Vector"],
        cylindrical_capped_normal_node.inputs["Face Normal"],
    )

    # Spherical (Normal) mapping - uses scale-adjusted normals
    node_tree.links.new(
        normalize_normal.outputs["Vector"], spherical_normal_node.inputs["Normal"]
    )
    node_tree.links.new(
        normalize_face_normal.outputs["Vector"],
        spherical_normal_node.inputs["Face Normal"],
    )

    # Shrink Wrap (Normal) mapping - uses scale-adjusted normals
    node_tree.links.new(
        normalize_normal.outputs["Vector"], shrink_wrap_normal_node.inputs["Normal"]
    )
    node_tree.links.new(
        normalize_face_normal.outputs["Vector"],
        shrink_wrap_normal_node.inputs["Face Normal"],
    )

    # Connect mapping outputs to UV switch
    # Note: Menu Switch socket inputs are named by the display name passed to enum_items.new()
    node_tree.links.new(
        input_node.outputs[SOCKET_MAPPING_TYPE], uv_switch.inputs["Menu"]
    )

    # Cylindrical cap switch: select between cylindrical and cylindrical_capped
    node_tree.links.new(
        input_node.outputs[SOCKET_CAP], cylindrical_cap_switch.inputs["Switch"]
    )
    node_tree.links.new(
        cylindrical_node.outputs["UV"], cylindrical_cap_switch.inputs["False"]
    )
    node_tree.links.new(
        cylindrical_capped_node.outputs["UV"], cylindrical_cap_switch.inputs["True"]
    )

    # Cylindrical (Normal) cap switch: select between cylindrical_normal and
    # cylindrical_capped_normal
    node_tree.links.new(
        input_node.outputs[SOCKET_CAP], cylindrical_normal_cap_switch.inputs["Switch"]
    )
    node_tree.links.new(
        cylindrical_normal_node.outputs["UV"],
        cylindrical_normal_cap_switch.inputs["False"],
    )
    node_tree.links.new(
        cylindrical_capped_normal_node.outputs["UV"],
        cylindrical_normal_cap_switch.inputs["True"],
    )

    # Cylindrical normal-based switch: select between position-based and normal-based
    node_tree.links.new(
        input_node.outputs[SOCKET_NORMAL_BASED],
        cylindrical_normal_switch.inputs["Switch"],
    )
    node_tree.links.new(
        cylindrical_cap_switch.outputs["Output"],
        cylindrical_normal_switch.inputs["False"],
    )
    node_tree.links.new(
        cylindrical_normal_cap_switch.outputs["Output"],
        cylindrical_normal_switch.inputs["True"],
    )

    # Spherical normal-based switch: select between position-based and normal-based
    node_tree.links.new(
        input_node.outputs[SOCKET_NORMAL_BASED],
        spherical_normal_switch.inputs["Switch"],
    )
    node_tree.links.new(
        spherical_node.outputs["UV"], spherical_normal_switch.inputs["False"]
    )
    node_tree.links.new(
        spherical_normal_node.outputs["UV"], spherical_normal_switch.inputs["True"]
    )

    # Shrink wrap normal-based switch: select between position-based and normal-based
    node_tree.links.new(
        input_node.outputs[SOCKET_NORMAL_BASED],
        shrink_wrap_normal_switch.inputs["Switch"],
    )
    node_tree.links.new(
        shrink_wrap_node.outputs["UV"], shrink_wrap_normal_switch.inputs["False"]
    )
    node_tree.links.new(
        shrink_wrap_normal_node.outputs["UV"], shrink_wrap_normal_switch.inputs["True"]
    )

    # Use indices for inputs since socket names may vary - inputs are: Menu, then one per enum item
    # Item 0 = Planar, Item 1 = Cylindrical (with cap and normal-based switches),
    # Item 2 = Spherical (with normal-based switch), Item 3 = Shrink Wrap (with normal-based switch),
    # Item 4 = Box
    node_tree.links.new(planar_node.outputs["UV"], uv_switch.inputs[1])
    node_tree.links.new(
        cylindrical_normal_switch.outputs["Output"], uv_switch.inputs[2]
    )
    node_tree.links.new(spherical_normal_switch.outputs["Output"], uv_switch.inputs[3])
    node_tree.links.new(
        shrink_wrap_normal_switch.outputs["Output"], uv_switch.inputs[4]
    )
    node_tree.links.new(box_node.outputs["UV"], uv_switch.inputs[5])

    # UV tiling and flip chain
    node_tree.links.new(uv_switch.outputs["Output"], separate_uv.inputs["Vector"])

    node_tree.links.new(separate_uv.outputs["X"], multiply_u.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_U_TILE], multiply_u.inputs[1])

    node_tree.links.new(separate_uv.outputs["Y"], multiply_v.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_V_TILE], multiply_v.inputs[1])

    # UV Rotation (applied after tiling, before flip)
    # u' = u*cos(θ) - v*sin(θ), v' = u*sin(θ) + v*cos(θ)
    node_tree.links.new(input_node.outputs[SOCKET_UV_ROTATION], rot_cos.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_UV_ROTATION], rot_sin.inputs[0])

    node_tree.links.new(multiply_u.outputs["Value"], rot_u_cos.inputs[0])
    node_tree.links.new(rot_cos.outputs["Value"], rot_u_cos.inputs[1])

    node_tree.links.new(multiply_v.outputs["Value"], rot_v_sin.inputs[0])
    node_tree.links.new(rot_sin.outputs["Value"], rot_v_sin.inputs[1])

    node_tree.links.new(rot_u_cos.outputs["Value"], rot_u_final.inputs[0])
    node_tree.links.new(rot_v_sin.outputs["Value"], rot_u_final.inputs[1])

    node_tree.links.new(multiply_u.outputs["Value"], rot_u_sin.inputs[0])
    node_tree.links.new(rot_sin.outputs["Value"], rot_u_sin.inputs[1])

    node_tree.links.new(multiply_v.outputs["Value"], rot_v_cos.inputs[0])
    node_tree.links.new(rot_cos.outputs["Value"], rot_v_cos.inputs[1])

    node_tree.links.new(rot_u_sin.outputs["Value"], rot_v_final.inputs[0])
    node_tree.links.new(rot_v_cos.outputs["Value"], rot_v_final.inputs[1])

    # Flip U: tile - rotated_u (when flip enabled)
    node_tree.links.new(input_node.outputs[SOCKET_U_TILE], flip_u_sub.inputs[0])
    node_tree.links.new(rot_u_final.outputs["Value"], flip_u_sub.inputs[1])
    node_tree.links.new(
        input_node.outputs[SOCKET_U_FLIP], flip_u_switch.inputs["Switch"]
    )
    node_tree.links.new(rot_u_final.outputs["Value"], flip_u_switch.inputs["False"])
    node_tree.links.new(flip_u_sub.outputs["Value"], flip_u_switch.inputs["True"])

    # Flip V: tile - rotated_v (when flip enabled)
    node_tree.links.new(input_node.outputs[SOCKET_V_TILE], flip_v_sub.inputs[0])
    node_tree.links.new(rot_v_final.outputs["Value"], flip_v_sub.inputs[1])
    node_tree.links.new(
        input_node.outputs[SOCKET_V_FLIP], flip_v_switch.inputs["Switch"]
    )
    node_tree.links.new(rot_v_final.outputs["Value"], flip_v_switch.inputs["False"])
    node_tree.links.new(flip_v_sub.outputs["Value"], flip_v_switch.inputs["True"])

    # Offset U and V (applied after flip)
    node_tree.links.new(flip_u_switch.outputs["Output"], offset_u.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_U_OFFSET], offset_u.inputs[1])
    node_tree.links.new(flip_v_switch.outputs["Output"], offset_v.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_V_OFFSET], offset_v.inputs[1])

    # Combine UV (from offset outputs)
    node_tree.links.new(offset_u.outputs["Value"], combine_uv.inputs["X"])
    node_tree.links.new(offset_v.outputs["Value"], combine_uv.inputs["Y"])

    # Store UV attribute
    node_tree.links.new(
        input_node.outputs[SOCKET_GEOMETRY], store_uv.inputs["Geometry"]
    )
    node_tree.links.new(
        input_node.outputs[SOCKET_SELECTION], store_uv.inputs["Selection"]
    )
    node_tree.links.new(input_node.outputs[SOCKET_UV_MAP], store_uv.inputs["Name"])
    node_tree.links.new(combine_uv.outputs["Vector"], store_uv.inputs["Value"])

    # === Gizmo System ===
    # Create multiple gizmos with different axis configurations
    # Note: Gizmos in Blender propagate backwards - they read from their Value input
    # and the Transform output joins into the geometry for visual display

    # Combine Transform to create transform matrix from Position/Rotation/Size
    combine_transform = node_tree.nodes.new("FunctionNodeCombineTransform")
    combine_transform.label = "Gizmo Transform"
    combine_transform.select = False

    # Connect Position/Rotation/Size to Combine Transform
    node_tree.links.new(
        input_node.outputs[SOCKET_POSITION], combine_transform.inputs["Translation"]
    )
    node_tree.links.new(
        input_node.outputs[SOCKET_ROTATION], combine_transform.inputs["Rotation"]
    )
    node_tree.links.new(
        input_node.outputs[SOCKET_SIZE], combine_transform.inputs["Scale"]
    )

    # Gizmo 1: All (position, rotation, size all enabled)
    gizmo_all = node_tree.nodes.new("GeometryNodeGizmoTransform")
    gizmo_all.label = "Gizmo All"
    gizmo_all.select = False
    # All axes enabled by default

    # Gizmo 2: Position only
    gizmo_position = node_tree.nodes.new("GeometryNodeGizmoTransform")
    gizmo_position.label = "Gizmo Position"
    gizmo_position.select = False
    gizmo_position.use_rotation_x = False  # type: ignore[attr-defined]
    gizmo_position.use_rotation_y = False  # type: ignore[attr-defined]
    gizmo_position.use_rotation_z = False  # type: ignore[attr-defined]
    gizmo_position.use_scale_x = False  # type: ignore[attr-defined]
    gizmo_position.use_scale_y = False  # type: ignore[attr-defined]
    gizmo_position.use_scale_z = False  # type: ignore[attr-defined]

    # Gizmo 3: Rotation only
    gizmo_rotation = node_tree.nodes.new("GeometryNodeGizmoTransform")
    gizmo_rotation.label = "Gizmo Rotation"
    gizmo_rotation.select = False
    gizmo_rotation.use_translation_x = False  # type: ignore[attr-defined]
    gizmo_rotation.use_translation_y = False  # type: ignore[attr-defined]
    gizmo_rotation.use_translation_z = False  # type: ignore[attr-defined]
    gizmo_rotation.use_scale_x = False  # type: ignore[attr-defined]
    gizmo_rotation.use_scale_y = False  # type: ignore[attr-defined]
    gizmo_rotation.use_scale_z = False  # type: ignore[attr-defined]

    # Gizmo 4: Size only
    gizmo_size = node_tree.nodes.new("GeometryNodeGizmoTransform")
    gizmo_size.label = "Gizmo Size"
    gizmo_size.select = False
    gizmo_size.use_translation_x = False  # type: ignore[attr-defined]
    gizmo_size.use_translation_y = False  # type: ignore[attr-defined]
    gizmo_size.use_translation_z = False  # type: ignore[attr-defined]
    gizmo_size.use_rotation_x = False  # type: ignore[attr-defined]
    gizmo_size.use_rotation_y = False  # type: ignore[attr-defined]
    gizmo_size.use_rotation_z = False  # type: ignore[attr-defined]

    # Connect transform to all gizmos
    node_tree.links.new(
        combine_transform.outputs["Transform"], gizmo_all.inputs["Value"]
    )
    node_tree.links.new(
        combine_transform.outputs["Transform"], gizmo_position.inputs["Value"]
    )
    node_tree.links.new(
        combine_transform.outputs["Transform"], gizmo_rotation.inputs["Value"]
    )
    node_tree.links.new(
        combine_transform.outputs["Transform"], gizmo_size.inputs["Value"]
    )

    # Connect Position and Rotation to gizmos for proper orientation
    for gizmo in [gizmo_all, gizmo_position, gizmo_rotation, gizmo_size]:
        node_tree.links.new(
            input_node.outputs[SOCKET_POSITION], gizmo.inputs["Position"]
        )
        node_tree.links.new(
            input_node.outputs[SOCKET_ROTATION], gizmo.inputs["Rotation"]
        )

    # Menu Switch for selecting which gizmo output to use
    gizmo_switch = node_tree.nodes.new("GeometryNodeMenuSwitch")
    gizmo_switch.label = "Gizmo Selection"
    gizmo_switch.data_type = "GEOMETRY"  # type: ignore[attr-defined]
    gizmo_switch.select = False

    # Configure gizmo switch menu items to match gizmo types
    gizmo_switch.enum_definition.enum_items.clear()  # type: ignore[attr-defined]
    for identifier, display_name, description in GIZMO_TYPES:
        item = gizmo_switch.enum_definition.enum_items.new(identifier)  # type: ignore[attr-defined]
        item.name = display_name
        item.description = description

    # Set default to first item (None)
    gizmo_switch.active_index = 2  # type: ignore[attr-defined]

    # Connect gizmo menu input
    node_tree.links.new(
        input_node.outputs[SOCKET_SHOW_GIZMO], gizmo_switch.inputs["Menu"]
    )

    # Connect gizmo outputs to switch
    # New order: Item 0 = None (empty), Item 1 = Position, Item 2 = Rotation, Item 3 = Size, Item 4 = All
    # Input 1 (None) left unconnected - will output empty geometry
    node_tree.links.new(gizmo_position.outputs["Transform"], gizmo_switch.inputs[2])
    node_tree.links.new(gizmo_rotation.outputs["Transform"], gizmo_switch.inputs[3])
    node_tree.links.new(gizmo_size.outputs["Transform"], gizmo_switch.inputs[4])
    node_tree.links.new(gizmo_all.outputs["Transform"], gizmo_switch.inputs[5])

    # Join selected gizmo transform output into geometry
    join_geometry = node_tree.nodes.new("GeometryNodeJoinGeometry")
    join_geometry.label = "Join Gizmo"
    join_geometry.select = False

    node_tree.links.new(store_uv.outputs["Geometry"], join_geometry.inputs["Geometry"])
    node_tree.links.new(
        gizmo_switch.outputs["Output"], join_geometry.inputs["Geometry"]
    )

    # Output
    node_tree.links.new(
        join_geometry.outputs["Geometry"], output_node.inputs[SOCKET_GEOMETRY]
    )

    # Layout nodes with tighter spacing
    layout_nodes_pcb_style(node_tree, cell_width=0.0, cell_height=150.0)


def create_uv_map_node_group() -> bpy.types.NodeTree:
    """Create the main UV Map node group with all mapping types.

    Returns:
        The newly created node tree.
    """
    # Generate unique name
    base_name = UV_MAP_NODE_GROUP_PREFIX
    name = base_name
    counter = 1
    while name in bpy.data.node_groups:
        name = f"{base_name}.{counter:03d}"
        counter += 1

    node_tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    node_tree.use_fake_user = False  # Let it be removed when unused
    node_tree.color_tag = "CONVERTER"  # Teal color for UV conversion

    # Tag the node group for identification
    node_tree[UV_MAP_NODE_GROUP_TAG] = True

    # Create interface
    _create_main_interface(node_tree)

    # Populate with nodes
    _populate_uv_map_node_group(node_tree)

    # Set default mapping type after nodes are created
    # (Menu enum items must exist first)
    _set_mapping_type_default(node_tree)
    _set_gizmo_default(node_tree)

    return node_tree


def regenerate_uv_map_node_group(  # noqa: PLR0912, PLR0915
    node_tree: bpy.types.NodeTree,
) -> None:
    """Regenerate an existing UV Map node group in place.

    Clears all nodes and the interface, then recreates them with the latest code.
    Preserves parameter values and connections from:
    - Modifiers using this node group directly
    - Group nodes in other node trees that reference this node group

    Note: Sub-groups (Planar, Cylindrical, etc.) should be deleted by the caller
    BEFORE calling this function if a full refresh is needed. This allows multiple
    main groups to share the same freshly created sub-groups.
    """
    import contextlib

    # Build socket name -> identifier mapping from current interface
    old_socket_map: dict[str, str] = {}
    old_output_socket_map: dict[str, str] = {}
    interface = node_tree.interface
    if interface is not None:
        for item in interface.items_tree:
            if getattr(item, "item_type", None) != "SOCKET":
                continue
            in_out = getattr(item, "in_out", None)
            if in_out == "INPUT":
                old_socket_map[item.name] = item.identifier  # type: ignore[union-attr]
            elif in_out == "OUTPUT":
                old_output_socket_map[item.name] = item.identifier  # type: ignore[union-attr]

    # Find all modifiers using this node group and save their values
    modifier_values: list[tuple[bpy.types.Modifier, dict[str, object]]] = []

    for obj in bpy.data.objects:
        for modifier in obj.modifiers:
            if modifier.type != "NODES":
                continue
            if getattr(modifier, "node_group", None) != node_tree:
                continue

            # Save values by socket NAME (not identifier, since those will change)
            values: dict[str, object] = {}
            for name, identifier in old_socket_map.items():
                value = modifier.get(identifier)
                if value is not None:
                    # Convert to a basic type to avoid issues with Blender types
                    if hasattr(value, "__len__") and not isinstance(value, str):
                        values[name] = tuple(value)
                    else:
                        values[name] = value

            modifier_values.append((modifier, values))

    # Find all group nodes in other node trees that reference this node group
    # Save their input values and connections
    # Type alias for group node data storage
    group_node_data: list[
        tuple[
            bpy.types.Node,  # The group node
            bpy.types.NodeTree,  # The parent node tree
            dict[str, object],  # Input socket default values by name
            list[tuple[str, bpy.types.Node, str]],  # Input links
            list[tuple[str, bpy.types.Node, str]],  # Output links
        ]
    ] = []

    for other_tree in bpy.data.node_groups:
        if other_tree == node_tree:
            continue
        for node in other_tree.nodes:
            if node.bl_idname != "GeometryNodeGroup":
                continue
            if getattr(node, "node_tree", None) != node_tree:
                continue

            # Save input default values
            input_values: dict[str, object] = {}
            for inp in node.inputs:
                default_val = getattr(inp, "default_value", None)
                if default_val is not None:
                    if hasattr(default_val, "__len__") and not isinstance(
                        default_val, str
                    ):
                        input_values[inp.name] = tuple(default_val)
                    else:
                        input_values[inp.name] = default_val

            # Save input connections
            input_links: list[tuple[str, bpy.types.Node, str]] = []
            for link in other_tree.links:
                if link.to_node == node:
                    from_node = link.from_node
                    from_socket = link.from_socket
                    to_socket = link.to_socket
                    if (
                        from_node is not None
                        and from_socket is not None
                        and to_socket is not None
                    ):
                        input_links.append(
                            (to_socket.name, from_node, from_socket.name)
                        )

            # Save output connections
            output_links: list[tuple[str, bpy.types.Node, str]] = []
            for link in other_tree.links:
                if link.from_node == node:
                    from_socket = link.from_socket
                    to_node = link.to_node
                    to_socket = link.to_socket
                    if (
                        from_socket is not None
                        and to_node is not None
                        and to_socket is not None
                    ):
                        output_links.append((from_socket.name, to_node, to_socket.name))

            group_node_data.append(
                (
                    node,
                    other_tree,
                    input_values,
                    input_links,
                    output_links,
                )
            )

    # Clear all existing nodes
    node_tree.nodes.clear()

    # Set color tag (may have been missing on older groups)
    node_tree.color_tag = "CONVERTER"

    # Clear and recreate interface (to pick up any socket changes)
    interface = node_tree.interface
    assert interface is not None
    interface.clear()
    _create_main_interface(node_tree)

    # Repopulate with fresh nodes
    _populate_uv_map_node_group(node_tree)

    # Set default mapping type after nodes are created
    # (Menu enum items must exist first)
    _set_mapping_type_default(node_tree)
    _set_gizmo_default(node_tree)

    # Build new socket name -> identifier mapping
    new_socket_map: dict[str, str] = {}
    for item in node_tree.interface.items_tree:  # type: ignore[union-attr]
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "in_out", None) != "INPUT":
            continue
        new_socket_map[item.name] = item.identifier  # type: ignore[union-attr]

    # Restore values to modifiers
    for modifier, values in modifier_values:
        for name, value in values.items():
            if name not in new_socket_map:
                continue  # Socket no longer exists
            new_identifier = new_socket_map[name]
            with contextlib.suppress(KeyError, TypeError):
                modifier[new_identifier] = value

    # Restore group node values and connections
    for node, parent_tree, input_values, input_links, output_links in group_node_data:
        # Restore input default values
        for inp in node.inputs:
            if inp.name in input_values:
                with contextlib.suppress(AttributeError, TypeError):
                    inp.default_value = input_values[inp.name]  # type: ignore[attr-defined]

        # Restore input connections
        for socket_name, from_node, from_socket_name in input_links:
            # Find the socket by name on the group node
            to_socket = node.inputs.get(socket_name)
            if to_socket is None:
                continue
            # Find the source socket
            from_socket = from_node.outputs.get(from_socket_name)
            if from_socket is None:
                continue
            with contextlib.suppress(RuntimeError):
                parent_tree.links.new(from_socket, to_socket)

        # Restore output connections
        for socket_name, to_node, to_socket_name in output_links:
            # Find the socket by name on the group node
            from_socket = node.outputs.get(socket_name)
            if from_socket is None:
                continue
            # Find the destination socket
            to_socket = to_node.inputs.get(to_socket_name)
            if to_socket is None:
                continue
            with contextlib.suppress(RuntimeError):
                parent_tree.links.new(from_socket, to_socket)


def is_uv_map_node_group(node_tree: bpy.types.NodeTree | None) -> bool:
    """Check if a node tree is a UV Map node group created by this addon."""
    if node_tree is None:
        return False
    return node_tree.get(UV_MAP_NODE_GROUP_TAG, False)


def get_uv_map_node_groups() -> list[bpy.types.NodeTree]:
    """Get all UV Map node groups in the blend file."""
    return [ng for ng in bpy.data.node_groups if is_uv_map_node_group(ng)]


def _cleanup_reference_groups(reference: bpy.types.NodeTree) -> None:
    """Clean up a reference node group and all its nested sub-groups."""
    # Collect all nested groups first
    groups_to_remove: list[bpy.types.NodeTree] = []
    for node in reference.nodes:
        node_tree = getattr(node, "node_tree", None)
        if node_tree is not None:
            groups_to_remove.append(node_tree)

    # Remove the main group
    bpy.data.node_groups.remove(reference)

    # Remove nested groups (they will have .001 etc suffixes from force_new)
    for group in groups_to_remove:
        if group.name in bpy.data.node_groups:
            bpy.data.node_groups.remove(group)


def needs_regeneration() -> bool:
    """Check if any UV Map node groups need regeneration.

    Creates a fresh reference node group and compares it structurally
    against existing UV Map node groups. If any differ, regeneration is needed.

    Returns:
        True if regeneration is needed, False otherwise.
    """
    global _force_new_subgroups

    existing_groups = get_uv_map_node_groups()
    if not existing_groups:
        return False

    # Create a fresh reference to compare against
    # Force creation of new sub-groups to get accurate comparison
    _force_new_subgroups = True
    try:
        reference = create_uv_map_node_group()
    finally:
        _force_new_subgroups = False

    try:
        for group in existing_groups:
            if not node_groups_match(group, reference):
                return True
        return False
    finally:
        # Clean up the reference group and its sub-groups
        _cleanup_reference_groups(reference)


# Classes to register (none for this module - it's utility only)
classes: list[type] = []
