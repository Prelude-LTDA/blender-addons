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
    MAPPING_TYPES,
    SOCKET_GEOMETRY,
    SOCKET_MAPPING_TYPE,
    SOCKET_POSITION,
    SOCKET_ROTATION,
    SOCKET_SIZE,
    SOCKET_U_FLIP,
    SOCKET_U_TILE,
    SOCKET_UV_MAP,
    SOCKET_V_FLIP,
    SOCKET_V_TILE,
    SOCKET_W_FLIP,
    SOCKET_W_TILE,
    UV_MAP_NODE_GROUP_PREFIX,
    UV_MAP_NODE_GROUP_TAG,
)
from .shared.node_layout import layout_nodes_pcb_style


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

    # Mapping type menu
    # Note: NodeSocketMenu default value is set by the Menu Switch node inside the group,
    # not on the interface socket itself
    interface.new_socket(
        name=SOCKET_MAPPING_TYPE,
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

    w_tile_socket = interface.new_socket(
        name=SOCKET_W_TILE,
        socket_type="NodeSocketFloat",
        in_out="INPUT",
    )
    w_tile_socket.default_value = DEFAULT_TILE[2]  # type: ignore[attr-defined]
    w_tile_socket.min_value = 0.001  # type: ignore[attr-defined]

    # Flip inputs
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

    interface.new_socket(
        name=SOCKET_W_FLIP,
        socket_type="NodeSocketBool",
        in_out="INPUT",
    )

    # UV Map attribute name
    uv_map_socket = interface.new_socket(
        name=SOCKET_UV_MAP,
        socket_type="NodeSocketString",
        in_out="INPUT",
    )
    uv_map_socket.default_value = "UVMap"  # type: ignore[attr-defined]


def _create_planar_mapping_group(
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for planar UV mapping."""
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Planar"

    # Check if group already exists
    if group_name in bpy.data.node_groups:
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

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # Connect
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])
    group.links.new(separate_xyz.outputs["X"], combine_xyz.inputs["X"])
    group.links.new(separate_xyz.outputs["Y"], combine_xyz.inputs["Y"])
    # Z stays 0 for UV
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_cylindrical_mapping_group(
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for cylindrical UV mapping."""
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Cylindrical"

    if group_name in bpy.data.node_groups:
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

    # Cylindrical: U = atan2(x, y) / (2*pi) + 0.5, V = z
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Position"

    # atan2 for angle - use Math node with ARCTAN2
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(X, Y)"

    # Divide by 2*pi
    divide_node = group.nodes.new("ShaderNodeMath")
    divide_node.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_node.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_node.label = "/ 2π"

    # Add 0.5 to shift from [-0.5, 0.5] to [0, 1]
    add_node = group.nodes.new("ShaderNodeMath")
    add_node.operation = "ADD"  # type: ignore[attr-defined]
    add_node.inputs[1].default_value = 0.5  # type: ignore[index]
    add_node.label = "+ 0.5"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # Connect
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_node.inputs[0])
    group.links.new(divide_node.outputs["Value"], add_node.inputs[0])
    group.links.new(add_node.outputs["Value"], combine_xyz.inputs["X"])  # U
    group.links.new(separate_xyz.outputs["Z"], combine_xyz.inputs["Y"])  # V = Z
    group.links.new(combine_xyz.outputs["Vector"], output_node.inputs["UV"])

    layout_nodes_pcb_style(group, cell_width=0.0, cell_height=150.0)
    return group


def _create_spherical_mapping_group(  # noqa: PLR0915
    node_tree: bpy.types.NodeTree,  # noqa: ARG001
) -> bpy.types.NodeTree:
    """Create a node group for spherical UV mapping."""
    group_name = f"{UV_MAP_NODE_GROUP_PREFIX} - Spherical"

    if group_name in bpy.data.node_groups:
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

    # Spherical: convert to spherical coordinates
    # U = atan2(x, y) / (2*pi) + 0.5
    # V = acos(z / length) / pi
    separate_xyz = group.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.label = "Separate Position"

    # Length of position vector
    length_node = group.nodes.new("ShaderNodeVectorMath")
    length_node.operation = "LENGTH"  # type: ignore[attr-defined]
    length_node.label = "Length"

    # atan2 for azimuth (U)
    atan2_node = group.nodes.new("ShaderNodeMath")
    atan2_node.operation = "ARCTAN2"  # type: ignore[attr-defined]
    atan2_node.label = "atan2(X, Y)"

    divide_2pi = group.nodes.new("ShaderNodeMath")
    divide_2pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_2pi.inputs[1].default_value = 2.0 * math.pi  # type: ignore[index]
    divide_2pi.label = "/ 2π"

    add_half = group.nodes.new("ShaderNodeMath")
    add_half.operation = "ADD"  # type: ignore[attr-defined]
    add_half.inputs[1].default_value = 0.5  # type: ignore[index]
    add_half.label = "+ 0.5"

    # z / length
    divide_z_len = group.nodes.new("ShaderNodeMath")
    divide_z_len.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_z_len.label = "Z / Length"

    # Clamp to [-1, 1] to avoid NaN in acos
    clamp_node = group.nodes.new("ShaderNodeClamp")
    clamp_node.inputs["Min"].default_value = -1.0  # type: ignore[index]
    clamp_node.inputs["Max"].default_value = 1.0  # type: ignore[index]
    clamp_node.label = "Clamp"

    # acos for elevation (V)
    acos_node = group.nodes.new("ShaderNodeMath")
    acos_node.operation = "ARCCOSINE"  # type: ignore[attr-defined]
    acos_node.label = "acos"

    divide_pi = group.nodes.new("ShaderNodeMath")
    divide_pi.operation = "DIVIDE"  # type: ignore[attr-defined]
    divide_pi.inputs[1].default_value = math.pi  # type: ignore[index]
    divide_pi.label = "/ π"

    combine_xyz = group.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz.label = "Combine UV"

    # Connect
    group.links.new(input_node.outputs["Position"], separate_xyz.inputs["Vector"])
    group.links.new(input_node.outputs["Position"], length_node.inputs[0])

    # U calculation
    group.links.new(separate_xyz.outputs["X"], atan2_node.inputs[0])
    group.links.new(separate_xyz.outputs["Y"], atan2_node.inputs[1])
    group.links.new(atan2_node.outputs["Value"], divide_2pi.inputs[0])
    group.links.new(divide_2pi.outputs["Value"], add_half.inputs[0])
    group.links.new(add_half.outputs["Value"], combine_xyz.inputs["X"])

    # V calculation
    group.links.new(separate_xyz.outputs["Z"], divide_z_len.inputs[0])
    group.links.new(length_node.outputs["Value"], divide_z_len.inputs[1])
    group.links.new(divide_z_len.outputs["Value"], clamp_node.inputs["Value"])
    group.links.new(clamp_node.outputs["Result"], acos_node.inputs[0])
    group.links.new(acos_node.outputs["Value"], divide_pi.inputs[0])
    group.links.new(divide_pi.outputs["Value"], combine_xyz.inputs["Y"])

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

    if group_name in bpy.data.node_groups:
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

    # |Nx| >= |Ny|
    nx_ge_ny = group.nodes.new("ShaderNodeMath")
    nx_ge_ny.operation = "COMPARE"  # type: ignore[attr-defined]
    nx_ge_ny.inputs[2].default_value = 0.0001  # type: ignore[index]
    nx_ge_ny.label = "|Nx| >= |Ny|"

    # |Nx| >= |Nz|
    nx_ge_nz = group.nodes.new("ShaderNodeMath")
    nx_ge_nz.operation = "COMPARE"  # type: ignore[attr-defined]
    nx_ge_nz.inputs[2].default_value = 0.0001  # type: ignore[index]
    nx_ge_nz.label = "|Nx| >= |Nz|"

    # |Ny| > |Nx|
    ny_gt_nx = group.nodes.new("ShaderNodeMath")
    ny_gt_nx.operation = "GREATER_THAN"  # type: ignore[attr-defined]
    ny_gt_nx.label = "|Ny| > |Nx|"

    # |Ny| >= |Nz|
    ny_ge_nz = group.nodes.new("ShaderNodeMath")
    ny_ge_nz.operation = "COMPARE"  # type: ignore[attr-defined]
    ny_ge_nz.inputs[2].default_value = 0.0001  # type: ignore[index]
    ny_ge_nz.label = "|Ny| >= |Nz|"

    # Combine conditions for X dominant: |Nx| >= |Ny| AND |Nx| >= |Nz|
    x_dominant = group.nodes.new("ShaderNodeMath")
    x_dominant.operation = "MULTIPLY"  # type: ignore[attr-defined]
    x_dominant.label = "X Dominant"

    # Combine conditions for Y dominant: |Ny| > |Nx| AND |Ny| >= |Nz|
    y_dominant = group.nodes.new("ShaderNodeMath")
    y_dominant.operation = "MULTIPLY"  # type: ignore[attr-defined]
    y_dominant.label = "Y Dominant"

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

    # Comparisons for X dominant
    group.links.new(abs_nx.outputs["Value"], nx_ge_ny.inputs[0])
    group.links.new(abs_ny.outputs["Value"], nx_ge_ny.inputs[1])
    group.links.new(abs_nx.outputs["Value"], nx_ge_nz.inputs[0])
    group.links.new(abs_nz.outputs["Value"], nx_ge_nz.inputs[1])

    # Comparisons for Y dominant
    group.links.new(abs_ny.outputs["Value"], ny_gt_nx.inputs[0])
    group.links.new(abs_nx.outputs["Value"], ny_gt_nx.inputs[1])
    group.links.new(abs_ny.outputs["Value"], ny_ge_nz.inputs[0])
    group.links.new(abs_nz.outputs["Value"], ny_ge_nz.inputs[1])

    # X dominant = |Nx| >= |Ny| AND |Nx| >= |Nz|
    group.links.new(nx_ge_ny.outputs["Value"], x_dominant.inputs[0])
    group.links.new(nx_ge_nz.outputs["Value"], x_dominant.inputs[1])

    # Y dominant = |Ny| > |Nx| AND |Ny| >= |Nz|
    group.links.new(ny_gt_nx.outputs["Value"], y_dominant.inputs[0])
    group.links.new(ny_ge_nz.outputs["Value"], y_dominant.inputs[1])

    # UV projections
    # X-dominant: U=Y, V=Z
    group.links.new(sep_pos.outputs["Y"], uv_x.inputs["X"])
    group.links.new(sep_pos.outputs["Z"], uv_x.inputs["Y"])

    # Y-dominant: U=X, V=Z
    group.links.new(sep_pos.outputs["X"], uv_y.inputs["X"])
    group.links.new(sep_pos.outputs["Z"], uv_y.inputs["Y"])

    # Z-dominant: U=X, V=Y
    group.links.new(sep_pos.outputs["X"], uv_z.inputs["X"])
    group.links.new(sep_pos.outputs["Y"], uv_z.inputs["Y"])

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


def create_uv_map_node_group() -> bpy.types.NodeTree:  # noqa: PLR0915
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

    # Tag the node group for identification
    node_tree[UV_MAP_NODE_GROUP_TAG] = True

    # Create interface
    _create_main_interface(node_tree)

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

    # Create mapping sub-groups
    planar_group = _create_planar_mapping_group(node_tree)
    cylindrical_group = _create_cylindrical_mapping_group(node_tree)
    spherical_group = _create_spherical_mapping_group(node_tree)
    box_group = _create_box_mapping_group(node_tree)

    # Create group nodes for each mapping type
    planar_node = node_tree.nodes.new("GeometryNodeGroup")
    planar_node.node_tree = planar_group  # type: ignore[attr-defined]
    planar_node.label = "Planar"
    planar_node.select = False

    cylindrical_node = node_tree.nodes.new("GeometryNodeGroup")
    cylindrical_node.node_tree = cylindrical_group  # type: ignore[attr-defined]
    cylindrical_node.label = "Cylindrical"
    cylindrical_node.select = False

    spherical_node = node_tree.nodes.new("GeometryNodeGroup")
    spherical_node.node_tree = spherical_group  # type: ignore[attr-defined]
    spherical_node.label = "Spherical"
    spherical_node.select = False

    box_node = node_tree.nodes.new("GeometryNodeGroup")
    box_node.node_tree = box_group  # type: ignore[attr-defined]
    box_node.label = "Box"
    box_node.select = False

    # Menu switch for UV output selection
    uv_switch = node_tree.nodes.new("GeometryNodeMenuSwitch")
    uv_switch.label = "Select UV"
    uv_switch.data_type = "VECTOR"  # type: ignore[attr-defined]
    uv_switch.select = False

    # Configure UV switch menu items to match mapping types
    # Use identifier as both name and display - this ensures consistent ordering
    uv_switch.enum_definition.enum_items.clear()  # type: ignore[attr-defined]
    for identifier, _display_name, _ in MAPPING_TYPES:
        uv_switch.enum_definition.enum_items.new(identifier)  # type: ignore[attr-defined]

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

    # Flip U (1 - u if flip enabled)
    flip_u_sub = node_tree.nodes.new("ShaderNodeMath")
    flip_u_sub.operation = "SUBTRACT"  # type: ignore[attr-defined]
    flip_u_sub.inputs[0].default_value = 1.0  # type: ignore[index]
    flip_u_sub.label = "1 - U"
    flip_u_sub.select = False

    flip_u_switch = node_tree.nodes.new("GeometryNodeSwitch")
    flip_u_switch.input_type = "FLOAT"  # type: ignore[attr-defined]
    flip_u_switch.label = "Flip U Switch"
    flip_u_switch.select = False

    # Flip V
    flip_v_sub = node_tree.nodes.new("ShaderNodeMath")
    flip_v_sub.operation = "SUBTRACT"  # type: ignore[attr-defined]
    flip_v_sub.inputs[0].default_value = 1.0  # type: ignore[index]
    flip_v_sub.label = "1 - V"
    flip_v_sub.select = False

    flip_v_switch = node_tree.nodes.new("GeometryNodeSwitch")
    flip_v_switch.input_type = "FLOAT"  # type: ignore[attr-defined]
    flip_v_switch.label = "Flip V Switch"
    flip_v_switch.select = False

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

    # Normal transformation
    node_tree.links.new(normal_node.outputs["Normal"], rotate_normal.inputs["Vector"])
    node_tree.links.new(
        invert_rot.outputs["Rotation"], rotate_normal.inputs["Rotation"]
    )

    # Connect transformed position/normal to mapping nodes
    node_tree.links.new(divide_size.outputs["Vector"], planar_node.inputs["Position"])
    node_tree.links.new(rotate_normal.outputs["Vector"], planar_node.inputs["Normal"])

    node_tree.links.new(
        divide_size.outputs["Vector"], cylindrical_node.inputs["Position"]
    )
    node_tree.links.new(
        rotate_normal.outputs["Vector"], cylindrical_node.inputs["Normal"]
    )

    node_tree.links.new(
        divide_size.outputs["Vector"], spherical_node.inputs["Position"]
    )
    node_tree.links.new(
        rotate_normal.outputs["Vector"], spherical_node.inputs["Normal"]
    )

    node_tree.links.new(divide_size.outputs["Vector"], box_node.inputs["Position"])
    node_tree.links.new(rotate_normal.outputs["Vector"], box_node.inputs["Normal"])

    # Connect mapping outputs to UV switch
    # Note: Menu Switch socket inputs are named by the display name passed to enum_items.new()
    node_tree.links.new(
        input_node.outputs[SOCKET_MAPPING_TYPE], uv_switch.inputs["Menu"]
    )
    # Use indices for inputs since socket names may vary - inputs are: Menu, then one per enum item
    # Item 0 = Planar, Item 1 = Cylindrical, Item 2 = Spherical, Item 3 = Box
    node_tree.links.new(planar_node.outputs["UV"], uv_switch.inputs[1])
    node_tree.links.new(cylindrical_node.outputs["UV"], uv_switch.inputs[2])
    node_tree.links.new(spherical_node.outputs["UV"], uv_switch.inputs[3])
    node_tree.links.new(box_node.outputs["UV"], uv_switch.inputs[4])

    # UV tiling and flip chain
    node_tree.links.new(uv_switch.outputs["Output"], separate_uv.inputs["Vector"])

    node_tree.links.new(separate_uv.outputs["X"], multiply_u.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_U_TILE], multiply_u.inputs[1])

    node_tree.links.new(separate_uv.outputs["Y"], multiply_v.inputs[0])
    node_tree.links.new(input_node.outputs[SOCKET_V_TILE], multiply_v.inputs[1])

    # Flip U
    node_tree.links.new(multiply_u.outputs["Value"], flip_u_sub.inputs[1])
    node_tree.links.new(
        input_node.outputs[SOCKET_U_FLIP], flip_u_switch.inputs["Switch"]
    )
    node_tree.links.new(multiply_u.outputs["Value"], flip_u_switch.inputs["False"])
    node_tree.links.new(flip_u_sub.outputs["Value"], flip_u_switch.inputs["True"])

    # Flip V
    node_tree.links.new(multiply_v.outputs["Value"], flip_v_sub.inputs[1])
    node_tree.links.new(
        input_node.outputs[SOCKET_V_FLIP], flip_v_switch.inputs["Switch"]
    )
    node_tree.links.new(multiply_v.outputs["Value"], flip_v_switch.inputs["False"])
    node_tree.links.new(flip_v_sub.outputs["Value"], flip_v_switch.inputs["True"])

    # Combine UV
    node_tree.links.new(flip_u_switch.outputs["Output"], combine_uv.inputs["X"])
    node_tree.links.new(flip_v_switch.outputs["Output"], combine_uv.inputs["Y"])

    # Store UV attribute
    node_tree.links.new(
        input_node.outputs[SOCKET_GEOMETRY], store_uv.inputs["Geometry"]
    )
    node_tree.links.new(input_node.outputs[SOCKET_UV_MAP], store_uv.inputs["Name"])
    node_tree.links.new(combine_uv.outputs["Vector"], store_uv.inputs["Value"])

    # Add Transform Gizmo for interactive viewport control (Blender 4.3+)
    # The gizmo's Value input connects to a Combine Transform that takes Position/Rotation/Size
    # The gizmo's Transform output joins into the geometry
    transform_gizmo = node_tree.nodes.new("GeometryNodeGizmoTransform")
    transform_gizmo.label = "UV Map Gizmo"
    transform_gizmo.select = False

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

    # Connect Combine Transform output to gizmo's Value input
    node_tree.links.new(
        combine_transform.outputs["Transform"], transform_gizmo.inputs["Value"]
    )

    # Also connect Position and Rotation directly to Gizmo's Position and Rotation inputs
    # This ensures the gizmo properly follows the overlay position/orientation
    node_tree.links.new(
        input_node.outputs[SOCKET_POSITION], transform_gizmo.inputs["Position"]
    )
    node_tree.links.new(
        input_node.outputs[SOCKET_ROTATION], transform_gizmo.inputs["Rotation"]
    )

    # Join gizmo transform output into geometry
    join_geometry = node_tree.nodes.new("GeometryNodeJoinGeometry")
    join_geometry.label = "Join Gizmo"
    join_geometry.select = False

    node_tree.links.new(store_uv.outputs["Geometry"], join_geometry.inputs["Geometry"])
    node_tree.links.new(
        transform_gizmo.outputs["Transform"], join_geometry.inputs["Geometry"]
    )

    # Output
    node_tree.links.new(
        join_geometry.outputs["Geometry"], output_node.inputs[SOCKET_GEOMETRY]
    )

    # Layout nodes with tighter spacing
    layout_nodes_pcb_style(node_tree, cell_width=0.0, cell_height=150.0)

    return node_tree


def is_uv_map_node_group(node_tree: bpy.types.NodeTree | None) -> bool:
    """Check if a node tree is a UV Map node group created by this addon."""
    if node_tree is None:
        return False
    return node_tree.get(UV_MAP_NODE_GROUP_TAG, False)


def get_uv_map_node_groups() -> list[bpy.types.NodeTree]:
    """Get all UV Map node groups in the blend file."""
    return [ng for ng in bpy.data.node_groups if is_uv_map_node_group(ng)]


# Classes to register (none for this module - it's utility only)
classes: list[type] = []
