"""
Operators module for UV Map addon.

Contains operators for adding UV Map modifier and inserting node groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from .constants import (
    MAPPING_PLANAR,
    MAPPING_TYPES,
    MODIFIER_NAME,
    SOCKET_MAPPING_TYPE,
    UV_MAP_NODE_GROUP_PREFIX,
    UV_MAP_NODE_GROUP_TAG,
)
from .nodes import (
    _SUB_GROUP_SUFFIXES,
    get_or_create_uv_map_node_group,
    get_uv_map_node_groups,
    is_uv_map_node_group,
    regenerate_uv_map_node_group,
)

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems
    from bpy.types import Context, Event


class UVMAP_OT_add_modifier(bpy.types.Operator):
    """Add a UV Map modifier to the selected object."""

    bl_idname = "uv_map.add_modifier"
    bl_label = "UV Map"
    bl_description = "Add a UV Map modifier with procedural UV mapping node group"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if the operator can be executed."""
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}

        # Get or create the UV Map node group (reuses existing if unmodified)
        node_tree = get_or_create_uv_map_node_group()

        # Add geometry nodes modifier
        modifier = obj.modifiers.new(name=MODIFIER_NAME, type="NODES")
        modifier.node_group = node_tree  # type: ignore[attr-defined]

        self.report({"INFO"}, f"Added UV Map modifier to {obj.name}")
        return {"FINISHED"}


class UVMAP_OT_insert_node_group(bpy.types.Operator):
    """Insert a UV Map node group into the current geometry nodes editor."""

    bl_idname = "uv_map.insert_node_group"
    bl_label = "UV Map"
    bl_description = "Insert a UV Map node group for procedural UV mapping"
    bl_options = {"REGISTER", "UNDO"}

    use_transform: bpy.props.BoolProperty(  # type: ignore[valid-type]
        name="Use Transform",
        description="Start transform operator after inserting the node",
        default=True,
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if the operator can be executed."""
        space = context.space_data
        if space is None or space.type != "NODE_EDITOR":
            return False
        # Check if we're in a geometry node tree
        node_tree = getattr(space, "edit_tree", None)
        if node_tree is None:
            return False
        return node_tree.type == "GEOMETRY"

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        space = context.space_data
        if space is None:
            self.report({"ERROR"}, "No active space")
            return {"CANCELLED"}

        node_tree = getattr(space, "edit_tree", None)
        if node_tree is None:
            self.report({"ERROR"}, "No active node tree")
            return {"CANCELLED"}

        # Get or create the UV Map node group (reuses existing if unmodified)
        uv_map_group = get_or_create_uv_map_node_group()

        # Add a group node referencing it
        group_node = node_tree.nodes.new("GeometryNodeGroup")
        group_node.node_tree = uv_map_group  # type: ignore[attr-defined]
        group_node.label = UV_MAP_NODE_GROUP_PREFIX
        group_node.width = 200  # Wider to avoid text clipping

        # Position at cursor location in node editor
        cursor_location = getattr(space, "cursor_location", (0.0, 0.0))
        group_node.location = cursor_location

        # Select the new node
        for node in node_tree.nodes:
            node.select = False
        group_node.select = True
        node_tree.nodes.active = group_node

        # Start transform if requested (enables "drop mode")
        if self.use_transform:
            bpy.ops.node.translate_attach_remove_on_cancel("INVOKE_DEFAULT")

        return {"FINISHED"}

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Invoke the operator - update cursor position from mouse."""
        space = context.space_data
        if space is not None:
            region = context.region
            if region is not None:
                view2d = region.view2d
                if view2d is not None:
                    # Convert mouse region coords to node editor view coords
                    view_x, view_y = view2d.region_to_view(
                        event.mouse_region_x, event.mouse_region_y
                    )
                    # Scale by UI scale factor (matches Blender's internal behavior)
                    prefs = context.preferences
                    if prefs is not None:
                        ui_scale = prefs.system.ui_scale
                        view_x /= ui_scale
                        view_y /= ui_scale
                    # Store in cursor_location for execute() to use
                    if hasattr(space, "cursor_location"):
                        space.cursor_location = (view_x, view_y)  # type: ignore[attr-defined]

        return self.execute(context)


class UVMAP_OT_select_uv_map_modifier(bpy.types.Operator):
    """Select the UV Map node group in the modifier for overlay display."""

    bl_idname = "uv_map.select_modifier"
    bl_label = "Select UV Map Modifier"
    bl_description = "Select the UV Map modifier's node group for overlay display"
    bl_options = {"INTERNAL"}

    modifier_name: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="Modifier Name",
        description="Name of the modifier to select",
        default="",
    )

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.active_object
        if obj is None:
            return {"CANCELLED"}

        modifier = obj.modifiers.get(self.modifier_name)
        if modifier is None:
            return {"CANCELLED"}

        # Set as active modifier (this enables overlay display)
        obj.modifiers.active = modifier
        return {"FINISHED"}


class UVMAP_OT_regenerate_node_groups(bpy.types.Operator):
    """Regenerate all UV Map node groups.

    This deletes and recreates the internal helper groups (Planar, Cylindrical,
    Spherical, Box) and regenerates all main UV Map node groups in place.
    Useful after code changes during development.
    """

    bl_idname = "uv_map.regenerate_node_groups"
    bl_label = "Regenerate UV Map Node Groups"
    bl_description = (
        "Regenerate all UV Map node groups (updates all existing UV Map setups)"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        # Helper group names (using constant from nodes.py)
        helper_names = [
            f"{UV_MAP_NODE_GROUP_PREFIX}{suffix}" for suffix in _SUB_GROUP_SUFFIXES
        ]

        # Find all main UV Map node groups before we delete helpers
        main_groups: list[bpy.types.NodeTree] = [
            ng for ng in bpy.data.node_groups if ng.get(UV_MAP_NODE_GROUP_TAG, False)
        ]

        # Delete old helper groups
        deleted_count = 0
        for name in helper_names:
            if name in bpy.data.node_groups:
                bpy.data.node_groups.remove(bpy.data.node_groups[name])
                deleted_count += 1

        # Regenerate all main UV Map node groups in place
        # This will also recreate the helper groups on first use
        for main_group in main_groups:
            regenerate_uv_map_node_group(main_group)

        self.report(
            {"INFO"},
            f"Regenerated {len(main_groups)} UV Map node groups",
        )
        return {"FINISHED"}


class UVMAP_OT_regenerate_confirm(bpy.types.Operator):
    """Confirmation dialog for regenerating UV Map node groups."""

    bl_idname = "uv_map.regenerate_confirm"
    bl_label = "Regenerate UV Map Node Groups?"
    bl_description = "Show confirmation dialog before regenerating node groups"
    bl_options = {"INTERNAL", "BLOCKING"}

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the regeneration."""
        return bpy.ops.uv_map.regenerate_node_groups()  # type: ignore[return-value]

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Show confirmation dialog."""
        wm = context.window_manager
        if wm is None:
            return {"CANCELLED"}
        return wm.invoke_props_dialog(self, width=400, confirm_text="Regenerate")

    def draw(self, context: Context) -> None:
        """Draw the dialog content."""
        layout = self.layout
        if layout is None:
            return

        existing_groups = get_uv_map_node_groups()

        layout.label(
            text=f"Found {len(existing_groups)} UV Map node group(s) that may be outdated.",
            icon="INFO",
        )
        layout.separator()
        layout.label(text="The UV Map addon has been updated. Regenerating will:")
        col = layout.column(align=True)
        col.label(text="• Update all UV Map node groups to the latest version")
        col.label(text="• Preserve your modifier settings where possible")
        layout.separator()
        layout.label(text="Click OK to regenerate, or Cancel to keep existing groups.")


def get_active_uv_map_node_group(context: Context) -> bpy.types.NodeTree | None:
    """Get the UV Map node group from the active modifier, if any.

    Returns the node tree if:
    - There's an active object
    - The object has an active modifier
    - The modifier is a geometry nodes modifier
    - The node group is a UV Map node group created by this addon
    """
    obj = context.active_object
    if obj is None:
        return None

    modifier = obj.modifiers.active
    if modifier is None:
        return None

    if modifier.type != "NODES":
        return None

    node_tree = getattr(modifier, "node_group", None)
    if node_tree is None:
        return None

    if not is_uv_map_node_group(node_tree):
        return None

    return node_tree


def get_uv_map_modifier_params(  # noqa: PLR0912, PLR0915
    obj: bpy.types.Object,  # noqa: ARG001
    modifier: bpy.types.Modifier,
) -> dict[str, object] | None:
    """Get the UV Map parameters from a modifier.

    Returns a dictionary with:
    - mapping_type: str
    - position: tuple[float, float, float]
    - rotation: tuple[float, float, float, float] (quaternion)
    - size: tuple[float, float, float]
    - u_tile, v_tile: float
    - u_flip, v_flip: bool
    - uv_map: str

    Returns None if the modifier is not a valid UV Map modifier.
    """
    if modifier.type != "NODES":
        return None

    node_tree = getattr(modifier, "node_group", None)
    if node_tree is None or not is_uv_map_node_group(node_tree):
        return None

    # Build socket identifier mapping
    socket_ids: dict[str, str] = {}
    for item in node_tree.interface.items_tree:  # type: ignore[union-attr]
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "in_out", None) != "INPUT":
            continue
        socket_ids[item.name] = item.identifier  # type: ignore[union-attr]

    # Read values from modifier
    params: dict[str, object] = {}

    # Mapping type (menu) - Menu Switch returns integer index offset by 2
    # (because the node has type selector and Menu input before the enum items)
    mapping_type_id = socket_ids.get(SOCKET_MAPPING_TYPE)
    if mapping_type_id:
        mapping_value = modifier.get(
            mapping_type_id, 2
        )  # Default to first item (index 2)
        if isinstance(mapping_value, int):
            # Subtract 2 to get 0-based index into MAPPING_TYPES
            adjusted_index = mapping_value - 2
            if 0 <= adjusted_index < len(MAPPING_TYPES):
                params["mapping_type"] = MAPPING_TYPES[adjusted_index][0]
            else:
                params["mapping_type"] = MAPPING_PLANAR
        elif isinstance(mapping_value, str):
            # Fallback if it's a string
            params["mapping_type"] = mapping_value.upper()
        else:
            params["mapping_type"] = MAPPING_PLANAR

    # Position
    pos_id = socket_ids.get("Position")
    if pos_id:
        pos = modifier.get(pos_id)
        if pos is not None and hasattr(pos, "__len__") and len(pos) >= 3:
            pos_list = list(pos)
            params["position"] = (
                float(pos_list[0]),
                float(pos_list[1]),
                float(pos_list[2]),
            )
        else:
            params["position"] = (0.0, 0.0, 0.0)

    # Rotation (stored as Euler or Quaternion depending on Blender version)
    rot_id = socket_ids.get("Rotation")
    if rot_id:
        rot = modifier.get(rot_id)
        if rot is not None and hasattr(rot, "__len__") and len(rot) >= 3:
            rot_list = list(rot)
            params["rotation"] = (
                float(rot_list[0]),
                float(rot_list[1]),
                float(rot_list[2]),
            )
        else:
            params["rotation"] = (0.0, 0.0, 0.0)

    # Size
    size_id = socket_ids.get("Size")
    if size_id:
        size = modifier.get(size_id)
        if size is not None and hasattr(size, "__len__") and len(size) >= 3:
            size_list = list(size)
            params["size"] = (
                float(size_list[0]),
                float(size_list[1]),
                float(size_list[2]),
            )
        else:
            params["size"] = (1.0, 1.0, 1.0)

    # Tiling
    for tile_name in ["U Tile", "V Tile"]:
        tile_id = socket_ids.get(tile_name)
        if tile_id:
            params[tile_name.lower().replace(" ", "_")] = modifier.get(tile_id, 1.0)

    # Flip
    for flip_name in ["U Flip", "V Flip"]:
        flip_id = socket_ids.get(flip_name)
        if flip_id:
            params[flip_name.lower().replace(" ", "_")] = modifier.get(flip_id, False)

    # Cap (for cylindrical mapping)
    cap_id = socket_ids.get("Cap")
    if cap_id:
        params["cap"] = modifier.get(cap_id, False)

    # Normal-based (for cylindrical, spherical, and shrink wrap mappings)
    normal_based_id = socket_ids.get("Normal-based")
    if normal_based_id:
        params["normal_based"] = modifier.get(normal_based_id, False)

    # UV Map name
    uv_map_id = socket_ids.get("UV Map")
    if uv_map_id:
        params["uv_map"] = modifier.get(uv_map_id, "UVMap")

    return params


# Classes to register
classes: list[type] = [
    UVMAP_OT_add_modifier,
    UVMAP_OT_insert_node_group,
    UVMAP_OT_select_uv_map_modifier,
    UVMAP_OT_regenerate_node_groups,
    UVMAP_OT_regenerate_confirm,
]
