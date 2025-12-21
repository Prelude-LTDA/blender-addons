"""
Operators module for Voxel Terrain.

Contains all operator classes for the addon.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import bpy

from .properties import VOXEL_TERRAIN_MODIFIER_NAME, _find_voxel_terrain_modifier
from .sockets import check_sockets_match, sync_sockets
from .typing_utils import get_object_props, get_scene_props

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems
    from bpy.types import Context, Event


class VOXELTERRAIN_OT_generate(bpy.types.Operator):
    """Generate voxel terrain."""

    bl_idname = "voxel_terrain.generate"
    bl_label = "Generate Terrain"
    bl_description = "Generate a new voxel terrain"
    bl_options = {"REGISTER", "UNDO"}

    # Example property with type annotation
    message: bpy.props.StringProperty(
        name="Message",
        description="Status message",
        default="Terrain generated!",
    )  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if the operator can be executed."""
        # This operator can always be executed
        return True

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        self.report({"INFO"}, self.message)
        return {"FINISHED"}

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Invoke the operator - called when the user triggers the operator."""
        # You could show a dialog here with:
        # return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)


class VOXELTERRAIN_OT_clear(bpy.types.Operator):
    """Clear the current voxel terrain."""

    bl_idname = "voxel_terrain.clear"
    bl_label = "Clear Terrain"
    bl_description = "Clear the current voxel terrain"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only allow execution when an object is selected."""
        return context.active_object is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.active_object
        if obj:
            self.report({"INFO"}, f"Cleared terrain from {obj.name}")
        return {"FINISHED"}


class VOXELTERRAIN_OT_export(bpy.types.Operator):
    """Export terrain data to the specified folder (quick export)."""

    bl_idname = "voxel_terrain.export"
    bl_label = "Export Terrain"
    bl_description = "Export terrain data to the configured folder"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if export is possible."""
        if not context.scene:
            return False
        props = get_scene_props(context.scene)
        # Require a valid export path
        return bool(props.export_path.strip())

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the export."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        # Resolve the path (handles // for relative paths)
        export_path = bpy.path.abspath(props.export_path)
        path = Path(export_path)

        return self._do_export(context, path)

    def _do_export(
        self, context: Context, path: Path
    ) -> set[OperatorReturnItems]:
        """Shared export logic."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.report({"ERROR"}, f"Failed to create export directory: {e}")
            return {"CANCELLED"}

        # Get settings
        chunk_size = props.chunk_size
        voxel_size = props.voxel_size
        lod_factor = props.lod_factor
        lod_levels = props.lod_levels

        # TODO: Implement actual export logic
        self.report(
            {"INFO"},
            f"Exported to {path} | Chunk: "
            f"({chunk_size[0]:.0f}, {chunk_size[1]:.0f}, {chunk_size[2]:.0f}) | "
            f"Voxel: {voxel_size:.3g} | LOD: {lod_levels}x (factor {lod_factor})"
        )

        return {"FINISHED"}


class VOXELTERRAIN_OT_export_dialog(bpy.types.Operator):
    """Export terrain data via file browser dialog."""

    bl_idname = "voxel_terrain.export_dialog"
    bl_label = "Export Voxel Terrain"
    bl_description = "Export terrain data to a selected folder"
    bl_options = {"REGISTER"}

    # File browser properties
    directory: bpy.props.StringProperty(
        name="Directory",
        description="Directory to export to",
        subtype="DIR_PATH",
    )  # type: ignore[valid-type]

    filter_folder: bpy.props.BoolProperty(
        default=True,
        options={"HIDDEN"},
    )  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if export is possible."""
        return context.scene is not None

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Open file browser dialog."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        # Pre-fill with current export path if set
        if props.export_path.strip():
            self.directory = bpy.path.abspath(props.export_path)

        wm = context.window_manager
        assert wm is not None
        wm.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the export after folder selection."""
        assert context.scene is not None
        props = get_scene_props(context.scene)
        path = Path(self.directory)

        # Optionally update the scene property with the selected path
        # (commented out - uncomment if you want selection to persist)
        # props.export_path = self.directory

        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.report({"ERROR"}, f"Failed to create export directory: {e}")
            return {"CANCELLED"}

        # Get settings
        chunk_size = props.chunk_size
        voxel_size = props.voxel_size
        lod_factor = props.lod_factor
        lod_levels = props.lod_levels

        # TODO: Implement actual export logic
        self.report(
            {"INFO"},
            f"Exported to {path} | Chunk: "
            f"({chunk_size[0]:.0f}, {chunk_size[1]:.0f}, {chunk_size[2]:.0f}) | "
            f"Voxel: {voxel_size:.3g} | LOD: {lod_levels}x (factor {lod_factor})"
        )

        return {"FINISHED"}


def menu_func_export(self: bpy.types.Menu, context: Context) -> None:  # noqa: ARG001
    """Add export option to File > Export menu."""
    layout = self.layout
    assert layout is not None
    layout.operator(
        VOXELTERRAIN_OT_export_dialog.bl_idname, text="Voxel Terrain"
    )


class VOXELTERRAIN_OT_set_view_lod(bpy.types.Operator):
    """Set view LOD level."""

    bl_idname = "voxel_terrain.set_view_lod"
    bl_label = "Set LOD Level"
    bl_description = "Set the LOD level to view"
    bl_options = {"INTERNAL"}

    level: bpy.props.IntProperty(
        name="Level",
        description="LOD level to set",
        default=0,
        min=0,
    )  # type: ignore[valid-type]

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        assert context.scene is not None
        props = get_scene_props(context.scene)
        max_lod = props.lod_levels - 1
        props.view_lod = min(self.level, max_lod)

        # Force viewport redraw
        for area in context.screen.areas if context.screen else []:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        return {"FINISHED"}


class VOXELTERRAIN_OT_toggle_grid_bounds(bpy.types.Operator):
    """Toggle grid bounds mode (Chunks/Selection)."""

    bl_idname = "voxel_terrain.toggle_grid_bounds"
    bl_label = "Toggle Grid Bounds"
    bl_description = "Toggle which grid bounds to display.\n\nShift: Switch exclusively"
    bl_options = {"INTERNAL"}

    mode: bpy.props.StringProperty(
        name="Mode",
        description="Which mode to toggle ('chunks' or 'selection')",
        default="chunks",
    )  # type: ignore[valid-type]

    def invoke(self, context: Context, event: bpy.types.Event) -> set[OperatorReturnItems]:
        """Handle the operator invocation with modifier key detection."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        is_chunks = self.mode == "chunks"
        chunks_on = props.voxel_grid_bounds_chunks
        selection_on = props.voxel_grid_bounds_selection
        grid_visible = props.show_voxel_grid
        this_on = chunks_on if is_chunks else selection_on
        other_on = selection_on if is_chunks else chunks_on

        # Shift+Click = exclusive mode (switch to only this one)
        if event.shift:
            props.show_voxel_grid = True
            props.voxel_grid_bounds_chunks = is_chunks
            props.voxel_grid_bounds_selection = not is_chunks
        elif not grid_visible:
            # Grid is off - enable grid and this mode (keep other as-is)
            props.show_voxel_grid = True
            self._set_mode(props, is_chunks, True)
        elif this_on and not other_on:
            # Only this one is on - disable grid entirely
            props.show_voxel_grid = False
        else:
            # Toggle this mode
            self._set_mode(props, is_chunks, not this_on)

        # Force viewport redraw
        for area in context.screen.areas if context.screen else []:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        return {"FINISHED"}

    def _set_mode(self, props: object, is_chunks: bool, value: bool) -> None:
        """Set the appropriate mode property."""
        if is_chunks:
            props.voxel_grid_bounds_chunks = value  # type: ignore[attr-defined]
        else:
            props.voxel_grid_bounds_selection = value  # type: ignore[attr-defined]

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute fallback (no event available)."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        if self.mode == "chunks":
            props.voxel_grid_bounds_chunks = not props.voxel_grid_bounds_chunks
        else:
            props.voxel_grid_bounds_selection = not props.voxel_grid_bounds_selection

        return {"FINISHED"}


class VOXELTERRAIN_OT_show_node_group(bpy.types.Operator):
    """Show node group in Geometry Nodes editor."""

    bl_idname = "voxel_terrain.show_node_group"
    bl_label = "Show in Editor"
    bl_description = "Open this node group in the Geometry Nodes editor"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only available if object has a node group assigned."""
        obj = context.object
        if obj is None:
            return False
        props = get_object_props(obj)
        return props.node_group is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)
        node_group = props.node_group

        if node_group is None:
            self.report({"WARNING"}, "No node group assigned")
            return {"CANCELLED"}

        # Find the Voxel Terrain modifier by name, or recreate if missing
        modifier = _find_voxel_terrain_modifier(obj)

        if modifier is None:
            # Recreate the modifier since it was manually deleted
            modifier = obj.modifiers.new(name=VOXEL_TERRAIN_MODIFIER_NAME, type="NODES")
            modifier.show_viewport = False
            modifier.show_render = False
            modifier.show_expanded = False
            modifier.node_group = node_group  # type: ignore[union-attr]
            self.report({"INFO"}, "Recreated Voxel Terrain modifier")

        # Make this modifier active so the editor shows it
        obj.modifiers.active = modifier

        # Switch Properties editor to Modifiers tab if visible
        for area in context.screen.areas if context.screen else []:
            if area.type == "PROPERTIES":
                for space in area.spaces:
                    if space.type == "PROPERTIES":
                        space.context = "MODIFIER"  # type: ignore[attr-defined]
                        break

        # Try to find a Geometry Nodes editor area and update it
        for area in context.screen.areas if context.screen else []:
            if area.type == "NODE_EDITOR":
                for space in area.spaces:
                    tree_type = getattr(space, "tree_type", None)
                    if tree_type == "GeometryNodeTree":
                        # The editor should automatically pick up the active modifier
                        area.tag_redraw()
                        break

        self.report({"INFO"}, f"Showing '{node_group.name}' in editor")
        return {"FINISHED"}


class VOXELTERRAIN_OT_new_node_group(bpy.types.Operator):
    """Create a new node group with the correct socket interface."""

    bl_idname = "voxel_terrain.new_node_group"
    bl_label = "New Voxel Terrain Node Group"
    bl_description = "Create a new Geometry Nodes group with the correct socket interface"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only enabled if we have an object selected."""
        return context.object is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)

        # Create new geometry node tree
        node_group = bpy.data.node_groups.new(
            name="Voxel Terrain",
            type="GeometryNodeTree",
        )

        # Sync sockets to match the required interface
        sync_sockets(node_group)

        # Add Group Input and Group Output nodes
        input_node = node_group.nodes.new("NodeGroupInput")
        input_node.location = (-300, 0)
        input_node.select = False

        output_node = node_group.nodes.new("NodeGroupOutput")
        output_node.location = (300, 0)
        output_node.select = False

        # Connect Geometry input to Geometry output
        # The first output of Group Input is the first socket (Geometry)
        # The first input of Group Output is the first socket (Geometry)
        if input_node.outputs and output_node.inputs:
            node_group.links.new(input_node.outputs[0], output_node.inputs[0])

        # Assign to the object property
        props.node_group = node_group

        self.report({"INFO"}, f"Created node group '{node_group.name}'")
        return {"FINISHED"}


class VOXELTERRAIN_OT_sync_sockets(bpy.types.Operator):
    """Sync node group sockets to match required interface."""

    bl_idname = "voxel_terrain.sync_sockets"
    bl_label = "Sync Sockets"
    bl_description = "Add missing input/output sockets to match the required interface"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only enabled if node group exists and sockets don't match."""
        obj = context.object
        if obj is None:
            return False
        props = get_object_props(obj)
        node_group = props.node_group
        if node_group is None:
            return False
        # Only enable if sockets don't match
        return not check_sockets_match(node_group)

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)
        node_group = props.node_group

        if node_group is None:
            self.report({"WARNING"}, "No node group assigned")
            return {"CANCELLED"}

        inputs_added, outputs_added = sync_sockets(node_group)
        total = inputs_added + outputs_added

        if total == 0:
            self.report({"INFO"}, "Sockets already match")
        else:
            self.report(
                {"INFO"},
                f"Synced sockets: {inputs_added} inputs, {outputs_added} outputs added",
            )
        return {"FINISHED"}


class VOXELTERRAIN_OT_sync_from_modifier(bpy.types.Operator):
    """Sync node group from the Voxel Terrain modifier."""

    bl_idname = "voxel_terrain.sync_from_modifier"
    bl_label = "Sync from Modifier"
    bl_description = "Update the node group property to match the modifier"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only enabled if modifier exists."""
        obj = context.object
        if obj is None:
            return False
        return _find_voxel_terrain_modifier(obj) is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)

        modifier = _find_voxel_terrain_modifier(obj)
        if modifier is None:
            self.report({"WARNING"}, "Voxel Terrain modifier not found")
            return {"CANCELLED"}

        modifier_ng = getattr(modifier, "node_group", None)
        props.node_group = modifier_ng

        if modifier_ng:
            self.report({"INFO"}, f"Synced from modifier: '{modifier_ng.name}'")
        else:
            self.report({"INFO"}, "Cleared node group (from modifier)")
        return {"FINISHED"}


class VOXELTERRAIN_OT_sync_to_modifier(bpy.types.Operator):
    """Sync node group to the Voxel Terrain modifier."""

    bl_idname = "voxel_terrain.sync_to_modifier"
    bl_label = "Sync to Modifier"
    bl_description = "Update the modifier's node group to match this property"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only enabled if modifier exists."""
        obj = context.object
        if obj is None:
            return False
        return _find_voxel_terrain_modifier(obj) is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)

        modifier = _find_voxel_terrain_modifier(obj)
        if modifier is None:
            self.report({"WARNING"}, "Voxel Terrain modifier not found")
            return {"CANCELLED"}

        modifier.node_group = props.node_group  # type: ignore[union-attr]

        if props.node_group:
            self.report({"INFO"}, f"Synced to modifier: '{props.node_group.name}'")
        else:
            self.report({"INFO"}, "Cleared modifier node group")
        return {"FINISHED"}


# List of all classes to register - used by __init__.py
classes: tuple[type, ...] = (
    VOXELTERRAIN_OT_generate,
    VOXELTERRAIN_OT_clear,
    VOXELTERRAIN_OT_export,
    VOXELTERRAIN_OT_export_dialog,
    VOXELTERRAIN_OT_set_view_lod,
    VOXELTERRAIN_OT_toggle_grid_bounds,
    VOXELTERRAIN_OT_show_node_group,
    VOXELTERRAIN_OT_new_node_group,
    VOXELTERRAIN_OT_sync_sockets,
    VOXELTERRAIN_OT_sync_from_modifier,
    VOXELTERRAIN_OT_sync_to_modifier,
)
