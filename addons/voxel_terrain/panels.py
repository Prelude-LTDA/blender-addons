"""
Panels module for Voxel Terrain.

Contains all panel classes for the addon UI, including the N-panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from .msgbus import subscribe_to_node_group
from .properties import _find_voxel_terrain_modifier
from .sockets import check_sockets_match
from .typing_utils import get_object_props, get_scene_props

if TYPE_CHECKING:
    from bpy.types import Context, UnitSettings


def _format_lod_size(size: float, unit_settings: UnitSettings) -> str:
    """Format a LOD size value with appropriate unit."""
    if unit_settings.system == "METRIC":
        if size >= 1000:
            result = f"{size / 1000:.3g} km"
        elif size >= 1:
            result = f"{size:.3g} m"
        elif size >= 0.01:
            result = f"{size * 100:.3g} cm"
        else:
            result = f"{size * 1000:.3g} mm"
    elif unit_settings.system == "IMPERIAL":
        feet = size * 3.28084
        if feet >= 5280:
            result = f"{feet / 5280:.3g} mi"
        elif feet >= 1:
            result = f"{feet:.3g} ft"
        else:
            result = f"{feet * 12:.3g} in"
    else:
        result = f"{size:.3g}"
    return result


class VOXELTERRAIN_PT_main_panel(bpy.types.Panel):
    """Main panel in the N-panel sidebar."""

    bl_idname = "VOXELTERRAIN_PT_main_panel"
    bl_label = "Overlays"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Voxel Terrain"

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        scene = context.scene
        assert scene is not None
        props = get_scene_props(scene)

        # Chunk visualization toggles
        layout.prop(props, "show_chunks", icon="MESH_GRID")
        row = layout.row()
        row.active = props.show_chunks and props.enable_skirt
        row.prop(props, "show_skirt", icon="SELECT_SUBTRACT")
        layout.prop(props, "show_voxel_grid", icon="SNAP_GRID")


class VOXELTERRAIN_PT_npanel_voxel_grid(bpy.types.Panel):
    """Voxel Grid subpanel in the N-panel sidebar."""

    bl_idname = "VOXELTERRAIN_PT_npanel_voxel_grid"
    bl_label = "Voxel Grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Voxel Terrain"
    bl_parent_id = "VOXELTERRAIN_PT_main_panel"

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        scene = context.scene
        assert scene is not None
        props = get_scene_props(scene)

        # Disable entire panel if voxel grid is not shown
        layout.active = props.show_voxel_grid

        # Grid Z source picker (segmented buttons)
        col = layout.column()
        col.label(text="Grid Z")
        row = col.row(align=True)
        row.prop_enum(props, "voxel_grid_z_source", "ORIGIN", text="Origin", icon="WORLD")
        row.prop_enum(props, "voxel_grid_z_source", "CURSOR", text="Cursor", icon="PIVOT_CURSOR")
        row.prop_enum(props, "voxel_grid_z_source", "SELECTION", text="Selection", icon="OBJECT_DATA")

        # Grid Bounds picker (toggle buttons via operators)
        col = layout.column()
        col.label(text="Grid Bounds")
        row = col.row(align=True)
        op = row.operator(
            "voxel_terrain.toggle_grid_bounds",
            text="Chunks",
            icon="MESH_GRID",
            depress=props.voxel_grid_bounds_chunks,
        )
        op.mode = "chunks"
        op = row.operator(
            "voxel_terrain.toggle_grid_bounds",
            text="Selection",
            icon="OBJECT_DATA",
            depress=props.voxel_grid_bounds_selection,
        )
        op.mode = "selection"

        # LOD picker (segmented buttons)
        max_lod = props.lod_levels - 1
        view_lod = min(props.view_lod, max_lod)

        col = layout.column()
        col.active = props.enable_lod and props.show_voxel_grid
        col.label(text="LOD Level")
        row = col.row(align=True)
        for i in range(props.lod_levels):
            op = row.operator(
                "voxel_terrain.set_view_lod",
                text=str(i),
                depress=(i == view_lod),
            )
            op.level = i


class VOXELTERRAIN_PT_npanel_about(bpy.types.Panel):
    """About subpanel in the N-panel sidebar."""

    bl_idname = "VOXELTERRAIN_PT_npanel_about"
    bl_label = "About"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Voxel Terrain"
    bl_parent_id = "VOXELTERRAIN_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        col = layout.column(align=True)
        assert col is not None
        col.label(text="Voxel Terrain v1.0.0")
        col.label(text="Voxel-based terrain tools")
        col.separator()
        col.label(text="For Blender 5.0+")


class VOXELTERRAIN_PT_scene_panel(bpy.types.Panel):
    """Scene properties panel for Voxel Terrain."""

    bl_idname = "VOXELTERRAIN_PT_scene_panel"
    bl_label = "Voxel Terrain"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        # Main panel is just a header, subpanels contain the content
        pass


class VOXELTERRAIN_PT_scene_settings(bpy.types.Panel):
    """Settings subpanel for Voxel Terrain."""

    bl_idname = "VOXELTERRAIN_PT_scene_settings"
    bl_label = "Settings"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_parent_id = "VOXELTERRAIN_PT_scene_panel"

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        # Use property split for label:value layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        assert scene is not None
        props = get_scene_props(scene)

        # Voxel Size (first)
        layout.prop(props, "voxel_size")

        layout.separator()

        # Chunk Size
        col = layout.column(align=True)
        col.prop(props, "chunk_size")

        layout.separator()

        # Skirt Settings
        layout.prop(props, "enable_skirt")
        row = layout.row()
        row.active = props.enable_skirt
        row.prop(props, "skirt_size")

        layout.separator()

        # LOD Settings
        layout.prop(props, "enable_lod")
        col = layout.column()
        col.active = props.enable_lod
        col.prop(props, "lod_factor")
        col.prop(props, "lod_levels")

        # Calculate LOD sizes
        voxel_size = props.voxel_size
        factor = props.lod_factor
        levels = props.lod_levels
        unit_settings = scene.unit_settings

        sizes = [_format_lod_size(voxel_size * (factor**i), unit_settings) for i in range(levels)]

        # LOD size preview (right-aligned in box)
        box = layout.box()
        box.active = props.enable_lod
        row = box.row()
        row.alignment = "RIGHT"
        row.label(text="LOD Sizes: " + ", ".join(sizes))


class VOXELTERRAIN_PT_scene_export(bpy.types.Panel):
    """Export subpanel for Voxel Terrain."""

    bl_idname = "VOXELTERRAIN_PT_scene_export"
    bl_label = "Export"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_parent_id = "VOXELTERRAIN_PT_scene_panel"

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        # Use property split for label:value layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        assert scene is not None
        props = get_scene_props(scene)

        # Export path
        layout.prop(props, "export_path")

        # Export button
        row = layout.row()
        row.operator("voxel_terrain.export", text="Export Voxel Terrain")


class VOXELTERRAIN_PT_scene_actions(bpy.types.Panel):
    """Actions subpanel for Voxel Terrain generation and baking."""

    bl_idname = "VOXELTERRAIN_PT_scene_actions"
    bl_label = "Actions"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_parent_id = "VOXELTERRAIN_PT_scene_panel"

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        # Action buttons - segmented row with delete
        row = layout.row(align=True)

        # Generate button - creates editable GeoNodes setup
        row.operator(
            "voxel_terrain.generate",
            text="Generate",
            icon="GEOMETRY_NODES",
        )

        # Bake button - creates static mesh copies
        row.operator(
            "voxel_terrain.bake",
            text="Bake",
            icon="RENDER_STILL",
        )

        # Delete button - removes generated terrain
        row.operator(
            "voxel_terrain.delete_terrain",
            text="",
            icon="TRASH",
        )

        # Info box
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="Generate: Creates editable GeoNodes", icon="INFO")
        col.label(text="Bake: Creates static mesh copies", icon="INFO")


class VOXELTERRAIN_PT_object_panel(bpy.types.Panel):
    """Object properties panel for Voxel Terrain."""

    bl_idname = "VOXELTERRAIN_PT_object_panel"
    bl_label = "Voxel Terrain"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only show for mesh objects."""
        return context.object is not None and context.object.type == "MESH"

    def draw_header(self, context: Context) -> None:
        """Draw the enable checkbox in the panel header."""
        layout = self.layout
        assert layout is not None
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)
        layout.prop(props, "enabled", text="")

    def _draw_modifier_sync_ui(
        self,
        layout: bpy.types.UILayout,
        props: object,
        modifier_ng: bpy.types.NodeTree | None,
    ) -> None:
        """Draw the modifier out-of-sync warning and sync buttons."""
        box = layout.box()
        box.alert = True
        row = box.row()
        row.alignment = "CENTER"
        row.scale_y = 0.8
        row.label(text="Modifier node group is out of sync", icon="ERROR")
        row = box.row(align=True)
        row.alert = False
        # Sync to modifier
        sub = row.row(align=True)
        sub.enabled = getattr(props, "node_group", None) is not None
        sub.operator(
            "voxel_terrain.sync_to_modifier",
            text="Sync Modifier",
            icon="MODIFIER",
        )
        # Sync from modifier
        sub = row.row(align=True)
        sub.enabled = modifier_ng is not None
        sub.operator(
            "voxel_terrain.sync_from_modifier",
            text="Sync Here",
            icon="OBJECT_DATA",
        )

    def _draw_modifier_missing_ui(
        self,
        layout: bpy.types.UILayout,
    ) -> None:
        """Draw the modifier missing warning and recreate button."""
        box = layout.box()
        box.alert = True
        row = box.row()
        row.alignment = "CENTER"
        row.scale_y = 0.8
        row.label(text="Modifier is missing", icon="ERROR")
        row = box.row()
        row.alert = False
        row.operator(
            "voxel_terrain.sync_to_modifier",
            text="Recreate Modifier",
            icon="MODIFIER",
        )

    def _draw_socket_sync_ui(
        self,
        layout: bpy.types.UILayout,
        sockets_match: bool,
    ) -> None:
        """Draw the socket sync status and button."""
        if sockets_match:
            row = layout.row()
            row.enabled = False
            row.operator(
                "voxel_terrain.sync_sockets",
                text="Sockets Synced",
                icon="CHECKMARK",
            )
        else:
            box = layout.box()
            box.alert = True
            row = box.row()
            row.alignment = "CENTER"
            row.scale_y = 0.8
            row.label(text="Sockets are out of sync", icon="ERROR")
            row = box.row()
            row.alert = False
            row.operator(
                "voxel_terrain.sync_sockets",
                text="Sync Sockets",
                icon="FILE_REFRESH",
            )

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        obj = context.object
        assert obj is not None
        props = get_object_props(obj)

        # Disable UI if not enabled
        layout.active = props.enabled

        # Use property split for label:value layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        # Check if modifier's node_group differs from our property
        modifier = _find_voxel_terrain_modifier(obj)
        modifier_ng = None
        modifier_missing = False
        out_of_sync = False
        if modifier is not None:
            modifier_ng = getattr(modifier, "node_group", None)
            out_of_sync = modifier_ng != props.node_group
        elif props.node_group is not None:
            # Modifier was removed but we still have a node group
            modifier_missing = True

        # Subscribe to the node group for real-time socket updates
        if props.node_group is not None:
            subscribe_to_node_group(props.node_group)
        if modifier_ng is not None and modifier_ng != props.node_group:
            subscribe_to_node_group(modifier_ng)

        # Node group selector using native template_ID picker
        layout.use_property_split = False  # template_ID doesn't work well with split
        layout.template_ID(
            props,
            "node_group",
            new="voxel_terrain.new_node_group",
        )
        layout.use_property_split = True

        # Show sync buttons if modifier differs, otherwise show "Show Modifier"
        if modifier_missing:
            self._draw_modifier_missing_ui(layout)
        elif out_of_sync:
            self._draw_modifier_sync_ui(layout, props, modifier_ng)
        elif props.node_group is not None:
            # Show Modifier button (only when in sync)
            layout.operator(
                "voxel_terrain.show_node_group",
                text="Show in Modifiers",
                icon="GEOMETRY_NODES",
            )

        # Use the effective node group (prefer modifier's if out of sync)
        effective_ng = modifier_ng if out_of_sync and modifier_ng else props.node_group

        # Sync sockets button - only enabled when sockets don't match
        if effective_ng is not None and not out_of_sync and not modifier_missing:
            sockets_match = check_sockets_match(effective_ng)
            self._draw_socket_sync_ui(layout, sockets_match)


# List of all classes to register - used by __init__.py
# Note: Order matters - parent panels must be registered before child panels
classes: tuple[type, ...] = (
    VOXELTERRAIN_PT_main_panel,
    VOXELTERRAIN_PT_npanel_voxel_grid,
    VOXELTERRAIN_PT_npanel_about,
    VOXELTERRAIN_PT_scene_panel,
    VOXELTERRAIN_PT_scene_settings,
    VOXELTERRAIN_PT_scene_actions,
    VOXELTERRAIN_PT_scene_export,
    VOXELTERRAIN_PT_object_panel,
)
