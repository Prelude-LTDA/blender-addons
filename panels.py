"""
Panels module for Voxel Terrain.

Contains all panel classes for the addon UI, including the N-panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

if TYPE_CHECKING:
    from bpy.types import Context


class VOXELTERRAIN_PT_main_panel(bpy.types.Panel):
    """Main panel in the N-panel sidebar."""

    bl_idname = "VOXELTERRAIN_PT_main_panel"
    bl_label = "Voxel Terrain"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Voxel"  # This creates the tab name in the N-panel

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        # Header section
        layout.label(text="Terrain Tools", icon="MESH_GRID")

        # Separator
        layout.separator()

        # Operators section
        box = layout.box()
        assert box is not None
        box.label(text="Actions:", icon="PLAY")

        # Add operator buttons
        row = box.row(align=True)
        row.operator("voxel_terrain.generate", text="Generate", icon="ADD")

        row = box.row(align=True)
        row.operator("voxel_terrain.clear", text="Clear", icon="TRASH")

        # Object info section
        layout.separator()

        obj = context.active_object
        if obj:
            info_box = layout.box()
            assert info_box is not None
            info_box.label(text="Selected Object:", icon="OBJECT_DATAMODE")

            col = info_box.column(align=True)
            assert col is not None
            col.label(text=f"Name: {obj.name}")
            col.label(text=f"Type: {obj.type}")

            # Show location
            loc = obj.location
            col.label(text=f"Location: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")
        else:
            layout.label(text="No terrain selected", icon="INFO")


class VOXELTERRAIN_PT_settings_subpanel(bpy.types.Panel):
    """Settings subpanel in the N-panel sidebar."""

    bl_idname = "VOXELTERRAIN_PT_settings_subpanel"
    bl_label = "Settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Voxel"
    bl_parent_id = "VOXELTERRAIN_PT_main_panel"  # Makes this a sub-panel
    bl_options = {"DEFAULT_CLOSED"}  # Collapsed by default

    def draw(self, context: Context) -> None:
        """Draw the panel contents."""
        layout = self.layout
        assert layout is not None

        # Example settings UI
        layout.label(text="Terrain Settings", icon="PREFERENCES")

        # You can add properties here when you create addon preferences
        # For example:
        # layout.prop(context.scene, "voxel_terrain_resolution")

        layout.label(text="(No settings yet)", icon="INFO")


class VOXELTERRAIN_PT_about_subpanel(bpy.types.Panel):
    """About subpanel in the N-panel sidebar."""

    bl_idname = "VOXELTERRAIN_PT_about_subpanel"
    bl_label = "About"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Voxel"
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


# List of all classes to register - used by __init__.py
# Note: Order matters - parent panels must be registered before child panels
classes: tuple[type, ...] = (
    VOXELTERRAIN_PT_main_panel,
    VOXELTERRAIN_PT_settings_subpanel,
    VOXELTERRAIN_PT_about_subpanel,
)
