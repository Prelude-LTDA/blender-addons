"""
Menus module for UV Map addon.

Handles menu integration for:
- Modifiers > Edit menu (appended at end)
- Geometry Nodes > Mesh > UV submenu
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import bpy

if TYPE_CHECKING:
    from bpy.types import Context


def _draw_modifier_menu(self: bpy.types.Menu, context: Context) -> None:  # noqa: ARG001
    """Draw UV Map item in the Modifiers > Edit menu."""
    layout = self.layout
    assert layout is not None
    layout.operator("uv_map.add_modifier", icon="MOD_UVPROJECT")


def _draw_geonodes_menu(self: bpy.types.Menu, context: Context) -> None:  # noqa: ARG001
    """Draw UV Map item in the Geometry Nodes > Mesh > UV submenu."""
    layout = self.layout
    assert layout is not None
    layout.operator("uv_map.insert_node_group", icon="MOD_UVPROJECT")


# Custom menu class for the UV submenu in geometry nodes
class NODE_MT_uv_map_submenu(bpy.types.Menu):
    """UV Map submenu for geometry nodes."""

    bl_idname = "NODE_MT_uv_map_submenu"
    bl_label = "UV"

    def draw(self, context: Context) -> None:
        """Draw the menu."""
        layout = self.layout
        assert layout is not None
        # Our UV Map item
        layout.operator("uv_map.insert_node_group", icon="MOD_UVPROJECT")
        layout.separator()


def register_menus() -> None:
    """Register menu items."""
    # Add to the Edit submenu of modifiers (appended at end)
    if hasattr(bpy.types, "OBJECT_MT_modifier_add_edit"):
        bpy.types.OBJECT_MT_modifier_add_edit.append(_draw_modifier_menu)

    # For geometry nodes, add to the UV submenu if it exists
    if hasattr(bpy.types, "NODE_MT_geometry_node_mesh_uv"):
        bpy.types.NODE_MT_geometry_node_mesh_uv.prepend(_draw_geonodes_menu)  # type: ignore[attr-defined]

    # Add to the node editor add menu's Mesh submenu
    if hasattr(bpy.types, "NODE_MT_category_GEO_NODE_MESH"):
        bpy.types.NODE_MT_category_GEO_NODE_MESH.append(_draw_geonodes_menu)  # type: ignore[attr-defined]


def unregister_menus() -> None:
    """Unregister menu items."""
    # Remove from Edit submenu
    if hasattr(bpy.types, "OBJECT_MT_modifier_add_edit"):
        with contextlib.suppress(ValueError):
            bpy.types.OBJECT_MT_modifier_add_edit.remove(_draw_modifier_menu)

    # Remove from geometry nodes UV menu
    if hasattr(bpy.types, "NODE_MT_geometry_node_mesh_uv"):
        with contextlib.suppress(ValueError):
            bpy.types.NODE_MT_geometry_node_mesh_uv.remove(_draw_geonodes_menu)  # type: ignore[attr-defined]

    # Remove from Mesh submenu
    if hasattr(bpy.types, "NODE_MT_category_GEO_NODE_MESH"):
        with contextlib.suppress(ValueError):
            bpy.types.NODE_MT_category_GEO_NODE_MESH.remove(_draw_geonodes_menu)  # type: ignore[attr-defined]


# Classes to register
classes: list[type] = [
    NODE_MT_uv_map_submenu,
]
