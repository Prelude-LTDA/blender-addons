"""
Menus module for UV Map addon.

Handles menu integration for:
- Modifiers > Edit menu (appended at end)
- Geometry Nodes > Add > Mesh > UV submenu
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


def _draw_geonodes_uv_menu(self: bpy.types.Menu, context: Context) -> None:  # noqa: ARG001
    """Draw UV Map item in the Geometry Nodes UV submenu."""
    layout = self.layout
    assert layout is not None
    layout.operator("uv_map.insert_node_group")


# The UV menu in geometry nodes is dynamically generated.
# In Blender's node_add_menu_geometry.py, the menus are generated with keys from add_menus dict.
# Looking at the source, there appears to be a naming swap where:
# - NODE_MT_category_PRIMITIVES_MESH uses NODE_MT_gn_mesh_uv_base
# - NODE_MT_category_GEO_UV uses NODE_MT_gn_mesh_primitives_base
# This seems to be a bug in Blender's source, but we handle it.
_GEONODES_ADD_MENU = (
    "NODE_MT_category_PRIMITIVES_MESH"  # Add menu (swapped naming in Blender)
)
_GEONODES_SWAP_MENU = "NODE_MT_gn_mesh_uv_swap"  # Swap menu

_registered_add_menu = False
_registered_swap_menu = False


def register_menus() -> None:
    """Register menu items."""
    global _registered_add_menu, _registered_swap_menu

    # Add to the Edit submenu of modifiers (appended at end)
    if hasattr(bpy.types, "OBJECT_MT_modifier_add_edit"):
        bpy.types.OBJECT_MT_modifier_add_edit.append(_draw_modifier_menu)

    # Append to the UV submenu in Geometry Nodes Add menu
    if hasattr(bpy.types, _GEONODES_ADD_MENU):
        menu_class = getattr(bpy.types, _GEONODES_ADD_MENU)
        menu_class.append(_draw_geonodes_uv_menu)
        _registered_add_menu = True
        print(f"[uv_map] Registered UV Map in Add menu: {_GEONODES_ADD_MENU}")

    # Also append to the Swap menu
    if hasattr(bpy.types, _GEONODES_SWAP_MENU):
        menu_class = getattr(bpy.types, _GEONODES_SWAP_MENU)
        menu_class.append(_draw_geonodes_uv_menu)
        _registered_swap_menu = True
        print(f"[uv_map] Registered UV Map in Swap menu: {_GEONODES_SWAP_MENU}")

    if not _registered_add_menu and not _registered_swap_menu:
        # Debug: print available menu types to help diagnose
        print("[uv_map] Warning: Could not find geometry nodes UV menus")
        node_menus = [
            name
            for name in dir(bpy.types)
            if name.startswith("NODE_MT")
            and ("mesh" in name.lower() or "uv" in name.lower())
        ]
        print(f"[uv_map] Available NODE_MT menus with 'mesh' or 'uv': {node_menus}")


def unregister_menus() -> None:
    """Unregister menu items."""
    global _registered_add_menu, _registered_swap_menu

    # Remove from Edit submenu
    if hasattr(bpy.types, "OBJECT_MT_modifier_add_edit"):
        with contextlib.suppress(ValueError):
            bpy.types.OBJECT_MT_modifier_add_edit.remove(_draw_modifier_menu)

    # Remove from Add menu
    if _registered_add_menu and hasattr(bpy.types, _GEONODES_ADD_MENU):
        menu_class = getattr(bpy.types, _GEONODES_ADD_MENU)
        with contextlib.suppress(ValueError):
            menu_class.remove(_draw_geonodes_uv_menu)
        _registered_add_menu = False

    # Remove from Swap menu
    if _registered_swap_menu and hasattr(bpy.types, _GEONODES_SWAP_MENU):
        menu_class = getattr(bpy.types, _GEONODES_SWAP_MENU)
        with contextlib.suppress(ValueError):
            menu_class.remove(_draw_geonodes_uv_menu)
        _registered_swap_menu = False


# Classes to register (none needed now since we use existing menu)
classes: list[type] = []
