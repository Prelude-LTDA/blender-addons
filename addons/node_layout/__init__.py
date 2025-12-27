"""
Node Layout - Automatically arrange nodes in a clean PCB-style layout.

This addon provides a single command to organize nodes in any node editor:
- Geometry Nodes
- Shader Nodes
- Compositor Nodes
- And any other node-based editors

Usage:
- Press the keyboard shortcut (default: Shift+L) in any node editor
- Or use the Node menu > Auto Layout Nodes
- Or search for "Auto Layout Nodes" in the command palette (F3)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import bpy

if TYPE_CHECKING:
    from collections.abc import Sequence

# Shared module names (need to be reloaded first, in dependency order)
_shared_module_names: tuple[str, ...] = (
    "shared",
    "shared.node_layout",
)

# Addon module names for registration
_module_names: tuple[str, ...] = ("operators",)

# Cache for loaded modules
_modules: dict[str, object] = {}


def _import_modules() -> None:
    """Import or reload all addon modules."""
    import sys

    # First, reload shared modules (they need to be reloaded before dependents)
    for name in _shared_module_names:
        full_name = f"{__package__}.{name}"
        if full_name in sys.modules:
            importlib.reload(sys.modules[full_name])

    # Then reload addon modules
    for name in _module_names:
        full_name = f"{__package__}.{name}"

        if full_name in sys.modules:
            _modules[name] = importlib.reload(sys.modules[full_name])
        else:
            _modules[name] = importlib.import_module(full_name)


def _get_classes() -> Sequence[type]:
    """Collect all registerable classes from modules."""
    classes: list[type] = []

    for name in _module_names:
        module = _modules.get(name)
        if module and hasattr(module, "classes"):
            classes.extend(module.classes)  # type: ignore[attr-defined]

    return classes


# Menu draw function
def draw_node_menu(self: bpy.types.Menu, context: bpy.types.Context) -> None:  # noqa: ARG001
    """Add Auto Layout to the Node menu."""
    layout = self.layout
    layout.separator()
    layout.operator("node.auto_layout", icon="SNAP_GRID")


# Keymap storage
addon_keymaps: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []


def register() -> None:
    """Register all addon classes with Blender."""
    _import_modules()

    for cls in _get_classes():
        bpy.utils.register_class(cls)

    # Add to Node Editor menus
    bpy.types.NODE_MT_node.append(draw_node_menu)

    # Add to context menu (right-click)
    from . import operators
    operators.register_menus()

    # Register keyboard shortcut
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon is not None:
        km = wm.keyconfigs.addon.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
        kmi = km.keymap_items.new("node.auto_layout", "V", "PRESS")
        addon_keymaps.append((km, kmi))


def unregister() -> None:
    """Unregister all addon classes from Blender."""
    # Remove keyboard shortcuts
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    # Remove from context menu
    from . import operators
    operators.unregister_menus()

    # Remove from menus
    bpy.types.NODE_MT_node.remove(draw_node_menu)

    # Unregister classes in reverse order
    for cls in reversed(_get_classes()):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
