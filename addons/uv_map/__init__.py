"""
UV Map - 3ds Max-style UVW Map modifier for Blender.

This addon provides procedural UV mapping similar to 3ds Max's UVW Map modifier:
- Multiple mapping types: Planar, Cylindrical, Spherical, Box
- Transform controls for UV map origin (position, rotation, scale)
- Tiling controls for U and V
- Overlay visualization of the UV map shape
- Gizmos for interactive transform editing (Blender 4.3+)

Usage:
- Add via Modifiers > Edit > UV Map menu
- Or insert the node group via Geometry Nodes > Mesh > UV > UV Map
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Shared module names (need to be reloaded first, in dependency order)
_shared_module_names: tuple[str, ...] = (
    "shared",
    "shared.node_layout",
    "shared.node_group_compare",
)

# Module names for registration (order matters for dependencies)
_module_names: tuple[str, ...] = (
    "constants",
    "nodes",
    "overlay",
    "operators",
    "menus",
)

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
            # Module already loaded - reload it
            _modules[name] = importlib.reload(sys.modules[full_name])
        else:
            # First time import
            _modules[name] = importlib.import_module(full_name)


def _get_classes() -> Sequence[type]:
    """Collect all registerable classes from modules."""
    classes: list[type] = []

    for name in _module_names:
        module = _modules.get(name)
        if module and hasattr(module, "classes"):
            classes.extend(module.classes)  # type: ignore[attr-defined]

    return classes


def _show_reload_popup() -> None:
    """Show a confirmation dialog when addon is reloaded and regeneration is needed.

    Only shows if:
    - UV Map node groups exist in the current blend file
    - The existing groups differ structurally from what the current code would create
    """
    import bpy

    from .nodes import get_uv_map_node_groups, needs_regeneration

    existing_groups = get_uv_map_node_groups()

    # Only show dialog if there are existing groups AND they need regeneration
    if existing_groups and needs_regeneration():
        # Use the confirmation dialog operator
        bpy.ops.uv_map.regenerate_confirm("INVOKE_DEFAULT")  # type: ignore[attr-defined]


def register() -> None:
    """Register all addon classes with Blender."""
    import bpy

    _import_modules()

    for cls in _get_classes():
        bpy.utils.register_class(cls)

    # Register menus
    from . import menus

    menus.register_menus()

    # Register overlay draw handler
    from . import overlay

    overlay.register_draw_handler()

    print(f"[{__package__}] Addon registered successfully")

    # Show popup after a short delay (to ensure UI is ready)
    bpy.app.timers.register(_show_reload_popup, first_interval=0.1)


def unregister() -> None:
    """Unregister all addon classes from Blender."""
    import contextlib

    import bpy

    # Unregister overlay draw handler
    from . import overlay

    overlay.unregister_draw_handler()

    # Unregister menus
    from . import menus

    menus.unregister_menus()

    # Unregister in reverse order
    for cls in reversed(_get_classes()):
        with contextlib.suppress(RuntimeError):
            bpy.utils.unregister_class(cls)

    print(f"[{__package__}] Addon unregistered")


# Allow running from Blender's text editor for development
if __name__ == "__main__":
    register()
