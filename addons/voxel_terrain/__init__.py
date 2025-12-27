"""
Voxel Terrain - Voxel-based terrain generation and editing tools for Blender.

This addon provides:
- Type annotations
- Proper module reloading support
- Separate modules for operators and panels
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
)

# Module names for registration (order matters for dependencies)
_module_names: tuple[str, ...] = (
    "properties",
    "typing_utils",
    "sockets",
    "msgbus",
    "chunks",
    "generation",
    "operators",
    "panels",
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
    """Show a popup to confirm addon was reloaded."""
    import bpy

    def draw(self: object, context: object) -> None:  # noqa: ARG001
        layout = self.layout  # type: ignore[attr-defined]
        layout.label(text="Voxel Terrain addon reloaded!")

    wm = bpy.context.window_manager
    if wm is not None:
        wm.popup_menu(draw, title="Reload Success", icon="INFO")


def register() -> None:
    """Register all addon classes with Blender."""
    import bpy

    from . import chunks, msgbus, operators, properties

    _import_modules()

    for cls in _get_classes():
        bpy.utils.register_class(cls)

    # Register scene properties
    properties.register_scene_properties()

    # Register viewport draw handler
    chunks.register_draw_handler()

    # Register message bus for real-time socket change detection
    msgbus.register_msgbus()

    # Add to File > Export menu
    bpy.types.TOPBAR_MT_file_export.append(operators.menu_func_export)

    print(f"[{__package__}] Addon registered successfully")

    # Show popup after a short delay (to ensure UI is ready)
    bpy.app.timers.register(_show_reload_popup, first_interval=0.1)


def unregister() -> None:
    """Unregister all addon classes from Blender."""
    import contextlib

    import bpy

    from . import chunks, msgbus, operators, properties

    # Unregister message bus subscriptions
    msgbus.unregister_msgbus()

    # Unregister viewport draw handler
    chunks.unregister_draw_handler()

    # Remove from File > Export menu
    bpy.types.TOPBAR_MT_file_export.remove(operators.menu_func_export)

    # Unregister scene properties first
    properties.unregister_scene_properties()

    # Unregister in reverse order
    for cls in reversed(_get_classes()):
        with contextlib.suppress(RuntimeError):
            bpy.utils.unregister_class(cls)

    print(f"[{__package__}] Addon unregistered")


# Allow running from Blender's text editor for development
if __name__ == "__main__":
    register()
