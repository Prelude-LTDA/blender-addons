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

# Module names for registration (order matters for dependencies)
_module_names: tuple[str, ...] = (
    "operators",
    "panels",
)

# Cache for loaded modules
_modules: dict[str, object] = {}


def _import_modules() -> None:
    """Import or reload all addon modules."""
    import sys

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

    _import_modules()

    for cls in _get_classes():
        bpy.utils.register_class(cls)

    print(f"[{__package__}] Addon registered successfully")

    # Show popup after a short delay (to ensure UI is ready)
    bpy.app.timers.register(_show_reload_popup, first_interval=0.1)


def unregister() -> None:
    """Unregister all addon classes from Blender."""
    import contextlib

    import bpy

    # Unregister in reverse order
    for cls in reversed(_get_classes()):
        with contextlib.suppress(RuntimeError):
            bpy.utils.unregister_class(cls)

    print(f"[{__package__}] Addon unregistered")


# Allow running from Blender's text editor for development
if __name__ == "__main__":
    register()
