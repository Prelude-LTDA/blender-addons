"""
Overlay package for UV Map addon.

Renders wireframe visualization of the UV map shape (plane, cylinder, sphere, box)
in the 3D viewport when a UV Map modifier is active.

Also renders UV direction indicators showing U and V axis directions as curved
lines that follow the projection surface.
"""

from __future__ import annotations

from .handlers import register_draw_handler, unregister_draw_handler

__all__ = [
    "register_draw_handler",
    "unregister_draw_handler",
]

# Classes to register (none for this module)
classes: list[type] = []
