"""
Addon-specific constants for UV Map overlay and UI.

For shared UV map constants, import from .shared.uv_map.constants directly.
"""

from __future__ import annotations

# Overlay colors (matching 3ds Max orange wireframe style)
OVERLAY_COLOR = (1.0, 0.5, 0.0, 1.0)  # Orange
OVERLAY_LINE_WIDTH = 1.0  # Match gridline thickness
OVERLAY_UV_DIRECTION_LINE_WIDTH = 2.0  # Thicker for UV direction visibility

# UV Direction colors - both yellow but slightly different shades for U vs V
OVERLAY_U_DIRECTION_COLOR = (1.0, 0.85, 0.0, 1.0)  # Yellow (slightly warmer for U)
OVERLAY_V_DIRECTION_COLOR = (0.9, 1.0, 0.0, 1.0)  # Yellow-green (slightly cooler for V)

# Text label settings
OVERLAY_LABEL_FONT_SIZE = 14
OVERLAY_LABEL_OFFSET_X = 5  # Offset from point in screen pixels
OVERLAY_LABEL_OFFSET_Y = 5

# Modifier name
MODIFIER_NAME = "UV Map"
