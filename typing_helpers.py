"""
Type stub helpers for Blender addon development.

This module provides additional type aliases and helpers that work
alongside fake-bpy-module for better type checking.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

# Common operator return types
OperatorReturn: TypeAlias = set[
    Literal[
        "RUNNING_MODAL",
        "CANCELLED",
        "FINISHED",
        "PASS_THROUGH",
        "INTERFACE",
    ]
]

# Common operator options
OperatorOptions: TypeAlias = set[
    Literal[
        "REGISTER",
        "UNDO",
        "UNDO_GROUPED",
        "BLOCKING",
        "MACRO",
        "GRAB_CURSOR",
        "GRAB_CURSOR_X",
        "GRAB_CURSOR_Y",
        "DEPENDS_ON_CURSOR",
        "PRESET",
        "INTERNAL",
    ]
]

# Space types for panels
SpaceType: TypeAlias = Literal[
    "EMPTY",
    "VIEW_3D",
    "IMAGE_EDITOR",
    "NODE_EDITOR",
    "SEQUENCE_EDITOR",
    "CLIP_EDITOR",
    "DOPESHEET_EDITOR",
    "GRAPH_EDITOR",
    "NLA_EDITOR",
    "TEXT_EDITOR",
    "CONSOLE",
    "INFO",
    "TOPBAR",
    "STATUSBAR",
    "OUTLINER",
    "PROPERTIES",
    "FILE_BROWSER",
    "SPREADSHEET",
    "PREFERENCES",
]

# Region types for panels
RegionType: TypeAlias = Literal[
    "WINDOW",
    "HEADER",
    "CHANNELS",
    "TEMPORARY",
    "UI",
    "TOOLS",
    "TOOL_PROPS",
    "ASSET_SHELF",
    "ASSET_SHELF_HEADER",
    "PREVIEW",
    "HUD",
    "NAVIGATION_BAR",
    "EXECUTE",
    "FOOTER",
    "TOOL_HEADER",
    "XR",
]
