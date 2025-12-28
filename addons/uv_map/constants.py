"""
Constants module for UV Map addon.

Contains constant values, identifiers, and configuration.
"""

from __future__ import annotations

# Node group identifier - used to tag procedurally generated UV Map node groups
UV_MAP_NODE_GROUP_PREFIX = "UV Map"
UV_MAP_NODE_GROUP_TAG = "uv_map.generated"

# Mapping type identifiers (matching 3ds Max UVW Map)
MAPPING_PLANAR = "PLANAR"
MAPPING_CYLINDRICAL = "CYLINDRICAL"
MAPPING_SPHERICAL = "SPHERICAL"
MAPPING_BOX = "BOX"

# Mapping type display names
MAPPING_TYPES = [
    (MAPPING_PLANAR, "Planar", "Project UVs from a flat plane"),
    (MAPPING_CYLINDRICAL, "Cylindrical", "Project UVs from a cylinder"),
    (MAPPING_SPHERICAL, "Spherical", "Project UVs from a sphere"),
    (MAPPING_BOX, "Box", "Project UVs from a box (tri-planar)"),
]

# Default values
DEFAULT_SIZE = (1.0, 1.0, 1.0)
DEFAULT_TILE = (1.0, 1.0, 1.0)
DEFAULT_POSITION = (0.0, 0.0, 0.0)
DEFAULT_ROTATION = (0.0, 0.0, 0.0)

# Overlay colors (matching 3ds Max orange wireframe style)
OVERLAY_COLOR = (1.0, 0.5, 0.0, 1.0)  # Orange
OVERLAY_LINE_WIDTH = 2.0

# Socket identifiers for the main node group inputs
SOCKET_GEOMETRY = "Geometry"
SOCKET_MAPPING_TYPE = "Mapping Type"
SOCKET_POSITION = "Position"
SOCKET_ROTATION = "Rotation"
SOCKET_SIZE = "Size"
SOCKET_U_TILE = "U Tile"
SOCKET_V_TILE = "V Tile"
SOCKET_W_TILE = "W Tile"
SOCKET_U_FLIP = "U Flip"
SOCKET_V_FLIP = "V Flip"
SOCKET_W_FLIP = "W Flip"
SOCKET_UV_MAP = "UV Map"

# Modifier name
MODIFIER_NAME = "UV Map"
