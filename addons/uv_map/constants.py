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
MAPPING_SHRINK_WRAP = "SHRINK_WRAP"
MAPPING_BOX = "BOX"

# Mapping type display names
MAPPING_CYLINDRICAL_CAPPED = "CYLINDRICAL_CAPPED"

# Normal-based mapping identifiers (use surface normal instead of position)
MAPPING_CYLINDRICAL_NORMAL = "CYLINDRICAL_NORMAL"
MAPPING_SPHERICAL_NORMAL = "SPHERICAL_NORMAL"
MAPPING_SHRINK_WRAP_NORMAL = "SHRINK_WRAP_NORMAL"

MAPPING_TYPES = [
    (MAPPING_PLANAR, "Planar", "Project UVs from a flat plane"),
    (MAPPING_CYLINDRICAL, "Cylindrical", "Project UVs from a cylinder"),
    (MAPPING_SPHERICAL, "Spherical", "Project UVs from a sphere"),
    (MAPPING_SHRINK_WRAP, "Shrink Wrap", "Project UVs with a single pole (azimuthal)"),
    (MAPPING_BOX, "Box", "Project UVs from a box (tri-planar)"),
]

# Default values
DEFAULT_SIZE = (1.0, 1.0, 1.0)
DEFAULT_TILE = (1.0, 1.0, 1.0)
DEFAULT_POSITION = (0.0, 0.0, 0.0)
DEFAULT_ROTATION = (0.0, 0.0, 0.0)

# Overlay colors (matching 3ds Max orange wireframe style)
OVERLAY_COLOR = (1.0, 0.5, 0.0, 1.0)  # Orange
OVERLAY_LINE_WIDTH = 1.0  # Match gridline thickness

# Socket identifiers for the main node group inputs
SOCKET_GEOMETRY = "Geometry"
SOCKET_SELECTION = "Selection"
SOCKET_MAPPING_TYPE = "Mapping Type"
SOCKET_GIZMO = "Gizmo"
SOCKET_POSITION = "Position"
SOCKET_ROTATION = "Rotation"
SOCKET_SIZE = "Size"
SOCKET_U_TILE = "U Tile"
SOCKET_V_TILE = "V Tile"
SOCKET_U_OFFSET = "U Offset"
SOCKET_V_OFFSET = "V Offset"
SOCKET_UV_ROTATION = "UV Rotation"
SOCKET_U_FLIP = "U Flip"
SOCKET_V_FLIP = "V Flip"
SOCKET_UV_MAP = "UV Map"
SOCKET_SMOOTH_NORMALS = "Smooth Normals"
SOCKET_NORMAL_BASED = "Normal-based"
SOCKET_CAP = "Cap"
SOCKET_SHOW_GIZMO = "Show Gizmo"

# Gizmo type identifiers
GIZMO_ALL = "ALL"
GIZMO_NONE = "NONE"
GIZMO_POSITION = "POSITION"
GIZMO_ROTATION = "ROTATION"
GIZMO_SIZE = "SIZE"

GIZMO_TYPES = [
    (GIZMO_NONE, "None", "Hide all gizmo controls"),
    (GIZMO_POSITION, "Position", "Show only position gizmo"),
    (GIZMO_ROTATION, "Rotation", "Show only rotation gizmo"),
    (GIZMO_SIZE, "Size", "Show only size gizmo"),
    (GIZMO_ALL, "All", "Show all gizmo controls (position, rotation, size)"),
]

# Modifier name
MODIFIER_NAME = "UV Map"
