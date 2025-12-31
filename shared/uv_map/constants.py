"""
Constants for UV Map node group generation.

These constants are shared across all addons that use UV mapping.
"""

from __future__ import annotations

# Node group identifier - used to tag procedurally generated UV Map node groups
UV_MAP_NODE_GROUP_PREFIX = "UV Map"
UV_MAP_NODE_GROUP_TAG = "uv_map.generated"

# Sub-group suffix constants (used when building nested group names)
SUB_GROUP_SUFFIX_PLANAR = " - Planar"
SUB_GROUP_SUFFIX_CYLINDRICAL = " - Cylindrical"
SUB_GROUP_SUFFIX_CYLINDRICAL_CAPPED = " - Cylindrical Capped"
SUB_GROUP_SUFFIX_SPHERICAL = " - Spherical"
SUB_GROUP_SUFFIX_SHRINK_WRAP = " - Shrink Wrap"
SUB_GROUP_SUFFIX_BOX = " - Box"
SUB_GROUP_SUFFIX_CYLINDRICAL_NORMAL = " - Cylindrical Normal"
SUB_GROUP_SUFFIX_CYLINDRICAL_CAPPED_NORMAL = " - Cylindrical Capped Normal"
SUB_GROUP_SUFFIX_SPHERICAL_NORMAL = " - Spherical Normal"
SUB_GROUP_SUFFIX_SHRINK_WRAP_NORMAL = " - Shrink Wrap Normal"

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
MAPPING_CYLINDRICAL_NORMAL_CAPPED = "CYLINDRICAL_NORMAL_CAPPED"
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

# Sub-group suffixes (used for cleanup during regeneration)
SUB_GROUP_SUFFIXES = [
    SUB_GROUP_SUFFIX_PLANAR,
    SUB_GROUP_SUFFIX_CYLINDRICAL,
    SUB_GROUP_SUFFIX_CYLINDRICAL_CAPPED,
    SUB_GROUP_SUFFIX_SPHERICAL,
    SUB_GROUP_SUFFIX_SHRINK_WRAP,
    SUB_GROUP_SUFFIX_BOX,
    SUB_GROUP_SUFFIX_CYLINDRICAL_NORMAL,
    SUB_GROUP_SUFFIX_CYLINDRICAL_CAPPED_NORMAL,
    SUB_GROUP_SUFFIX_SPHERICAL_NORMAL,
    SUB_GROUP_SUFFIX_SHRINK_WRAP_NORMAL,
]
