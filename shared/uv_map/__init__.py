"""
Shared UV Map module for procedural UV mapping in Blender.

This module provides the core functionality for creating UV Map node groups
that can be used across multiple addons. It is designed to work standalone
or in conjunction with the UV Map addon which provides additional features
like overlays and regeneration.

Usage:
    from shared.uv_map import get_or_create_uv_map_node_group

    # Create or get existing UV Map node group
    node_group = get_or_create_uv_map_node_group()

    # Add as a modifier
    modifier = obj.modifiers.new(name="UV Map", type="NODES")
    modifier.node_group = node_group

    # Or insert into a geometry node tree
    group_node = node_tree.nodes.new("GeometryNodeGroup")
    group_node.node_tree = node_group
"""

from __future__ import annotations

from .constants import (
    DEFAULT_POSITION,
    DEFAULT_ROTATION,
    DEFAULT_SIZE,
    DEFAULT_TILE,
    GIZMO_ALL,
    GIZMO_NONE,
    GIZMO_POSITION,
    GIZMO_ROTATION,
    GIZMO_SIZE,
    GIZMO_TYPES,
    MAPPING_BOX,
    MAPPING_CYLINDRICAL,
    MAPPING_PLANAR,
    MAPPING_SHRINK_WRAP,
    MAPPING_SPHERICAL,
    MAPPING_TYPES,
    SOCKET_CAP,
    SOCKET_GEOMETRY,
    SOCKET_MAPPING_TYPE,
    SOCKET_NORMAL_BASED,
    SOCKET_POSITION,
    SOCKET_ROTATION,
    SOCKET_SELECTION,
    SOCKET_SHOW_GIZMO,
    SOCKET_SIZE,
    SOCKET_SMOOTH_NORMALS,
    SOCKET_U_FLIP,
    SOCKET_U_OFFSET,
    SOCKET_U_TILE,
    SOCKET_UV_MAP,
    SOCKET_UV_ROTATION,
    SOCKET_V_FLIP,
    SOCKET_V_OFFSET,
    SOCKET_V_TILE,
    SUB_GROUP_SUFFIXES,
    UV_MAP_NODE_GROUP_PREFIX,
    UV_MAP_NODE_GROUP_TAG,
)
from .nodes import (
    create_uv_map_node_group,
    get_or_create_uv_map_node_group,
    needs_regeneration,
    regenerate_uv_map_node_group,
)
from .utils import (
    get_uv_map_node_groups,
    is_uv_map_node_group,
)

__all__ = [
    "DEFAULT_POSITION",
    "DEFAULT_ROTATION",
    "DEFAULT_SIZE",
    "DEFAULT_TILE",
    "GIZMO_ALL",
    "GIZMO_NONE",
    "GIZMO_POSITION",
    "GIZMO_ROTATION",
    "GIZMO_SIZE",
    "GIZMO_TYPES",
    "MAPPING_BOX",
    "MAPPING_CYLINDRICAL",
    "MAPPING_PLANAR",
    "MAPPING_SHRINK_WRAP",
    "MAPPING_SPHERICAL",
    "MAPPING_TYPES",
    "SOCKET_CAP",
    "SOCKET_GEOMETRY",
    "SOCKET_MAPPING_TYPE",
    "SOCKET_NORMAL_BASED",
    "SOCKET_POSITION",
    "SOCKET_ROTATION",
    "SOCKET_SELECTION",
    "SOCKET_SHOW_GIZMO",
    "SOCKET_SIZE",
    "SOCKET_SMOOTH_NORMALS",
    "SOCKET_UV_MAP",
    "SOCKET_UV_ROTATION",
    "SOCKET_U_FLIP",
    "SOCKET_U_OFFSET",
    "SOCKET_U_TILE",
    "SOCKET_V_FLIP",
    "SOCKET_V_OFFSET",
    "SOCKET_V_TILE",
    "SUB_GROUP_SUFFIXES",
    "UV_MAP_NODE_GROUP_PREFIX",
    "UV_MAP_NODE_GROUP_TAG",
    "create_uv_map_node_group",
    "get_or_create_uv_map_node_group",
    "get_uv_map_node_groups",
    "is_uv_map_node_group",
    "needs_regeneration",
    "regenerate_uv_map_node_group",
]
