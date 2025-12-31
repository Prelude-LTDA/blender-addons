"""
Nodes module for UV Map addon.

Re-exports the shared UV map node generation functionality and provides
additional addon-specific features like overlay integration.
"""

from __future__ import annotations

# Re-export everything from the shared module
# This allows the addon to work as before while using shared code
from .shared.uv_map import (
    SUB_GROUP_SUFFIXES as _SUB_GROUP_SUFFIXES,
)
from .shared.uv_map.nodes import (
    _cleanup_reference_groups,
    _force_new_subgroups,
    create_uv_map_node_group,
    get_or_create_uv_map_node_group,
    get_uv_map_node_groups,
    is_uv_map_node_group,
    needs_regeneration,
    regenerate_uv_map_node_group,
)

# Re-export for backwards compatibility
__all__ = [
    "_SUB_GROUP_SUFFIXES",
    "_cleanup_reference_groups",
    "_force_new_subgroups",
    "create_uv_map_node_group",
    "get_or_create_uv_map_node_group",
    "get_uv_map_node_groups",
    "is_uv_map_node_group",
    "needs_regeneration",
    "regenerate_uv_map_node_group",
]

# Classes to register (none for this module - it's utility only)
classes: list[type] = []
