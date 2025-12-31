"""
Utility functions for UV Map node groups.

Provides functions to check and identify UV Map node groups.
These are re-exported from the nodes module for convenience.
"""

from __future__ import annotations

# Re-export utility functions from nodes module
# This avoids code duplication while providing a clean API
from .nodes import get_uv_map_node_groups, is_uv_map_node_group

__all__ = [
    "get_uv_map_node_groups",
    "is_uv_map_node_group",
]
