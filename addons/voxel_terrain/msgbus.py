"""
Message bus subscriptions for Voxel Terrain.

Handles real-time updates when node group sockets change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import bpy

if TYPE_CHECKING:
    from bpy.types import Depsgraph, Scene

# Subscription owner (must be kept alive while subscription is active)
_msgbus_owner: object = object()

# Track subscribed node groups to avoid duplicate subscriptions
_subscribed_node_groups: set[str] = set()


def _redraw_properties_panels() -> None:
    """Tag all Properties areas for redraw."""
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "PROPERTIES":
                area.tag_redraw()


def _on_node_tree_update(*_args: Any) -> None:
    """Called when any node tree is modified. Triggers panel redraw."""
    _redraw_properties_panels()


def _on_depsgraph_update(_scene: Scene, depsgraph: Depsgraph) -> None:
    """Check for node tree changes and trigger panel redraw."""
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.NodeTree):
            _redraw_properties_panels()
            return


def subscribe_to_node_group(node_group: bpy.types.NodeTree) -> None:
    """Subscribe to a specific node group's interface changes."""
    if node_group.name in _subscribed_node_groups:
        return

    _subscribed_node_groups.add(node_group.name)

    # Subscribe to changes on this specific node tree's interface
    bpy.msgbus.subscribe_rna(
        key=node_group.path_resolve("interface", False),  # type: ignore[arg-type]
        owner=_msgbus_owner,
        args=(),
        notify=_on_node_tree_update,
        options={"PERSISTENT"},
    )


def register_msgbus() -> None:
    """Register message bus subscriptions for node tree changes."""
    _subscribed_node_groups.clear()

    # Subscribe to changes on all NodeTree data (generic)
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.NodeTree, "interface"),  # type: ignore[arg-type]
        owner=_msgbus_owner,
        args=(),
        notify=_on_node_tree_update,
        options={"PERSISTENT"},
    )

    # Use depsgraph handler as backup
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister_msgbus() -> None:
    """Unregister message bus subscriptions."""
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    _subscribed_node_groups.clear()

    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
