"""Frame management for node layout.

Handles saving, removing, and restoring frame nodes during layout operations.
"""

from __future__ import annotations

import bpy

from .types import SavedFrame

__all__ = [
    "assign_reroutes_to_frames",
    "get_all_frames_containing",
    "get_connected_non_reroute_nodes",
    "remove_all_frames",
    "restore_all_frames",
    "save_all_frames",
]


def save_all_frames(
    node_tree: bpy.types.NodeTree,
    nodes_to_layout: set[bpy.types.Node] | None = None,
) -> list[SavedFrame]:
    """Save information about frames containing nodes to be laid out.

    Args:
        node_tree: The node tree
        nodes_to_layout: If provided, only save frames that contain these nodes

    Returns frames sorted so parent frames come before child frames,
    which is needed for proper restoration.
    """
    saved_frames: list[SavedFrame] = []

    # Collect frames to save
    if nodes_to_layout is None:
        frames = [n for n in node_tree.nodes if n.type == "FRAME"]
    else:
        # Find frames that contain any of our nodes
        frames_to_save: set[bpy.types.Node] = set()
        for node in nodes_to_layout:
            parent = node.parent
            while parent is not None:
                frames_to_save.add(parent)
                parent = parent.parent
        frames = list(frames_to_save)

    for frame in frames:
        # Find direct children of this frame that are in our layout set
        if nodes_to_layout is None:
            child_names = [
                n.name
                for n in node_tree.nodes
                if n.parent == frame and n.type != "FRAME"
            ]
        else:
            child_names = [
                n.name
                for n in node_tree.nodes
                if n.parent == frame and n.type != "FRAME" and n in nodes_to_layout
            ]

        # Get parent frame name if nested
        parent_name = frame.parent.name if frame.parent is not None else None

        saved_frames.append(
            SavedFrame(
                name=frame.name,
                label=frame.label,
                color=(frame.color.r, frame.color.g, frame.color.b),
                use_custom_color=frame.use_custom_color,
                label_size=frame.label_size,  # type: ignore[attr-defined]
                child_node_names=child_names,
                parent_frame_name=parent_name,
            )
        )

    # Sort so parent frames come before their children
    # (frames with no parent first, then frames whose parents are already in the list)
    sorted_frames: list[SavedFrame] = []
    remaining = saved_frames.copy()

    while remaining:
        # Find frames whose parent is already processed (or has no parent)
        processed_names = {f.name for f in sorted_frames}
        ready = [
            f
            for f in remaining
            if f.parent_frame_name is None or f.parent_frame_name in processed_names
        ]

        if not ready:
            # Cycle or orphan - just add remaining
            sorted_frames.extend(remaining)
            break

        for f in ready:
            sorted_frames.append(f)
            remaining.remove(f)

    return sorted_frames


def remove_all_frames(
    node_tree: bpy.types.NodeTree,
    nodes_to_layout: set[bpy.types.Node] | None = None,
) -> None:
    """Remove frame nodes and detach their children.

    Args:
        node_tree: The node tree to modify
        nodes_to_layout: If provided, only detach these nodes from frames
                        and only remove frames that become empty
    """
    if nodes_to_layout is None:
        # Detach all nodes from frames
        for node in node_tree.nodes:
            if node.parent is not None:
                node.parent = None

        # Remove all frame nodes
        frames = [n for n in node_tree.nodes if n.type == "FRAME"]
        for frame in frames:
            node_tree.nodes.remove(frame)
    else:
        # Only detach nodes we're laying out
        for node in nodes_to_layout:
            if node.parent is not None:
                node.parent = None

        # Don't remove frames - they may still have other children


def restore_all_frames(
    node_tree: bpy.types.NodeTree,
    saved_frames: list[SavedFrame],
) -> None:
    """Recreate all frames and re-parent nodes to them.

    saved_frames must be sorted so parent frames come before children.
    If a frame with the same name already exists, reuse it instead of creating a new one.
    Also adds reroute nodes to frames if they only connect nodes within that frame.
    """
    # Create or reuse frames in order (parents before children)
    frame_nodes: dict[str, bpy.types.Node] = {}

    for saved in saved_frames:
        # Check if frame already exists (e.g., in selected-only mode where
        # frames with non-selected children are preserved)
        existing_frame = node_tree.nodes.get(saved.name)
        if existing_frame is not None and existing_frame.type == "FRAME":
            frame = existing_frame
        else:
            frame = node_tree.nodes.new("NodeFrame")
            frame.name = saved.name
            frame.label = saved.label
            frame.color = saved.color
            frame.use_custom_color = saved.use_custom_color
            frame.label_size = saved.label_size  # type: ignore[attr-defined]
        frame_nodes[saved.name] = frame

    # Set parent relationships between frames
    for saved in saved_frames:
        if saved.parent_frame_name is not None:
            parent_frame = frame_nodes.get(saved.parent_frame_name)
            if parent_frame is not None:
                frame_nodes[saved.name].parent = parent_frame

    # Re-parent child nodes to their frames
    for saved in saved_frames:
        frame = frame_nodes[saved.name]
        for child_name in saved.child_node_names:
            child_node = node_tree.nodes.get(child_name)
            if child_node is not None:
                child_node.parent = frame

    # Now handle reroutes: add them to frames if they ONLY connect nodes within that frame
    assign_reroutes_to_frames(node_tree, frame_nodes)


def get_all_frames_containing(node: bpy.types.Node) -> set[bpy.types.Node]:
    """Get all frames containing this node (from immediate parent up to root)."""
    frames: set[bpy.types.Node] = set()
    parent = node.parent
    while parent is not None and parent.type == "FRAME":
        frames.add(parent)
        parent = parent.parent
    return frames


def assign_reroutes_to_frames(
    node_tree: bpy.types.NodeTree,
    frame_nodes: dict[str, bpy.types.Node],
) -> None:
    """Assign reroute nodes to frames if all their connections are within that frame.

    A reroute is added to a frame if:
    1. All nodes it connects to (directly or through other reroutes) are in the same frame
    2. This includes both upstream and downstream connections

    For nested frames, reroutes are assigned to the innermost common frame.
    """
    if not frame_nodes:
        return

    # Get all reroutes
    reroutes = [n for n in node_tree.nodes if n.type == "REROUTE"]

    for reroute in reroutes:
        # Skip if already parented
        if reroute.parent is not None:
            continue

        # Find all non-reroute nodes this reroute connects to (both directions)
        connected_nodes = get_connected_non_reroute_nodes(node_tree, reroute)

        if not connected_nodes:
            continue

        # Find the common frame for all connected nodes
        # Start with frames containing the first node
        first_node = connected_nodes[0]
        common_frames = get_all_frames_containing(first_node)

        # Intersect with frames of all other nodes
        for node in connected_nodes[1:]:
            node_frames = get_all_frames_containing(node)
            common_frames &= node_frames

        if not common_frames:
            # No common frame - don't parent the reroute
            continue

        # Find the innermost (most deeply nested) common frame
        innermost_frame = None
        max_depth = -1
        for frame in common_frames:
            depth = 0
            parent = frame.parent
            while parent is not None and parent.type == "FRAME":
                depth += 1
                parent = parent.parent
            if depth > max_depth:
                max_depth = depth
                innermost_frame = frame

        if innermost_frame is not None:
            reroute.parent = innermost_frame


def get_connected_non_reroute_nodes(
    _node_tree: bpy.types.NodeTree,
    reroute: bpy.types.Node,
) -> list[bpy.types.Node]:
    """Get all non-reroute nodes connected to a reroute (following reroute chains)."""
    connected: list[bpy.types.Node] = []
    visited_reroutes: set[bpy.types.Node] = set()

    def trace_connections(current_reroute: bpy.types.Node) -> None:  # noqa: PLR0912
        if current_reroute in visited_reroutes:
            return
        visited_reroutes.add(current_reroute)

        # Check upstream (input)
        if current_reroute.inputs:
            input_links = current_reroute.inputs[0].links
            if input_links is not None:
                for link in input_links:
                    if link.from_node is not None:
                        if link.from_node.type == "REROUTE":
                            trace_connections(link.from_node)
                        else:
                            connected.append(link.from_node)

        # Check downstream (output)
        if current_reroute.outputs:
            output_links = current_reroute.outputs[0].links
            if output_links is not None:
                for link in output_links:
                    if link.to_node is not None:
                        if link.to_node.type == "REROUTE":
                            trace_connections(link.to_node)
                        else:
                            connected.append(link.to_node)

    trace_connections(reroute)
    return connected
