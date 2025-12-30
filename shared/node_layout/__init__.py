"""
PCB-Style Node Layout System.

A general-purpose layout algorithm that arranges nodes like an FPGA:
- Nodes are placed in discrete cells on an X/Y grid
- Each cell can contain 0-1 nodes plus multiple reroute nodes
- Connections are routed taxicab-style: first Y, then X
- Reroutes from the same source are reused when possible
- Results in clean, readable node graphs with organized routing lanes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy


def _trace_to_real_source(socket: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
    """Trace back through reroute nodes to find the real source socket."""
    current = socket
    visited: set[bpy.types.NodeSocket] = set()

    while current.node is not None and current.node.type == "REROUTE":
        if current in visited:
            # Cycle detected, return current to avoid infinite loop
            break
        visited.add(current)

        # Reroute nodes have one input and one output
        # If we're on the output, find what's connected to the input
        reroute_node = current.node
        if not reroute_node.inputs or not reroute_node.inputs[0].links:
            # Dead end - reroute has no input connection
            break

        # Follow the link to the source
        link = reroute_node.inputs[0].links[0]
        if link.from_socket is None:
            break
        current = link.from_socket

    return current


def _remove_all_reroutes(
    node_tree: bpy.types.NodeTree,
    nodes_to_layout: set[bpy.types.Node] | None = None,
) -> None:
    """Remove reroute nodes and reconnect through them.

    For each connection that goes through reroutes, traces back to find
    the real source and creates a direct connection.

    Args:
        node_tree: The node tree to modify
        nodes_to_layout: If provided, only remove reroutes where both source and dest are in this set
    """
    # Pre-build adjacency map for O(1) lookups: node -> list of (to_node, link)
    outgoing_links: dict[
        bpy.types.Node, list[tuple[bpy.types.Node, bpy.types.NodeLink]]
    ] = {}
    for link in node_tree.links:
        if not link.is_valid:
            continue
        from_node = link.from_node
        to_node = link.to_node
        if from_node is None or to_node is None:
            continue
        if from_node not in outgoing_links:
            outgoing_links[from_node] = []
        outgoing_links[from_node].append((to_node, link))

    # First, collect all connections that need to be preserved
    # Map: (real_source_socket, destination_socket)
    connections_to_restore: list[tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]] = []

    # Track which reroutes are part of connections we're restoring (both ends in our set)
    reroutes_to_potentially_remove: set[bpy.types.Node] = set()

    for link in node_tree.links:
        if not link.is_valid:
            continue

        to_socket = link.to_socket
        to_node = link.to_node

        # Skip if destination is a reroute (we'll handle it from the final destination)
        if to_node is None or to_node.type == "REROUTE":
            continue

        # Skip if no destination socket
        if to_socket is None:
            continue

        # Trace back through any reroutes to find real source
        from_socket = link.from_socket
        if from_socket is None:
            continue

        real_source = _trace_to_real_source(from_socket)

        # Only process if we found a real (non-reroute) source
        if real_source.node is None or real_source.node.type == "REROUTE":
            continue

        # If filtering, only restore connections where BOTH endpoints are in our set
        if nodes_to_layout is not None:
            if (
                real_source.node not in nodes_to_layout
                or to_node not in nodes_to_layout
            ):
                continue

        connections_to_restore.append((real_source, to_socket))

        # Mark reroutes in this chain for potential removal
        current = from_socket
        while current.node is not None and current.node.type == "REROUTE":
            reroutes_to_potentially_remove.add(current.node)
            reroute_node = current.node
            if reroute_node.inputs and reroute_node.inputs[0].links:
                link_to_input = reroute_node.inputs[0].links[0]
                if link_to_input.from_socket is not None:
                    current = link_to_input.from_socket
                else:
                    break
            else:
                break

    # Determine which reroutes to actually remove
    if nodes_to_layout is None:
        # Remove all reroutes
        reroutes_to_remove = [n for n in node_tree.nodes if n.type == "REROUTE"]
    else:
        # Only remove reroutes that are exclusively part of internal connections
        # A reroute is safe to remove only if ALL its connections lead to nodes in our set
        reroutes_to_remove = []
        for reroute in reroutes_to_potentially_remove:
            safe_to_remove = True
            # Check all outgoing connections from this reroute (O(1) lookup)
            for to_node, _link in outgoing_links.get(reroute, []):
                # Trace forward to final destination
                dest = to_node
                while dest is not None and dest.type == "REROUTE":
                    # Find where this reroute outputs to (O(1) lookup)
                    next_links = outgoing_links.get(dest, [])
                    if next_links:
                        dest = next_links[0][0]
                    else:
                        dest = None
                # If final dest is outside our set, don't remove this reroute
                if dest is not None and dest not in nodes_to_layout:
                    safe_to_remove = False
                    break
            if safe_to_remove:
                reroutes_to_remove.append(reroute)

    for reroute in reroutes_to_remove:
        node_tree.nodes.remove(reroute)

    # Build a set of existing connections for O(1) lookup
    existing_connections: set[tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]] = set()
    for link in node_tree.links:
        if (
            link.is_valid
            and link.from_socket is not None
            and link.to_socket is not None
        ):
            existing_connections.add((link.from_socket, link.to_socket))

    # Restore direct connections
    for from_socket, to_socket in connections_to_restore:
        # Check if connection already exists (O(1) lookup)
        if (from_socket, to_socket) not in existing_connections:
            node_tree.links.new(from_socket, to_socket)


@dataclass
class SavedFrame:
    """Information needed to recreate a frame after layout."""

    name: str
    label: str
    color: tuple[float, float, float]
    use_custom_color: bool
    label_size: int
    # Children stored by name since node references become invalid after removal
    child_node_names: list[str]
    # Parent frame name (for nested frames), or None if top-level
    parent_frame_name: str | None


def _save_all_frames(
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


def _remove_all_frames(
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


def _restore_all_frames(
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
    _assign_reroutes_to_frames(node_tree, frame_nodes)


def _get_all_frames_containing(node: bpy.types.Node) -> set[bpy.types.Node]:
    """Get all frames containing this node (from immediate parent up to root)."""
    frames: set[bpy.types.Node] = set()
    parent = node.parent
    while parent is not None and parent.type == "FRAME":
        frames.add(parent)
        parent = parent.parent
    return frames


def _assign_reroutes_to_frames(
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
        connected_nodes = _get_connected_non_reroute_nodes(node_tree, reroute)

        if not connected_nodes:
            continue

        # Find the common frame for all connected nodes
        # Start with frames containing the first node
        first_node = connected_nodes[0]
        common_frames = _get_all_frames_containing(first_node)

        # Intersect with frames of all other nodes
        for node in connected_nodes[1:]:
            node_frames = _get_all_frames_containing(node)
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


def _get_connected_non_reroute_nodes(
    node_tree: bpy.types.NodeTree,
    reroute: bpy.types.Node,
) -> list[bpy.types.Node]:
    """Get all non-reroute nodes connected to a reroute (following reroute chains)."""
    connected: list[bpy.types.Node] = []
    visited_reroutes: set[bpy.types.Node] = set()

    def trace_connections(current_reroute: bpy.types.Node) -> None:
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


@dataclass
class VirtualReroute:
    """A reroute node in the virtual grid, before realization."""

    source_key: tuple[int, int, str]  # (source_cell_x, source_cell_y, socket_id)
    blender_node: bpy.types.Node | None = None
    used: bool = False  # Whether this reroute is actually used after optimization


@dataclass
class GridCell:
    """A cell in the virtual grid."""

    x: int
    y: int
    node: bpy.types.Node | None = None
    # Reroutes keyed by source for reuse: (src_x, src_y, socket_id) -> VirtualReroute
    reroutes: dict[tuple[int, int, str], VirtualReroute] = field(default_factory=dict)

    def get_or_create_reroute(self, source_key: tuple[int, int, str]) -> VirtualReroute:
        """Get existing reroute for source or create new one."""
        if source_key not in self.reroutes:
            self.reroutes[source_key] = VirtualReroute(source_key=source_key)
        return self.reroutes[source_key]

    @property
    def reroute_count(self) -> int:
        """Number of reroutes in this cell."""
        return len(self.reroutes)


@dataclass
class PendingConnection:
    """A connection to be routed through the grid."""

    from_socket: bpy.types.NodeSocket
    to_socket: bpy.types.NodeSocket
    from_cell: tuple[int, int]
    to_cell: tuple[int, int]
    source_key: tuple[int, int, str]  # For reroute reuse


class VirtualGrid:
    """Virtual grid for planning node layout before realization."""

    def __init__(self) -> None:
        self.cells: dict[tuple[int, int], GridCell] = {}
        self.pending_connections: list[PendingConnection] = []
        self.node_to_cell: dict[bpy.types.Node, tuple[int, int]] = {}

    def get_or_create_cell(self, x: int, y: int) -> GridCell:
        """Get existing cell or create new one."""
        key = (x, y)
        if key not in self.cells:
            self.cells[key] = GridCell(x=x, y=y)
        return self.cells[key]

    def place_node(self, node: bpy.types.Node, x: int, y: int) -> None:
        """Place a node in the grid."""
        cell = self.get_or_create_cell(x, y)
        cell.node = node
        self.node_to_cell[node] = (x, y)

    def add_connection(
        self,
        from_socket: bpy.types.NodeSocket,
        to_socket: bpy.types.NodeSocket,
        from_node: bpy.types.Node,
        to_node: bpy.types.Node,
    ) -> None:
        """Add a connection to be routed."""
        if from_node not in self.node_to_cell or to_node not in self.node_to_cell:
            return

        from_cell = self.node_to_cell[from_node]
        to_cell = self.node_to_cell[to_node]

        # Source key for reroute reuse: based on originating cell and socket
        source_key = (from_cell[0], from_cell[1], from_socket.identifier)

        self.pending_connections.append(
            PendingConnection(
                from_socket=from_socket,
                to_socket=to_socket,
                from_cell=from_cell,
                to_cell=to_cell,
                source_key=source_key,
            )
        )

    def route_all_connections(self) -> None:
        """Route all pending connections through the grid using taxicab geometry."""
        for conn in self.pending_connections:
            self._route_connection(conn)

    def _route_connection(self, conn: PendingConnection) -> None:
        """Route a single connection: exit to X+1, then Y, then X to dest cell."""
        from_x, from_y = conn.from_cell
        to_x, to_y = conn.to_cell

        # Start at cell to the right of source (X+1) for output routing
        current_x = from_x + 1
        current_y = from_y

        # Create/get reroute in the first routing cell (right of source)
        cell = self.get_or_create_cell(current_x, current_y)
        cell.get_or_create_reroute(conn.source_key)

        # Navigate Y first (vertical)
        y_step = 1 if to_y > current_y else -1
        while current_y != to_y:
            current_y += y_step
            cell = self.get_or_create_cell(current_x, current_y)
            cell.get_or_create_reroute(conn.source_key)

        # Navigate X (horizontal towards destination)
        # to_x > current_x means destination is further right
        x_step = 1 if to_x > current_x else -1
        while current_x != to_x:
            current_x += x_step
            cell = self.get_or_create_cell(current_x, current_y)
            cell.get_or_create_reroute(conn.source_key)

    def get_max_reroutes(self) -> int:
        """Get the maximum number of reroutes in any cell."""
        if not self.cells:
            return 0
        return max(cell.reroute_count for cell in self.cells.values())

    def get_max_used_reroutes_per_column(self) -> dict[int, int]:
        """Get the maximum number of used reroutes for each column (X)."""
        col_max: dict[int, int] = {}
        for (x, _y), cell in self.cells.items():
            used_count = sum(1 for r in cell.reroutes.values() if r.used)
            col_max[x] = max(col_max.get(x, 0), used_count)
        return col_max

    def get_max_used_reroutes_per_row(self) -> dict[int, int]:
        """Get the maximum number of used reroutes for each row (Y)."""
        row_max: dict[int, int] = {}
        for (_x, y), cell in self.cells.items():
            used_count = sum(1 for r in cell.reroutes.values() if r.used)
            row_max[y] = max(row_max.get(y, 0), used_count)
        return row_max

    def get_grid_bounds(self) -> tuple[int, int, int, int]:
        """Get (min_x, max_x, min_y, max_y) of the grid."""
        if not self.cells:
            return (0, 0, 0, 0)
        xs = [k[0] for k in self.cells]
        ys = [k[1] for k in self.cells]
        return (min(xs), max(xs), min(ys), max(ys))


def _build_link_adjacency(
    node_tree: bpy.types.NodeTree,
) -> tuple[
    dict[bpy.types.Node, list[bpy.types.Node]],
    dict[bpy.types.Node, list[bpy.types.Node]],
    set[bpy.types.Node],
]:
    """Pre-build adjacency lists from links for O(1) lookups.

    Returns:
        - outgoing: dict mapping node -> list of nodes it outputs to
        - incoming: dict mapping node -> list of nodes it receives from
        - has_outgoing: set of nodes that have outgoing links
    """
    outgoing: dict[bpy.types.Node, list[bpy.types.Node]] = {}
    incoming: dict[bpy.types.Node, list[bpy.types.Node]] = {}
    has_outgoing: set[bpy.types.Node] = set()

    for link in node_tree.links:
        if not link.is_valid:
            continue
        from_node = link.from_node
        to_node = link.to_node
        if from_node is None or to_node is None:
            continue

        has_outgoing.add(from_node)

        if from_node not in outgoing:
            outgoing[from_node] = []
        outgoing[from_node].append(to_node)

        if to_node not in incoming:
            incoming[to_node] = []
        incoming[to_node].append(from_node)

    return outgoing, incoming, has_outgoing


def compute_node_depths(
    node_tree: bpy.types.NodeTree,
    outgoing: dict[bpy.types.Node, list[bpy.types.Node]] | None = None,
    incoming: dict[bpy.types.Node, list[bpy.types.Node]] | None = None,
    has_outgoing: set[bpy.types.Node] | None = None,
) -> dict[bpy.types.Node, int]:
    """Compute the depth of each node (distance from output nodes).

    Output nodes have depth 0, their inputs have depth 1, etc.

    Args:
        node_tree: The node tree
        outgoing: Pre-built adjacency list (optional, built if not provided)
        incoming: Pre-built adjacency list (optional, built if not provided)
        has_outgoing: Set of nodes with outgoing links (optional)
    """
    # Build adjacency if not provided
    if outgoing is None or incoming is None or has_outgoing is None:
        outgoing, incoming, has_outgoing = _build_link_adjacency(node_tree)

    depths: dict[bpy.types.Node, int] = {}

    # Find output nodes (nodes with no output connections, or Group Output)
    output_nodes: list[bpy.types.Node] = []
    for node in node_tree.nodes:
        if node.type == "GROUP_OUTPUT":
            output_nodes.append(node)
        elif node.type == "FRAME":
            continue  # Skip frame nodes
        elif node not in has_outgoing:
            # Node has no outgoing links
            output_nodes.append(node)

    # BFS from output nodes (traversing backwards)
    queue: list[tuple[bpy.types.Node, int]] = [(n, 0) for n in output_nodes]
    for node, depth in queue:
        if node in depths and depths[node] <= depth:
            # Already visited with a shorter path
            continue
        depths[node] = depth

        # Find all nodes that feed into this node (O(1) lookup)
        for from_node in incoming.get(node, []):
            queue.append((from_node, depth + 1))

    # Handle disconnected nodes (place at max depth + 1)
    max_depth = max(depths.values()) if depths else 0
    for node in node_tree.nodes:
        if node not in depths and node.type != "FRAME":
            depths[node] = max_depth + 1

    return depths


def compute_node_input_depths(
    node_tree: bpy.types.NodeTree,
    outgoing: dict[bpy.types.Node, list[bpy.types.Node]] | None = None,
) -> dict[bpy.types.Node, int]:
    """Compute the depth of each node from input nodes (distance from inputs).

    Input nodes (Group Input, nodes with no incoming connections) have depth 0,
    nodes they connect to have depth 1, etc.

    Args:
        node_tree: The node tree
        outgoing: Pre-built adjacency list (optional, built if not provided)
    """
    # Build adjacency if not provided
    if outgoing is None:
        outgoing, _, _ = _build_link_adjacency(node_tree)

    depths: dict[bpy.types.Node, int] = {}

    # Find input nodes (only Group Input)
    input_nodes: list[bpy.types.Node] = []
    for node in node_tree.nodes:
        if node.type == "GROUP_INPUT":
            input_nodes.append(node)

    # BFS from input nodes (traversing forwards)
    queue: list[tuple[bpy.types.Node, int]] = [(n, 0) for n in input_nodes]
    for node, depth in queue:
        if node in depths and depths[node] <= depth:
            # Already visited with a shorter path
            continue
        depths[node] = depth

        # Find all nodes this node connects to (O(1) lookup)
        for to_node in outgoing.get(node, []):
            queue.append((to_node, depth + 1))

    # Handle disconnected nodes
    max_depth = max(depths.values()) if depths else 0
    for node in node_tree.nodes:
        if node not in depths and node.type != "FRAME":
            depths[node] = max_depth + 1

    return depths


def _build_connection_maps(
    node_tree: bpy.types.NodeTree,
    valid_nodes: set[bpy.types.Node],
) -> tuple[
    dict[bpy.types.Node, set[bpy.types.Node]], dict[bpy.types.Node, set[bpy.types.Node]]
]:
    """Build maps of node connections for quick lookup.

    Returns (outputs_to, inputs_from) where:
    - outputs_to[node] = set of nodes this node outputs to
    - inputs_from[node] = set of nodes this node receives input from
    """
    outputs_to: dict[bpy.types.Node, set[bpy.types.Node]] = {}
    inputs_from: dict[bpy.types.Node, set[bpy.types.Node]] = {}

    for link in node_tree.links:
        if not link.is_valid:
            continue
        from_node = link.from_node
        to_node = link.to_node
        if from_node is None or to_node is None:
            continue
        if from_node not in valid_nodes or to_node not in valid_nodes:
            continue
        if from_node not in outputs_to:
            outputs_to[from_node] = set()
        outputs_to[from_node].add(to_node)
        if to_node not in inputs_from:
            inputs_from[to_node] = set()
        inputs_from[to_node].add(from_node)

    return outputs_to, inputs_from


def _shaker_sorting_pass(
    raw_positions: dict[bpy.types.Node, float],
    outputs_to: dict[bpy.types.Node, set[bpy.types.Node]],
    inputs_from: dict[bpy.types.Node, set[bpy.types.Node]],
    nudge: float,
) -> None:
    """Enforce correct ordering: sources must be left of their targets."""
    # Backward pass: if node outputs to a node at same or higher position, move LEFT
    for node in list(raw_positions.keys()):
        if node not in outputs_to:
            continue
        pos = raw_positions[node]
        for target in outputs_to[node]:
            target_pos = raw_positions.get(target, float("-inf"))
            if pos <= target_pos:
                raw_positions[node] += nudge  # Move left
                break

    # Forward pass: if node receives from a node at same or lower position, move RIGHT
    for node in list(raw_positions.keys()):
        if node not in inputs_from:
            continue
        pos = raw_positions[node]
        for source in inputs_from[node]:
            source_pos = raw_positions.get(source, float("inf"))
            if pos >= source_pos:
                raw_positions[node] -= nudge  # Move right
                break


def _shaker_gravity_pass(
    raw_positions: dict[bpy.types.Node, float],
    outputs_to: dict[bpy.types.Node, set[bpy.types.Node]],
    inputs_from: dict[bpy.types.Node, set[bpy.types.Node]],
    nudge: float,
) -> None:
    """Pull nodes closer if they're too far apart (gap > 1)."""
    # Gravity pass 1: if node is too far LEFT of its targets (gap > 1), pull RIGHT
    for node in list(raw_positions.keys()):
        if node not in outputs_to:
            continue
        pos = raw_positions[node]
        for target in outputs_to[node]:
            target_pos = raw_positions.get(target, float("-inf"))
            if pos - target_pos > 1.0:
                raw_positions[node] -= nudge  # Pull right (closer to target)
                break

    # Gravity pass 2: if node is too far RIGHT of its sources (gap > 1), pull LEFT
    for node in list(raw_positions.keys()):
        if node not in inputs_from:
            continue
        pos = raw_positions[node]
        for source in inputs_from[node]:
            source_pos = raw_positions.get(source, float("inf"))
            if source_pos - pos > 1.0:
                raw_positions[node] += nudge  # Pull left (closer to source)
                break


def _refine_columns_shaker(
    raw_positions: dict[bpy.types.Node, float],
    outputs_to: dict[bpy.types.Node, set[bpy.types.Node]],
    inputs_from: dict[bpy.types.Node, set[bpy.types.Node]],
    iterations: int = 5,
    use_gravity: bool = False,
) -> None:
    """Refine column positions using shaker-sort style passes.

    Modifies raw_positions in place.
    Higher raw_position = more to the left (closer to inputs).
    Lower raw_position = more to the right (closer to outputs).

    Uses progressively smaller increments to create in-between positions without large jumps.

    Args:
        raw_positions: Node positions to refine (modified in place)
        outputs_to: Map of node -> nodes it outputs to
        inputs_from: Map of node -> nodes it receives input from
        iterations: Number of refinement iterations
        use_gravity: Whether to pull nodes closer together
    """
    nudge = 1.0

    for _ in range(iterations):
        if use_gravity:
            _shaker_gravity_pass(raw_positions, outputs_to, inputs_from, nudge)
        _shaker_sorting_pass(raw_positions, outputs_to, inputs_from, nudge)
        nudge = nudge * 0.75  # Decrease nudge for finer adjustments


def compute_node_columns(
    node_tree: bpy.types.NodeTree,
    nodes_to_layout: set[bpy.types.Node] | None = None,
    sorting_method: str = "combined",
    use_gravity: bool = False,
    original_positions: dict[bpy.types.Node, float] | None = None,
    column_width: float = 250.0,
) -> dict[bpy.types.Node, int]:
    """Compute column position for each node using both input and output distances.

    Uses output_depth - input_depth to order nodes along the flow,
    then refines with shaker-sort style passes to enforce correct ordering,
    then assigns sequential column numbers to collapse any gaps.

    Args:
        node_tree: The node tree
        nodes_to_layout: If provided, only compute columns for these nodes
        sorting_method: How to compute initial positions:
            - "combined": output_depth - input_depth (default, balanced)
            - "output": distance from outputs only (outputs on right)
            - "input": distance from inputs only (inputs on left)
            - "position": use original X positions divided by column_width
        use_gravity: Whether to pull nodes closer together in refinement
    """
    # Build adjacency once, reuse for both depth computations
    outgoing, incoming, has_outgoing = _build_link_adjacency(node_tree)

    output_depths = compute_node_depths(node_tree, outgoing, incoming, has_outgoing)
    input_depths = compute_node_input_depths(node_tree, outgoing)

    # Compute raw position value for each node (as float for fine-grained adjustment)
    raw_positions: dict[bpy.types.Node, float] = {}
    for node in node_tree.nodes:
        if node.type == "FRAME":
            continue
        if nodes_to_layout is not None and node not in nodes_to_layout:
            continue

        out_d = output_depths.get(node, 0)
        in_d = input_depths.get(node, 0)

        if sorting_method == "output":
            raw_positions[node] = float(out_d)
        elif sorting_method == "input":
            raw_positions[node] = float(
                -in_d
            )  # Negate so higher input depth = more left
        elif sorting_method == "position" and original_positions is not None:
            # Use original X position - will be processed below
            raw_positions[node] = original_positions.get(node, 0.0)
        else:  # "combined"
            raw_positions[node] = float(out_d - in_d)

    # For "position" method, convert X positions to column indices directly
    if sorting_method == "position" and original_positions is not None:
        # Find the leftmost (minimum X) position
        if raw_positions:
            min_x = min(raw_positions.values())
            # Convert to column indices: (x - min_x) / column_width, rounded
            # Higher X = more to the right = lower column number (outputs on right)
            # So we negate to flip the order
            for node in raw_positions:
                relative_x = raw_positions[node] - min_x
                col_index = round(relative_x / column_width) if column_width > 0 else 0
                raw_positions[node] = float(
                    -col_index
                )  # Negate so left = higher column

        # Skip shaker refinement for position-based sorting
        # (we want to preserve the original spatial arrangement)
    else:
        # Build connection maps and refine with shaker-sort passes
        outputs_to, inputs_from = _build_connection_maps(
            node_tree, set(raw_positions.keys())
        )
        _refine_columns_shaker(
            raw_positions,
            outputs_to,
            inputs_from,
            use_gravity=use_gravity,
        )

    # Get unique position values and sort them
    unique_positions = sorted(set(raw_positions.values()))

    # Map raw positions to sequential column numbers (collapse gaps)
    position_to_column = {pos: idx for idx, pos in enumerate(unique_positions)}

    # Assign final column numbers
    columns: dict[bpy.types.Node, int] = {}
    for node, raw_pos in raw_positions.items():
        columns[node] = position_to_column[raw_pos]

    return columns


def layout_nodes_pcb_style(
    node_tree: bpy.types.NodeTree,
    cell_width: float = 200.0,
    cell_height: float = 200.0,
    lane_width: float = 20.0,
    lane_gap: float = 50.0,
    nodes_to_layout: set[bpy.types.Node] | None = None,
    anchor_nodes: set[bpy.types.Node] | None = None,
    sorting_method: str = "combined",
    use_gravity: bool = False,
    vertical_align: str = "CENTER",
    collapse_vertical: bool = True,
    collapse_horizontal: bool = True,
    collapse_adjacent: bool = True,
    snap_to_grid: bool = False,
    grid_size: float = 20.0,
    respect_dimensions: bool = True,
    original_positions: dict[bpy.types.Node, float] | None = None,
) -> list[bpy.types.Node]:
    """Layout nodes in a PCB/FPGA-style grid with routed connections.

    Args:
        node_tree: The node tree to layout
        cell_width: Width of each grid cell (for nodes)
        cell_height: Height of each grid cell (for nodes)
        lane_width: Width allocated per reroute lane in the diagonal
        lane_gap: Gap before and after the lane area
        nodes_to_layout: If provided, only layout these specific nodes. If None, layout all.
        anchor_nodes: If provided, use these nodes for centroid calculation instead of all
            nodes being laid out. Useful when expanding selection to upstream nodes but
            wanting to anchor the result to the originally selected nodes.
        sorting_method: How to compute column positions:
            - "combined": balanced using both input and output distance (default)
            - "output": prioritize distance from outputs
            - "input": prioritize distance from inputs
        use_gravity: Pull nodes closer together when gaps are large
        vertical_align: Vertical alignment of nodes in cells (TOP, CENTER, BOTTOM)
        collapse_vertical: Whether to collapse vertical runs of reroutes (default: True)
        collapse_adjacent: Whether to collapse adjacent reroutes in neighboring columns (default: True)
        snap_to_grid: Whether to snap final positions to the editor grid (default: False)
        grid_size: Size of the grid to snap to (default: 20.0, Blender's default)
        respect_dimensions: If True, measure actual node heights and have tall nodes span
            multiple grid rows to prevent vertical overlapping (default: False).
            Note: Requires nodes to have been drawn at least once for dimensions to be available.
        original_positions: Pre-captured X positions for "position" sorting method.
            If None and sorting_method is "position", positions are captured at start of layout.
            Pass this to preserve original positions across multiple layout calls.

    Returns:
        List of reroute nodes created during the layout operation.
    """
    if not node_tree.nodes:
        return []

    # Capture original X positions BEFORE any modifications
    # Used by "position" sorting method to preserve spatial arrangement
    # If original_positions was passed in, use that instead (for re-layouts)
    if original_positions is None:
        original_positions = {}
        for node in node_tree.nodes:
            if node.type not in ("FRAME", "REROUTE"):
                if nodes_to_layout is None or node in nodes_to_layout:
                    original_positions[node] = node.location.x

    # Step 0a: Remove existing reroutes and restore direct connections
    # This ensures repeated layouts produce consistent results
    _remove_all_reroutes(node_tree, nodes_to_layout)

    # Step 0b: Save frame info and remove frames
    # Frames interfere with layout since node positions are relative to parent frame
    saved_frames = _save_all_frames(node_tree, nodes_to_layout)
    _remove_all_frames(node_tree, nodes_to_layout)

    # Compute column width for position-based sorting
    # Use min(cell_width, max_node_width) so we don't produce fewer columns than needed
    # (we can produce more columns if user chooses smaller cell_width, but not fewer)
    ui_scale = 1.0
    if bpy.context is not None and bpy.context.preferences is not None:
        ui_scale = bpy.context.preferences.system.ui_scale
    max_node_width = 0.0
    for node in node_tree.nodes:
        if node.type not in ("REROUTE", "FRAME"):
            if nodes_to_layout is None or node in nodes_to_layout:
                width = node.dimensions[0] / ui_scale
                max_node_width = max(max_node_width, width)
    if max_node_width <= 0:
        max_node_width = 200.0  # Fallback if nodes not drawn yet
    column_width = min(cell_width, max_node_width) + lane_gap

    # Step 1: Compute column positions using both input and output distances
    columns = compute_node_columns(
        node_tree,
        nodes_to_layout,
        sorting_method=sorting_method,
        use_gravity=use_gravity,
        original_positions=original_positions,
        column_width=column_width,
    )
    if not columns:
        # Restore frames even if no columns to compute
        _restore_all_frames(node_tree, saved_frames)
        return []

    # Compute centroid of nodes before layout
    # Use anchor_nodes if provided, otherwise use all nodes being laid out
    centroid_nodes = list(anchor_nodes) if anchor_nodes else list(columns.keys())
    # Filter to only nodes actually in the layout
    centroid_nodes = [n for n in centroid_nodes if n in columns]
    if not centroid_nodes:
        centroid_nodes = list(columns.keys())
    old_centroid_x = sum(n.location.x for n in centroid_nodes) / len(centroid_nodes)
    old_centroid_y = sum(n.location.y for n in centroid_nodes) / len(centroid_nodes)

    # Step 2: Build virtual grid with initial node placement
    grid = _build_virtual_grid(
        columns,
        vertical_align,
        respect_dimensions=respect_dimensions,
        cell_height=cell_height,
        lane_gap=lane_gap,
    )

    # Step 3: Collect all connections and add to grid
    _collect_connections(node_tree, grid, columns)

    # Step 4: Route all connections through the grid
    grid.route_all_connections()

    # Step 5: Mark which reroutes are actually used after optimization
    _mark_used_reroutes(
        grid,
        collapse_vertical=collapse_vertical,
        collapse_horizontal=collapse_horizontal,
        collapse_adjacent=collapse_adjacent,
    )

    # Step 6: Realize the layout in Blender
    _realize_layout(
        node_tree,
        grid,
        cell_width,
        cell_height,
        lane_width,
        lane_gap,
        nodes_to_layout,
        collapse_vertical=collapse_vertical,
        collapse_horizontal=collapse_horizontal,
        collapse_adjacent=collapse_adjacent,
    )

    # Step 7: Compute new centroid and shift to match old position
    laid_out_nodes = list(columns.keys())

    # Collect reroutes created during this layout (from the grid cells)
    created_reroutes: list[bpy.types.Node] = []
    for cell in grid.cells.values():
        for virtual_reroute in cell.reroutes.values():
            if virtual_reroute.blender_node is not None:
                created_reroutes.append(virtual_reroute.blender_node)

    all_laid_out = laid_out_nodes + created_reroutes

    if laid_out_nodes:
        # Use the same nodes for new centroid as we used for old centroid
        new_centroid_nodes = [n for n in centroid_nodes if n in columns]
        if not new_centroid_nodes:
            new_centroid_nodes = laid_out_nodes
        new_centroid_x = sum(n.location.x for n in new_centroid_nodes) / len(
            new_centroid_nodes
        )
        new_centroid_y = sum(n.location.y for n in new_centroid_nodes) / len(
            new_centroid_nodes
        )

        offset_x = old_centroid_x - new_centroid_x
        offset_y = old_centroid_y - new_centroid_y

        for node in all_laid_out:
            node.location.x += offset_x
            node.location.y += offset_y

    # Step 8: Snap to grid if requested
    if snap_to_grid and grid_size > 0:
        for node in all_laid_out:
            node.location.x = round(node.location.x / grid_size) * grid_size
            node.location.y = round(node.location.y / grid_size) * grid_size

    # Step 9: Restore frames and re-parent nodes
    _restore_all_frames(node_tree, saved_frames)

    return created_reroutes


# Socket type multipliers for height estimation (when NOT connected)
# Some socket types render larger UI elements when displaying input fields
# When connected, all sockets are effectively 1 row tall
_SOCKET_HEIGHT_MULTIPLIERS: dict[str, float] = {
    # Vector types (show X, Y, Z fields when not connected)
    "VECTOR": 3.0,
    "NodeSocketVector": 3.0,
    "NodeSocketVectorDirection": 3.0,
    "NodeSocketVectorEuler": 3.0,
    "NodeSocketVectorTranslation": 3.0,
    "NodeSocketVectorVelocity": 3.0,
    "NodeSocketVectorAcceleration": 3.0,
    "NodeSocketVectorXYZ": 3.0,
    # Rotation (shows X, Y, Z, W or euler when not connected)
    "ROTATION": 3.0,
    "NodeSocketRotation": 3.0,
    # Color types - inline color picker, always 1 row
    # (no multiplier needed, defaults to 1.0)
    # Matrix (4x4 = 16 values, but usually collapsed)
    "MATRIX": 2.0,
    "NodeSocketMatrix": 2.0,
    # Default for standard types (float, int, bool, string, color, etc.) is 1.0
}

# Base height estimates (in pixels)
_NODE_HEADER_HEIGHT = 30.0  # Node title bar
_SOCKET_BASE_HEIGHT = 22.0  # Height per socket row


def _estimate_node_height(node: bpy.types.Node) -> float:
    """Estimate node height based on socket count and types.

    Used as a fallback when node.dimensions returns (0, 0).

    Args:
        node: The node to estimate height for

    Returns:
        Estimated height in pixels
    """
    height = _NODE_HEADER_HEIGHT

    # Count input sockets with their type multipliers
    for socket in node.inputs:
        if socket.enabled:
            # Connected sockets are always 1 row tall (no expanded fields)
            if socket.is_linked:
                height += _SOCKET_BASE_HEIGHT
            else:
                # Get socket type - try both .type and bl_idname
                socket_type = getattr(socket, "type", "") or ""
                socket_idname = getattr(socket, "bl_idname", "") or ""

                multiplier = _SOCKET_HEIGHT_MULTIPLIERS.get(
                    socket_type, _SOCKET_HEIGHT_MULTIPLIERS.get(socket_idname, 1.0)
                )
                height += _SOCKET_BASE_HEIGHT * multiplier

    # Count output sockets (usually simpler, no expanded fields)
    for socket in node.outputs:
        if socket.enabled:
            height += _SOCKET_BASE_HEIGHT

    # Add some padding
    height += 10.0

    return height


def _calculate_node_row_span(
    node: bpy.types.Node,
    cell_height: float,
    lane_gap: float = 0.0,
    row_index: int = 0,
) -> int:
    """Calculate how many grid rows a node should span based on its dimensions.

    Uses node.dimensions[1] (actual rendered height) to determine if the node
    is taller than a single cell and needs to occupy multiple rows.

    Falls back to estimating height based on socket count and types if
    dimensions are not available (returns 0, 0).

    Args:
        node: The node to measure
        cell_height: Height of a single grid cell
        lane_gap: Gap between cells (adds to available space before requiring
            an additional row)
        row_index: The node's visual index within its column (0 = top).
            Used to stagger reflow thresholds so nodes don't all jump at once.

    Returns:
        Number of rows this node should span (minimum 1)
    """
    import math

    if cell_height <= 0:
        return 1

    # Get the actual rendered height of the node
    # Note: dimensions may be (0, 0) if the node hasn't been drawn yet
    # Dimensions are in screen pixels, so we need to divide by ui_scale
    # to get the actual node size in Blender units
    ui_scale = 1.0
    ctx = bpy.context
    if ctx is not None and ctx.preferences is not None:
        ui_scale = ctx.preferences.system.ui_scale
    node_height = node.dimensions[1] / ui_scale

    # Fallback: estimate based on socket count and types
    if node_height <= 0:
        node_height = _estimate_node_height(node)

    if node_height <= 0:
        return 1

    # Calculate how many cells this node needs
    # A node in row 0 has cell_height space. If it extends into the lane_gap
    # of the next row but not into the cell itself, it still fits in 1 row.
    # For n rows: available = n * cell_height + (n-1) * lane_gap
    # Rearranging: n >= (node_height + lane_gap) / (cell_height + lane_gap)
    #
    # To prevent all similar-height nodes from reflowing at once, we add a
    # row-dependent stagger. Using lane_gap / (1 + row_index * 0.1) gives a
    # smooth decay that starts almost linear but asymptotically approaches 0
    stagger_bonus = lane_gap / (1.0 + row_index * 0.1) if row_index >= 0 else lane_gap
    effective_cell = cell_height + lane_gap + stagger_bonus
    rows_needed = (
        math.ceil((node_height + lane_gap) / effective_cell)
        if effective_cell > 0
        else 1
    )

    return max(1, rows_needed)


def _build_virtual_grid(
    columns: dict[bpy.types.Node, int],
    vertical_align: str = "CENTER",
    respect_dimensions: bool = False,
    cell_height: float = 200.0,
    lane_gap: float = 0.0,
) -> VirtualGrid:
    """Build virtual grid with initial node placement based on column positions and vertical alignment.

    Args:
        columns: Dict mapping nodes to their column indices
        vertical_align: Vertical alignment (TOP, CENTER, BOTTOM)
        respect_dimensions: If True, tall nodes will span multiple rows to avoid overlapping
        cell_height: Height of each grid cell (used when respect_dimensions is True)
    """
    grid = VirtualGrid()
    max_col = max(columns.values())

    # Group nodes by column for Y ordering
    nodes_by_column: dict[int, list[bpy.types.Node]] = {}
    for node, col in columns.items():
        if col not in nodes_by_column:
            nodes_by_column[col] = []
        nodes_by_column[col].append(node)

    if respect_dimensions:
        # Calculate row spans for each node and place with gaps for tall nodes
        # First pass: calculate total rows needed per column (accounting for tall nodes)
        column_total_rows: dict[int, int] = {}
        for col, nodes in nodes_by_column.items():
            nodes.sort(key=lambda n: (-n.location.y, n.name))
            total_rows = 0
            for i, node in enumerate(nodes):
                span = _calculate_node_row_span(node, cell_height, lane_gap, i)
                total_rows += span
            column_total_rows[col] = total_rows

        max_h = max(column_total_rows.values()) if column_total_rows else 0

        # Second pass: place nodes, skipping rows for tall nodes
        for col, nodes in nodes_by_column.items():
            nodes.sort(key=lambda n: (-n.location.y, n.name))
            grid_x = max_col - col
            total_rows = column_total_rows[col]

            if vertical_align == "TOP":
                y_offset = 0
            elif vertical_align == "BOTTOM":
                y_offset = max_h - total_rows
            else:  # CENTER
                y_offset = (max_h - total_rows) // 2

            current_y = y_offset
            for i, node in enumerate(nodes):
                grid.place_node(node, grid_x, current_y)
                # Skip rows based on node height
                span = _calculate_node_row_span(node, cell_height, lane_gap, i)
                current_y += span
    else:
        # Original behavior: all nodes occupy exactly one row
        max_h = (
            max(len(nodes) for nodes in nodes_by_column.values())
            if nodes_by_column
            else 0
        )

        for col, nodes in nodes_by_column.items():
            # Sort by original Y position (higher Y = top of screen, should come first)
            # Use name as secondary sort key for consistent ordering when Y is equal
            nodes.sort(key=lambda n: (-n.location.y, n.name))
            grid_x = max_col - col  # Flip so inputs are on left, outputs are on right
            h = len(nodes)
            if vertical_align == "TOP":
                y_offset = 0
            elif vertical_align == "BOTTOM":
                y_offset = max_h - h
            else:  # CENTER
                y_offset = (max_h - h) // 2
            for y_idx, node in enumerate(nodes):
                grid.place_node(node, grid_x, y_idx + y_offset)

    return grid


def _collect_connections(
    node_tree: bpy.types.NodeTree,
    grid: VirtualGrid,
    columns: dict[bpy.types.Node, int],
) -> None:
    """Collect all valid connections from the node tree and add to grid."""
    for link in node_tree.links:
        if not link.is_valid:
            continue
        from_node = link.from_node
        to_node = link.to_node
        from_socket = link.from_socket
        to_socket = link.to_socket

        if from_node is None or to_node is None:
            continue
        if from_socket is None or to_socket is None:
            continue
        if from_node not in columns or to_node not in columns:
            continue

        grid.add_connection(from_socket, to_socket, from_node, to_node)


def _mark_used_reroutes(
    grid: VirtualGrid,
    collapse_vertical: bool = True,
    collapse_horizontal: bool = True,
    collapse_adjacent: bool = True,
) -> None:
    """Mark which reroutes are actually used after path optimization."""
    for conn in grid.pending_connections:
        from_x, from_y = conn.from_cell
        to_x, to_y = conn.to_cell

        # Build the path of cells (same logic as _realize_connection)
        path: list[tuple[int, int]] = []
        current_x = from_x + 1
        current_y = from_y
        path.append((current_x, current_y))

        y_step = 1 if to_y > current_y else -1
        while current_y != to_y:
            current_y += y_step
            path.append((current_x, current_y))

        x_step = 1 if to_x > current_x else -1
        while current_x != to_x:
            current_x += x_step
            path.append((current_x, current_y))

        # Apply optimization
        optimized_path = _optimize_routing_path(
            path,
            conn.from_cell,
            conn.to_cell,
            collapse_vertical=collapse_vertical,
            collapse_horizontal=collapse_horizontal,
            collapse_adjacent=collapse_adjacent,
        )

        # Mark reroutes in the optimized path as used
        for cell_coord in optimized_path:
            cell = grid.cells.get(cell_coord)
            if cell is None:
                continue
            reroute = cell.reroutes.get(conn.source_key)
            if reroute is not None:
                reroute.used = True


def _compute_lane_areas(
    grid: VirtualGrid,
    lane_width: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute lane area for each column and row based on max reroutes.

    Returns (col_lane_area, row_lane_area) dictionaries.
    """
    col_max_reroutes = grid.get_max_used_reroutes_per_column()
    row_max_reroutes = grid.get_max_used_reroutes_per_row()
    min_x, max_x, min_y, max_y = grid.get_grid_bounds()

    # For each column, compute the lane area needed
    col_lane_area: dict[int, float] = {}
    for x in range(min_x, max_x + 1):
        col_reroutes = col_max_reroutes.get(x, 0)
        # Find max row reroutes that intersect this column
        max_row_reroutes = max(
            (
                row_max_reroutes.get(y, 0)
                for y in range(min_y, max_y + 1)
                if (x, y) in grid.cells
            ),
            default=0,
        )
        lane_count = max(col_reroutes, max_row_reroutes, 1)
        col_lane_area[x] = lane_width * lane_count

    # For each row, compute the lane area needed
    row_lane_area: dict[int, float] = {}
    for y in range(min_y, max_y + 1):
        row_reroutes = row_max_reroutes.get(y, 0)
        # Find max col reroutes that intersect this row
        max_col_reroutes = max(
            (
                col_max_reroutes.get(x, 0)
                for x in range(min_x, max_x + 1)
                if (x, y) in grid.cells
            ),
            default=0,
        )
        lane_count = max(row_reroutes, max_col_reroutes, 1)
        row_lane_area[y] = lane_width * lane_count

    return col_lane_area, row_lane_area


def _compute_cell_positions(
    grid: VirtualGrid,
    col_lane_area: dict[int, float],
    row_lane_area: dict[int, float],
    cell_width: float,
    cell_height: float,
    lane_width: float,
    lane_gap: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute cumulative X/Y positions for each column/row.

    Returns (col_x_start, row_y_start) dictionaries.
    """
    min_x, max_x, min_y, max_y = grid.get_grid_bounds()

    # Compute cumulative X positions for each column
    col_x_start: dict[int, float] = {}
    current_x = 0.0
    for x in range(min_x, max_x + 1):
        col_x_start[x] = current_x
        lane_area = col_lane_area.get(x, lane_width)
        current_x += lane_gap + lane_area + cell_width

    # Compute cumulative Y positions for each row
    row_y_start: dict[int, float] = {}
    current_y = 0.0
    for y in range(min_y, max_y + 1):
        row_y_start[y] = current_y
        lane_area = row_lane_area.get(y, lane_width)
        current_y += lane_gap + lane_area + cell_height

    return col_x_start, row_y_start


def _realize_layout(
    node_tree: bpy.types.NodeTree,
    grid: VirtualGrid,
    cell_width: float,
    cell_height: float,
    lane_width: float,
    lane_gap: float,
    nodes_to_layout: set[bpy.types.Node] | None = None,
    collapse_vertical: bool = True,
    collapse_horizontal: bool = True,
    collapse_adjacent: bool = True,
) -> None:
    """Realize the virtual grid layout in Blender."""
    # Calculate per-column and per-row lane areas
    col_lane_area, row_lane_area = _compute_lane_areas(grid, lane_width)

    # Calculate cumulative positions
    col_x_start, row_y_start = _compute_cell_positions(
        grid,
        col_lane_area,
        row_lane_area,
        cell_width,
        cell_height,
        lane_width,
        lane_gap,
    )

    # Remove links only between nodes we're laying out
    if nodes_to_layout is None:
        # Remove all links
        for link in list(node_tree.links):
            node_tree.links.remove(link)
    else:
        # Only remove links where BOTH endpoints are in our layout set
        for link in list(node_tree.links):
            if not link.is_valid:
                continue
            from_node = link.from_node
            to_node = link.to_node
            if from_node is None or to_node is None:
                continue
            # Only remove if both nodes are in our set
            if from_node in nodes_to_layout and to_node in nodes_to_layout:
                node_tree.links.remove(link)

    # Place actual nodes in their cells
    for cell in grid.cells.values():
        if cell.node is None:
            continue
        lane_area_x = col_lane_area.get(cell.x, lane_width)
        lane_area_y = row_lane_area.get(cell.y, lane_width)
        # Node origin: after lane_gap + lanes
        cell.node.location.x = col_x_start[cell.x] + lane_gap + lane_area_x
        cell.node.location.y = -(row_y_start[cell.y] + lane_gap + lane_area_y)

    # Create reroute nodes in each cell - only for used reroutes, positioned diagonally
    _create_reroute_nodes(
        node_tree, grid, col_lane_area, row_lane_area, col_x_start, row_y_start
    )

    # Create all connections through reroute chains
    for conn in grid.pending_connections:
        _realize_connection(
            node_tree,
            grid,
            conn,
            collapse_vertical=collapse_vertical,
            collapse_horizontal=collapse_horizontal,
            collapse_adjacent=collapse_adjacent,
        )


def _create_reroute_nodes(
    node_tree: bpy.types.NodeTree,
    grid: VirtualGrid,
    col_lane_area: dict[int, float],
    row_lane_area: dict[int, float],
    col_x_start: dict[int, float],
    row_y_start: dict[int, float],
) -> None:
    """Create reroute nodes in each cell, positioned diagonally."""
    reroute_spacing = 20.0  # Spacing between reroutes along the diagonal
    default_lane = 20.0  # Default lane width

    for cell in grid.cells.values():
        lane_area_x = col_lane_area.get(cell.x, default_lane)
        lane_area_y = row_lane_area.get(cell.y, default_lane)
        lane_area = max(lane_area_x, lane_area_y)  # Use the larger for diagonal

        # Lane area starts after the first gap
        lane_start_x = col_x_start[cell.x] + default_lane
        lane_start_y = -(row_y_start[cell.y] + default_lane)  # Top of lane area

        used_index = 0
        for virtual_reroute in cell.reroutes.values():
            if not virtual_reroute.used:
                continue  # Skip unused reroutes - they were melded away
            # Diagonal position: bottom-left to top-right
            offset = (used_index + 1) * reroute_spacing
            reroute = node_tree.nodes.new("NodeReroute")
            reroute.location.x = lane_start_x + offset
            reroute.location.y = lane_start_y - lane_area + offset
            reroute.select = False
            virtual_reroute.blender_node = reroute
            used_index += 1


def _realize_connection(
    node_tree: bpy.types.NodeTree,
    grid: VirtualGrid,
    conn: PendingConnection,
    collapse_vertical: bool = True,
    collapse_horizontal: bool = True,
    collapse_adjacent: bool = True,
) -> None:
    """Create the actual Blender links for a routed connection."""
    from_x, from_y = conn.from_cell
    to_x, to_y = conn.to_cell

    # Build the path of cells this connection travels through
    path: list[tuple[int, int]] = []

    # Start at cell to the right of source (X+1)
    current_x = from_x + 1
    current_y = from_y
    path.append((current_x, current_y))

    # Y movement first (taxicab: vertical then horizontal)
    y_step = 1 if to_y > current_y else -1
    while current_y != to_y:
        current_y += y_step
        path.append((current_x, current_y))

    # X movement (horizontal towards destination)
    x_step = 1 if to_x > current_x else -1
    while current_x != to_x:
        current_x += x_step
        path.append((current_x, current_y))

    # Optimize path by melding unnecessary reroutes
    path = _optimize_routing_path(
        path,
        conn.from_cell,
        conn.to_cell,
        collapse_vertical=collapse_vertical,
        collapse_horizontal=collapse_horizontal,
        collapse_adjacent=collapse_adjacent,
    )

    # Create links: source -> first reroute -> ... -> last reroute -> dest
    prev_socket: bpy.types.NodeSocket = conn.from_socket

    for cell_coord in path:
        cell = grid.cells.get(cell_coord)
        if cell is None:
            continue
        reroute = cell.reroutes.get(conn.source_key)
        if reroute is None or reroute.blender_node is None:
            continue

        # Link from previous to this reroute's input
        node_tree.links.new(prev_socket, reroute.blender_node.inputs[0])
        prev_socket = reroute.blender_node.outputs[0]

    # Final link to destination
    node_tree.links.new(prev_socket, conn.to_socket)


def _optimize_routing_path(
    path: list[tuple[int, int]],
    from_cell: tuple[int, int],
    to_cell: tuple[int, int],
    collapse_vertical: bool = True,
    collapse_horizontal: bool = True,
    collapse_adjacent: bool = True,
) -> list[tuple[int, int]]:
    """Optimize routing path by melding unnecessary reroutes.

    Rule 1: Collapse vertical runs - if reroutes are stacked vertically (same X),
            keep only the last one in the run.
    Rule 2: Collapse horizontal runs of 3+ - if reroutes are in a horizontal line (same Y),
            keep only the first and last, removing internal reroutes.
    Rule 3: Adjacent column meld - if after optimization only one reroute remains
            at the destination cell, and the source is in the adjacent column (X-1),
            we can skip it entirely for a direct node-to-node connection.
    """
    if not path:
        return path

    optimized: list[tuple[int, int]] = []
    if collapse_vertical:
        # Rule 1: Collapse vertical runs - keep only the last reroute of each vertical segment
        i = 0
        while i < len(path):
            j = i
            while j + 1 < len(path) and path[j + 1][0] == path[i][0]:
                j += 1
            optimized.append(path[j])
            i = j + 1
    else:
        optimized = path[:]

    if collapse_horizontal:
        # Rule 2: Collapse horizontal runs of 3+ reroutes - keep first and last
        horiz_optimized: list[tuple[int, int]] = []
        i = 0
        while i < len(optimized):
            # Find the end of the current horizontal run (same Y)
            j = i
            while j + 1 < len(optimized) and optimized[j + 1][1] == optimized[i][1]:
                j += 1
            run_length = j - i + 1
            if run_length >= 3:
                # Keep only first and last of this horizontal run
                horiz_optimized.append(optimized[i])
                horiz_optimized.append(optimized[j])
            else:
                # Keep all (1 or 2 reroutes)
                for k in range(i, j + 1):
                    horiz_optimized.append(optimized[k])
            i = j + 1
        optimized = horiz_optimized

    if collapse_adjacent:
        # Rule 3: If after optimization, only one reroute remains at dest cell,
        # and source is at X-1 (adjacent column), we can remove it entirely
        if len(optimized) == 1 and from_cell[0] + 1 == to_cell[0]:
            return []

    return optimized
