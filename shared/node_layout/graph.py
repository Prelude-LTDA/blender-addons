"""Graph analysis and traversal utilities for node layout.

Includes reroute tracing, adjacency building, depth computation, and column assignment.
"""

from __future__ import annotations

import bpy

__all__ = [
    "build_connection_maps",
    "build_link_adjacency",
    "compute_node_columns",
    "compute_node_depths",
    "compute_node_input_depths",
    "remove_all_reroutes",
    "trace_to_real_source",
]


def trace_to_real_source(socket: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
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


def remove_all_reroutes(  # noqa: PLR0912, PLR0915
    node_tree: bpy.types.NodeTree,
    nodes_to_layout: set[bpy.types.Node] | None = None,
) -> None:
    """Remove reroute nodes and reconnect through them.

    For each connection that goes through reroutes, traces back to find
    the real source and creates a direct connection.

    Args:
        node_tree: The node tree to modify
        nodes_to_layout: If provided, only remove reroutes where both source and dest
            are in this set
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

        real_source = trace_to_real_source(from_socket)

        # Only process if we found a real (non-reroute) source
        if real_source.node is None or real_source.node.type == "REROUTE":
            continue

        # If filtering, only restore connections where BOTH endpoints are in our set
        if nodes_to_layout is not None and (
            real_source.node not in nodes_to_layout or to_node not in nodes_to_layout
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
                    dest = next_links[0][0] if next_links else None
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


def build_link_adjacency(
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
        outgoing, incoming, has_outgoing = build_link_adjacency(node_tree)

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

    Group Input nodes have depth 0, nodes they connect to have depth 1, etc.
    Disconnected nodes (not reachable from Group Input) get max_depth + 1.

    Args:
        node_tree: The node tree
        outgoing: Pre-built adjacency list (optional, built if not provided)
    """
    # Build adjacency if not provided
    if outgoing is None:
        outgoing, _, _ = build_link_adjacency(node_tree)

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


def build_connection_maps(
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

    Uses progressively smaller increments to create in-between positions without
    large jumps.

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
        original_positions: Original X positions for position-based sorting
        column_width: Width used to convert X positions to columns
    """
    # Build adjacency once, reuse for both depth computations
    outgoing, incoming, has_outgoing = build_link_adjacency(node_tree)

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
            for node, pos in raw_positions.items():
                relative_x = pos - min_x
                col_index = round(relative_x / column_width) if column_width > 0 else 0
                raw_positions[node] = float(
                    -col_index
                )  # Negate so left = higher column

        # Skip shaker refinement for position-based sorting
        # (we want to preserve the original spatial arrangement)
    else:
        # Build connection maps and refine with shaker-sort passes
        outputs_to, inputs_from = build_connection_maps(
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
