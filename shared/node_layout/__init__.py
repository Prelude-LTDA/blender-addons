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


def _remove_all_reroutes(node_tree: bpy.types.NodeTree) -> None:
    """Remove all reroute nodes and reconnect through them.

    For each connection that goes through reroutes, traces back to find
    the real source and creates a direct connection.
    """
    # First, collect all connections that need to be preserved
    # Map: (real_source_socket, destination_socket)
    connections_to_restore: list[tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]] = []

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

        # Only add if we found a real (non-reroute) source
        if real_source.node is not None and real_source.node.type != "REROUTE":
            connections_to_restore.append((real_source, to_socket))

    # Remove all reroute nodes
    reroutes_to_remove = [n for n in node_tree.nodes if n.type == "REROUTE"]
    for reroute in reroutes_to_remove:
        node_tree.nodes.remove(reroute)

    # Restore direct connections
    for from_socket, to_socket in connections_to_restore:
        # Check if connection already exists (from non-reroute path)
        exists = any(
            link.from_socket == from_socket and link.to_socket == to_socket
            for link in node_tree.links
            if link.is_valid
        )
        if not exists:
            node_tree.links.new(from_socket, to_socket)


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

    def get_or_create_reroute(
        self, source_key: tuple[int, int, str]
    ) -> VirtualReroute:
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


def compute_node_depths(node_tree: bpy.types.NodeTree) -> dict[bpy.types.Node, int]:
    """Compute the depth of each node (distance from output nodes).

    Output nodes have depth 0, their inputs have depth 1, etc.
    """
    depths: dict[bpy.types.Node, int] = {}

    # Find output nodes (nodes with no output connections, or Group Output)
    output_nodes: list[bpy.types.Node] = []
    for node in node_tree.nodes:
        if node.type == "GROUP_OUTPUT":
            output_nodes.append(node)
        elif node.type == "FRAME":
            continue  # Skip frame nodes
        elif not any(
            link.from_node == node for link in node_tree.links if link.is_valid
        ):
            # Node has no outgoing links
            output_nodes.append(node)

    # BFS from output nodes (traversing backwards)
    queue: list[tuple[bpy.types.Node, int]] = [(n, 0) for n in output_nodes]
    for node, depth in queue:
        if node in depths and depths[node] <= depth:
            # Already visited with a shorter path
            continue
        depths[node] = depth

        # Find all nodes that feed into this node
        for link in node_tree.links:
            if not link.is_valid:
                continue
            if link.to_node == node and link.from_node is not None:
                queue.append((link.from_node, depth + 1))

    # Handle disconnected nodes (place at max depth + 1)
    max_depth = max(depths.values()) if depths else 0
    for node in node_tree.nodes:
        if node not in depths and node.type != "FRAME":
            depths[node] = max_depth + 1

    return depths


def compute_node_input_depths(node_tree: bpy.types.NodeTree) -> dict[bpy.types.Node, int]:
    """Compute the depth of each node from input nodes (distance from inputs).

    Input nodes (Group Input, nodes with no incoming connections) have depth 0,
    nodes they connect to have depth 1, etc.
    """
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

        # Find all nodes this node connects to
        for link in node_tree.links:
            if not link.is_valid:
                continue
            if link.from_node == node and link.to_node is not None:
                queue.append((link.to_node, depth + 1))

    # Handle disconnected nodes
    max_depth = max(depths.values()) if depths else 0
    for node in node_tree.nodes:
        if node not in depths and node.type != "FRAME":
            depths[node] = max_depth + 1

    return depths


def _build_connection_maps(
    node_tree: bpy.types.NodeTree,
    valid_nodes: set[bpy.types.Node],
) -> tuple[dict[bpy.types.Node, set[bpy.types.Node]], dict[bpy.types.Node, set[bpy.types.Node]]]:
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


def _shaker_push_pass(
    raw_positions: dict[bpy.types.Node, float],
    outputs_to: dict[bpy.types.Node, set[bpy.types.Node]],
    inputs_from: dict[bpy.types.Node, set[bpy.types.Node]],
    nudge: float,
) -> None:
    """Push nodes apart if they're in wrong order or same position."""
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
) -> None:
    """Refine column positions using shaker-sort style passes.

    Modifies raw_positions in place.
    Higher raw_position = more to the left (closer to inputs).
    Lower raw_position = more to the right (closer to outputs).

    Uses progressively smaller increments to create in-between positions without large jumps.
    """
    nudge = 1.0

    for _ in range(iterations):
        _shaker_gravity_pass(raw_positions, outputs_to, inputs_from, nudge)
        _shaker_push_pass(raw_positions, outputs_to, inputs_from, nudge)
        nudge = nudge * 0.75  # Decrease nudge for finer adjustments


def compute_node_columns(
    node_tree: bpy.types.NodeTree,
) -> dict[bpy.types.Node, int]:
    """Compute column position for each node using both input and output distances.

    Uses output_depth - input_depth to order nodes along the flow,
    then refines with shaker-sort style passes to separate same-column connections,
    then assigns sequential column numbers to collapse any gaps.
    """
    output_depths = compute_node_depths(node_tree)
    input_depths = compute_node_input_depths(node_tree)

    # Compute raw position value for each node (as float for fine-grained adjustment)
    raw_positions: dict[bpy.types.Node, float] = {}
    for node in node_tree.nodes:
        if node.type == "FRAME":
            continue
        out_d = output_depths.get(node, 0)
        in_d = input_depths.get(node, 0)
        raw_positions[node] = float(out_d - in_d)

    # Build connection maps and refine with shaker-sort passes
    outputs_to, inputs_from = _build_connection_maps(node_tree, set(raw_positions.keys()))
    _refine_columns_shaker(raw_positions, outputs_to, inputs_from)

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
) -> None:
    """Layout nodes in a PCB/FPGA-style grid with routed connections.

    Args:
        node_tree: The node tree to layout
        cell_width: Width of each grid cell (for nodes)
        cell_height: Height of each grid cell (for nodes)
        lane_width: Width allocated per reroute lane in the diagonal
        lane_gap: Gap before and after the lane area
    """
    if not node_tree.nodes:
        return

    # Step 0: Remove existing reroutes and restore direct connections
    # This ensures repeated layouts produce consistent results
    _remove_all_reroutes(node_tree)

    # Step 1: Compute column positions using both input and output distances
    columns = compute_node_columns(node_tree)
    if not columns:
        return

    # Step 2: Build virtual grid with initial node placement
    grid = _build_virtual_grid(columns)

    # Step 3: Collect all connections and add to grid
    _collect_connections(node_tree, grid, columns)

    # Step 4: Route all connections through the grid
    grid.route_all_connections()

    # Step 5: Mark which reroutes are actually used after optimization
    _mark_used_reroutes(grid)

    # Step 6: Realize the layout in Blender
    _realize_layout(node_tree, grid, cell_width, cell_height, lane_width, lane_gap)


def _build_virtual_grid(columns: dict[bpy.types.Node, int]) -> VirtualGrid:
    """Build virtual grid with initial node placement based on column positions."""
    grid = VirtualGrid()
    max_col = max(columns.values())

    # Group nodes by column for Y ordering
    nodes_by_column: dict[int, list[bpy.types.Node]] = {}
    for node, col in columns.items():
        if col not in nodes_by_column:
            nodes_by_column[col] = []
        nodes_by_column[col].append(node)

    # Calculate max height across all columns for vertical centering
    max_h = max(len(nodes) for nodes in nodes_by_column.values()) if nodes_by_column else 0

    # Place nodes: X based on column, Y based on order within column (centered)
    for col, nodes in nodes_by_column.items():
        nodes.sort(key=lambda n: n.name)  # Consistent ordering
        grid_x = max_col - col  # Flip so inputs are on left, outputs on right

        # Vertical centering: shift down by floor((max_h - h) / 2)
        h = len(nodes)
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


def _mark_used_reroutes(grid: VirtualGrid) -> None:
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
        optimized_path = _optimize_routing_path(path, conn.from_cell, conn.to_cell)

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
            (row_max_reroutes.get(y, 0) for y in range(min_y, max_y + 1) if (x, y) in grid.cells),
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
            (col_max_reroutes.get(x, 0) for x in range(min_x, max_x + 1) if (x, y) in grid.cells),
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
) -> None:
    """Realize the virtual grid layout in Blender."""
    # Calculate per-column and per-row lane areas
    col_lane_area, row_lane_area = _compute_lane_areas(grid, lane_width)

    # Calculate cumulative positions
    col_x_start, row_y_start = _compute_cell_positions(
        grid, col_lane_area, row_lane_area, cell_width, cell_height, lane_width, lane_gap
    )

    # Remove all existing links (we'll recreate them through reroutes)
    for link in list(node_tree.links):
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
    _create_reroute_nodes(node_tree, grid, col_lane_area, row_lane_area, col_x_start, row_y_start)

    # Create all connections through reroute chains
    for conn in grid.pending_connections:
        _realize_connection(node_tree, grid, conn)


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
    path = _optimize_routing_path(path, conn.from_cell, conn.to_cell)

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
) -> list[tuple[int, int]]:
    """Optimize routing path by melding unnecessary reroutes.

    Rule 1: Collapse vertical runs - if reroutes are stacked vertically (same X),
            keep only the last one in the run.

    Rule 2: Adjacent column meld - if after optimization only one reroute remains
            at the destination cell, and the source is in the adjacent column (X-1),
            we can skip it entirely for a direct node-to-node connection.
    """

    if not path:
        return path

    # Rule 1: Collapse vertical runs - keep only the last reroute of each vertical segment
    optimized: list[tuple[int, int]] = []
    i = 0
    while i < len(path):
        # Find the end of the current vertical run (same X)
        j = i
        while j + 1 < len(path) and path[j + 1][0] == path[i][0]:
            j += 1
        # Add only the first and last cell of this vertical run
        optimized.append(path[j])
        i = j + 1

    # Rule 2: If after optimization, only one reroute remains at dest cell,
    # and source is at X-1 (adjacent column), we can remove it entirely
    if (
        len(path) == 1 and
        from_cell[0] + 1 == to_cell[0]
    ):
        return []

    return optimized
