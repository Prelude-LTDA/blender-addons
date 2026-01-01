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

import bpy

from .consts import (
    NODE_HEADER_HEIGHT,
    SOCKET_BASE_HEIGHT,
    SOCKET_HEIGHT_MULTIPLIERS,
)
from .frames import (
    remove_all_frames,
    restore_all_frames,
    save_all_frames,
)
from .graph import (
    build_connection_maps,
    build_link_adjacency,
    compute_node_columns,
    compute_node_depths,
    compute_node_input_depths,
    remove_all_reroutes,
    trace_to_real_source,
)
from .types import (
    PendingConnection,
    SavedFrame,
    VirtualGrid,
)

# Re-export public API
__all__ = [
    # Constants (from consts.py)
    "NODE_HEADER_HEIGHT",
    "SOCKET_BASE_HEIGHT",
    "SOCKET_HEIGHT_MULTIPLIERS",
    # Types (from types.py)
    "PendingConnection",
    "SavedFrame",
    "VirtualGrid",
    # Graph utilities (from graph.py)
    "build_connection_maps",
    "build_link_adjacency",
    "compute_node_columns",
    "compute_node_depths",
    "compute_node_input_depths",
    # Main layout function
    "layout_nodes_pcb_style",
    # Frame utilities (from frames.py)
    "remove_all_frames",
    "remove_all_reroutes",
    "restore_all_frames",
    "save_all_frames",
    "trace_to_real_source",
]


def layout_nodes_pcb_style(  # noqa: PLR0912, PLR0915
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
            if node.type not in ("FRAME", "REROUTE") and (
                nodes_to_layout is None or node in nodes_to_layout
            ):
                original_positions[node] = node.location.x

    # Step 0a: Remove existing reroutes and restore direct connections
    # This ensures repeated layouts produce consistent results
    remove_all_reroutes(node_tree, nodes_to_layout)

    # Step 0b: Save frame info and remove frames
    # Frames interfere with layout since node positions are relative to parent frame
    saved_frames = save_all_frames(node_tree, nodes_to_layout)
    remove_all_frames(node_tree, nodes_to_layout)

    # Compute column width for position-based sorting
    # Use min(cell_width, max_node_width) so we don't produce fewer columns than needed
    # (we can produce more columns if user chooses smaller cell_width, but not fewer)
    ui_scale = 1.0
    if bpy.context is not None and bpy.context.preferences is not None:
        ui_scale = bpy.context.preferences.system.ui_scale
    max_node_width = 0.0
    for node in node_tree.nodes:
        if node.type not in ("REROUTE", "FRAME") and (
            nodes_to_layout is None or node in nodes_to_layout
        ):
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
        restore_all_frames(node_tree, saved_frames)
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
    restore_all_frames(node_tree, saved_frames)

    return created_reroutes


def _estimate_node_height(node: bpy.types.Node) -> float:
    """Estimate node height based on socket count and types.

    Used as a fallback when node.dimensions returns (0, 0).

    Args:
        node: The node to estimate height for

    Returns:
        Estimated height in pixels
    """
    height = NODE_HEADER_HEIGHT

    # Count input sockets with their type multipliers
    for socket in node.inputs:
        if socket.enabled:
            # Connected sockets are always 1 row tall (no expanded fields)
            if socket.is_linked:
                height += SOCKET_BASE_HEIGHT
            else:
                # Get socket type - try both .type and bl_idname
                socket_type = getattr(socket, "type", "") or ""
                socket_idname = getattr(socket, "bl_idname", "") or ""

                multiplier = SOCKET_HEIGHT_MULTIPLIERS.get(
                    socket_type, SOCKET_HEIGHT_MULTIPLIERS.get(socket_idname, 1.0)
                )
                height += SOCKET_BASE_HEIGHT * multiplier

    # Count output sockets (usually simpler, no expanded fields)
    for socket in node.outputs:
        if socket.enabled:
            height += SOCKET_BASE_HEIGHT

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


def _build_virtual_grid(  # noqa: PLR0912
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

    # Rule 3: If after optimization, only one reroute remains at dest cell,
    # and source is at X-1 (adjacent column), we can remove it entirely
    if collapse_adjacent and len(optimized) == 1 and from_cell[0] + 1 == to_cell[0]:
        return []

    return optimized
