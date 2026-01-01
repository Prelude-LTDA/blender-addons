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
from .grid import (
    build_virtual_grid,
    calculate_node_row_span,
    collect_connections,
    estimate_node_height,
)
from .routing import (
    build_routing_path,
    compute_cell_positions,
    compute_lane_areas,
    mark_used_reroutes,
    optimize_routing_path,
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
    # Routing utilities (from routing.py)
    "build_routing_path",
    # Grid utilities (from grid.py)
    "build_virtual_grid",
    "calculate_node_row_span",
    "collect_connections",
    "compute_cell_positions",
    "compute_lane_areas",
    "compute_node_columns",
    "compute_node_depths",
    "compute_node_input_depths",
    "estimate_node_height",
    # Main layout function
    "layout_nodes_pcb_style",
    "mark_used_reroutes",
    "optimize_routing_path",
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
    grid = build_virtual_grid(
        columns,
        vertical_align,
        respect_dimensions=respect_dimensions,
        cell_height=cell_height,
        lane_gap=lane_gap,
    )

    # Step 3: Collect all connections and add to grid
    collect_connections(node_tree, grid, columns)

    # Step 4: Route all connections through the grid
    grid.route_all_connections()

    # Step 5: Mark which reroutes are actually used after optimization
    mark_used_reroutes(
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
    col_lane_area, row_lane_area = compute_lane_areas(grid, lane_width)

    # Calculate cumulative positions
    col_x_start, row_y_start = compute_cell_positions(
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
    # Build the path of cells this connection travels through
    path = build_routing_path(conn)

    # Optimize path by melding unnecessary reroutes
    path = optimize_routing_path(
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
