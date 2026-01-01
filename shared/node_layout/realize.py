"""Layout realization - converting virtual grid to actual Blender nodes and links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from .routing import (
    build_routing_path,
    compute_cell_positions,
    compute_lane_areas,
    optimize_routing_path,
)

if TYPE_CHECKING:
    from .types import PendingConnection, VirtualGrid

__all__ = [
    "create_reroute_nodes",
    "realize_connection",
    "realize_layout",
]


def realize_layout(
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
    """Realize the virtual grid layout in Blender.

    This function takes the virtual grid representation and creates the actual
    Blender node positions and reroute connections.

    Args:
        node_tree: The Blender node tree to modify
        grid: The virtual grid containing node placements and connections
        cell_width: Width of each grid cell
        cell_height: Height of each grid cell
        lane_width: Width allocated per reroute lane
        lane_gap: Gap before and after the lane area
        nodes_to_layout: If provided, only layout these specific nodes
        collapse_vertical: Whether to collapse vertical runs of reroutes
        collapse_horizontal: Whether to collapse horizontal runs of reroutes
        collapse_adjacent: Whether to collapse adjacent reroutes
    """
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
    create_reroute_nodes(
        node_tree, grid, col_lane_area, row_lane_area, col_x_start, row_y_start
    )

    # Create all connections through reroute chains
    for conn in grid.pending_connections:
        realize_connection(
            node_tree,
            grid,
            conn,
            collapse_vertical=collapse_vertical,
            collapse_horizontal=collapse_horizontal,
            collapse_adjacent=collapse_adjacent,
        )


def create_reroute_nodes(
    node_tree: bpy.types.NodeTree,
    grid: VirtualGrid,
    col_lane_area: dict[int, float],
    row_lane_area: dict[int, float],
    col_x_start: dict[int, float],
    row_y_start: dict[int, float],
) -> None:
    """Create reroute nodes in each cell, positioned diagonally.

    Args:
        node_tree: The Blender node tree to add reroutes to
        grid: The virtual grid containing cell information
        col_lane_area: Lane area for each column
        row_lane_area: Lane area for each row
        col_x_start: Starting X position for each column
        row_y_start: Starting Y position for each row
    """
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


def realize_connection(
    node_tree: bpy.types.NodeTree,
    grid: VirtualGrid,
    conn: PendingConnection,
    collapse_vertical: bool = True,
    collapse_horizontal: bool = True,
    collapse_adjacent: bool = True,
) -> None:
    """Create the actual Blender links for a routed connection.

    Args:
        node_tree: The Blender node tree to add links to
        grid: The virtual grid containing reroute nodes
        conn: The pending connection to realize
        collapse_vertical: Whether to collapse vertical runs of reroutes
        collapse_horizontal: Whether to collapse horizontal runs of reroutes
        collapse_adjacent: Whether to collapse adjacent reroutes
    """
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
