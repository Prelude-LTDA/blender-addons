"""Grid building and node placement utilities."""

from __future__ import annotations

import math

import bpy

from .consts import (
    NODE_HEADER_HEIGHT,
    SOCKET_BASE_HEIGHT,
    SOCKET_HEIGHT_MULTIPLIERS,
)
from .types import VirtualGrid

__all__ = [
    "build_virtual_grid",
    "calculate_node_row_span",
    "collect_connections",
    "estimate_node_height",
]


def estimate_node_height(node: bpy.types.Node) -> float:
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


def calculate_node_row_span(
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
        node_height = estimate_node_height(node)

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


def build_virtual_grid(  # noqa: PLR0912
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
        lane_gap: Gap between cells (used when respect_dimensions is True)

    Returns:
        A VirtualGrid with nodes placed in their grid positions
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
                span = calculate_node_row_span(node, cell_height, lane_gap, i)
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
                span = calculate_node_row_span(node, cell_height, lane_gap, i)
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


def collect_connections(
    node_tree: bpy.types.NodeTree,
    grid: VirtualGrid,
    columns: dict[bpy.types.Node, int],
) -> None:
    """Collect all valid connections from the node tree and add to grid.

    Args:
        node_tree: The node tree to collect connections from
        grid: The virtual grid to add connections to
        columns: Dict mapping nodes to their column indices (used to filter connections)
    """
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
