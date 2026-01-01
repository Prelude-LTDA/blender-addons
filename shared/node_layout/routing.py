"""Routing utilities for connection path optimization and lane computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import PendingConnection, VirtualGrid

__all__ = [
    "compute_cell_positions",
    "compute_lane_areas",
    "mark_used_reroutes",
    "optimize_routing_path",
]


def optimize_routing_path(
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

    Args:
        path: List of (x, y) cell coordinates representing the routing path
        from_cell: Source cell coordinates
        to_cell: Destination cell coordinates
        collapse_vertical: Whether to collapse vertical runs (default: True)
        collapse_horizontal: Whether to collapse horizontal runs of 3+ (default: True)
        collapse_adjacent: Whether to remove single reroute in adjacent column (default: True)

    Returns:
        Optimized list of cell coordinates
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


def build_routing_path(conn: PendingConnection) -> list[tuple[int, int]]:
    """Build the raw routing path for a connection (before optimization).

    Args:
        conn: The pending connection to build a path for

    Returns:
        List of (x, y) cell coordinates representing the routing path
    """
    from_x, from_y = conn.from_cell
    to_x, to_y = conn.to_cell

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

    return path


def mark_used_reroutes(
    grid: VirtualGrid,
    collapse_vertical: bool = True,
    collapse_horizontal: bool = True,
    collapse_adjacent: bool = True,
) -> None:
    """Mark which reroutes are actually used after path optimization.

    Args:
        grid: The virtual grid containing cells and pending connections
        collapse_vertical: Whether to collapse vertical runs (default: True)
        collapse_horizontal: Whether to collapse horizontal runs (default: True)
        collapse_adjacent: Whether to collapse adjacent reroutes (default: True)
    """
    for conn in grid.pending_connections:
        # Build the path of cells
        path = build_routing_path(conn)

        # Apply optimization
        optimized_path = optimize_routing_path(
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


def compute_lane_areas(
    grid: VirtualGrid,
    lane_width: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute lane area for each column and row based on max reroutes.

    Args:
        grid: The virtual grid containing cells and reroutes
        lane_width: Width allocated per reroute lane

    Returns:
        Tuple of (col_lane_area, row_lane_area) dictionaries mapping
        column/row indices to their lane area in pixels
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


def compute_cell_positions(
    grid: VirtualGrid,
    col_lane_area: dict[int, float],
    row_lane_area: dict[int, float],
    cell_width: float,
    cell_height: float,
    lane_width: float,
    lane_gap: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute cumulative X/Y positions for each column/row.

    Args:
        grid: The virtual grid containing cells
        col_lane_area: Lane area for each column (from compute_lane_areas)
        row_lane_area: Lane area for each row (from compute_lane_areas)
        cell_width: Width of each grid cell
        cell_height: Height of each grid cell
        lane_width: Default lane width
        lane_gap: Gap before and after the lane area

    Returns:
        Tuple of (col_x_start, row_y_start) dictionaries mapping
        column/row indices to their starting position in pixels
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
