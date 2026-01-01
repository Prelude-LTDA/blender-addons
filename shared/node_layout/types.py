"""Type definitions for the node layout system.

Contains dataclasses and core types used across the layout modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import bpy

__all__ = [
    "CellCoord",
    "GridCell",
    "PendingConnection",
    "SavedFrame",
    "SourceKey",
    "VirtualGrid",
    "VirtualReroute",
]


class CellCoord(NamedTuple):
    """Coordinates of a cell in the virtual grid."""

    x: int
    y: int


class SourceKey(NamedTuple):
    """Key identifying a connection source for reroute reuse."""

    cell: CellCoord
    socket_id: str


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


@dataclass
class VirtualReroute:
    """A reroute node in the virtual grid, before realization."""

    source_key: SourceKey
    blender_node: bpy.types.Node | None = None
    used: bool = False  # Whether this reroute is actually used after optimization


@dataclass
class GridCell:
    """A cell in the virtual grid."""

    x: int
    y: int
    node: bpy.types.Node | None = None
    # Reroutes keyed by source for reuse
    reroutes: dict[SourceKey, VirtualReroute] = field(default_factory=dict)

    def get_or_create_reroute(self, source_key: SourceKey) -> VirtualReroute:
        """Get existing reroute for source or create new one."""
        if source_key not in self.reroutes:
            self.reroutes[source_key] = VirtualReroute(source_key=source_key)
        return self.reroutes[source_key]


@dataclass
class PendingConnection:
    """A connection to be routed through the grid."""

    from_socket: bpy.types.NodeSocket
    to_socket: bpy.types.NodeSocket
    from_cell: CellCoord
    to_cell: CellCoord
    source_key: SourceKey


class VirtualGrid:
    """Virtual grid for planning node layout before realization."""

    def __init__(self) -> None:
        self.cells: dict[CellCoord, GridCell] = {}
        self.pending_connections: list[PendingConnection] = []
        self.node_to_cell: dict[bpy.types.Node, CellCoord] = {}

    def get_or_create_cell(self, x: int, y: int) -> GridCell:
        """Get existing cell or create new one."""
        key = CellCoord(x, y)
        if key not in self.cells:
            self.cells[key] = GridCell(x=x, y=y)
        return self.cells[key]

    def place_node(self, node: bpy.types.Node, x: int, y: int) -> None:
        """Place a node in the grid."""
        cell = self.get_or_create_cell(x, y)
        cell.node = node
        self.node_to_cell[node] = CellCoord(x, y)

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
        source_key = SourceKey(from_cell, from_socket.identifier)

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
