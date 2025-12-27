"""
Socket shape definitions and sync utilities for Voxel Terrain.

Defines the expected input/output socket interface for terrain node groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import bpy

# Socket type mapping from friendly names to Blender socket types
SOCKET_TYPES = {
    "Geometry": "NodeSocketGeometry",
    "Vector": "NodeSocketVector",
    "Float": "NodeSocketFloat",
    "Int": "NodeSocketInt",
    "Bool": "NodeSocketBool",
    "Color": "NodeSocketColor",
    "String": "NodeSocketString",
    "Material": "NodeSocketMaterial",
    "Object": "NodeSocketObject",
    "Collection": "NodeSocketCollection",
    "Image": "NodeSocketImage",
}


@dataclass
class SocketDef:
    """Definition of a single socket."""

    name: str
    socket_type: str  # Friendly name like "Geometry", "Float", etc.

    @property
    def bl_socket_type(self) -> str:
        """Get the Blender socket type name."""
        return SOCKET_TYPES.get(self.socket_type, self.socket_type)


# ============================================================================
# EXPECTED SOCKET SHAPE - Edit these lists to change the interface
# ============================================================================

INPUT_SOCKETS: list[SocketDef] = [
    SocketDef("Geometry", "Geometry"),
    SocketDef("Chunk Bounding Box", "Geometry"),
    SocketDef("Chunk Min", "Vector"),
    SocketDef("Chunk Max", "Vector"),
    SocketDef("Voxel Size", "Float"),
    SocketDef("LOD Level", "Int"),
]

OUTPUT_SOCKETS: list[SocketDef] = [
    SocketDef("Geometry", "Geometry"),
]

# ============================================================================


def get_existing_sockets(
    node_group: bpy.types.NodeTree,
    in_out: Literal["INPUT", "OUTPUT"],
) -> dict[str, str]:
    """
    Get existing sockets from a node group.

    Returns a dict mapping socket name to socket type.
    """
    result: dict[str, str] = {}
    for item in node_group.interface.items_tree:  # type: ignore[union-attr]
        if item.item_type == "SOCKET" and item.in_out == in_out:  # type: ignore[attr-defined]
            result[item.name] = item.socket_type  # type: ignore[attr-defined]
    return result


def check_sockets_match(node_group: bpy.types.NodeTree) -> bool:
    """
    Check if a node group's sockets match the expected shape.

    Returns True if all required sockets exist with correct types.
    """
    existing_inputs = get_existing_sockets(node_group, "INPUT")
    existing_outputs = get_existing_sockets(node_group, "OUTPUT")

    # Check all required inputs exist with correct type
    for socket_def in INPUT_SOCKETS:
        existing_type = existing_inputs.get(socket_def.name)
        if existing_type != socket_def.bl_socket_type:
            return False

    # Check all required outputs exist with correct type
    for socket_def in OUTPUT_SOCKETS:
        existing_type = existing_outputs.get(socket_def.name)
        if existing_type != socket_def.bl_socket_type:
            return False

    return True


def sync_sockets(node_group: bpy.types.NodeTree) -> tuple[int, int]:
    """
    Sync a node group's sockets to match the expected shape.

    Adds missing sockets but does not remove extra ones.

    Returns (inputs_added, outputs_added) count.
    """
    existing_inputs = get_existing_sockets(node_group, "INPUT")
    existing_outputs = get_existing_sockets(node_group, "OUTPUT")

    inputs_added = 0
    outputs_added = 0

    # Add missing input sockets
    for socket_def in INPUT_SOCKETS:
        existing_type = existing_inputs.get(socket_def.name)
        if existing_type is None:
            # Socket doesn't exist, create it
            node_group.interface.new_socket(  # type: ignore[union-attr]
                name=socket_def.name,
                socket_type=socket_def.bl_socket_type,
                in_out="INPUT",
            )
            inputs_added += 1
        elif existing_type != socket_def.bl_socket_type:
            # Socket exists but wrong type - remove and recreate
            for item in node_group.interface.items_tree:  # type: ignore[union-attr]
                if (
                    item.item_type == "SOCKET"
                    and item.in_out == "INPUT"  # type: ignore[attr-defined]
                    and item.name == socket_def.name  # type: ignore[attr-defined]
                ):
                    node_group.interface.remove(item)  # type: ignore[union-attr]
                    break
            node_group.interface.new_socket(  # type: ignore[union-attr]
                name=socket_def.name,
                socket_type=socket_def.bl_socket_type,
                in_out="INPUT",
            )
            inputs_added += 1

    # Add missing output sockets
    for socket_def in OUTPUT_SOCKETS:
        existing_type = existing_outputs.get(socket_def.name)
        if existing_type is None:
            # Socket doesn't exist, create it
            node_group.interface.new_socket(  # type: ignore[union-attr]
                name=socket_def.name,
                socket_type=socket_def.bl_socket_type,
                in_out="OUTPUT",
            )
            outputs_added += 1
        elif existing_type != socket_def.bl_socket_type:
            # Socket exists but wrong type - remove and recreate
            for item in node_group.interface.items_tree:  # type: ignore[union-attr]
                if (
                    item.item_type == "SOCKET"
                    and item.in_out == "OUTPUT"  # type: ignore[attr-defined]
                    and item.name == socket_def.name  # type: ignore[attr-defined]
                ):
                    node_group.interface.remove(item)  # type: ignore[union-attr]
                    break
            node_group.interface.new_socket(  # type: ignore[union-attr]
                name=socket_def.name,
                socket_type=socket_def.bl_socket_type,
                in_out="OUTPUT",
            )
            outputs_added += 1

    return inputs_added, outputs_added
