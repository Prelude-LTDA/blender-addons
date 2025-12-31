"""Simple, reliable node group structural comparison.

This module provides a fingerprint-based approach to comparing Blender node groups.
Instead of complex matching algorithms, it computes a deterministic fingerprint
of all structural elements and compares those.

Usage:
    from shared.node_group_compare import node_groups_match

    if node_groups_match(group_a, group_b):
        print("Groups are structurally identical")
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import bpy


def _strip_numeric_suffix(name: str) -> str:
    """Strip Blender's .001, .002 etc. suffixes from a name."""
    return re.sub(r"\.\d{3}$", "", name)


def _get_socket_value(socket: bpy.types.NodeSocket) -> Any:
    """Extract the default value from a socket as a JSON-serializable type."""
    val = getattr(socket, "default_value", None)
    if val is None:
        return None
    # Convert Blender vector/color/euler types to lists
    if hasattr(val, "__iter__") and not isinstance(val, str):
        try:
            return [round(v, 6) if isinstance(v, float) else v for v in val]
        except TypeError:
            # Not iterable after all, fall through
            pass
    # Round floats to avoid floating point comparison issues
    if isinstance(val, float):
        return round(val, 6)
    # Handle int, bool, str directly
    if isinstance(val, (int, bool, str)):
        return val
    # For any other type, convert to string representation
    return str(val)


def _compute_node_fingerprint(node: bpy.types.Node) -> dict[str, Any]:
    """Compute a fingerprint for a single node."""
    fingerprint: dict[str, Any] = {
        "type": node.bl_idname,
    }

    # For group nodes, include the nested group's base name
    node_tree = getattr(node, "node_tree", None)
    if node_tree is not None:
        fingerprint["group_name"] = _strip_numeric_suffix(node_tree.name)

    # Include all input socket default values
    inputs: dict[str, Any] = {}
    for inp in node.inputs:
        val = _get_socket_value(inp)
        if val is not None:
            inputs[inp.name] = val
    if inputs:
        fingerprint["inputs"] = inputs

    return fingerprint


def _compute_fingerprint(group: bpy.types.NodeTree, recurse: bool = True) -> str:
    """Compute a deterministic fingerprint for a node group.

    The fingerprint captures:
    - Interface sockets (name, type, direction, flags)
    - All nodes (type, nested group name if applicable)
    - All input socket default values
    - All links (as node_label:socket_name pairs)
    - Recursively, nested group fingerprints (if recurse=True)

    Args:
        group: The node group to fingerprint
        recurse: Whether to include nested group fingerprints

    Returns:
        A deterministic string fingerprint (MD5 hash of JSON structure)
    """
    data: dict[str, Any] = {
        "type": group.type,
    }

    # Interface sockets
    interface = group.interface
    if interface is not None:
        sockets = []
        for item in interface.items_tree:
            if getattr(item, "item_type", None) != "SOCKET":
                continue
            sockets.append({
                "name": getattr(item, "name", ""),
                "socket_type": getattr(item, "socket_type", ""),
                "in_out": getattr(item, "in_out", ""),
                "hide_value": getattr(item, "hide_value", False),
                "hide_in_modifier": getattr(item, "hide_in_modifier", False),
            })
        data["interface"] = sockets

    # Nodes - use label as stable identifier if set, otherwise generate one
    # based on type + index within that type
    nodes_by_type: dict[str, list[bpy.types.Node]] = {}
    for node in group.nodes:
        type_key = node.bl_idname
        node_tree = getattr(node, "node_tree", None)
        if node_tree is not None:
            type_key = f"{type_key}:{_strip_numeric_suffix(node_tree.name)}"
        if type_key not in nodes_by_type:
            nodes_by_type[type_key] = []
        nodes_by_type[type_key].append(node)

    # Assign stable IDs to nodes
    node_ids: dict[bpy.types.Node, str] = {}
    for type_key in sorted(nodes_by_type.keys()):
        nodes = nodes_by_type[type_key]
        # Sort by label if available, then by some stable property
        nodes.sort(key=lambda n: (n.label or "", len(n.inputs), len(n.outputs)))
        for i, node in enumerate(nodes):
            node_ids[node] = f"{type_key}#{i}"

    # Build node data
    nodes_data: dict[str, dict[str, Any]] = {}
    for node, node_id in sorted(node_ids.items(), key=lambda x: x[1]):
        nodes_data[node_id] = _compute_node_fingerprint(node)
    data["nodes"] = nodes_data

    # Links - use stable node IDs
    links = []
    for link in group.links:
        if link.from_node is None or link.to_node is None:
            continue
        if link.from_socket is None or link.to_socket is None:
            continue
        from_id = node_ids.get(link.from_node)
        to_id = node_ids.get(link.to_node)
        if from_id is None or to_id is None:
            continue
        links.append({
            "from": f"{from_id}:{link.from_socket.name}",
            "to": f"{to_id}:{link.to_socket.name}",
        })
    # Sort links for determinism
    links.sort(key=lambda x: (x["from"], x["to"]))
    data["links"] = links

    # Nested group fingerprints (recursive)
    if recurse:
        nested: dict[str, str] = {}
        for node in group.nodes:
            node_tree = getattr(node, "node_tree", None)
            if node_tree is not None:
                base_name = _strip_numeric_suffix(node_tree.name)
                if base_name not in nested:
                    nested[base_name] = _compute_fingerprint(node_tree, recurse=True)
        if nested:
            data["nested"] = nested

    # Convert to JSON and hash
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(json_str.encode()).hexdigest()


def node_groups_match(
    group_a: bpy.types.NodeTree | None,
    group_b: bpy.types.NodeTree | None,
) -> bool:
    """Check if two node groups are structurally identical.

    This compares all structural aspects:
    - Interface sockets (name, type, direction, flags)
    - Node types and counts
    - All input socket default values
    - All link connections
    - Nested node groups (recursively)

    Does NOT compare:
    - Node positions/locations
    - Node names (only labels and types)
    - Custom properties

    Args:
        group_a: First node group
        group_b: Second node group

    Returns:
        True if the groups match structurally, False otherwise.
    """
    if group_a is None or group_b is None:
        return group_a is None and group_b is None

    fp_a = _compute_fingerprint(group_a)
    fp_b = _compute_fingerprint(group_b)

    return fp_a == fp_b


def get_fingerprint(group: bpy.types.NodeTree) -> str:
    """Get the fingerprint of a node group.

    Useful for debugging or caching purposes.

    Args:
        group: The node group to fingerprint

    Returns:
        The fingerprint string (MD5 hash)
    """
    return _compute_fingerprint(group)
