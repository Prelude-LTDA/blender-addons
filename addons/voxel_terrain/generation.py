"""
Generation and baking logic for Voxel Terrain.

Provides a reusable iterator for processing chunks across LODs,
with support for Generate, Bake, and Export modes.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import bpy
from mathutils import Vector

from .chunks import calculate_chunk_bounds
from .shared.node_layout import layout_nodes_pcb_style
from .typing_utils import get_object_props, get_scene_props

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class GenerationMode(Enum):
    """Mode for the generation process."""

    GENERATE = auto()  # Create editable geometry nodes setup
    BAKE = auto()  # Bake to static mesh
    EXPORT = auto()  # Export to file (future)


@dataclass
class ChunkInfo:
    """Information about a single chunk being processed."""

    lod_level: int
    chunk_index: tuple[int, int, int]
    chunk_min: Vector
    chunk_max: Vector
    voxel_size: float
    skirt_min: Vector
    skirt_max: Vector


@dataclass
class GenerationProgress:
    """Progress information for the generation process."""

    lod_index: int  # Which LOD iteration (0-based, in processing order)
    lod_level: int  # The actual LOD level (0=highest quality)
    total_lods: int
    current_chunk: int
    total_chunks: int
    chunk_info: ChunkInfo | None
    message: str
    # Per-LOD chunk counts for ETA calculation
    chunks_completed: int = 0  # Total chunks completed so far
    chunks_total: int = 1  # Total chunks across all LODs
    # LOD-aware progress info: list of (lod_level, chunks_in_lod) in processing order
    lod_chunk_counts: list[tuple[int, int]] | None = None

    @property
    def progress(self) -> float:
        """Overall progress from 0.0 to 1.0, based on chunk count."""
        if self.chunks_total <= 0:
            return 0.0
        return min(1.0, self.chunks_completed / self.chunks_total)


@dataclass
class GenerationResult:
    """Result of the generation process."""

    success: bool
    cancelled: bool
    message: str
    chunks_processed: int


# Names for hierarchy objects
TERRAIN_ROOT_NAME = "Voxel Terrain"


def get_or_create_empty(
    name: str,
    parent: bpy.types.Object | None = None,
) -> bpy.types.Object:
    """Get or create an empty object with the given name.

    Empty objects are used to organize the hierarchy and appear
    collapsed by default in the outliner.
    """
    if name in bpy.data.objects:
        empty = bpy.data.objects[name]
    else:
        empty = bpy.data.objects.new(name, None)  # None = Empty
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.5
        # Link to scene collection
        scene = bpy.context.scene
        assert scene is not None
        scene.collection.objects.link(empty)

    # Set parent if specified
    if parent is not None and empty.parent != parent:
        empty.parent = parent

    return empty


def get_chunk_empty_name(x: int, y: int, z: int) -> str:
    """Get the name for a chunk empty object."""
    return f"Chunk ({x}, {y}, {z})"


def get_lod_object_name(x: int, y: int, z: int, lod_level: int) -> str:
    """Get the name for a LOD object (globally unique)."""
    return f"Voxel Terrain - Chunk ({x}, {y}, {z}) - LOD {lod_level}"


def calculate_chunks_for_lod(
    scene_props: object,
    lod_level: int,
) -> list[ChunkInfo]:
    """Calculate all chunk infos for a given LOD level."""
    chunks: list[ChunkInfo] = []

    # Get chunk bounds from terrain objects
    scene = bpy.context.scene
    assert scene is not None
    chunk_bounds = calculate_chunk_bounds(scene)
    if chunk_bounds is None:
        return chunks  # No terrain objects

    # Get chunk_size as a tuple of floats (it's a Vector from FloatVectorProperty)
    chunk_size_prop = getattr(scene_props, "chunk_size", (16.0, 16.0, 16.0))
    chunk_size = (
        float(chunk_size_prop[0]),
        float(chunk_size_prop[1]),
        float(chunk_size_prop[2]),
    )
    base_voxel_size = float(getattr(scene_props, "voxel_size", 1.0))
    lod_factor = float(getattr(scene_props, "lod_factor", 2.0))
    skirt_size = int(getattr(scene_props, "skirt_size", 1))

    # Voxel size increases with LOD factor for each level
    voxel_size = base_voxel_size * (lod_factor**lod_level)

    # Skirt extends by skirt_size voxels in each direction
    skirt_voxels = skirt_size * voxel_size

    # Iterate over chunk indices from calculated bounds (Z, X, Y order for 3D printer effect)
    min_chunk = chunk_bounds.min_chunk
    max_chunk = chunk_bounds.max_chunk

    for z in range(min_chunk[2], max_chunk[2] + 1):
        for x in range(min_chunk[0], max_chunk[0] + 1):
            for y in range(min_chunk[1], max_chunk[1] + 1):
                chunk_min = Vector(
                    (
                        x * chunk_size[0],
                        y * chunk_size[1],
                        z * chunk_size[2],
                    )
                )
                chunk_max = Vector(
                    (
                        (x + 1) * chunk_size[0],
                        (y + 1) * chunk_size[1],
                        (z + 1) * chunk_size[2],
                    )
                )
                # Skirt extends by skirt_voxels in each direction
                skirt_min = Vector(
                    (
                        chunk_min[0] - skirt_voxels,
                        chunk_min[1] - skirt_voxels,
                        chunk_min[2] - skirt_voxels,
                    )
                )
                skirt_max = Vector(
                    (
                        chunk_max[0] + skirt_voxels,
                        chunk_max[1] + skirt_voxels,
                        chunk_max[2] + skirt_voxels,
                    )
                )

                chunks.append(
                    ChunkInfo(
                        lod_level=lod_level,
                        chunk_index=(x, y, z),
                        chunk_min=chunk_min,
                        chunk_max=chunk_max,
                        voxel_size=voxel_size,
                        skirt_min=skirt_min,
                        skirt_max=skirt_max,
                    )
                )

    return chunks


def get_voxel_terrain_objects() -> list[bpy.types.Object]:
    """Get all objects with Voxel Terrain enabled that are in the scene."""
    objects: list[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        # Skip objects that aren't in any scene (deleted but still referenced)
        if len(obj.users_scene) == 0:
            continue
        props = get_object_props(obj)
        if props.enabled and props.node_group is not None:
            objects.append(obj)
    return objects


def _create_chunk_input_sockets(node_tree: bpy.types.NodeTree) -> None:
    """Create all input sockets for the chunk node group."""
    interface = node_tree.interface  # type: ignore[union-attr]

    # Check existing sockets
    existing_inputs = {
        item.name  # type: ignore[union-attr]
        for item in interface.items_tree  # type: ignore[union-attr]
        if getattr(item, "item_type", None) == "SOCKET"
        and getattr(item, "in_out", None) == "INPUT"
    }

    # Define input sockets with their types and defaults
    # Chunk Translation: center of the chunk (including skirt)
    # Chunk Size: full size of the chunk (including skirt)
    input_sockets = [
        ("Chunk Translation", "NodeSocketVector", (0.0, 0.0, 0.0)),
        ("Chunk Size", "NodeSocketVector", (16.0, 16.0, 16.0)),
        ("Voxel Size", "NodeSocketFloat", 1.0),
        ("LOD Level", "NodeSocketInt", 0),
    ]

    for name, socket_type, default in input_sockets:
        if name not in existing_inputs:
            socket = interface.new_socket(  # type: ignore[union-attr]
                name=name,
                socket_type=socket_type,
                in_out="INPUT",
            )
            socket.default_value = default  # type: ignore[union-attr]


def _add_terrain_object_nodes(
    node_tree: bpy.types.NodeTree,
    terrain_obj: bpy.types.Object,
    terrain_node_group: bpy.types.NodeTree,
    input_node: bpy.types.Node,
    bbox_node: bpy.types.Node,
    bounds_bbox_node: bpy.types.Node,
    join_node: bpy.types.Node,
) -> None:
    """Add nodes for a single terrain object to the chunk node setup."""
    # Create Object Info node to get the object's geometry
    obj_info_node = node_tree.nodes.new("GeometryNodeObjectInfo")
    obj_info_node.inputs["Object"].default_value = terrain_obj  # type: ignore[index]
    obj_info_node.transform_space = "RELATIVE"  # type: ignore[attr-defined]
    obj_info_node.label = f"Source: {terrain_obj.name}"
    obj_info_node.select = False

    # Create Group node for the terrain node group
    group_node = node_tree.nodes.new("GeometryNodeGroup")
    group_node.node_tree = terrain_node_group  # type: ignore[attr-defined]
    group_node.label = f"Terrain: {terrain_obj.name}"
    group_node.select = False

    # Connect object geometry to group input
    if "Geometry" in group_node.inputs:
        node_tree.links.new(
            obj_info_node.outputs["Geometry"],
            group_node.inputs["Geometry"],
        )

    # Connect chunk bounds geometry if the input exists
    if "Chunk Bounding Box" in group_node.inputs:
        node_tree.links.new(
            bbox_node.outputs["Geometry"],
            group_node.inputs["Chunk Bounding Box"],
        )

    # Connect Chunk Min/Max from Bounding Box node
    if "Chunk Min" in group_node.inputs:
        node_tree.links.new(
            bounds_bbox_node.outputs["Min"],
            group_node.inputs["Chunk Min"],
        )
    if "Chunk Max" in group_node.inputs:
        node_tree.links.new(
            bounds_bbox_node.outputs["Max"],
            group_node.inputs["Chunk Max"],
        )

    # Connect Voxel Size and LOD Level from Group Input
    if "Voxel Size" in group_node.inputs:
        node_tree.links.new(
            input_node.outputs["Voxel Size"],
            group_node.inputs["Voxel Size"],
        )
    if "LOD Level" in group_node.inputs:
        node_tree.links.new(
            input_node.outputs["LOD Level"],
            group_node.inputs["LOD Level"],
        )

    # Connect group output to join
    if group_node.outputs:
        node_tree.links.new(
            group_node.outputs["Geometry"],
            join_node.inputs["Geometry"],
        )


def _create_chunk_bounds_nodes(
    node_tree: bpy.types.NodeTree,
    input_node: bpy.types.Node,
) -> bpy.types.Node:
    """Create nodes for chunk bounds geometry.

    The Chunk Translation and Chunk Size inputs already include the skirt,
    so we just create a cube at the right position with the right size.

    Returns the transform node that outputs the positioned chunk bounds.
    """
    # Create cube with Chunk Size (already includes skirt from Python calculation)
    cube_node = node_tree.nodes.new("GeometryNodeMeshCube")
    cube_node.label = "Chunk Bounds"
    cube_node.select = False
    node_tree.links.new(input_node.outputs["Chunk Size"], cube_node.inputs["Size"])

    # Transform cube to Chunk Translation (already calculated in Python)
    transform_node = node_tree.nodes.new("GeometryNodeTransform")
    transform_node.label = "Position Chunk"
    transform_node.select = False
    node_tree.links.new(cube_node.outputs["Mesh"], transform_node.inputs["Geometry"])
    node_tree.links.new(
        input_node.outputs["Chunk Translation"], transform_node.inputs["Translation"]
    )

    return transform_node


def get_or_create_chunk_node_group(
    terrain_objects: list[bpy.types.Object],
) -> bpy.types.NodeTree:
    """Get or create the shared geometry nodes setup for all chunks.

    Since all chunk-specific values are passed via modifier inputs,
    we can reuse a single node group across all chunks.
    """
    node_group_name = "VT_ChunkGenerator"

    # Remove existing node group to ensure fresh setup
    # (Socket layout may have changed between addon versions)
    if node_group_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[node_group_name])

    node_tree = bpy.data.node_groups.new(node_group_name, "GeometryNodeTree")

    # Keep the node group even if no modifiers reference it temporarily
    node_tree.use_fake_user = True

    # Create output socket
    node_tree.interface.new_socket(  # type: ignore[union-attr]
        name="Geometry",
        socket_type="NodeSocketGeometry",
        in_out="OUTPUT",
    )

    # Create input sockets for chunk parameters
    _create_chunk_input_sockets(node_tree)

    # Create Group Input node
    input_node = node_tree.nodes.new("NodeGroupInput")
    input_node.select = False

    # Create Group Output node
    output_node = node_tree.nodes.new("NodeGroupOutput")
    output_node.select = False

    # Create Join Geometry node
    join_node = node_tree.nodes.new("GeometryNodeJoinGeometry")
    join_node.label = "Join All Terrain"
    join_node.select = False

    # Connect join to output
    node_tree.links.new(join_node.outputs["Geometry"], output_node.inputs[0])

    # Create chunk bounds nodes
    transform_node = _create_chunk_bounds_nodes(node_tree, input_node)

    # Create Bounding Box node to get min/max from the chunk bounds geometry (shared)
    bounds_bbox_node = node_tree.nodes.new("GeometryNodeBoundBox")
    bounds_bbox_node.label = "Chunk Bounds"
    bounds_bbox_node.select = False
    node_tree.links.new(
        transform_node.outputs["Geometry"],
        bounds_bbox_node.inputs["Geometry"],
    )

    # Add terrain object nodes
    for terrain_obj in terrain_objects:
        obj_props = get_object_props(terrain_obj)
        terrain_node_group = obj_props.node_group

        if terrain_node_group is None:
            continue

        _add_terrain_object_nodes(
            node_tree,
            terrain_obj,
            terrain_node_group,
            input_node,
            transform_node,
            bounds_bbox_node,
            join_node,
        )

    # Layout nodes using PCB-style arrangement
    layout_nodes_pcb_style(node_tree)

    return node_tree


def _setup_chunk_modifier(
    chunk_obj: bpy.types.Object,
    node_tree: bpy.types.NodeTree,
    chunk_info: ChunkInfo,
) -> bpy.types.Modifier:
    """Set up a geometry nodes modifier on a chunk object with proper inputs.

    Returns the modifier.
    """
    modifier_name = "VoxelTerrain"

    # Remove existing modifier to ensure clean socket setup
    # (Old modifiers may have incompatible IDProperty types from previous versions)
    existing = chunk_obj.modifiers.get(modifier_name)
    if existing is not None:
        chunk_obj.modifiers.remove(existing)

    # Create fresh modifier
    modifier = chunk_obj.modifiers.new(modifier_name, "NODES")
    modifier.node_group = node_tree  # type: ignore[union-attr]

    # Build a mapping of socket name -> identifier from the node tree interface
    socket_ids: dict[str, str] = {}
    for item in node_tree.interface.items_tree:  # type: ignore[union-attr]
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "in_out", None) != "INPUT":
            continue
        socket_ids[item.name] = item.identifier  # type: ignore[union-attr]

    # Set input socket values on the modifier using the correct identifiers
    # Calculate chunk translation (center of skirt bounds)
    chunk_translation = (
        (chunk_info.skirt_min[0] + chunk_info.skirt_max[0]) / 2,
        (chunk_info.skirt_min[1] + chunk_info.skirt_max[1]) / 2,
        (chunk_info.skirt_min[2] + chunk_info.skirt_max[2]) / 2,
    )
    # Calculate chunk size (full extent including skirt)
    chunk_size = (
        chunk_info.skirt_max[0] - chunk_info.skirt_min[0],
        chunk_info.skirt_max[1] - chunk_info.skirt_min[1],
        chunk_info.skirt_max[2] - chunk_info.skirt_min[2],
    )

    modifier[socket_ids["Chunk Translation"]] = chunk_translation
    modifier[socket_ids["Chunk Size"]] = chunk_size
    modifier[socket_ids["Voxel Size"]] = chunk_info.voxel_size
    modifier[socket_ids["LOD Level"]] = chunk_info.lod_level

    return modifier


def get_or_create_chunk_object(
    name: str,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    """Get or create a chunk object parented to the given empty."""
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        # Clear existing mesh data
        mesh = obj.data
        if isinstance(mesh, bpy.types.Mesh):
            mesh.clear_geometry()
    else:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        # Link to scene collection
        scene = bpy.context.scene
        assert scene is not None
        scene.collection.objects.link(obj)

    # Set parent
    if obj.parent != parent:
        obj.parent = parent

    # Enable shade smooth
    obj.data.shade_smooth()  # type: ignore[union-attr]

    return obj


def _bake_chunk_to_mesh(
    chunk_obj: bpy.types.Object,
    modifier: bpy.types.Modifier,
) -> None:
    """Evaluate geometry nodes and bake to static mesh."""
    # Get evaluated mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = chunk_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    # Capture per-face smooth shading from evaluated mesh (vectorized)
    num_polys = len(eval_mesh.polygons)
    face_smooth = [False] * num_polys
    eval_mesh.polygons.foreach_get("use_smooth", face_smooth)

    # Copy to the object's mesh
    chunk_mesh = chunk_obj.data
    if isinstance(chunk_mesh, bpy.types.Mesh):
        chunk_mesh.clear_geometry()
        chunk_mesh.from_pydata(
            [v.co[:] for v in eval_mesh.vertices],
            [(e.vertices[0], e.vertices[1]) for e in eval_mesh.edges],
            [tuple(p.vertices) for p in eval_mesh.polygons],
        )
        chunk_mesh.update()
        # Restore per-face smooth shading (vectorized)
        chunk_mesh.polygons.foreach_set("use_smooth", face_smooth)

    # Clean up evaluated mesh
    eval_obj.to_mesh_clear()

    # Remove modifier (node group is shared, don't remove it here)
    chunk_obj.modifiers.remove(modifier)


def _precalculate_chunks(
    scene_props: object,
    lod_levels: int,
) -> tuple[list[int], list[list[ChunkInfo]], int, list[tuple[int, int]]]:
    """
    Pre-calculate chunks for all LODs.

    Returns:
        Tuple of (lod_order, chunks_per_lod, total_chunks, lod_chunk_counts)
        lod_chunk_counts is list of (lod_level, chunk_count) in processing order
    """
    # We process in reverse order (highest lod_level first = lowest detail)
    lod_order = list(range(lod_levels - 1, -1, -1))
    chunks_per_lod: list[list[ChunkInfo]] = []
    lod_chunk_counts: list[tuple[int, int]] = []
    total_chunks = 0

    for lod_level in lod_order:
        chunks = calculate_chunks_for_lod(scene_props, lod_level)
        chunks_per_lod.append(chunks)
        lod_chunk_counts.append((lod_level, len(chunks)))
        total_chunks += len(chunks)

    return lod_order, chunks_per_lod, total_chunks, lod_chunk_counts


def generation_iterator(
    mode: GenerationMode,
    cancel_check: Callable[[], bool] | None = None,
) -> Generator[GenerationProgress, None, GenerationResult]:
    """
    Iterator that yields progress updates during generation.

    Args:
        mode: The generation mode (GENERATE, BAKE, EXPORT)
        cancel_check: Optional callable that returns True if cancelled


    Yields:
        GenerationProgress objects with current status

    Returns:
        GenerationResult with final status
    """
    scene = bpy.context.scene
    assert scene is not None
    scene_props = get_scene_props(scene)

    # Use lod_levels from scene properties (defaults to 4)
    # LOD 0 is highest detail, LOD (lod_levels-1) is lowest
    enable_lod = getattr(scene_props, "enable_lod", False)
    lod_levels = getattr(scene_props, "lod_levels", 4) if enable_lod else 1
    total_lods = lod_levels

    # Get terrain objects
    terrain_objects = get_voxel_terrain_objects()
    if not terrain_objects:
        return GenerationResult(
            success=False,
            cancelled=False,
            message="No objects with Voxel Terrain enabled",
            chunks_processed=0,
        )

    # Create main empty as root of hierarchy
    terrain_root = get_or_create_empty(TERRAIN_ROOT_NAME)

    # Create shared node group once (before processing any chunks)
    node_tree = get_or_create_chunk_node_group(terrain_objects)

    chunks_processed = 0

    # Pre-calculate all chunks for all LODs
    lod_order, chunks_per_lod, total_chunks_all_lods, lod_chunk_counts = (
        _precalculate_chunks(scene_props, lod_levels)
    )

    # Iterate over LODs in reverse order (lowest quality first for faster preview)
    # LOD 0 is highest detail, LOD (lod_levels-1) is lowest detail
    for lod_index, lod_level in enumerate(lod_order):
        # Get pre-calculated chunks for this LOD
        chunks = chunks_per_lod[lod_index]
        total_chunks = len(chunks)

        # Iterate over chunks
        for chunk_idx, chunk_info in enumerate(chunks):
            # Check for cancellation
            if cancel_check is not None and cancel_check():
                return GenerationResult(
                    success=False,
                    cancelled=True,
                    message="Generation cancelled by user",
                    chunks_processed=chunks_processed,
                )

            # Yield progress
            yield GenerationProgress(
                lod_index=lod_index,
                lod_level=lod_level,
                total_lods=total_lods,
                current_chunk=chunk_idx,
                total_chunks=total_chunks,
                chunk_info=chunk_info,
                message=f"Chunk {chunk_info.chunk_index} - LOD {lod_level}",
                chunks_completed=chunks_processed,
                chunks_total=total_chunks_all_lods,
                lod_chunk_counts=lod_chunk_counts,
            )

            # Get or create chunk empty (Voxel Terrain > Chunk (x, y, z))
            chunk_empty_name = get_chunk_empty_name(*chunk_info.chunk_index)
            chunk_empty = get_or_create_empty(
                chunk_empty_name,
                parent=terrain_root,
            )

            # Get or create LOD object parented to chunk empty
            x, y, z = chunk_info.chunk_index
            lod_obj_name = get_lod_object_name(x, y, z, chunk_info.lod_level)

            # Delete stale higher quality LODs (lower lod_level numbers)
            # since we're regenerating and they'd be out of date
            for lower_lod in range(0, lod_level):
                lower_lod_name = get_lod_object_name(x, y, z, lower_lod)
                lower_lod_obj = bpy.data.objects.get(lower_lod_name)
                if lower_lod_obj is not None:
                    # Remove mesh data first
                    mesh = lower_lod_obj.data
                    bpy.data.objects.remove(lower_lod_obj, do_unlink=True)
                    if isinstance(mesh, bpy.types.Mesh) and mesh.users == 0:
                        bpy.data.meshes.remove(mesh)

            # Hide lower quality LODs (higher lod_level numbers) immediately
            # so user sees them disappear as soon as the new LOD is created
            for higher_lod in range(lod_level + 1, lod_levels):
                higher_lod_name = get_lod_object_name(x, y, z, higher_lod)
                higher_lod_obj = bpy.data.objects.get(higher_lod_name)
                if higher_lod_obj is not None:
                    higher_lod_obj.hide_set(True)
                    higher_lod_obj.hide_render = True

            chunk_obj = get_or_create_chunk_object(lod_obj_name, chunk_empty)

            # Make the current LOD visible while generating
            chunk_obj.hide_set(False)
            chunk_obj.hide_render = False

            # Set up modifier with chunk-specific values (using shared node group)
            modifier = _setup_chunk_modifier(chunk_obj, node_tree, chunk_info)

            # If baking, evaluate and apply
            if mode == GenerationMode.BAKE:
                _bake_chunk_to_mesh(chunk_obj, modifier)

            chunks_processed += 1

    # Final progress
    yield GenerationProgress(
        lod_index=total_lods,
        lod_level=0,
        total_lods=total_lods,
        current_chunk=0,
        total_chunks=0,
        chunk_info=None,
        message="Complete",
        chunks_completed=total_chunks_all_lods,
        chunks_total=total_chunks_all_lods,
        lod_chunk_counts=lod_chunk_counts,
    )

    return GenerationResult(
        success=True,
        cancelled=False,
        message=f"Generated {chunks_processed} chunks",
        chunks_processed=chunks_processed,
    )
