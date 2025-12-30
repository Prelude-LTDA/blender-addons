"""
Chunk calculation and visualization for Voxel Terrain.

This module handles:
- Calculating chunk bounds from terrain-enabled objects
- Drawing chunk wireframes in the viewport
- Highlighting the currently processing chunk during generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

if TYPE_CHECKING:
    from bpy.types import Scene

from .typing_utils import get_object_props, get_scene_props

# Draw handler reference (set during registration)
_draw_handler: object | None = None

# Current processing chunk state (set by generation operators)
# Contains: (chunk_min, chunk_max, skirt_min, skirt_max)
_current_processing_chunk: (
    tuple[
        tuple[float, float, float],  # chunk min bounds
        tuple[float, float, float],  # chunk max bounds
        tuple[float, float, float],  # skirt min bounds
        tuple[float, float, float],  # skirt max bounds
    ]
    | None
) = None


def set_current_processing_chunk(
    chunk_min: tuple[float, float, float],
    chunk_max: tuple[float, float, float],
    skirt_min: tuple[float, float, float],
    skirt_max: tuple[float, float, float],
) -> None:
    """Set the current chunk being processed (for overlay visualization)."""
    global _current_processing_chunk  # noqa: PLW0603
    _current_processing_chunk = (chunk_min, chunk_max, skirt_min, skirt_max)


def clear_current_processing_chunk() -> None:
    """Clear the current processing chunk highlight."""
    global _current_processing_chunk  # noqa: PLW0603
    _current_processing_chunk = None


@dataclass
class ChunkBounds:
    """Represents the min/max chunk indices for the terrain."""

    min_chunk: tuple[int, int, int]
    max_chunk: tuple[int, int, int]

    @property
    def chunk_count(self) -> tuple[int, int, int]:
        """Return the number of chunks in each dimension."""
        return (
            self.max_chunk[0] - self.min_chunk[0] + 1,
            self.max_chunk[1] - self.min_chunk[1] + 1,
            self.max_chunk[2] - self.min_chunk[2] + 1,
        )

    @property
    def total_chunks(self) -> int:
        """Return the total number of chunks."""
        count = self.chunk_count
        return count[0] * count[1] * count[2]


@dataclass
class WorldBounds:
    """Represents min/max world coordinates for arbitrary bounds."""

    min_pos: tuple[float, float, float]
    max_pos: tuple[float, float, float]


def get_terrain_objects(scene: Scene) -> list[bpy.types.Object]:
    """Get all terrain-enabled objects in the scene."""
    terrain_objects: list[bpy.types.Object] = []
    for obj in scene.objects:
        if obj.type == "MESH":
            props = get_object_props(obj)
            if props.enabled:
                terrain_objects.append(obj)
    return terrain_objects


def calculate_world_bounds(
    objects: list[bpy.types.Object],
) -> tuple[Vector, Vector] | None:
    """
    Calculate the combined world-space bounding box of all objects.

    Returns (min_corner, max_corner) or None if no objects.
    """
    if not objects:
        return None

    # Initialize with first object's bounds
    first_obj = objects[0]
    bbox_corners = [
        first_obj.matrix_world @ Vector(corner) for corner in first_obj.bound_box
    ]
    min_corner = Vector(
        (
            min(c.x for c in bbox_corners),
            min(c.y for c in bbox_corners),
            min(c.z for c in bbox_corners),
        )
    )
    max_corner = Vector(
        (
            max(c.x for c in bbox_corners),
            max(c.y for c in bbox_corners),
            max(c.z for c in bbox_corners),
        )
    )

    # Expand bounds with remaining objects
    for obj in objects[1:]:
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        for corner in bbox_corners:
            min_corner.x = min(min_corner.x, corner.x)
            min_corner.y = min(min_corner.y, corner.y)
            min_corner.z = min(min_corner.z, corner.z)
            max_corner.x = max(max_corner.x, corner.x)
            max_corner.y = max(max_corner.y, corner.y)
            max_corner.z = max(max_corner.z, corner.z)

    return min_corner, max_corner


def calculate_chunk_bounds(scene: Scene) -> ChunkBounds | None:
    """
    Calculate chunk bounds based on terrain-enabled objects.

    Returns ChunkBounds or None if no terrain objects exist.
    """
    terrain_objects = get_terrain_objects(scene)
    world_bounds = calculate_world_bounds(terrain_objects)

    if world_bounds is None:
        return None

    min_corner, max_corner = world_bounds
    props = get_scene_props(scene)
    chunk_size = props.chunk_size

    # Calculate chunk indices (floor for min, ceil for max to fully contain bounds)
    min_chunk = (
        math.floor(min_corner.x / chunk_size[0]),
        math.floor(min_corner.y / chunk_size[1]),
        math.floor(min_corner.z / chunk_size[2]),
    )
    max_chunk = (
        math.ceil(max_corner.x / chunk_size[0]) - 1,
        math.ceil(max_corner.y / chunk_size[1]) - 1,
        math.ceil(max_corner.z / chunk_size[2]) - 1,
    )

    # Ensure at least one chunk
    max_chunk = (
        max(max_chunk[0], min_chunk[0]),
        max(max_chunk[1], min_chunk[1]),
        max(max_chunk[2], min_chunk[2]),
    )

    return ChunkBounds(min_chunk=min_chunk, max_chunk=max_chunk)


def get_selection_world_bounds(context: bpy.types.Context) -> WorldBounds | None:
    """
    Get world-space bounds of selected objects, snapped to voxel grid.

    Returns WorldBounds or None if no objects are selected.
    """
    selected = context.selected_objects
    if not selected:
        return None

    world_bounds = calculate_world_bounds(list(selected))
    if world_bounds is None:
        return None

    min_corner, max_corner = world_bounds
    return WorldBounds(
        min_pos=(min_corner.x, min_corner.y, min_corner.z),
        max_pos=(max_corner.x, max_corner.y, max_corner.z),
    )


def generate_chunk_wireframe_vertices(
    chunk_bounds: ChunkBounds,
    chunk_size: tuple[float, float, float],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    """
    Generate vertices and edges for chunk wireframe visualization.

    Returns (vertices, edges) for use with GPU shader.
    """
    vertices: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []

    min_c = chunk_bounds.min_chunk
    max_c = chunk_bounds.max_chunk

    for cx in range(min_c[0], max_c[0] + 2):
        for cy in range(min_c[1], max_c[1] + 2):
            for cz in range(min_c[2], max_c[2] + 2):
                # World position of this chunk corner
                x = cx * chunk_size[0]
                y = cy * chunk_size[1]
                z = cz * chunk_size[2]
                vertices.append((x, y, z))

    # Generate edges along each axis
    count = chunk_bounds.chunk_count
    nx, ny, nz = count[0] + 1, count[1] + 1, count[2] + 1

    def idx(ix: int, iy: int, iz: int) -> int:
        return ix * ny * nz + iy * nz + iz

    # X-axis edges
    for ix in range(nx - 1):
        for iy in range(ny):
            for iz in range(nz):
                edges.append((idx(ix, iy, iz), idx(ix + 1, iy, iz)))

    # Y-axis edges
    for ix in range(nx):
        for iy in range(ny - 1):
            for iz in range(nz):
                edges.append((idx(ix, iy, iz), idx(ix, iy + 1, iz)))

    # Z-axis edges
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz - 1):
                edges.append((idx(ix, iy, iz), idx(ix, iy, iz + 1)))

    return vertices, edges


def generate_skirt_wireframe_vertices(
    chunk_bounds: ChunkBounds,
    chunk_size: tuple[float, float, float],
    voxel_size: float,
    skirt_voxels: int,
    corner_fraction: float = 0.15,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    """
    Generate vertices and edges for skirt wireframe visualization.

    The skirt is drawn outset from the chunk boundaries by skirt_voxels * voxel_size.
    Only corner markers are drawn (short lines along each axis from each corner).

    Args:
        corner_fraction: Length of corner markers as fraction of smallest chunk dimension.

    Returns (vertices, edges) for use with GPU shader.
    """
    vertices: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []

    min_c = chunk_bounds.min_chunk
    max_c = chunk_bounds.max_chunk

    # Skirt offset in world units (same in all directions)
    skirt_offset = skirt_voxels * voxel_size

    # Uniform corner marker size based on smallest chunk dimension
    min_chunk_dim = min(chunk_size[0], chunk_size[1], chunk_size[2])
    corner_size = min_chunk_dim * corner_fraction

    def add_edge(
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
    ) -> None:
        """Add an edge."""
        i = len(vertices)
        vertices.append(p1)
        vertices.append(p2)
        edges.append((i, i + 1))

    # Draw skirt corners for each chunk (outset from chunk boundaries)
    for cx in range(min_c[0], max_c[0] + 1):
        for cy in range(min_c[1], max_c[1] + 1):
            for cz in range(min_c[2], max_c[2] + 1):
                # Chunk corners in world space - outset by skirt amount
                x0 = cx * chunk_size[0] - skirt_offset
                y0 = cy * chunk_size[1] - skirt_offset
                z0 = cz * chunk_size[2] - skirt_offset
                x1 = (cx + 1) * chunk_size[0] + skirt_offset
                y1 = (cy + 1) * chunk_size[1] + skirt_offset
                z1 = (cz + 1) * chunk_size[2] + skirt_offset

                # 8 corners with their outward directions (toward box interior)
                # Each corner has 3 edges going in +/- X, Y, Z directions
                corner_data = [
                    # (corner_pos, x_dir, y_dir, z_dir)
                    ((x0, y0, z0), +1, +1, +1),  # 0: min corner
                    ((x1, y0, z0), -1, +1, +1),  # 1
                    ((x1, y1, z0), -1, -1, +1),  # 2
                    ((x0, y1, z0), +1, -1, +1),  # 3
                    ((x0, y0, z1), +1, +1, -1),  # 4
                    ((x1, y0, z1), -1, +1, -1),  # 5
                    ((x1, y1, z1), -1, -1, -1),  # 6: max corner
                    ((x0, y1, z1), +1, -1, -1),  # 7
                ]

                for corner, sx, sy, sz in corner_data:
                    cx0, cy0, cz0 = corner
                    # X-axis marker (uniform size)
                    add_edge(corner, (cx0 + sx * corner_size, cy0, cz0))
                    # Y-axis marker (uniform size)
                    add_edge(corner, (cx0, cy0 + sy * corner_size, cz0))
                    # Z-axis marker (uniform size)
                    add_edge(corner, (cx0, cy0, cz0 + sz * corner_size))

    return vertices, edges


def generate_voxel_grid_vertices(
    chunk_bounds: ChunkBounds,
    chunk_size: tuple[float, float, float],
    voxel_size: float,
    floor_z: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    """
    Generate vertices and edges for a 2D voxel grid on the floor.

    Draws lines spanning the full terrain extent in X and Y directions.

    Args:
        floor_z: Z position for the grid (e.g., from 3D cursor).

    Returns (vertices, edges) for use with GPU shader.
    """
    vertices: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []

    min_c = chunk_bounds.min_chunk
    max_c = chunk_bounds.max_chunk

    # World-space bounds of all chunks
    x_min = min_c[0] * chunk_size[0]
    y_min = min_c[1] * chunk_size[1]
    x_max = (max_c[0] + 1) * chunk_size[0]
    y_max = (max_c[1] + 1) * chunk_size[1]

    # Floor Z position from parameter
    z = floor_z

    # Calculate voxel grid line positions
    # Start from first voxel boundary at or before x_min
    x_start = math.floor(x_min / voxel_size) * voxel_size
    y_start = math.floor(y_min / voxel_size) * voxel_size

    # Lines parallel to Y axis (varying X)
    x = x_start
    while x <= x_max:
        i = len(vertices)
        vertices.append((x, y_min, z))
        vertices.append((x, y_max, z))
        edges.append((i, i + 1))
        x += voxel_size

    # Lines parallel to X axis (varying Y)
    y = y_start
    while y <= y_max:
        i = len(vertices)
        vertices.append((x_min, y, z))
        vertices.append((x_max, y, z))
        edges.append((i, i + 1))
        y += voxel_size

    return vertices, edges


def generate_world_bounds_grid_vertices(
    bounds: WorldBounds,
    voxel_size: float,
    floor_z: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    """
    Generate vertices and edges for a 2D voxel grid from arbitrary world bounds.

    Grid lines snap to nearest voxel boundaries that contain the selection.

    Args:
        bounds: World-space bounds to generate grid for.
        voxel_size: Size of each voxel.
        floor_z: Z position for the grid.

    Returns (vertices, edges) for use with GPU shader.
    """
    vertices: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []

    # Snap bounds to voxel grid (expand to fully contain selection)
    x_min = math.floor(bounds.min_pos[0] / voxel_size) * voxel_size
    y_min = math.floor(bounds.min_pos[1] / voxel_size) * voxel_size
    x_max = math.ceil(bounds.max_pos[0] / voxel_size) * voxel_size
    y_max = math.ceil(bounds.max_pos[1] / voxel_size) * voxel_size

    z = floor_z

    # Lines parallel to Y axis (varying X)
    x = x_min
    while x <= x_max:
        i = len(vertices)
        vertices.append((x, y_min, z))
        vertices.append((x, y_max, z))
        edges.append((i, i + 1))
        x += voxel_size

    # Lines parallel to X axis (varying Y)
    y = y_min
    while y <= y_max:
        i = len(vertices)
        vertices.append((x_min, y, z))
        vertices.append((x_max, y, z))
        edges.append((i, i + 1))
        y += voxel_size

    return vertices, edges


def _get_grid_z(
    props: object, scene: bpy.types.Scene, context: bpy.types.Context
) -> float:
    """Get the Z position for the voxel grid based on user settings."""
    z_source = getattr(props, "voxel_grid_z_source", "ORIGIN")
    if z_source == "CURSOR":
        return scene.cursor.location.z
    if z_source == "SELECTION":
        active = context.active_object
        return active.location.z if active else 0.0
    return 0.0


def _draw_chunk_wireframes(
    props: object,
    chunk_bounds: ChunkBounds,
    chunk_size: tuple[float, float, float],
    shader: gpu.types.GPUShader,
) -> None:
    """Draw chunk and skirt wireframes."""
    vertices, edges = generate_chunk_wireframe_vertices(chunk_bounds, chunk_size)
    if vertices and edges:
        # Teal color (R, G, B, A)
        color = (0.0, 0.8, 0.8, 0.8)
        batch = batch_for_shader(shader, "LINES", {"pos": vertices}, indices=edges)
        shader.uniform_float("color", color)
        batch.draw(shader)

    # Draw skirt if enabled
    enable_skirt = getattr(props, "enable_skirt", False)
    show_skirt = getattr(props, "show_skirt", False)
    skirt_size = getattr(props, "skirt_size", 0)
    voxel_size = getattr(props, "voxel_size", 1.0)

    if enable_skirt and show_skirt and skirt_size > 0:
        skirt_vertices, skirt_edges = generate_skirt_wireframe_vertices(
            chunk_bounds, chunk_size, voxel_size, skirt_size
        )
        if skirt_vertices and skirt_edges:
            skirt_color = (0.0, 0.6, 0.6, 0.6)
            skirt_batch = batch_for_shader(
                shader, "LINES", {"pos": skirt_vertices}, indices=skirt_edges
            )
            shader.uniform_float("color", skirt_color)
            skirt_batch.draw(shader)


def _draw_voxel_grids(
    props: object,
    context: bpy.types.Context,
    scene: bpy.types.Scene,
    chunk_bounds: ChunkBounds | None,
    chunk_size: tuple[float, float, float],
    shader: gpu.types.GPUShader,
) -> None:
    """Draw voxel grid overlays."""
    # Calculate voxel size for selected LOD level
    enable_lod = getattr(props, "enable_lod", False)
    view_lod = getattr(props, "view_lod", 0)
    lod_levels = getattr(props, "lod_levels", 1)
    voxel_size = getattr(props, "voxel_size", 1.0)
    lod_factor = getattr(props, "lod_factor", 2.0)

    lod_level = min(view_lod, lod_levels - 1) if enable_lod else 0
    grid_voxel_size = voxel_size * (lod_factor**lod_level)
    grid_z = _get_grid_z(props, scene, context)

    show_chunks_grid = getattr(props, "voxel_grid_bounds_chunks", True)
    show_selection_grid = getattr(props, "voxel_grid_bounds_selection", False)
    chunk_grid_color = (0.0, 0.8, 0.8, 0.3)  # Teal (transparent)

    # Get selection color from theme (default to orange if unavailable)
    selection_grid_color = (1.0, 0.6, 0.0, 0.3)  # Default orange
    if context.preferences and context.preferences.themes:
        theme = context.preferences.themes[0]
        sel_color = theme.view_3d.object_selected
        selection_grid_color = (sel_color.r, sel_color.g, sel_color.b, 0.3)

    # Draw chunk-based grid if enabled
    if show_chunks_grid and chunk_bounds is not None:
        grid_vertices, grid_edges = generate_voxel_grid_vertices(
            chunk_bounds, chunk_size, grid_voxel_size, floor_z=grid_z
        )
        if grid_vertices and grid_edges:
            grid_batch = batch_for_shader(
                shader, "LINES", {"pos": grid_vertices}, indices=grid_edges
            )
            shader.uniform_float("color", chunk_grid_color)
            shader.uniform_float("lineWidth", 1.0)
            grid_batch.draw(shader)

    # Draw selection-based grid if enabled
    if show_selection_grid:
        sel_bounds = get_selection_world_bounds(context)
        if sel_bounds is not None:
            sel_grid_vertices, sel_grid_edges = generate_world_bounds_grid_vertices(
                sel_bounds, grid_voxel_size, floor_z=grid_z
            )
            if sel_grid_vertices and sel_grid_edges:
                sel_grid_batch = batch_for_shader(
                    shader, "LINES", {"pos": sel_grid_vertices}, indices=sel_grid_edges
                )
                shader.uniform_float("color", selection_grid_color)
                shader.uniform_float("lineWidth", 2.0)
                sel_grid_batch.draw(shader)


def _draw_processing_chunk_highlight(
    shader: gpu.types.GPUShader,
) -> None:
    """Draw a highlight around the currently processing chunk."""
    if _current_processing_chunk is None:
        return

    chunk_min, chunk_max, skirt_min, skirt_max = _current_processing_chunk

    # Draw the chunk bounds as a solid wireframe box
    x0, y0, z0 = chunk_min
    x1, y1, z1 = chunk_max

    # 8 corners of the chunk box
    chunk_vertices: list[tuple[float, float, float]] = [
        (x0, y0, z0),  # 0
        (x1, y0, z0),  # 1
        (x1, y1, z0),  # 2
        (x0, y1, z0),  # 3
        (x0, y0, z1),  # 4
        (x1, y0, z1),  # 5
        (x1, y1, z1),  # 6
        (x0, y1, z1),  # 7
    ]

    # 12 edges of the chunk box
    chunk_edges: list[tuple[int, int]] = [
        # Bottom face
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # Top face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # Vertical edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Draw chunk box with bright yellow/gold color
    color = (1.0, 0.85, 0.0, 1.0)  # Gold color
    chunk_batch = batch_for_shader(
        shader, "LINES", {"pos": chunk_vertices}, indices=chunk_edges
    )
    shader.uniform_float("color", color)
    shader.uniform_float("lineWidth", 3.0)  # Thicker line for visibility
    chunk_batch.draw(shader)

    # Draw skirt as corner markers (similar to regular chunk skirt visualization)
    sx0, sy0, sz0 = skirt_min
    sx1, sy1, sz1 = skirt_max

    # Calculate corner marker size based on chunk dimensions
    chunk_size_x = x1 - x0
    chunk_size_y = y1 - y0
    chunk_size_z = z1 - z0
    min_dim = min(chunk_size_x, chunk_size_y, chunk_size_z)
    corner_size = min_dim * 0.15  # 15% of smallest dimension

    skirt_vertices: list[tuple[float, float, float]] = []
    skirt_edges: list[tuple[int, int]] = []

    def add_skirt_edge(
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
    ) -> None:
        """Add an edge to the skirt."""
        i = len(skirt_vertices)
        skirt_vertices.append(p1)
        skirt_vertices.append(p2)
        skirt_edges.append((i, i + 1))

    # 8 corners of skirt box with their inward directions
    corner_data = [
        ((sx0, sy0, sz0), +1, +1, +1),  # 0: min corner
        ((sx1, sy0, sz0), -1, +1, +1),  # 1
        ((sx1, sy1, sz0), -1, -1, +1),  # 2
        ((sx0, sy1, sz0), +1, -1, +1),  # 3
        ((sx0, sy0, sz1), +1, +1, -1),  # 4
        ((sx1, sy0, sz1), -1, +1, -1),  # 5
        ((sx1, sy1, sz1), -1, -1, -1),  # 6: max corner
        ((sx0, sy1, sz1), +1, -1, -1),  # 7
    ]

    for corner, dx, dy, dz in corner_data:
        cx, cy, cz = corner
        # X-axis marker
        add_skirt_edge(corner, (cx + dx * corner_size, cy, cz))
        # Y-axis marker
        add_skirt_edge(corner, (cx, cy + dy * corner_size, cz))
        # Z-axis marker
        add_skirt_edge(corner, (cx, cy, cz + dz * corner_size))

    # Draw skirt corners with same color but thinner
    skirt_batch = batch_for_shader(
        shader, "LINES", {"pos": skirt_vertices}, indices=skirt_edges
    )
    shader.uniform_float("color", color)
    shader.uniform_float("lineWidth", 2.0)
    skirt_batch.draw(shader)


def draw_chunks() -> None:
    """Draw chunk wireframes in the viewport."""
    context = bpy.context
    scene = context.scene

    if scene is None:
        return

    props = get_scene_props(scene)

    # Check if anything needs to be drawn
    has_processing_chunk = _current_processing_chunk is not None
    if not props.show_chunks and not props.show_voxel_grid and not has_processing_chunk:
        return

    chunk_bounds = calculate_chunk_bounds(scene)
    chunk_size = (props.chunk_size[0], props.chunk_size[1], props.chunk_size[2])

    # For chunk visualization, we need chunk_bounds
    # For voxel grid in SELECTION mode, we can draw without chunk_bounds
    show_selection_grid = getattr(props, "voxel_grid_bounds_selection", False)
    can_draw_selection_grid = props.show_voxel_grid and show_selection_grid

    if (
        chunk_bounds is None
        and not can_draw_selection_grid
        and not has_processing_chunk
    ):
        return

    # Get viewport dimensions for the shader
    region = context.region
    viewport_size = (region.width, region.height) if region else (1920, 1080)

    # Enable depth testing so overlays are occluded by objects
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(False)  # Don't write to depth buffer
    gpu.state.blend_set("ALPHA")

    # Use POLYLINE shader for smooth anti-aliased lines
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    shader.bind()
    shader.uniform_float("lineWidth", 1.0)
    shader.uniform_float("viewportSize", viewport_size)

    # Draw chunk wireframes if enabled (requires chunk_bounds)
    if props.show_chunks and chunk_bounds is not None:
        _draw_chunk_wireframes(props, chunk_bounds, chunk_size, shader)

    # Draw voxel grid on floor if enabled
    if props.show_voxel_grid:
        _draw_voxel_grids(props, context, scene, chunk_bounds, chunk_size, shader)

    # Restore state for regular drawing
    gpu.state.blend_set("NONE")
    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set("NONE")

    # Draw processing chunk highlight AFTER restoring state,
    # with depth test disabled so it's always visible through objects
    if _current_processing_chunk is not None:
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")  # Always visible
        _draw_processing_chunk_highlight(shader)
        gpu.state.blend_set("NONE")


def register_draw_handler() -> None:
    """Register the viewport draw handler."""
    global _draw_handler  # noqa: PLW0603
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_chunks, (), "WINDOW", "POST_VIEW"
        )


def unregister_draw_handler() -> None:
    """Unregister the viewport draw handler."""
    global _draw_handler  # noqa: PLW0603
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
