"""
Properties module for Voxel Terrain.

Contains PropertyGroups for storing addon data on Blender objects.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import NodeTree, PropertyGroup


def _clamp_view_lod(self: VoxelTerrainSceneProperties, _context: bpy.types.Context) -> None:
    """Clamp view_lod to valid range when lod_levels changes."""
    max_lod = self.lod_levels - 1
    self.view_lod = min(self.view_lod, max_lod)


class VoxelTerrainSceneProperties(PropertyGroup):
    """Properties stored on each Scene for Voxel Terrain."""

    chunk_size: FloatVectorProperty(
        name="Chunk Size",
        description="Size of terrain chunks (X, Y, Z)",
        size=3,
        default=(128.0, 128.0, 1024.0),
        min=1.0,
        soft_max=4096.0,
        subtype="XYZ",
        unit="LENGTH",
    )  # type: ignore[valid-type]

    voxel_size: FloatProperty(
        name="Voxel Size",
        description="Size of each voxel in scene units",
        default=1.0,
        min=0.001,
        soft_max=100.0,
        unit="LENGTH",
    )  # type: ignore[valid-type]

    lod_factor: FloatProperty(
        name="LOD Factor",
        description="Scale factor between LOD levels",
        default=2.0,
        min=1.1,
        soft_max=4.0,
    )  # type: ignore[valid-type]

    lod_levels: IntProperty(
        name="LOD Levels",
        description="Number of level-of-detail levels to generate",
        default=4,
        min=1,
        soft_max=8,
        update=_clamp_view_lod,
    )  # type: ignore[valid-type]

    enable_lod: BoolProperty(
        name="LOD",
        description="Enable level-of-detail generation",
        default=True,
    )  # type: ignore[valid-type]

    view_lod: IntProperty(
        name="View LOD",
        description="LOD level to display in viewport (0 = highest detail)",
        default=0,
        min=0,
        soft_max=7,
    )  # type: ignore[valid-type]

    show_chunks: BoolProperty(
        name="Chunks",
        description="Display chunk boundaries in the viewport",
        default=False,
    )  # type: ignore[valid-type]

    show_skirt: BoolProperty(
        name="Skirt",
        description="Display chunk skirt boundaries in the viewport",
        default=False,
    )  # type: ignore[valid-type]

    show_voxel_grid: BoolProperty(
        name="Voxel Grid",
        description="Display voxel grid on the floor",
        default=False,
    )  # type: ignore[valid-type]

    voxel_grid_z_source: bpy.props.EnumProperty(
        name="Grid Z Source",
        description="Where to get the Z position for the voxel grid",
        items=[
            ("ORIGIN", "Origin", "Draw grid at Z=0"),
            ("CURSOR", "Cursor", "Draw grid at 3D cursor Z position"),
            ("SELECTION", "Selection", "Draw grid at selected object's origin Z"),
        ],
        default="ORIGIN",
    )  # type: ignore[valid-type]

    voxel_grid_bounds_chunks: BoolProperty(
        name="Chunks",
        description="Show voxel grid based on chunk boundaries",
        default=True,
    )  # type: ignore[valid-type]

    voxel_grid_bounds_selection: BoolProperty(
        name="Selection",
        description="Show voxel grid based on selected object bounds",
        default=False,
    )  # type: ignore[valid-type]

    enable_skirt: BoolProperty(
        name="Skirt",
        description="Enable skirt (overlap region) for chunks",
        default=True,
    )  # type: ignore[valid-type]

    skirt_size: IntProperty(
        name="Skirt Size",
        description="Skirt size in voxels (overlap region between chunks)",
        default=8,
        min=0,
        soft_max=64,
    )  # type: ignore[valid-type]

    export_path: StringProperty(
        name="Export Path",
        description="Folder to export terrain data to",
        default="//terrain_export/",
        subtype="DIR_PATH",
    )  # type: ignore[valid-type]


# Modifier name used to identify our managed modifier
VOXEL_TERRAIN_MODIFIER_NAME = "Voxel Terrain"


def _find_voxel_terrain_modifier(obj: bpy.types.Object) -> bpy.types.Modifier | None:
    """Find the Voxel Terrain modifier by name."""
    modifier = obj.modifiers.get(VOXEL_TERRAIN_MODIFIER_NAME)
    if modifier is not None and modifier.type == "NODES":
        return modifier
    return None


def _on_node_group_update(self: object, context: bpy.types.Context) -> None:
    """Update or remove the VoxelTerrain modifier when node_group changes."""
    obj = context.object
    if obj is None:
        return

    node_group = getattr(self, "node_group", None)
    modifier = _find_voxel_terrain_modifier(obj)

    if node_group is None:
        # Remove the modifier if it exists
        if modifier is not None:
            obj.modifiers.remove(modifier)
    else:
        # Create or update the modifier
        if modifier is None:
            modifier = obj.modifiers.new(name=VOXEL_TERRAIN_MODIFIER_NAME, type="NODES")
            modifier.show_viewport = False
            modifier.show_render = False
            modifier.show_expanded = False  # Collapse by default
        modifier.node_group = node_group  # type: ignore[union-attr]


class VoxelTerrainObjectProperties(PropertyGroup):
    """Properties stored on each Object for Voxel Terrain."""

    enabled: BoolProperty(
        name="Enable Voxel Terrain",
        description="Enable this object as a Voxel Terrain source",
        default=False,
    )  # type: ignore[valid-type]

    node_group: PointerProperty(
        name="Node Group",
        description="Geometry Nodes group defining the voxel terrain",
        type=NodeTree,
        update=_on_node_group_update,
    )  # type: ignore[valid-type]


# List of all classes to register
classes: tuple[type, ...] = (
    VoxelTerrainSceneProperties,
    VoxelTerrainObjectProperties,
)


def register_scene_properties() -> None:
    """Register the addon properties on Scene and Object types."""
    bpy.types.Scene.voxel_terrain = bpy.props.PointerProperty(  # type: ignore[attr-defined]
        type=VoxelTerrainSceneProperties
    )
    bpy.types.Object.voxel_terrain = bpy.props.PointerProperty(  # type: ignore[attr-defined]
        type=VoxelTerrainObjectProperties
    )


def unregister_scene_properties() -> None:
    """Unregister the addon properties from Scene and Object types."""
    if hasattr(bpy.types.Object, "voxel_terrain"):
        del bpy.types.Object.voxel_terrain  # type: ignore[attr-defined]
    if hasattr(bpy.types.Scene, "voxel_terrain"):
        del bpy.types.Scene.voxel_terrain  # type: ignore[attr-defined]
