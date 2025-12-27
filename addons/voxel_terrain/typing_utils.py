"""
Type definitions for the Voxel Terrain addon.

This module provides type aliases and protocols for type checking
dynamically added Blender properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from bpy.types import Object as BpyObject, Scene as BpyScene

    from .properties import VoxelTerrainObjectProperties, VoxelTerrainSceneProperties


class VoxelTerrainScene(Protocol):
    """Protocol for Scene with voxel_terrain property."""

    voxel_terrain: VoxelTerrainSceneProperties


class VoxelTerrainObject(Protocol):
    """Protocol for Object with voxel_terrain property."""

    voxel_terrain: VoxelTerrainObjectProperties


def get_scene_props(scene: BpyScene) -> VoxelTerrainSceneProperties:
    """
    Get the voxel terrain properties from a scene.

    This helper provides proper typing for the dynamically added property.
    """
    return scene.voxel_terrain  # type: ignore[attr-defined, return-value]


def get_object_props(obj: BpyObject) -> VoxelTerrainObjectProperties:
    """
    Get the voxel terrain properties from an object.

    This helper provides proper typing for the dynamically added property.
    """
    return obj.voxel_terrain  # type: ignore[attr-defined, return-value]
