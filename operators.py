"""
Operators module for Voxel Terrain.

Contains all operator classes for the addon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems
    from bpy.types import Context, Event


class VOXELTERRAIN_OT_generate(bpy.types.Operator):
    """Generate voxel terrain."""

    bl_idname = "voxel_terrain.generate"
    bl_label = "Generate Terrain"
    bl_description = "Generate a new voxel terrain"
    bl_options = {"REGISTER", "UNDO"}

    # Example property with type annotation
    message: bpy.props.StringProperty(
        name="Message",
        description="Status message",
        default="Terrain generated!",
    )  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if the operator can be executed."""
        # This operator can always be executed
        return True

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        self.report({"INFO"}, self.message)
        return {"FINISHED"}

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Invoke the operator - called when the user triggers the operator."""
        # You could show a dialog here with:
        # return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)


class VOXELTERRAIN_OT_clear(bpy.types.Operator):
    """Clear the current voxel terrain."""

    bl_idname = "voxel_terrain.clear"
    bl_label = "Clear Terrain"
    bl_description = "Clear the current voxel terrain"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only allow execution when an object is selected."""
        return context.active_object is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.active_object
        if obj:
            self.report({"INFO"}, f"Cleared terrain from {obj.name}")
        return {"FINISHED"}


# List of all classes to register - used by __init__.py
classes: tuple[type, ...] = (
    VOXELTERRAIN_OT_generate,
    VOXELTERRAIN_OT_clear,
)
