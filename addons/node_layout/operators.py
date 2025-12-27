"""
Operators for the Node Layout addon.

Provides operators to automatically arrange nodes in various node editors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from .shared.node_layout import layout_nodes_pcb_style

if TYPE_CHECKING:
    from bpy.types import Context, Event


class NODE_OT_auto_layout(bpy.types.Operator):
    """Automatically arrange nodes in a clean PCB-style layout"""

    bl_idname = "node.auto_layout"
    bl_label = "Auto Layout Nodes"
    bl_description = "Arrange nodes in a clean PCB-style grid layout with organized connections"
    bl_options = {"REGISTER", "UNDO"}

    # Layout parameters as operator properties
    cell_width: bpy.props.FloatProperty(
        name="Cell Width",
        description="Width of each grid cell",
        default=200.0,
        min=50.0,
        max=1000.0,
    )  # type: ignore[valid-type]

    cell_height: bpy.props.FloatProperty(
        name="Cell Height",
        description="Height of each grid cell",
        default=200.0,
        min=50.0,
        max=1000.0,
    )  # type: ignore[valid-type]

    lane_width: bpy.props.FloatProperty(
        name="Lane Width",
        description="Width allocated per reroute lane",
        default=20.0,
        min=5.0,
        max=100.0,
    )  # type: ignore[valid-type]

    lane_gap: bpy.props.FloatProperty(
        name="Lane Gap",
        description="Gap before and after the lane area",
        default=50.0,
        min=0.0,
        max=200.0,
    )  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if operator can run."""
        # Must be in a node editor with an active node tree
        space = context.space_data
        return (
            context.area is not None
            and context.area.type == "NODE_EDITOR"
            and space is not None
            and hasattr(space, "edit_tree")
            and space.edit_tree is not None  # type: ignore[union-attr]
        )

    def execute(self, context: Context) -> set[str]:
        """Execute the layout operation."""
        space = context.space_data
        node_tree = space.edit_tree  # type: ignore[union-attr]

        if node_tree is None:
            self.report({"ERROR"}, "No active node tree")
            return {"CANCELLED"}

        if not node_tree.nodes:
            self.report({"WARNING"}, "Node tree is empty")
            return {"CANCELLED"}

        # Check if we should only layout selected nodes
        selected_nodes = [n for n in node_tree.nodes if n.select and n.type not in ("FRAME", "REROUTE")]
        selected_only = len(selected_nodes) > 1

        # Count nodes for reporting
        if selected_only:
            node_count = len(selected_nodes)
        else:
            node_count = len([n for n in node_tree.nodes if n.type not in ("REROUTE", "FRAME")])

        # Apply the layout
        layout_nodes_pcb_style(
            node_tree,
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            lane_width=self.lane_width,
            lane_gap=self.lane_gap,
            selected_only=selected_only,
        )

        if selected_only:
            self.report({"INFO"}, f"Arranged {node_count} selected nodes")
        else:
            self.report({"INFO"}, f"Arranged {node_count} nodes")
        return {"FINISHED"}

    def invoke(self, context: Context, event: Event) -> set[str]:  # noqa: ARG002
        """Show options dialog before executing."""
        # Run directly without dialog for quick access
        # Use F6 or operator panel to adjust settings after
        return self.execute(context)


# List of classes to register
classes: list[type] = [
    NODE_OT_auto_layout,
]
