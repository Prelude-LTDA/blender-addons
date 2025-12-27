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

    # =========================
    # Column Assignment
    # =========================

    sorting_method: bpy.props.EnumProperty(
        name="Column Assignment",
        description="How to determine column positions",
        items=[
            ("combined", "Input & Output Distance", "Balance between input and output distance"),
            ("output", "Output Distance", "Prioritize distance from outputs"),
            ("input", "Input Distance", "Prioritize distance from inputs"),
        ],
        default="combined",
    )  # type: ignore[valid-type]

    use_gravity: bpy.props.BoolProperty(
        name="Gravity",
        description="Pull nodes closer together when gaps are large",
        default=False,
    )  # type: ignore[valid-type]

    vertical_align: bpy.props.EnumProperty(
        name="Vertical Alignment",
        description="How to align nodes vertically within each column",
        items=[
            ("CENTER", "Center", "Center nodes vertically (default)"),
            ("TOP", "Top", "Align nodes to the top of the grid"),
            ("BOTTOM", "Bottom", "Align nodes to the bottom of the grid"),
        ],
        default="CENTER",
    )  # type: ignore[valid-type]

    # =========================
    # Spacing
    # =========================

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

    # =========================
    # Reroute Optimization
    # =========================

    collapse_vertical: bpy.props.BoolProperty(
        name="Collapse Vertical",
        description="Collapse vertical runs of reroutes into a single reroute",
        default=True,
    )  # type: ignore[valid-type]

    collapse_horizontal: bpy.props.BoolProperty(
        name="Collapse Horizontal",
        description="Collapse horizontal runs of 3+ reroutes, keeping only the first and last",
        default=True,
    )  # type: ignore[valid-type]

    collapse_adjacent: bpy.props.BoolProperty(
        name="Collapse Adjacent",
        description="Remove single reroutes between adjacent columns",
        default=True,
    )  # type: ignore[valid-type]

    # =========================
    # Snapping
    # =========================

    snap_to_grid: bpy.props.BoolProperty(
        name="Snap to Grid",
        description="Snap final node positions to the editor grid",
        default=False,
    )  # type: ignore[valid-type]

    grid_size: bpy.props.FloatProperty(
        name="Grid Size",
        description="Size of the grid to snap to (Blender's default is 20)",
        default=20.0,
        min=1.0,
        max=100.0,
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

        # Determine which nodes to layout
        selected_nodes = [n for n in node_tree.nodes if n.select and n.type not in ("FRAME", "REROUTE")]
        
        # If only frames are selected, layout their contents
        selected_frames = [n for n in node_tree.nodes if n.select and n.type == "FRAME"]
        if not selected_nodes and selected_frames:
            # Get all children of selected frames (including nested frames)
            frame_children: set[bpy.types.Node] = set()
            for frame in selected_frames:
                for node in node_tree.nodes:
                    if node.type in ("FRAME", "REROUTE"):
                        continue
                    # Check if node is inside this frame (direct or nested)
                    parent = node.parent
                    while parent is not None:
                        if parent == frame:
                            frame_children.add(node)
                            break
                        parent = parent.parent
            selected_nodes = list(frame_children)
        
        # Only use subset if we have more than 1 node to layout
        nodes_to_layout: set[bpy.types.Node] | None = None
        if len(selected_nodes) > 1:
            nodes_to_layout = set(selected_nodes)

        # Count nodes for reporting
        if nodes_to_layout is not None:
            node_count = len(nodes_to_layout)
        else:
            node_count = len([n for n in node_tree.nodes if n.type not in ("REROUTE", "FRAME")])

        # Apply the layout
        layout_nodes_pcb_style(
            node_tree,
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            lane_width=self.lane_width,
            lane_gap=self.lane_gap,
            nodes_to_layout=nodes_to_layout,
            sorting_method=self.sorting_method,
            use_gravity=self.use_gravity,
            vertical_align=self.vertical_align,
            collapse_vertical=self.collapse_vertical,
            collapse_horizontal=self.collapse_horizontal,
            collapse_adjacent=self.collapse_adjacent,
            snap_to_grid=self.snap_to_grid,
            grid_size=self.grid_size,
        )

        if nodes_to_layout is not None:
            self.report({"INFO"}, f"Arranged {node_count} selected nodes")
        else:
            self.report({"INFO"}, f"Arranged {node_count} nodes")
        return {"FINISHED"}

    def invoke(self, context: Context, event: Event) -> set[str]:  # noqa: ARG002
        """Show options dialog before executing."""
        # Run directly without dialog for quick access
        # Press F9 or use View > Adjust Last Operation to tweak settings after
        return self.execute(context)


def _draw_context_menu(self: bpy.types.Menu, context: Context) -> None:  # noqa: ARG001
    """Draw the Auto Layout option in the node editor context menu."""
    layout = self.layout
    layout.separator()
    layout.operator(NODE_OT_auto_layout.bl_idname, icon="SNAP_GRID")


# List of classes to register
classes: list[type] = [
    NODE_OT_auto_layout,
]


def register_menus() -> None:
    """Register context menu entries."""
    bpy.types.NODE_MT_context_menu.append(_draw_context_menu)


def unregister_menus() -> None:
    """Unregister context menu entries."""
    bpy.types.NODE_MT_context_menu.remove(_draw_context_menu)
