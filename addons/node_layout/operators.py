"""
Operators for the Node Layout addon.

Provides operators to automatically arrange nodes in various node editors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from .shared.node_layout import layout_nodes_pcb_style

if TYPE_CHECKING:
    from bpy.types import Context, Event, Node, NodeTree


# =============================================================================
# Status Bar with Icons
# =============================================================================

# Global state for the modal status bar display
_status_state: dict[str, str | bool] = {
    "active": False,
    "vertical_align": "CENTER",
    "sorting_method": "combined",
    "use_gravity": False,
    "snap_to_grid": False,
    "cell_width": "200",
    "cell_height": "200",
    "lane_width": "20",
    "grid_size": "20",
}


def _draw_modal_status_bar(self: bpy.types.Header, _context: bpy.types.Context) -> None:
    """Draw the modal status bar with proper icons."""
    layout = self.layout
    if layout is None or not _status_state["active"]:
        return

    # Main row - don't set ui_units_x on the main row to avoid stretching
    row = layout.row(align=True)

    # ── Confirm/Cancel ──
    sub = row.row(align=True)
    sub.label(text="", icon="MOUSE_LMB")
    sub.label(text="Confirm")

    sub = row.row(align=True)
    sub.label(text="", icon="MOUSE_RMB")
    sub.label(text="Cancel")

    # ── Shift+Tab: Snap Toggle, Ctrl: Snap Invert (like native grab) ──
    sub = row.row(align=True)
    sub.label(text="", icon="EVENT_CTRL")
    sub.label(text=f"Snap Invert ({_status_state['grid_size']})")
    sub.label(text="", icon="EVENT_SHIFT")
    sub.label(text="", icon="EVENT_TAB")
    sub.label(text="Snap Toggle")


    # ── A: Vertical Alignment ──
    align_label = {"TOP": "Top", "CENTER": "Center", "BOTTOM": "Bottom"}.get(
        str(_status_state["vertical_align"]), "Center"
    )
    sub = row.row(align=True)
    sub.label(text="", icon="EVENT_A")
    sub.label(text=f"Align ({align_label})")

    # ── C: Column Assignment ──
    method_label = {"combined": "In+Out", "output": "Output", "input": "Input"}.get(
        str(_status_state["sorting_method"]), "In+Out"
    )
    sub = row.row(align=True)
    sub.label(text="", icon="EVENT_C")
    sub.label(text=f"Column ({method_label})")

    # ── G: Gravity ──
    gravity_on = _status_state["use_gravity"]
    sub = row.row(align=True)
    sub.label(text="", icon="EVENT_G")
    sub.label(text=f"Gravity ({'On' if gravity_on else 'Off'})")

    # ── Arrows: Width/Height ──
    sub = row.row(align=True)
    sub.label(text="", icon="EVENT_LEFT_ARROW")
    sub.label(text="", icon="EVENT_RIGHT_ARROW")
    sub.label(text=f"Width ({_status_state['cell_width']})")

    sub = row.row(align=True)
    sub.label(text="", icon="EVENT_UP_ARROW")
    sub.label(text="", icon="EVENT_DOWN_ARROW")
    sub.label(text=f"Height ({_status_state['cell_height']})")

    # ── Bracket keys: Lane Width ──
    sub = row.row(align=True)
    sub.label(text="", icon="EVENT_LEFTBRACKET")
    sub.label(text="", icon="EVENT_RIGHTBRACKET")
    sub.label(text=f"Lane ({_status_state['lane_width']})")


def _enable_status_bar(context: bpy.types.Context) -> None:
    """Enable the custom modal status bar."""
    if not _status_state["active"]:
        _status_state["active"] = True
        bpy.types.STATUSBAR_HT_header.prepend(_draw_modal_status_bar)
        # Set empty status text to hide default mouse hints
        if context.workspace is not None:
            context.workspace.status_text_set(text=" ")


def _disable_status_bar(context: bpy.types.Context) -> None:
    """Disable the custom modal status bar."""
    if _status_state["active"]:
        _status_state["active"] = False
        bpy.types.STATUSBAR_HT_header.remove(_draw_modal_status_bar)
        # Clear the status text override
        if context.workspace is not None:
            context.workspace.status_text_set(text=None)


def _update_status_state(
    vertical_align: str,
    sorting_method: str,
    use_gravity: bool,
    snap_to_grid: bool,
    cell_width: float,
    cell_height: float,
    lane_width: float,
    grid_size: float,
) -> None:
    """Update the status bar state with current values."""
    _status_state["vertical_align"] = vertical_align
    _status_state["sorting_method"] = sorting_method
    _status_state["use_gravity"] = use_gravity
    _status_state["snap_to_grid"] = snap_to_grid
    _status_state["cell_width"] = f"{cell_width:.0f}"
    _status_state["cell_height"] = f"{cell_height:.0f}"
    _status_state["lane_width"] = f"{lane_width:.0f}"
    _status_state["grid_size"] = f"{grid_size:.0f}"


# =============================================================================
# Node Traversal Helpers
# =============================================================================


def _get_upstream_nodes(node_tree: NodeTree, start_nodes: set[Node]) -> set[Node]:
    """
    Get all nodes that flow into the given start nodes (upstream/input direction).

    This recursively finds all nodes connected via inputs, similar to
    Blender's "Select Linked From" operation.
    """
    result: set[Node] = set(start_nodes)
    to_visit: list[Node] = list(start_nodes)

    # Build a map of node -> nodes that feed into it (via inputs)
    # For efficiency, we build this once
    input_sources: dict[Node, set[Node]] = {}
    for link in node_tree.links:
        if not link.is_valid:
            continue
        to_node = link.to_node
        from_node = link.from_node
        if to_node is None or from_node is None:
            continue
        if to_node not in input_sources:
            input_sources[to_node] = set()
        input_sources[to_node].add(from_node)

    # BFS to find all upstream nodes
    while to_visit:
        current = to_visit.pop()
        for upstream_node in input_sources.get(current, []):
            if upstream_node not in result:
                result.add(upstream_node)
                to_visit.append(upstream_node)

    return result


def _get_downstream_nodes(node_tree: NodeTree, start_nodes: set[Node]) -> set[Node]:
    """
    Get all nodes that the given start nodes flow into (downstream/output direction).

    This recursively finds all nodes connected via outputs, similar to
    Blender's "Select Linked To" operation. Stops traversal when hitting
    a node with 2+ connected inputs (to avoid affecting sibling branches).
    """
    result: set[Node] = set(start_nodes)
    to_visit: list[Node] = list(start_nodes)

    # Build a map of node -> nodes it feeds into (via outputs)
    # For efficiency, we build this once
    output_targets: dict[Node, set[Node]] = {}
    for link in node_tree.links:
        if not link.is_valid:
            continue
        from_node = link.from_node
        to_node = link.to_node
        if from_node is None or to_node is None:
            continue
        if from_node not in output_targets:
            output_targets[from_node] = set()
        output_targets[from_node].add(to_node)

    # Count connected inputs per node (not just input sockets, but actual links)
    connected_inputs: dict[Node, int] = {}
    for link in node_tree.links:
        if not link.is_valid:
            continue
        to_node = link.to_node
        if to_node is None:
            continue
        connected_inputs[to_node] = connected_inputs.get(to_node, 0) + 1

    # BFS to find all downstream nodes
    # Stop when we hit a node with 2+ connected inputs (merge point) - don't include it
    while to_visit:
        current = to_visit.pop()
        for downstream_node in output_targets.get(current, []):
            if downstream_node not in result:
                # Only include and continue if this node has fewer than 2 connected inputs
                if connected_inputs.get(downstream_node, 0) < 2:
                    result.add(downstream_node)
                    to_visit.append(downstream_node)

    return result


def _has_internal_connections(node_tree: NodeTree, nodes: set[Node]) -> bool:
    """
    Check if there are any connections between nodes in the given set.

    Returns True if at least one link connects two nodes both in the set.
    """
    for link in node_tree.links:
        if not link.is_valid:
            continue
        if link.from_node in nodes and link.to_node in nodes:
            return True
    return False


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

    # =========================
    # Move (set during modal)
    # =========================

    move_offset: bpy.props.FloatVectorProperty(
        name="Move",
        description="Offset applied after layout",
        default=(0.0, 0.0),
        size=2,
        subtype="TRANSLATION",
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

    def execute(self, context: Context) -> set[str]:  # type: ignore[override]
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
        # Track original selection for anchoring when we expand
        nodes_to_layout: set[bpy.types.Node] | None = None
        anchor_nodes: set[bpy.types.Node] | None = None
        expansion_type: str | None = None  # "upstream", "downstream", or None
        original_count = len(selected_nodes)

        if len(selected_nodes) > 1:
            nodes_to_layout = set(selected_nodes)
            # Check if selected nodes have connections between them
            # If not, expand selection to include all upstream nodes
            if not _has_internal_connections(node_tree, nodes_to_layout):
                # Remember original selection for anchoring
                anchor_nodes = set(selected_nodes)
                nodes_to_layout = _get_upstream_nodes(node_tree, nodes_to_layout)
                # Filter out frames and reroutes from the result
                nodes_to_layout = {n for n in nodes_to_layout if n.type not in ("FRAME", "REROUTE")}
                # If no upstream nodes found (only the original nodes), try downstream instead
                if nodes_to_layout == anchor_nodes:
                    nodes_to_layout = _get_downstream_nodes(node_tree, anchor_nodes)
                    nodes_to_layout = {n for n in nodes_to_layout if n.type not in ("FRAME", "REROUTE")}
                    if nodes_to_layout != anchor_nodes:
                        expansion_type = "downstream"
                else:
                    expansion_type = "upstream"
        elif len(selected_nodes) == 1:
            # Single node selected: get all upstream nodes
            # Anchor to the originally selected node
            anchor_nodes = set(selected_nodes)
            nodes_to_layout = _get_upstream_nodes(node_tree, anchor_nodes)
            # Filter out frames and reroutes from the result
            nodes_to_layout = {n for n in nodes_to_layout if n.type not in ("FRAME", "REROUTE")}
            # If no upstream nodes found (only the original node), try downstream instead
            if nodes_to_layout == anchor_nodes:
                nodes_to_layout = _get_downstream_nodes(node_tree, anchor_nodes)
                nodes_to_layout = {n for n in nodes_to_layout if n.type not in ("FRAME", "REROUTE")}
                if nodes_to_layout != anchor_nodes:
                    expansion_type = "downstream"
            else:
                expansion_type = "upstream"

        # Count nodes for reporting
        if nodes_to_layout is not None:
            node_count = len(nodes_to_layout)
        else:
            node_count = len([n for n in node_tree.nodes if n.type not in ("REROUTE", "FRAME")])

        # Apply the layout
        created_reroutes = layout_nodes_pcb_style(
            node_tree,
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            lane_width=self.lane_width,
            lane_gap=self.lane_gap,
            nodes_to_layout=nodes_to_layout,
            anchor_nodes=anchor_nodes,
            sorting_method=self.sorting_method,
            use_gravity=self.use_gravity,
            vertical_align=self.vertical_align,
            collapse_vertical=self.collapse_vertical,
            collapse_horizontal=self.collapse_horizontal,
            collapse_adjacent=self.collapse_adjacent,
            snap_to_grid=self.snap_to_grid,
            grid_size=self.grid_size,
        )

        # If we expanded the selection (upstream nodes or frame contents),
        # select all laid out nodes so user can easily move them
        if nodes_to_layout is not None:
            reroute_set = set(created_reroutes)
            for node in node_tree.nodes:
                node.select = node in nodes_to_layout or node in reroute_set
        else:
            # Whole graph layout - deselect all
            for node in node_tree.nodes:
                node.select = False

        # Collect all affected nodes
        all_affected_nodes = list(nodes_to_layout) + created_reroutes if nodes_to_layout else []

        # Apply move offset (set during modal, or via F9 redo)
        if all_affected_nodes and (self.move_offset[0] != 0 or self.move_offset[1] != 0):
            for node in all_affected_nodes:
                node.location.x += self.move_offset[0]
                node.location.y += self.move_offset[1]

        # Build report message
        if nodes_to_layout is not None:
            if expansion_type == "upstream":
                extra_count = node_count - original_count
                report_msg = f"Arranged {original_count} selected + {extra_count} upstream nodes"
            elif expansion_type == "downstream":
                extra_count = node_count - original_count
                report_msg = f"Arranged {original_count} selected + {extra_count} downstream nodes"
            else:
                report_msg = f"Arranged {node_count} selected nodes"
        else:
            report_msg = f"Arranged {node_count} nodes"

        # Store state for modal grab mode
        self._enter_grab_mode = nodes_to_layout is not None
        self._grab_nodes = all_affected_nodes
        self._anchor_nodes = anchor_nodes
        self._nodes_to_layout = nodes_to_layout
        self._report_msg = report_msg

        # Only report immediately if not entering grab mode
        # (grab mode will report on confirm)

        return {"FINISHED"}

    def invoke(self, context: Context, event: Event) -> set[str]:  # type: ignore[override]
        """Execute layout and optionally enter grab mode for subsets."""
        # Reset move offset for fresh invocation
        self.move_offset = (0.0, 0.0)

        result = self.execute(context)

        if result != {"FINISHED"}:
            return result

        # Enter modal grab mode if we laid out a subset (not the whole graph)
        # Similar to Shift+D behavior
        if self._enter_grab_mode and self._grab_nodes:
            # Don't report yet - will report on confirm
            pass
        else:
            # Report immediately for non-modal execution
            self.report({"INFO"}, self._report_msg)

        if self._enter_grab_mode and self._grab_nodes:
            self._initial_mouse_x = event.mouse_region_x
            self._initial_mouse_y = event.mouse_region_y
            self._current_offset = (0.0, 0.0)  # Track cumulative offset for incremental movement

            # Set move cursor
            if context.window is not None:
                context.window.cursor_modal_set("SCROLL_XY")

            # Enable custom status bar with icons
            _enable_status_bar(context)
            self._update_status_text(context)

            if context.window_manager is not None:
                context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}

        return {"FINISHED"}

    def _update_status_text(self, context: Context) -> None:
        """Update status bar with current settings and available keys."""
        _update_status_state(
            vertical_align=self.vertical_align,
            sorting_method=self.sorting_method,
            use_gravity=self.use_gravity,
            snap_to_grid=self.snap_to_grid,
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            lane_width=self.lane_width,
            grid_size=self.grid_size,
        )
        # Trigger redraw of status bar
        if context.screen is not None:
            for area in context.screen.areas:
                if area.type == "STATUSBAR":
                    area.tag_redraw()
                    break

    def modal(self, context: Context, event: Event) -> set[str]:  # type: ignore[override]
        """Handle modal grab mode."""
        if event.type == "MOUSEMOVE":
            # Convert mouse positions to view space (accounts for zoom/pan)
            region = context.region
            if region is not None:
                view2d = region.view2d
                # Convert initial and current mouse positions to view coordinates
                init_vx, init_vy = view2d.region_to_view(self._initial_mouse_x, self._initial_mouse_y)
                curr_vx, curr_vy = view2d.region_to_view(event.mouse_region_x, event.mouse_region_y)
                node_dx = curr_vx - init_vx
                node_dy = curr_vy - init_vy

                # Account for pixel size (HiDPI/Retina displays)
                pixel_size = context.preferences.system.pixel_size if context.preferences is not None else 1.0
                node_dx /= pixel_size
                node_dy /= pixel_size
            else:
                dx = event.mouse_region_x - self._initial_mouse_x
                dy = event.mouse_region_y - self._initial_mouse_y
                node_dx = dx
                node_dy = -dy

            # Apply snap to grid during grab (Ctrl inverts snap behavior)
            snap_active = self.snap_to_grid != event.ctrl  # XOR: ctrl inverts
            if snap_active:
                grid = self.grid_size
                node_dx = round(node_dx / grid) * grid
                node_dy = round(node_dy / grid) * grid

            # Calculate incremental delta from last frame
            prev_offset = getattr(self, "_current_offset", (0.0, 0.0))
            inc_dx = node_dx - prev_offset[0]
            inc_dy = node_dy - prev_offset[1]

            # Store current offset for F9 redo and next frame
            self._current_offset = (node_dx, node_dy)

            # Apply incremental offset to all grabbed nodes
            if inc_dx != 0 or inc_dy != 0:
                for node in self._grab_nodes:
                    node.location.x += inc_dx
                    node.location.y += inc_dy

            if context.area:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        elif event.type in {"LEFTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            # Confirm placement - store offset in property for F9 redo
            if hasattr(self, "_current_offset"):
                self.move_offset = self._current_offset
            # Restore cursor and disable status bar
            if context.window is not None:
                context.window.cursor_modal_restore()
            _disable_status_bar(context)
            # Report success now
            if hasattr(self, "_report_msg"):
                self.report({"INFO"}, self._report_msg)
            return {"FINISHED"}

        elif event.type in {"RIGHTMOUSE", "ESC"} and event.value == "PRESS":
            # Cancel - undo the movement by applying negative of current offset
            current_offset = getattr(self, "_current_offset", (0.0, 0.0))
            if current_offset[0] != 0 or current_offset[1] != 0:
                for node in self._grab_nodes:
                    node.location.x -= current_offset[0]
                    node.location.y -= current_offset[1]
            if context.area:
                context.area.tag_redraw()
            # Restore cursor and disable status bar
            if context.window is not None:
                context.window.cursor_modal_restore()
            _disable_status_bar(context)
            return {"CANCELLED"}

        # Shift+Tab: Toggle snap to grid (like native grab)
        elif event.type == "TAB" and event.value == "PRESS" and event.shift:
            self.snap_to_grid = not self.snap_to_grid
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}

        # A: Cycle vertical alignment (Top -> Center -> Bottom -> Top)
        elif event.type == "A" and event.value == "PRESS":
            alignments = ["TOP", "CENTER", "BOTTOM"]
            current_idx = alignments.index(self.vertical_align) if self.vertical_align in alignments else 0
            self.vertical_align = alignments[(current_idx + 1) % 3]
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}

        # G: Toggle gravity
        elif event.type == "G" and event.value == "PRESS":
            self.use_gravity = not self.use_gravity
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}

        # C: Cycle column assignment method
        elif event.type == "C" and event.value == "PRESS":
            methods = ["combined", "output", "input"]
            current_idx = methods.index(self.sorting_method) if self.sorting_method in methods else 0
            self.sorting_method = methods[(current_idx + 1) % 3]
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}

        # Up/Down arrows: Adjust cell height
        elif event.type == "UP_ARROW" and event.value == "PRESS":
            self.cell_height += 20.0 if not event.shift else 5.0
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}
        elif event.type == "DOWN_ARROW" and event.value == "PRESS":
            self.cell_height = max(50.0, self.cell_height - (20.0 if not event.shift else 5.0))
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}

        # Left/Right arrows: Adjust cell width
        elif event.type == "RIGHT_ARROW" and event.value == "PRESS":
            self.cell_width += 20.0 if not event.shift else 5.0
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}
        elif event.type == "LEFT_ARROW" and event.value == "PRESS":
            self.cell_width = max(50.0, self.cell_width - (20.0 if not event.shift else 5.0))
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}

        # [ and ]: Adjust lane width
        elif event.type == "LEFT_BRACKET" and event.value == "PRESS":
            self.lane_width = max(5.0, self.lane_width - (5.0 if not event.shift else 1.0))
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}
        elif event.type == "RIGHT_BRACKET" and event.value == "PRESS":
            self.lane_width += 5.0 if not event.shift else 1.0
            self._re_layout_and_grab(context, event)
            self._update_status_text(context)
            return {"RUNNING_MODAL"}

        return {"RUNNING_MODAL"}

    def _re_layout_and_grab(self, context: Context, event: Event) -> None:
        """Re-run layout with current settings and update grab positions."""
        space = context.space_data
        node_tree = space.edit_tree  # type: ignore[union-attr]

        # Get the nodes we're working with (excluding reroutes we created)
        nodes_to_layout = getattr(self, "_nodes_to_layout", None)

        # Re-run layout
        created_reroutes = layout_nodes_pcb_style(
            node_tree,
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            lane_width=self.lane_width,
            lane_gap=self.lane_gap,
            nodes_to_layout=nodes_to_layout,
            anchor_nodes=getattr(self, "_anchor_nodes", None),
            sorting_method=self.sorting_method,
            use_gravity=self.use_gravity,
            vertical_align=self.vertical_align,
            collapse_vertical=self.collapse_vertical,
            collapse_horizontal=self.collapse_horizontal,
            collapse_adjacent=self.collapse_adjacent,
            snap_to_grid=self.snap_to_grid,
            grid_size=self.grid_size,
        )

        # Update grab nodes
        if nodes_to_layout:
            self._grab_nodes = list(nodes_to_layout) + created_reroutes
        else:
            self._grab_nodes = created_reroutes

        # Reset mouse tracking to current mouse position
        # (so movement continues smoothly from new layout)
        self._initial_mouse_x = event.mouse_region_x
        self._initial_mouse_y = event.mouse_region_y
        self._current_offset = (0.0, 0.0)

        if context.area:
            context.area.tag_redraw()


def _draw_context_menu(self: bpy.types.Menu, context: Context) -> None:  # noqa: ARG001
    """Draw the Auto Layout option in the node editor context menu."""
    layout = self.layout  # type: ignore[union-attr]
    layout.separator()  # type: ignore[union-attr]
    layout.operator(NODE_OT_auto_layout.bl_idname, icon="SNAP_GRID")  # type: ignore[union-attr]


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
