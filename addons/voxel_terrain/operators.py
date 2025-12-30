"""
Operators module for Voxel Terrain.

Contains all operator classes for the addon.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import bpy

from .chunks import clear_current_processing_chunk, set_current_processing_chunk
from .generation import (
    GenerationMode,
    GenerationProgress,
    GenerationResult,
    generation_iterator,
)
from .properties import VOXEL_TERRAIN_MODIFIER_NAME, _find_voxel_terrain_modifier
from .sockets import check_sockets_match, sync_sockets
from .typing_utils import get_object_props, get_scene_props

if TYPE_CHECKING:
    from collections.abc import Generator

    from bpy.stub_internal.rna_enums import OperatorReturnItems
    from bpy.types import Context, Event


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "--:--"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


# Global progress state for the status bar
_progress_state: dict[str, float | str | bool] = {
    "factor": 0.0,
    "text": "",
    "active": False,
    "operation": "",  # "generate", "bake", or "delete"
    "elapsed": "",  # Elapsed time string
    "eta": "",  # ETA string
}


def _draw_progress_bar(self: bpy.types.Header, _context: bpy.types.Context) -> None:
    """Draw the progress bar in the status bar."""
    layout = self.layout
    if layout is None:
        return

    if _progress_state["active"]:
        row = layout.row(align=True)

        # Icon based on operation type
        op_type = str(_progress_state["operation"])
        if op_type == "generate":
            icon = "MESH_GRID"
        elif op_type == "bake":
            icon = "RENDER_STILL"
        else:  # delete
            icon = "TRASH"

        # Text label (LOD / Chunk info) - fixed width to prevent jumping
        label_row = row.row()
        label_row.ui_units_x = 14  # Fixed width for label
        label_row.label(text=str(_progress_state["text"]), icon=icon)

        # Progress bar with percentage inside
        factor = float(_progress_state["factor"])
        percent_text = f"{factor * 100:.0f}%"
        sub = row.row(align=True)
        sub.progress(
            factor=factor,
            type="BAR",
            text=percent_text,
        )
        sub.scale_x = 1.5

        # Time info: Elapsed | ETA
        elapsed = str(_progress_state["elapsed"])
        eta = str(_progress_state["eta"])
        if elapsed or eta:
            time_row = row.row()
            time_row.ui_units_x = 8  # Fixed width for time
            if elapsed and eta:
                time_row.label(text=f"{elapsed} | ETA: {eta}")
            elif elapsed:
                time_row.label(text=elapsed)

        # ESC hint
        row.label(text="(ESC to cancel)")


def _enable_progress_bar(operation: str) -> None:
    """Enable the progress bar in the status bar."""
    if not _progress_state["active"]:
        _progress_state["active"] = True
        _progress_state["operation"] = operation
        _progress_state["elapsed"] = ""
        _progress_state["eta"] = ""
        # Prepend our draw function to the status bar header
        bpy.types.STATUSBAR_HT_header.prepend(_draw_progress_bar)


def _disable_progress_bar() -> None:
    """Disable the progress bar in the status bar."""
    if _progress_state["active"]:
        _progress_state["active"] = False
        _progress_state["factor"] = 0.0
        _progress_state["text"] = ""
        _progress_state["operation"] = ""
        _progress_state["elapsed"] = ""
        _progress_state["eta"] = ""
        # Remove our draw function
        bpy.types.STATUSBAR_HT_header.remove(_draw_progress_bar)


def _update_progress_bar(
    factor: float, text: str, elapsed: str = "", eta: str = ""
) -> None:
    """Update the progress bar state."""
    _progress_state["factor"] = factor
    _progress_state["text"] = text
    _progress_state["elapsed"] = elapsed
    _progress_state["eta"] = eta


class VOXELTERRAIN_OT_generate(bpy.types.Operator):
    """Generate voxel terrain with editable geometry nodes setup."""

    bl_idname = "voxel_terrain.generate"
    bl_label = "Generate Terrain"
    bl_description = "Generate voxel terrain chunks with geometry nodes setup"
    bl_options = {"REGISTER", "UNDO"}

    # Internal state
    _timer: bpy.types.Timer | None = None
    _iterator: Generator[GenerationProgress, None, GenerationResult] | None = None
    _cancelled: bool = False
    _progress: GenerationProgress | None = None
    _start_time: float = 0.0
    _last_chunk_time: float = 0.0
    # Per-LOD timing: {lod_level: [chunk_times]}
    _lod_chunk_times: dict[int, list[float]] = {}
    _current_lod: int = -1
    _smoothed_eta: float = -1.0  # Smoothed ETA value

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if the operator can be executed."""
        from .generation import get_voxel_terrain_objects

        return len(get_voxel_terrain_objects()) > 0

    def cancel_check(self) -> bool:
        """Check if the operation was cancelled."""
        return self._cancelled

    def _update_viewport_overlay(self) -> None:
        """Update the processing chunk highlight in the viewport."""
        if self._progress is None:
            return
        chunk_info = self._progress.chunk_info
        if chunk_info is not None:
            set_current_processing_chunk(
                chunk_info.chunk_min,
                chunk_info.chunk_max,
                chunk_info.skirt_min,
                chunk_info.skirt_max,
            )
        else:
            clear_current_processing_chunk()

    def _calculate_observed_lod_ratio(
        self, lod_avg_times: dict[int, float]
    ) -> float | None:
        """
        Calculate the observed ratio between consecutive LOD levels.

        Returns the average ratio of time(LOD N-1) / time(LOD N) for all
        consecutive LOD pairs we have data for. Returns None if no
        consecutive pairs are available yet.
        """
        if len(lod_avg_times) < 2:
            return None  # Need at least 2 LODs to calculate ratio

        # Sort LOD levels (higher number = lower detail = faster)
        sorted_lods = sorted(lod_avg_times.keys(), reverse=True)

        ratios: list[float] = []
        for i in range(len(sorted_lods) - 1):
            higher_lod = sorted_lods[i]  # Lower detail, faster
            lower_lod = sorted_lods[i + 1]  # Higher detail, slower

            # Only consider consecutive LOD levels
            if higher_lod - lower_lod == 1:
                higher_time = lod_avg_times[higher_lod]
                lower_time = lod_avg_times[lower_lod]
                if higher_time > 0:
                    ratio = lower_time / higher_time
                    ratios.append(ratio)

        if ratios:
            return sum(ratios) / len(ratios)
        return None

    def _estimate_lod_time(
        self,
        lod_level: int,
        chunks_remaining: int,
        lod_avg_times: dict[int, float],
        observed_ratio: float,
    ) -> float:
        """Estimate time for remaining chunks in a specific LOD."""
        if lod_level in lod_avg_times:
            return lod_avg_times[lod_level] * chunks_remaining
        # Extrapolate from known LODs
        known_lods = sorted(lod_avg_times.keys())
        if not known_lods:
            return 0.0
        closest_lod = min(known_lods)
        closest_time = lod_avg_times[closest_lod]
        lod_diff = closest_lod - lod_level
        scale = observed_ratio**lod_diff
        return closest_time * scale * chunks_remaining

    def _calculate_lod_aware_eta(self) -> float:
        """
        Calculate ETA using per-LOD timing data.

        For LODs we've already processed, we know exact avg time per chunk.
        For upcoming LODs, we extrapolate using observed ratios between LOD levels.
        Returns -1 if we're still on the first LOD (no observed ratios yet).
        """
        if self._progress is None or not self._progress.lod_chunk_counts:
            return -1.0

        lod_chunk_counts = self._progress.lod_chunk_counts
        current_lod_index = self._progress.lod_index
        current_chunk_in_lod = self._progress.current_chunk

        # Calculate average time per chunk for each LOD we have data for
        lod_avg_times: dict[int, float] = {}
        for lod_level, times in self._lod_chunk_times.items():
            if times:
                lod_avg_times[lod_level] = sum(times) / len(times)

        if not lod_avg_times:
            return -1.0  # No data yet

        # Calculate observed ratio between LOD levels
        observed_ratio = self._calculate_observed_lod_ratio(lod_avg_times)

        # Don't show ETA until we have observed ratio (completed at least one LOD)
        if observed_ratio is None:
            return -1.0

        # Estimate time for remaining work
        total_remaining_time = 0.0

        for idx, (lod_level, chunk_count) in enumerate(lod_chunk_counts):
            if idx < current_lod_index:
                continue
            if idx == current_lod_index:
                chunks_remaining = chunk_count - current_chunk_in_lod - 1
            else:
                chunks_remaining = chunk_count
            if chunks_remaining > 0:
                total_remaining_time += self._estimate_lod_time(
                    lod_level, chunks_remaining, lod_avg_times, observed_ratio
                )

        return total_remaining_time

    def _smooth_eta(self, raw_eta: float) -> float:
        """Apply exponential smoothing to ETA to reduce fluctuations."""
        if raw_eta < 0:
            return -1.0

        if self._smoothed_eta < 0:
            # First valid ETA, use it directly
            self._smoothed_eta = raw_eta
        else:
            # Exponential moving average with alpha=0.01 (slow smoothing)
            alpha = 0.01
            self._smoothed_eta = alpha * raw_eta + (1 - alpha) * self._smoothed_eta

        return self._smoothed_eta

    def modal(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Handle modal events."""
        if event.type == "ESC":
            self._cancelled = True

        if event.type == "TIMER":
            if self._iterator is None:
                self.finish(context)
                return {"FINISHED"}

            try:
                # Get next progress update
                self._progress = next(self._iterator)
                self._update_viewport_overlay()

                # Track chunk processing time per LOD
                now = time.time()
                lod_level = self._progress.lod_level

                # Initialize timing list for new LOD
                if lod_level not in self._lod_chunk_times:
                    self._lod_chunk_times[lod_level] = []

                # Record chunk time (only if we have a previous timestamp for same LOD)
                if self._last_chunk_time > 0 and self._current_lod == lod_level:
                    chunk_time = now - self._last_chunk_time
                    self._lod_chunk_times[lod_level].append(chunk_time)

                self._last_chunk_time = now
                self._current_lod = lod_level

                # Calculate elapsed time and smoothed LOD-aware ETA
                elapsed = now - self._start_time
                raw_eta = self._calculate_lod_aware_eta()
                eta = self._smooth_eta(raw_eta)

                _update_progress_bar(
                    self._progress.progress,
                    f"Generating {self._progress.message}",
                    _format_time(elapsed),
                    _format_time(eta),
                )

                # Tag for redraw (all areas for status bar)
                if context.screen:
                    for area in context.screen.areas:
                        area.tag_redraw()

            except StopIteration as e:
                # Generation complete
                result: GenerationResult = e.value
                self.finish(context)

                if result.cancelled:
                    self.report({"WARNING"}, result.message)
                    return {"CANCELLED"}
                if result.success:
                    self.report({"INFO"}, result.message)
                    return {"FINISHED"}
                self.report({"ERROR"}, result.message)
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def finish(self, context: Context) -> None:
        """Clean up after generation."""
        wm = context.window_manager
        if self._timer is not None and wm is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        _disable_progress_bar()
        clear_current_processing_chunk()
        self._iterator = None
        self._progress = None
        self._start_time = 0.0
        self._last_chunk_time = 0.0
        self._lod_chunk_times = {}
        self._current_lod = -1
        self._smoothed_eta = -1.0

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        wm = context.window_manager
        assert wm is not None
        assert context.window is not None
        assert context.workspace is not None

        # Start the generation iterator
        self._cancelled = False
        self._start_time = time.time()
        self._iterator = generation_iterator(
            mode=GenerationMode.GENERATE,
            cancel_check=self.cancel_check,
        )

        # Set up timer for modal
        self._timer = wm.event_timer_add(
            0.01,  # 10ms interval for responsiveness
            window=context.window,
        )

        wm.modal_handler_add(self)
        _enable_progress_bar("generate")
        _update_progress_bar(0.0, "Starting...", "00:00", "--:--")

        return {"RUNNING_MODAL"}

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Invoke the operator."""
        return self.execute(context)


class VOXELTERRAIN_OT_bake(bpy.types.Operator):
    """Bake voxel terrain to static meshes."""

    bl_idname = "voxel_terrain.bake"
    bl_label = "Bake Terrain"
    bl_description = "Bake voxel terrain chunks to static meshes"
    bl_options = {"REGISTER", "UNDO"}

    # Internal state
    _timer: bpy.types.Timer | None = None
    _iterator: Generator[GenerationProgress, None, GenerationResult] | None = None
    _cancelled: bool = False
    _progress: GenerationProgress | None = None
    _start_time: float = 0.0
    _last_chunk_time: float = 0.0
    _lod_chunk_times: dict[int, list[float]] = {}
    _current_lod: int = -1
    _smoothed_eta: float = -1.0

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if the operator can be executed."""
        from .generation import get_voxel_terrain_objects

        return len(get_voxel_terrain_objects()) > 0

    def cancel_check(self) -> bool:
        """Check if the operation was cancelled."""
        return self._cancelled

    def _update_viewport_overlay(self) -> None:
        """Update the processing chunk highlight in the viewport."""
        if self._progress is None:
            return
        chunk_info = self._progress.chunk_info
        if chunk_info is not None:
            set_current_processing_chunk(
                chunk_info.chunk_min,
                chunk_info.chunk_max,
                chunk_info.skirt_min,
                chunk_info.skirt_max,
            )
        else:
            clear_current_processing_chunk()

    def _calculate_observed_lod_ratio(
        self, lod_avg_times: dict[int, float]
    ) -> float | None:
        """
        Calculate the observed ratio between consecutive LOD levels.

        Returns the average ratio of time(LOD N-1) / time(LOD N) for all
        consecutive LOD pairs we have data for. Returns None if no
        consecutive pairs are available yet.
        """
        if len(lod_avg_times) < 2:
            return None  # Need at least 2 LODs to calculate ratio

        # Sort LOD levels (higher number = lower detail = faster)
        sorted_lods = sorted(lod_avg_times.keys(), reverse=True)

        ratios: list[float] = []
        for i in range(len(sorted_lods) - 1):
            higher_lod = sorted_lods[i]  # Lower detail, faster
            lower_lod = sorted_lods[i + 1]  # Higher detail, slower

            # Only consider consecutive LOD levels
            if higher_lod - lower_lod == 1:
                higher_time = lod_avg_times[higher_lod]
                lower_time = lod_avg_times[lower_lod]
                if higher_time > 0:
                    ratio = lower_time / higher_time
                    ratios.append(ratio)

        if ratios:
            return sum(ratios) / len(ratios)
        return None

    def _estimate_lod_time(
        self,
        lod_level: int,
        chunks_remaining: int,
        lod_avg_times: dict[int, float],
        observed_ratio: float,
    ) -> float:
        """Estimate time for remaining chunks in a specific LOD."""
        if lod_level in lod_avg_times:
            return lod_avg_times[lod_level] * chunks_remaining
        # Extrapolate from known LODs
        known_lods = sorted(lod_avg_times.keys())
        if not known_lods:
            return 0.0
        closest_lod = min(known_lods)
        closest_time = lod_avg_times[closest_lod]
        lod_diff = closest_lod - lod_level
        scale = observed_ratio**lod_diff
        return closest_time * scale * chunks_remaining

    def _calculate_lod_aware_eta(self) -> float:
        """
        Calculate ETA using per-LOD timing data.

        For LODs we've already processed, we know exact avg time per chunk.
        For upcoming LODs, we extrapolate using observed ratios between LOD levels.
        Returns -1 if we're still on the first LOD (no observed ratios yet).
        """
        if self._progress is None or not self._progress.lod_chunk_counts:
            return -1.0

        lod_chunk_counts = self._progress.lod_chunk_counts
        current_lod_index = self._progress.lod_index
        current_chunk_in_lod = self._progress.current_chunk

        # Calculate average time per chunk for each LOD we have data for
        lod_avg_times: dict[int, float] = {}
        for lod_level, times in self._lod_chunk_times.items():
            if times:
                lod_avg_times[lod_level] = sum(times) / len(times)

        if not lod_avg_times:
            return -1.0  # No data yet

        # Calculate observed ratio between LOD levels
        observed_ratio = self._calculate_observed_lod_ratio(lod_avg_times)

        # Don't show ETA until we have observed ratio (completed at least one LOD)
        if observed_ratio is None:
            return -1.0

        # Estimate time for remaining work
        total_remaining_time = 0.0

        for idx, (lod_level, chunk_count) in enumerate(lod_chunk_counts):
            if idx < current_lod_index:
                continue
            if idx == current_lod_index:
                chunks_remaining = chunk_count - current_chunk_in_lod - 1
            else:
                chunks_remaining = chunk_count
            if chunks_remaining > 0:
                total_remaining_time += self._estimate_lod_time(
                    lod_level, chunks_remaining, lod_avg_times, observed_ratio
                )

        return total_remaining_time

    def _smooth_eta(self, raw_eta: float) -> float:
        """Apply exponential smoothing to ETA to reduce fluctuations."""
        if raw_eta < 0:
            return -1.0

        if self._smoothed_eta < 0:
            # First valid ETA, use it directly
            self._smoothed_eta = raw_eta
        else:
            # Exponential moving average with alpha=0.1 (slow smoothing)
            alpha = 0.1
            self._smoothed_eta = alpha * raw_eta + (1 - alpha) * self._smoothed_eta

        return self._smoothed_eta

    def modal(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Handle modal events."""
        if event.type == "ESC":
            self._cancelled = True

        if event.type == "TIMER":
            if self._iterator is None:
                self.finish(context)
                return {"FINISHED"}

            try:
                # Get next progress update
                self._progress = next(self._iterator)
                self._update_viewport_overlay()

                # Track chunk processing time per LOD
                now = time.time()
                lod_level = self._progress.lod_level

                if lod_level not in self._lod_chunk_times:
                    self._lod_chunk_times[lod_level] = []

                if self._last_chunk_time > 0 and self._current_lod == lod_level:
                    chunk_time = now - self._last_chunk_time
                    self._lod_chunk_times[lod_level].append(chunk_time)

                self._last_chunk_time = now
                self._current_lod = lod_level

                # Calculate elapsed time and smoothed LOD-aware ETA
                elapsed = now - self._start_time
                raw_eta = self._calculate_lod_aware_eta()
                eta = self._smooth_eta(raw_eta)

                _update_progress_bar(
                    self._progress.progress,
                    f"Baking {self._progress.message}",
                    _format_time(elapsed),
                    _format_time(eta),
                )

                # Tag for redraw (all areas for status bar)
                if context.screen:
                    for area in context.screen.areas:
                        area.tag_redraw()

            except StopIteration as e:
                # Baking complete
                result: GenerationResult = e.value
                self.finish(context)

                if result.cancelled:
                    self.report({"WARNING"}, result.message)
                    return {"CANCELLED"}
                if result.success:
                    self.report({"INFO"}, result.message)
                    return {"FINISHED"}
                self.report({"ERROR"}, result.message)
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def finish(self, context: Context) -> None:
        """Clean up after baking."""
        wm = context.window_manager
        if self._timer is not None and wm is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        _disable_progress_bar()
        clear_current_processing_chunk()
        self._iterator = None
        self._lod_chunk_times = {}
        self._current_lod = -1
        self._smoothed_eta = -1.0
        self._progress = None
        self._start_time = 0.0
        self._last_chunk_time = 0.0
        self._lod_chunk_times = {}
        self._current_lod = -1

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        wm = context.window_manager
        assert wm is not None
        assert context.window is not None
        assert context.workspace is not None

        # Start the generation iterator in bake mode
        self._cancelled = False
        self._start_time = time.time()
        self._iterator = generation_iterator(
            mode=GenerationMode.BAKE,
            cancel_check=self.cancel_check,
        )

        # Set up timer for modal
        self._timer = wm.event_timer_add(
            0.01,  # 10ms interval for responsiveness
            window=context.window,
        )

        wm.modal_handler_add(self)
        _enable_progress_bar("bake")
        _update_progress_bar(0.0, "Starting...", "00:00", "--:--")

        return {"RUNNING_MODAL"}

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Invoke the operator."""
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


class VOXELTERRAIN_OT_export(bpy.types.Operator):
    """Export terrain data to the specified folder (quick export)."""

    bl_idname = "voxel_terrain.export"
    bl_label = "Export Terrain"
    bl_description = "Export terrain data to the configured folder"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if export is possible."""
        if not context.scene:
            return False
        props = get_scene_props(context.scene)
        # Require a valid export path
        return bool(props.export_path.strip())

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the export."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        # Resolve the path (handles // for relative paths)
        export_path = bpy.path.abspath(props.export_path)
        path = Path(export_path)

        return self._do_export(context, path)

    def _do_export(self, context: Context, path: Path) -> set[OperatorReturnItems]:
        """Shared export logic."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.report({"ERROR"}, f"Failed to create export directory: {e}")
            return {"CANCELLED"}

        # Get settings
        chunk_size = props.chunk_size
        voxel_size = props.voxel_size
        lod_factor = props.lod_factor
        lod_levels = props.lod_levels

        # TODO: Implement actual export logic
        self.report(
            {"INFO"},
            f"Exported to {path} | Chunk: "
            f"({chunk_size[0]:.0f}, {chunk_size[1]:.0f}, {chunk_size[2]:.0f}) | "
            f"Voxel: {voxel_size:.3g} | LOD: {lod_levels}x (factor {lod_factor})",
        )

        return {"FINISHED"}


class VOXELTERRAIN_OT_export_dialog(bpy.types.Operator):
    """Export terrain data via file browser dialog."""

    bl_idname = "voxel_terrain.export_dialog"
    bl_label = "Export Voxel Terrain"
    bl_description = "Export terrain data to a selected folder"
    bl_options = {"REGISTER"}

    # File browser properties
    directory: bpy.props.StringProperty(
        name="Directory",
        description="Directory to export to",
        subtype="DIR_PATH",
    )  # type: ignore[valid-type]

    filter_folder: bpy.props.BoolProperty(
        default=True,
        options={"HIDDEN"},
    )  # type: ignore[valid-type]

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if export is possible."""
        return context.scene is not None

    def invoke(self, context: Context, event: Event) -> set[OperatorReturnItems]:
        """Open file browser dialog."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        # Pre-fill with current export path if set
        if props.export_path.strip():
            self.directory = bpy.path.abspath(props.export_path)

        wm = context.window_manager
        assert wm is not None
        wm.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the export after folder selection."""
        assert context.scene is not None
        props = get_scene_props(context.scene)
        path = Path(self.directory)

        # Optionally update the scene property with the selected path
        # (commented out - uncomment if you want selection to persist)
        # props.export_path = self.directory

        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.report({"ERROR"}, f"Failed to create export directory: {e}")
            return {"CANCELLED"}

        # Get settings
        chunk_size = props.chunk_size
        voxel_size = props.voxel_size
        lod_factor = props.lod_factor
        lod_levels = props.lod_levels

        # TODO: Implement actual export logic
        self.report(
            {"INFO"},
            f"Exported to {path} | Chunk: "
            f"({chunk_size[0]:.0f}, {chunk_size[1]:.0f}, {chunk_size[2]:.0f}) | "
            f"Voxel: {voxel_size:.3g} | LOD: {lod_levels}x (factor {lod_factor})",
        )

        return {"FINISHED"}


def menu_func_export(self: bpy.types.Menu, context: Context) -> None:  # noqa: ARG001
    """Add export option to File > Export menu."""
    layout = self.layout
    assert layout is not None
    layout.operator(VOXELTERRAIN_OT_export_dialog.bl_idname, text="Voxel Terrain")


class VOXELTERRAIN_OT_set_view_lod(bpy.types.Operator):
    """Set view LOD level."""

    bl_idname = "voxel_terrain.set_view_lod"
    bl_label = "Set LOD Level"
    bl_description = "Set the LOD level to view"
    bl_options = {"INTERNAL"}

    level: bpy.props.IntProperty(
        name="Level",
        description="LOD level to set",
        default=0,
        min=0,
    )  # type: ignore[valid-type]

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        assert context.scene is not None
        props = get_scene_props(context.scene)
        max_lod = props.lod_levels - 1
        props.view_lod = min(self.level, max_lod)

        # Force viewport redraw
        for area in context.screen.areas if context.screen else []:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        return {"FINISHED"}


class VOXELTERRAIN_OT_toggle_grid_bounds(bpy.types.Operator):
    """Toggle grid bounds mode (Chunks/Selection)."""

    bl_idname = "voxel_terrain.toggle_grid_bounds"
    bl_label = "Toggle Grid Bounds"
    bl_description = "Toggle which grid bounds to display.\n\nShift: Switch exclusively"
    bl_options = {"INTERNAL"}

    mode: bpy.props.StringProperty(
        name="Mode",
        description="Which mode to toggle ('chunks' or 'selection')",
        default="chunks",
    )  # type: ignore[valid-type]

    def invoke(
        self, context: Context, event: bpy.types.Event
    ) -> set[OperatorReturnItems]:
        """Handle the operator invocation with modifier key detection."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        is_chunks = self.mode == "chunks"
        chunks_on = props.voxel_grid_bounds_chunks
        selection_on = props.voxel_grid_bounds_selection
        grid_visible = props.show_voxel_grid
        this_on = chunks_on if is_chunks else selection_on
        other_on = selection_on if is_chunks else chunks_on

        # Shift+Click = exclusive mode (switch to only this one)
        if event.shift:
            props.show_voxel_grid = True
            props.voxel_grid_bounds_chunks = is_chunks
            props.voxel_grid_bounds_selection = not is_chunks
        elif not grid_visible:
            # Grid is off - enable grid and this mode (keep other as-is)
            props.show_voxel_grid = True
            self._set_mode(props, is_chunks, True)
        elif this_on and not other_on:
            # Only this one is on - disable grid entirely
            props.show_voxel_grid = False
        else:
            # Toggle this mode
            self._set_mode(props, is_chunks, not this_on)

        # Force viewport redraw
        for area in context.screen.areas if context.screen else []:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        return {"FINISHED"}

    def _set_mode(self, props: object, is_chunks: bool, value: bool) -> None:
        """Set the appropriate mode property."""
        if is_chunks:
            props.voxel_grid_bounds_chunks = value  # type: ignore[attr-defined]
        else:
            props.voxel_grid_bounds_selection = value  # type: ignore[attr-defined]

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute fallback (no event available)."""
        assert context.scene is not None
        props = get_scene_props(context.scene)

        if self.mode == "chunks":
            props.voxel_grid_bounds_chunks = not props.voxel_grid_bounds_chunks
        else:
            props.voxel_grid_bounds_selection = not props.voxel_grid_bounds_selection

        return {"FINISHED"}


class VOXELTERRAIN_OT_show_node_group(bpy.types.Operator):
    """Show node group in Geometry Nodes editor."""

    bl_idname = "voxel_terrain.show_node_group"
    bl_label = "Show in Editor"
    bl_description = "Open this node group in the Geometry Nodes editor"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only available if object has a node group assigned."""
        obj = context.object
        if obj is None:
            return False
        props = get_object_props(obj)
        return props.node_group is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)
        node_group = props.node_group

        if node_group is None:
            self.report({"WARNING"}, "No node group assigned")
            return {"CANCELLED"}

        # Find the Voxel Terrain modifier by name, or recreate if missing
        modifier = _find_voxel_terrain_modifier(obj)

        if modifier is None:
            # Recreate the modifier since it was manually deleted
            modifier = obj.modifiers.new(name=VOXEL_TERRAIN_MODIFIER_NAME, type="NODES")
            modifier.show_viewport = False
            modifier.show_render = False
            modifier.show_expanded = False
            modifier.node_group = node_group  # type: ignore[union-attr]
            self.report({"INFO"}, "Recreated Voxel Terrain modifier")

        # Make this modifier active so the editor shows it
        obj.modifiers.active = modifier

        # Switch Properties editor to Modifiers tab if visible
        for area in context.screen.areas if context.screen else []:
            if area.type == "PROPERTIES":
                for space in area.spaces:
                    if space.type == "PROPERTIES":
                        space.context = "MODIFIER"  # type: ignore[attr-defined]
                        break

        # Try to find a Geometry Nodes editor area and update it
        for area in context.screen.areas if context.screen else []:
            if area.type == "NODE_EDITOR":
                for space in area.spaces:
                    tree_type = getattr(space, "tree_type", None)
                    if tree_type == "GeometryNodeTree":
                        # The editor should automatically pick up the active modifier
                        area.tag_redraw()
                        break

        self.report({"INFO"}, f"Showing '{node_group.name}' in editor")
        return {"FINISHED"}


class VOXELTERRAIN_OT_new_node_group(bpy.types.Operator):
    """Create a new node group with the correct socket interface."""

    bl_idname = "voxel_terrain.new_node_group"
    bl_label = "New Voxel Terrain Node Group"
    bl_description = (
        "Create a new Geometry Nodes group with the correct socket interface"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only enabled if we have an object selected."""
        return context.object is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)

        # Create new geometry node tree
        node_group = bpy.data.node_groups.new(
            name="Voxel Terrain",
            type="GeometryNodeTree",
        )

        # Sync sockets to match the required interface
        sync_sockets(node_group)

        # Add Group Input and Group Output nodes
        input_node = node_group.nodes.new("NodeGroupInput")
        input_node.location = (-300, 0)
        input_node.select = False

        output_node = node_group.nodes.new("NodeGroupOutput")
        output_node.location = (300, 0)
        output_node.select = False

        # Connect Geometry input to Geometry output
        # The first output of Group Input is the first socket (Geometry)
        # The first input of Group Output is the first socket (Geometry)
        if input_node.outputs and output_node.inputs:
            node_group.links.new(input_node.outputs[0], output_node.inputs[0])

        # Assign to the object property
        props.node_group = node_group

        self.report({"INFO"}, f"Created node group '{node_group.name}'")
        return {"FINISHED"}


class VOXELTERRAIN_OT_sync_sockets(bpy.types.Operator):
    """Sync node group sockets to match required interface."""

    bl_idname = "voxel_terrain.sync_sockets"
    bl_label = "Sync Sockets"
    bl_description = "Add missing input/output sockets to match the required interface"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only enabled if node group exists and sockets don't match."""
        obj = context.object
        if obj is None:
            return False
        props = get_object_props(obj)
        node_group = props.node_group
        if node_group is None:
            return False
        # Only enable if sockets don't match
        return not check_sockets_match(node_group)

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)
        node_group = props.node_group

        if node_group is None:
            self.report({"WARNING"}, "No node group assigned")
            return {"CANCELLED"}

        inputs_added, outputs_added = sync_sockets(node_group)
        total = inputs_added + outputs_added

        if total == 0:
            self.report({"INFO"}, "Sockets already match")
        else:
            self.report(
                {"INFO"},
                f"Synced sockets: {inputs_added} inputs, {outputs_added} outputs added",
            )
        return {"FINISHED"}


class VOXELTERRAIN_OT_sync_from_modifier(bpy.types.Operator):
    """Sync node group from the Voxel Terrain modifier."""

    bl_idname = "voxel_terrain.sync_from_modifier"
    bl_label = "Sync from Modifier"
    bl_description = "Update the node group property to match the modifier"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only enabled if modifier exists."""
        obj = context.object
        if obj is None:
            return False
        return _find_voxel_terrain_modifier(obj) is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)

        modifier = _find_voxel_terrain_modifier(obj)
        if modifier is None:
            self.report({"WARNING"}, "Voxel Terrain modifier not found")
            return {"CANCELLED"}

        modifier_ng = getattr(modifier, "node_group", None)
        props.node_group = modifier_ng

        if modifier_ng:
            self.report({"INFO"}, f"Synced from modifier: '{modifier_ng.name}'")
        else:
            self.report({"INFO"}, "Cleared node group (from modifier)")
        return {"FINISHED"}


class VOXELTERRAIN_OT_sync_to_modifier(bpy.types.Operator):
    """Sync node group to the Voxel Terrain modifier."""

    bl_idname = "voxel_terrain.sync_to_modifier"
    bl_label = "Sync to Modifier"
    bl_description = "Update the modifier's node group to match this property"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled if we have a node group to sync."""
        obj = context.object
        if obj is None:
            return False
        props = get_object_props(obj)
        # Allow if we have a node group (modifier can be recreated)
        return props.node_group is not None

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Execute the operator."""
        obj = context.object
        assert obj is not None
        props = get_object_props(obj)

        modifier = _find_voxel_terrain_modifier(obj)

        # Recreate modifier if it was deleted
        if modifier is None:
            modifier = obj.modifiers.new(name=VOXEL_TERRAIN_MODIFIER_NAME, type="NODES")
            modifier.show_viewport = False
            modifier.show_render = False
            modifier.show_expanded = False
            self.report({"INFO"}, "Recreated Voxel Terrain modifier")

        modifier.node_group = props.node_group  # type: ignore[union-attr]

        if props.node_group:
            self.report({"INFO"}, f"Synced to modifier: '{props.node_group.name}'")
        else:
            self.report({"INFO"}, "Cleared modifier node group")
        return {"FINISHED"}


class VOXELTERRAIN_OT_delete_terrain(bpy.types.Operator):
    """Delete generated terrain objects and empties."""

    bl_idname = "voxel_terrain.delete_terrain"
    bl_label = "Delete Terrain"
    bl_description = "Delete all generated terrain objects and empties"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Check if terrain root empty exists."""
        return "Voxel Terrain" in bpy.data.objects

    def execute(self, context: Context) -> set[OperatorReturnItems]:
        """Delete all terrain objects and empties."""
        terrain_root_name = "Voxel Terrain"

        if terrain_root_name not in bpy.data.objects:
            self.report({"WARNING"}, "No terrain to delete")
            return {"CANCELLED"}

        terrain_root = bpy.data.objects[terrain_root_name]

        # Collect all objects recursively (children of the root empty)
        all_objects: list[bpy.types.Object] = []

        def collect_children(obj: bpy.types.Object) -> None:
            for child in obj.children:
                collect_children(child)
                all_objects.append(child)

        collect_children(terrain_root)
        all_objects.append(terrain_root)  # Add root last

        objects_count = len(all_objects)

        # Collect mesh data to clean up
        mesh_data: list[bpy.types.Mesh] = [
            obj.data
            for obj in all_objects
            if obj.data is not None and isinstance(obj.data, bpy.types.Mesh)
        ]

        # Use batch_remove for efficient bulk deletion
        bpy.data.batch_remove(all_objects)  # type: ignore[arg-type]

        # Clean up orphan mesh data
        orphan_meshes = [mesh for mesh in mesh_data if mesh.users == 0]
        if orphan_meshes:
            bpy.data.batch_remove(orphan_meshes)  # type: ignore[arg-type]

        self.report(
            {"INFO"},
            f"Deleted {objects_count} terrain objects",
        )
        return {"FINISHED"}


# List of all classes to register - used by __init__.py
classes: tuple[type, ...] = (
    VOXELTERRAIN_OT_generate,
    VOXELTERRAIN_OT_bake,
    VOXELTERRAIN_OT_delete_terrain,
    VOXELTERRAIN_OT_clear,
    VOXELTERRAIN_OT_export,
    VOXELTERRAIN_OT_export_dialog,
    VOXELTERRAIN_OT_set_view_lod,
    VOXELTERRAIN_OT_toggle_grid_bounds,
    VOXELTERRAIN_OT_show_node_group,
    VOXELTERRAIN_OT_new_node_group,
    VOXELTERRAIN_OT_sync_sockets,
    VOXELTERRAIN_OT_sync_from_modifier,
    VOXELTERRAIN_OT_sync_to_modifier,
)
