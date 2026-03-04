from __future__ import annotations

import contextlib
import random
from typing import TYPE_CHECKING, ClassVar, cast

import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIcon, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from sqlmodel import Session, col, select
from superqt import QIconifyIcon

from cali.plot._main_plot import (
    ANALYSIS_PRODUCTS,
    AnalysisGroup,
    PipelineStage,
    get_available_plots,
    plot_multi_well_data,
    plot_single_well_data,
    requires_active_rois,
)
from cali.sqlmodel import FOV, ROI, AnalysisSettings, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Engine


RED = "#C33"
SECTION_ROLE = Qt.ItemDataRole.UserRole + 1


class _SingleWellGraphWidget(QWidget):
    roiSelected = Signal(object)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        # Global pg config tweaks (optional)
        pg.setConfigOptions(antialias=False, background="w", foreground="k")

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(200)

        self._database_path: str | None = None
        self._engine: Engine | None = None
        self._run_id: int | None = None
        self._fov: str = ""
        self._experiment_type: str | None = None

        # Connectivity settings
        self._connectivity_threshold: float = 0.3
        self._connectivity_method: str = "calcium_den_dff_corr"

        # ------------------------------------------------------------------ #
        # Top combo + save button
        # ------------------------------------------------------------------ #
        self._combo = QComboBox(self)
        self._rebuild_combo_box()

        self._save_btn = QPushButton("Save Image", self)
        self._save_btn.setIcon(QIcon(QIconifyIcon("mdi:content-save-outline")))
        self._save_btn.clicked.connect(self._on_save)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._save_btn, 0)

        self._choose_dysplayed_traces = _DisplaySingleWellTraces(self)

        # Connectivity threshold widget (only visible for connectivity plots)
        self._connectivity_threshold_widget = _ConnectivityThresholdWidget(self)
        self._connectivity_threshold_widget.setVisible(False)

        # ------------------------------------------------------------------ #
        # pyqtgraph canvas replacement
        # ------------------------------------------------------------------ #
        # This replaces: self.figure, self.canvas, self.toolbar
        self.plot_widget = pg.PlotWidget(self)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=False, alpha=0.3)

        # Create a shared legend for this widget (reused across plots)
        self.legend = pg.LegendItem(
            offset=(-10, 10),  # near top-right
            horSpacing=10,
            verSpacing=0,
        )
        self.legend.setParentItem(self.plot_item.graphicsItem())
        self.legend.setVisible(False)

        # Colorbar for raster plots (initially None, created when needed)
        self.colorbar: pg.ColorBarItem | None = None

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.addLayout(top)
        layout.addWidget(self._choose_dysplayed_traces)
        layout.addWidget(self._connectivity_threshold_widget)
        layout.addWidget(self.plot_widget)

        self._combo.currentTextChanged.connect(self._on_combo_changed)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def database_path(self) -> str | None:
        return self._database_path

    @database_path.setter
    def database_path(self, path: Path | str | None) -> None:
        self._database_path = str(path) if path is not None else None

    @property
    def fov(self) -> str:
        return self._fov

    @fov.setter
    def fov(self, fov: str) -> None:
        old_fov = self._fov
        self._fov = fov

        # Check if current selection was disabled before update
        current_text = self._combo.currentText()
        was_disabled = False
        if current_text and current_text != "None":
            model = self._combo.model()
            if isinstance(model, QStandardItemModel):
                idx = self._combo.findText(current_text)
                if idx >= 0:
                    item = model.item(idx)
                    if item:
                        was_disabled = not (item.flags() & Qt.ItemFlag.ItemIsEnabled)

        # Update combo item availability based on new FOV data
        self._update_combo_item_availability()

        # Check if current selection is now enabled (after update)
        is_now_enabled = False
        if current_text and current_text != "None":
            model = self._combo.model()
            if isinstance(model, QStandardItemModel):
                idx = self._combo.findText(current_text)
                if idx >= 0:
                    item = model.item(idx)
                    if item:
                        is_now_enabled = bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)

        # Reload plot if:
        # 1. FOV changed AND new FOV is not empty, OR
        # 2. Current selection went from disabled to enabled
        should_reload = (old_fov != fov and fov) or (was_disabled and is_now_enabled)
        if should_reload:
            self._reload_current_plot()

    @property
    def run_id(self) -> int | None:
        """Return the current run ID (CaliResult.id)."""
        return self._run_id

    @run_id.setter
    def run_id(self, run_id: int | None) -> None:
        """Set the current run ID and refresh the plot."""
        old_run_id = self._run_id
        self._run_id = run_id
        self._update_experiment_type()
        # Rebuild combo box when run changes (experiment type may have changed)
        self._rebuild_combo_box(preserve_selection=True)
        # Reload plot if run changed
        if old_run_id != run_id:
            self._reload_current_plot()

    @property
    def engine(self) -> Engine | None:
        return self._engine

    @engine.setter
    def engine(self, engine: Engine | None) -> None:
        self._engine = engine

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _update_experiment_type(self) -> None:
        """Query the current run's experiment type from the database."""
        if self._engine is None or self._run_id is None:
            self._experiment_type = None
            return

        with Session(self._engine) as session:
            stmt = (
                select(AnalysisSettings.experiment_type)
                .join(CaliResult)
                .where(col(CaliResult.id) == self._run_id)
            )
            result = session.exec(stmt).first()
            self._experiment_type = result if result else None

    def _check_pipeline_stage_availability(self) -> tuple[bool, bool, bool]:
        """Check which pipeline stages have been completed.

        Returns
        -------
        tuple[bool, bool, bool]
            (has_detection, has_extraction, has_analysis)
        """
        if self._engine is None or self._run_id is None or not self._fov:
            return (False, False, False)

        with Session(self._engine) as session:
            # Check for ROIs (detection)
            has_detection = bool(
                session.exec(
                    select(ROI.id).join(FOV).where(col(FOV.name) == self._fov).limit(1)
                ).first()
            )

            # Check for Traces (extraction)
            has_extraction = bool(
                session.exec(
                    select(Traces.id)
                    .join(ROI)
                    .join(FOV)
                    .where(col(FOV.name) == self._fov)
                    .where(col(Traces.analysis_result_id) == self._run_id)
                    .limit(1)
                ).first()
            )

            # Check for DataAnalysis (analysis)
            has_analysis = bool(
                session.exec(
                    select(DataAnalysis.id)
                    .join(ROI)
                    .join(FOV)
                    .where(col(FOV.name) == self._fov)
                    .where(col(DataAnalysis.analysis_result_id) == self._run_id)
                    .limit(1)
                ).first()
            )

        return (has_detection, has_extraction, has_analysis)

    def _rebuild_combo_box(self, preserve_selection: bool = False) -> None:
        """Rebuild combo box based on experiment type and data availability.

        Parameters
        ----------
        preserve_selection : bool
            If True, attempt to preserve the current selection.
            If False, reset to "None".
        """
        # Check which pipeline stages are available
        has_detection, has_extraction, has_analysis = (
            self._check_pipeline_stage_availability()
        )

        # Get ALL possible plots for this experiment type
        # (we'll disable items individually based on data availability)
        combo_options = get_available_plots(
            group=AnalysisGroup.SINGLE_WELL,
            has_detection=True,
            has_extraction=True,
            has_analysis=True,
            experiment_type=self._experiment_type,
        )

        # Store current selection if preserving
        current_text = self._combo.currentText() if preserve_selection else "None"

        # Rebuild the combo box model
        model = QStandardItemModel()
        self._combo.setModel(model)

        # Add "None" option
        none_item = QStandardItem("None")
        model.appendRow(none_item)

        # Create a mapping of plot names to their pipeline stage requirements
        plot_requirements = {
            product.name: product.pipeline_stage
            for product in ANALYSIS_PRODUCTS
            if product.group == AnalysisGroup.SINGLE_WELL
        }

        # Add categorized plots
        for key, value in combo_options.items():
            section = QStandardItem(key)
            section.setFlags(Qt.ItemFlag.NoItemFlags)
            section.setData(True, SECTION_ROLE)
            model.appendRow(section)
            for plot_name in value:
                item = QStandardItem(plot_name)

                # Check if this plot's required stage is available
                required_stage = plot_requirements.get(plot_name)
                is_available = True

                if required_stage == PipelineStage.DETECTION:
                    is_available = has_detection
                elif required_stage == PipelineStage.EXTRACTION:
                    is_available = has_detection and has_extraction
                elif required_stage == PipelineStage.ANALYSIS:
                    is_available = has_detection and has_extraction and has_analysis

                # Disable item if data not available
                if not is_available:
                    item.setFlags(Qt.ItemFlag.NoItemFlags)

                model.appendRow(item)

        # Try to restore previous selection if preserving and still valid
        if preserve_selection:
            idx = self._combo.findText(current_text)
            if idx >= 0:
                # Check if the item is enabled
                item = model.item(idx)
                if item and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                    self._combo.setCurrentIndex(idx)
                    return
            self._combo.setCurrentIndex(0)
            return

        # Default to "None" if not preserving or selection not found
        self._combo.setCurrentIndex(0)

    def _update_combo_item_availability(self) -> None:
        """Update combo items enabled/disabled state based on FOV data.

        This method does NOT rebuild the combo or change the selection.
        It only updates which items are clickable based on data availability.
        """
        # Check which pipeline stages are available for current FOV
        has_detection, has_extraction, has_analysis = (
            self._check_pipeline_stage_availability()
        )

        # Get the model
        model = self._combo.model()
        if not isinstance(model, QStandardItemModel):
            return

        # Create a mapping of plot names to their pipeline stage requirements
        plot_requirements = {
            product.name: product.pipeline_stage
            for product in ANALYSIS_PRODUCTS
            if product.group == AnalysisGroup.SINGLE_WELL
        }

        # Update each item's enabled/disabled state
        for i in range(model.rowCount()):
            item = model.item(i)
            if not item:
                continue

            # Skip "None" and section headers
            if item.text() == "None" or item.data(SECTION_ROLE):
                continue

            # Check if this plot's required stage is available
            required_stage = plot_requirements.get(item.text())
            is_available = True

            if required_stage == PipelineStage.DETECTION:
                is_available = has_detection
            elif required_stage == PipelineStage.EXTRACTION:
                is_available = has_detection and has_extraction
            elif required_stage == PipelineStage.ANALYSIS:
                is_available = has_detection and has_extraction and has_analysis

            # Update item state
            if is_available:
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(Qt.ItemFlag.NoItemFlags)

    def _reload_current_plot(self) -> None:
        """Reload the currently selected plot with new data.

        Or clear if data unavailable. This preserves the combo selection
        but updates the plot content.
        """
        current_text = self._combo.currentText()
        if not current_text or current_text == "None":
            self.clear_plot()
            return

        # Check if current selection is enabled
        model = self._combo.model()
        if isinstance(model, QStandardItemModel):
            idx = self._combo.findText(current_text)
            if idx >= 0:
                item = model.item(idx)
                if item and not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
                    # Selection is disabled - clear plot but keep selection
                    self.clear_plot()
                    return

        # Reload the plot with current selection
        self._on_combo_changed(current_text)

    # ------------------------------------------------------------------ #
    # Public helpers used by plot functions
    # ------------------------------------------------------------------ #
    def clear_plot(self) -> None:
        """Completely reset the plot to default state before drawing a new one.

        This is the SINGLE SOURCE OF TRUTH for plot reset. All plot functions
        should call this at the start to ensure consistent baseline state.

        Default state:
        - Origin at bottom-left (0,0): invertY(False)
        - No aspect lock: setAspectLocked(False)
        - No ViewBox limits: all set to None
        - Axes show numeric values
        - No custom ticks
        - Empty title and labels
        - Legend hidden
        - No colorbar
        - All event handlers disconnected

        Individual plots can override these settings (e.g., heatmaps may call
        invertY(True), rasters may set limits), but they start from this clean state.
        """
        plot = self.plot_item
        if plot is None:
            return

        # 1) Disconnect any custom handlers we attached
        from cali.plot._util import disconnect_hover_handlers

        disconnect_hover_handlers(plot)

        # 2) Clear all items (curves, images, lines, regions, etc.)
        plot.clear()

        # 3) Reset ViewBox to default state
        vb = plot.getViewBox()
        # Remove all limits
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        # Default orientation: origin at bottom-left (standard math/cartesian)
        vb.invertY(False)
        # Allow non-square aspect by default
        vb.setAspectLocked(False)
        # Enable auto-ranging for next plot
        vb.enableAutoRange(x=True, y=True)

        # 4) Reset axes: ticks + value visibility
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            # Remove any custom ticks
            axis.setTicks(None)
            # Show numeric labels by default
            axis.setStyle(showValues=True)
            # Reset to default auto SI prefix behavior
            axis.enableAutoSIPrefix(True)

        # 5) Reset labels & title
        plot.setTitle("")
        plot.setLabel("left", "")
        plot.setLabel("bottom", "")

        # 6) Hide shared legend
        self.legend.clear()
        self.legend.setVisible(False)

        # 7) Remove colorbar if present
        if self.colorbar is not None:
            self.plot_item.layout.removeItem(self.colorbar)
            self.colorbar = None

        # 8) Hide connectivity threshold widget
        self._connectivity_threshold_widget.setVisible(False)

    # ------------------------------------------------------------------ #
    # Internal slots
    # ------------------------------------------------------------------ #
    def _on_combo_changed(self, text: str) -> None:
        """Update the graph when the combo box is changed."""
        # clear the plot
        self.clear_plot()
        if (
            not text
            or text == "None"
            or not self._fov
            or not self._engine
            or self._run_id is None
        ):
            return

        # Show/hide connectivity threshold widget based on plot type
        is_connectivity_plot = "Connectivity" in text and "Clustering" not in text
        self._connectivity_threshold_widget.setVisible(is_connectivity_plot)

        plot_single_well_data(
            self, self._engine, self._fov, text, rois=None, run_id=self._run_id
        )
        if self._choose_dysplayed_traces.isChecked():
            self._choose_dysplayed_traces._update()

    def _on_save(self) -> None:
        """Save the current plot as an image file."""
        name = self._combo.currentText().replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tiff)",
        )
        if not filename:
            return

        # Easiest: grab the widget pixels and save
        pixmap = self.plot_widget.grab()
        pixmap.save(filename)
        # Alternatively, use pyqtgraph exporters if you want vector formats.\


class _MultilWellGraphWidget(QWidget):
    """Multi-well graph widget using pyqtgraph for bar plots across conditions."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        # Global pg config tweaks (optional)
        pg.setConfigOptions(antialias=False, background="w", foreground="k")

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(200)

        self._database_path: str | None = None
        self._engine: Engine | None = None
        self._run_id: int | None = None
        self._experiment_type: str | None = None
        self._conditions: dict[str, dict[str, bool | str]] = {}
        self._pca_features: list[str] | None = None
        self._last_plot_data: dict[str, object] | None = None

        # ------------------------------------------------------------------ #
        # Top combo + conditions button + save button
        # ------------------------------------------------------------------ #
        self._combo = QComboBox(self)
        self._rebuild_combo_box()  # Initialize with no experiment type filter

        self._conditions_btn = QPushButton("Conditions...", self)
        self._conditions_btn.setEnabled(False)
        self._conditions_btn.clicked.connect(self._show_conditions_menu)
        self._conditions_btn.setToolTip(
            "Use the Conditions dialog to reorder and toggle visibility of conditions."
        )

        self._pca_features_btn = QPushButton("PCA Features...", self)
        self._pca_features_btn.setToolTip(
            "Select which features to include in the PCA computation."
        )
        self._pca_features_btn.clicked.connect(self._show_pca_features_dialog)
        self._pca_features_btn.hide()

        self._save_btn = QPushButton("Save Image", self)
        self._save_btn.setIcon(QIcon(QIconifyIcon("mdi:content-save-outline")))
        self._save_btn.clicked.connect(self._on_save)

        self._save_csv_btn = QPushButton("Save as CSV", self)
        self._save_csv_btn.setIcon(QIcon(QIconifyIcon("mdi:file-delimited-outline")))
        self._save_csv_btn.setToolTip("Export the current plot data as a CSV file.")
        self._save_csv_btn.clicked.connect(self._on_save_csv)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._conditions_btn, 0)
        top.addWidget(self._pca_features_btn, 0)
        top.addWidget(self._save_btn, 0)
        top.addWidget(self._save_csv_btn, 0)

        # ------------------------------------------------------------------ #
        # pyqtgraph canvas for bar plots
        # ------------------------------------------------------------------ #
        self.plot_widget = pg.PlotWidget(self)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=False, y=True, alpha=0.3)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addLayout(top)
        layout.addWidget(self.plot_widget)

        self._combo.currentTextChanged.connect(self._on_combo_changed)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def database_path(self) -> str | None:
        return self._database_path

    @database_path.setter
    def database_path(self, path: Path | str | None) -> None:
        self._database_path = str(path) if path is not None else None

    @property
    def conditions(self) -> dict[str, dict[str, bool | str]]:
        """Return the dict of conditions and their enabled state."""
        return self._conditions

    @conditions.setter
    def conditions(self, conditions: dict[str, dict[str, bool | str]]) -> None:
        self._conditions = conditions

    @property
    def run_id(self) -> int | None:
        """Return the current run ID (CaliResult.id)."""
        return self._run_id

    @run_id.setter
    def run_id(self, run_id: int | None) -> None:
        """Set the current run ID and refresh the plot."""
        old_run_id = self._run_id
        self._run_id = run_id
        self._update_experiment_type()
        # Rebuild combo box when run changes (experiment type may have changed)
        self._rebuild_combo_box(preserve_selection=True)
        # Reload plot if run changed
        if old_run_id != run_id:
            self._on_combo_changed(self._combo.currentText())

    @property
    def engine(self) -> Engine | None:
        return self._engine

    @engine.setter
    def engine(self, engine: Engine | None) -> None:
        self._engine = engine

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _update_experiment_type(self) -> None:
        """Query the current run's experiment type from the database."""
        if self._engine is None or self._run_id is None:
            self._experiment_type = None
            return

        with Session(self._engine) as session:
            stmt = (
                select(AnalysisSettings.experiment_type)
                .join(CaliResult)
                .where(col(CaliResult.id) == self._run_id)
            )
            result = session.exec(stmt).first()
            self._experiment_type = result if result else None

    def _rebuild_combo_box(self, preserve_selection: bool = False) -> None:
        """Rebuild combo box based on experiment type.

        Parameters
        ----------
        preserve_selection : bool
            If True, attempt to preserve the current selection.
            If False, reset to "None".
        """
        # Get available plots filtered by experiment type
        combo_options = get_available_plots(
            group=AnalysisGroup.MULTI_WELL,
            has_detection=True,
            has_extraction=True,
            has_analysis=True,
            experiment_type=self._experiment_type,
        )

        # Store current selection if preserving
        current_text = self._combo.currentText() if preserve_selection else "None"

        # Rebuild the combo box model
        model = QStandardItemModel()
        self._combo.setModel(model)

        # Add "None" option
        none_item = QStandardItem("None")
        model.appendRow(none_item)

        # Add categorized plots
        for key, value in combo_options.items():
            section = QStandardItem(key)
            section.setFlags(Qt.ItemFlag.NoItemFlags)
            section.setData(True, SECTION_ROLE)
            model.appendRow(section)
            for plot_name in value:
                item = QStandardItem(plot_name)
                model.appendRow(item)

        # Try to restore previous selection if preserving and still valid
        if preserve_selection:
            idx = self._combo.findText(current_text)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
                return

        # Default to "None" if not preserving or selection not found
        self._combo.setCurrentIndex(0)

    # ------------------------------------------------------------------ #
    # Public helpers used by plot functions
    # ------------------------------------------------------------------ #
    def clear_plot(self) -> None:
        """Completely reset the plot before drawing a new one."""
        self._last_plot_data = None

        plot = self.plot_item
        if plot is None:
            return

        # Disconnect any custom handlers we attached
        from cali.plot._util import disconnect_hover_handlers

        disconnect_hover_handlers(plot)

        # Clear all items
        plot.clear()

        # Clear legend items (created by scatter/PCA plots via addLegend)
        if getattr(plot, "legend", None) is not None:
            plot.legend.clear()

        # Reset ViewBox transforms and ranges
        vb = plot.getViewBox()
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        vb.invertY(False)
        vb.setAspectLocked(False)
        vb.enableAutoRange(x=True, y=True)

        # Reset axes
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setTicks(None)
            axis.setStyle(showValues=True)

        # Reset labels & title
        plot.setTitle("")
        plot.setLabel("left", "")
        plot.setLabel("bottom", "")

    # ------------------------------------------------------------------ #
    # Internal slots
    # ------------------------------------------------------------------ #
    def _on_combo_changed(self, text: str) -> None:
        """Update the graph when the combo box is changed."""
        self.clear_plot()
        self._conditions_btn.setEnabled(text != "None")

        # Show PCA Features button only for PCA plots
        if text.startswith("PCA"):
            self._pca_features_btn.show()
        else:
            self._pca_features_btn.hide()

        if text == "None" or not self._engine:
            return

        plot_multi_well_data(self, text, self._engine, run_id=self._run_id)

    def _on_save(self) -> None:
        """Save the current plot as an image file."""
        name = self._combo.currentText().replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tiff)",
        )
        if not filename:
            return

        pixmap = self.plot_widget.grab()
        pixmap.save(filename)

    def _on_save_csv(self) -> None:
        """Export the current plot data as a CSV file."""
        if self._last_plot_data is None:
            return

        name = self._combo.currentText().replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            name,
            "CSV Files (*.csv)",
        )
        if not filename:
            return

        import pandas as pd

        data: dict[str, object] = self._last_plot_data
        if "fov_values" in data:
            # Bar plot data — build rows with fov columns
            fov_vals = cast("list[list[float]]", data["fov_values"])
            conditions = cast("list[str]", data["conditions"])
            means = cast("list[float]", data["means"])
            sems = cast("list[float]", data["sems"])
            max_fovs = max((len(fv) for fv in fov_vals), default=0)
            rows: list[dict[str, object]] = []
            for cond, mean, sem, fv in zip(conditions, means, sems, fov_vals):
                row: dict[str, object] = {"condition": cond, "mean": mean, "sem": sem}
                for i, val in enumerate(fv):
                    row[f"fov_{i + 1}"] = val
                for i in range(len(fv), max_fovs):
                    row[f"fov_{i + 1}"] = float("nan")
                rows.append(row)
            df = pd.DataFrame(rows)
        else:
            # PCA or other dict-based data — just pass columns directly
            cols = {k: v for k, v in data.items() if k != "parameter"}
            df = pd.DataFrame(cols)

        df.to_csv(filename, index=False)

    def _show_conditions_menu(self) -> None:
        """Show a dialog for reordering and toggling conditions."""
        if not self._conditions:
            return

        dialog = _ConditionsDialog(self._conditions, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get the new condition order and states
            new_conditions = dialog.get_conditions()
            self._conditions = new_conditions
            # Redraw the plot with the new order
            self._on_combo_changed(self._combo.currentText())

    def _show_pca_features_dialog(self) -> None:
        """Show a dialog for selecting PCA features, then refresh the plot."""
        # Query whether rising edge analysis is enabled for this run
        enable_rising_edge = False
        if self._engine is not None and self._run_id is not None:
            with Session(self._engine) as session:
                stmt = (
                    select(AnalysisSettings.enable_rising_edge_analysis)
                    .join(CaliResult)
                    .where(col(CaliResult.id) == self._run_id)
                )
                result = session.exec(stmt).first()
                if result is not None:
                    enable_rising_edge = bool(result)

        dialog = _PCAFeaturesDialog(
            current_features=self._pca_features,
            experiment_type=self._experiment_type,
            enable_rising_edge=enable_rising_edge,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._pca_features = dialog.get_features()
            self._on_combo_changed(self._combo.currentText())


class _DisplaySingleWellTraces(QGroupBox):
    def __init__(self, parent: _SingleWellGraphWidget) -> None:
        super().__init__(parent)
        self.setTitle("Choose which ROI to display")
        self.setCheckable(True)
        self.setChecked(False)

        self.setToolTip(
            "By default, the widget will display the traces form all the ROIs from the "
            "current FOV. Here you can choose to only display a subset of ROIs. You "
            "can input a range (e.g. 1-10 to plot the first 10 ROIs), single ROIs "
            "(e.g. 30, 33 to plot ROI 30 and 33) or, if you only want to pick n random "
            "ROIs, you can type 'rnd' followed by the number or ROIs you want to "
            "display (e.g. rnd10 to plot 10 random ROIs)."
        )

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        )

        self._graph: _SingleWellGraphWidget = parent

        self._roi_le = QLineEdit()
        self._roi_le.setPlaceholderText("e.g. 1-10, 30, 33 or rnd10")
        # when pressing enter in the line edit, update the graph
        self._roi_le.returnPressed.connect(self._update)
        self._update_btn = QPushButton("Update", self)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(QLabel("ROIs:"))
        main_layout.addWidget(self._roi_le)
        main_layout.addWidget(self._update_btn)
        self._update_btn.clicked.connect(self._update)

        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, state: bool) -> None:
        """Enable or disable the random spin box and the update button."""
        if not state:
            self._graph._on_combo_changed(self._graph._combo.currentText())
        else:
            self._update()

    def _update(self) -> None:
        """Update the graph with random traces."""
        self._graph.clear_plot()
        text = self._graph._combo.currentText()

        # Get database path and FOV name
        if not self._graph._database_path or not self._graph._fov:
            return

        # Get ROI selection
        rois = self._parse_roi_selection()

        if rois is None or not self._graph._engine or self._graph._run_id is None:
            return

        plot_single_well_data(
            self._graph,
            self._graph._engine,
            self._graph._fov,
            text,
            rois=rois,
            run_id=self._graph._run_id,
        )

    def _parse_roi_selection(self) -> list[int] | None:
        """Return the list of ROIs to be displayed."""
        text = self._roi_le.text()
        if not text:
            return None

        # Handle random ROI selection (e.g., "rnd10")
        # This queries the database to get all available ROIs for the current FOV
        # and randomly selects the requested number
        if text[:3] == "rnd" and text[3:].isdigit():
            num_rois = int(text[3:])
            if not self._graph._database_path or not self._graph._fov:
                return None

            # Check if the current plot requires only active ROIs
            plot_name = self._graph._combo.currentText()
            active_only = requires_active_rois(plot_name)

            # Query database to get all available ROI label values for this FOV
            if not self._graph._engine:
                return None

            with Session(self._graph._engine) as session:
                # Get all ROI label values for this FOV
                stmt = (
                    select(ROI.label_value)
                    .join(FOV)
                    .where(col(FOV.name) == self._graph._fov)
                    .order_by(col(ROI.label_value))
                )

                # Filter for active ROIs if the plot requires it
                if active_only:
                    stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

                roi_label_values = session.exec(stmt).all()

                if not roi_label_values:
                    return None

                # Randomly select the requested number of ROIs
                selected_rois = random.sample(
                    roi_label_values, min(num_rois, len(roi_label_values))
                )
                return sorted(selected_rois)

        # Parse the input string for specific ROI numbers
        rois = self._parse_input(text)

        return rois or None

    def _parse_input(self, input_str: str) -> list[int]:
        """Parse the input string and return a list of ROIs."""
        parts = input_str.split(",")
        numbers: list[int] = []
        for part in parts:
            part = part.strip()  # remove any leading/trailing whitespace
            if "-" in part:
                with contextlib.suppress(ValueError):
                    start, end = map(int, part.split("-"))
                    numbers.extend(range(start, end + 1))
            else:
                with contextlib.suppress(ValueError):
                    numbers.append(int(part))
        return numbers


class _ConnectivityThresholdWidget(QGroupBox):
    """Widget for adjusting connectivity method and threshold."""

    def __init__(self, parent: _SingleWellGraphWidget) -> None:
        super().__init__(parent)
        self.setTitle("Connectivity Settings")
        self.setCheckable(False)

        self.setToolTip(
            "Select a functional connectivity method and adjust the edge threshold.\n\n"
            "Functional connectivity here is computed as zero-lag Pearson correlation "
            "between calcium activity traces.\n\n"
            "• DF/F: uses raw ΔF/F traces.\n"
            "• Denoised DF/F: uses OASIS-denoised ΔF/F traces for a cleaner, "
            "more stable network."
        )

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        )

        self._graph: _SingleWellGraphWidget = parent

        # Method combo box
        self._method_combo = QComboBox(self)
        self._method_combo.addItems(
            [
                "Denoised DF/F Correlation",
                "DF/F Correlation",
            ]
        )
        self._method_combo.setToolTip("Select connectivity metric method")

        # Threshold slider
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setMinimum(0)
        self._threshold_slider.setMaximum(100)
        self._threshold_slider.setValue(30)  # Default 0.3
        self._threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._threshold_slider.setTickInterval(10)

        # Threshold value label
        self._threshold_label = QLabel("0.30")
        self._threshold_label.setMinimumWidth(40)
        self._threshold_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        # Update button
        self._update_btn = QPushButton("Update", self)
        self._update_btn.clicked.connect(self._update_connectivity)

        # Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(QLabel("Method:"))
        main_layout.addWidget(self._method_combo)
        main_layout.addWidget(QLabel("Threshold:"))
        main_layout.addWidget(self._threshold_slider)
        main_layout.addWidget(self._threshold_label)
        main_layout.addWidget(self._update_btn)

        # Connect signals
        self._threshold_slider.valueChanged.connect(self._on_slider_changed)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)

    def _get_method_from_combo(self) -> str:
        """Map combo box selection to ConnectivityMethod string."""
        method_map = {
            0: "calcium_den_dff_corr",
            1: "calcium_dff_corr",
        }
        return method_map[self._method_combo.currentIndex()]

    def _on_method_changed(self, index: int) -> None:
        """Update connectivity method when combo box changes."""
        method = self._get_method_from_combo()
        self._graph._connectivity_method = method
        # Update the plot with the new method
        self._update_connectivity()

    def _on_slider_changed(self, value: int) -> None:
        """Update threshold label when slider changes."""
        threshold = value / 100.0
        self._threshold_label.setText(f"{threshold:.2f}")

    def _update_connectivity(self) -> None:
        """Update the connectivity graph with new threshold."""
        threshold = self._threshold_slider.value() / 100.0

        # Store threshold on the parent widget so the plot function can access it
        self._graph._connectivity_threshold = threshold

        # Re-plot with new threshold
        text = self._graph._combo.currentText()
        if "Connectivity" in text and self._graph._engine and self._graph._fov:
            self._graph.clear_plot()
            plot_single_well_data(
                self._graph,
                self._graph._engine,
                self._graph._fov,
                text,
                rois=None,
                run_id=self._graph._run_id,
            )


class _ConditionItemWidget(QWidget):
    """Widget for a single condition in the conditions dialog."""

    # Available colors for conditions
    COLORS: ClassVar[dict[str, str]] = {
        "gray": "#808080",
        "green": "#00FF00",
        "magenta": "#FF00FF",
        "red": "#FF0000",
        "blue": "#0000FF",
        "cyan": "#00FFFF",
        "yellow": "#FFFF00",
        "orange": "#FFA500",
    }

    def __init__(self, name: str, visible: bool, color: str, parent: QWidget) -> None:
        """Initialize the condition item widget.

        Parameters
        ----------
        name : str
            Condition name
        visible : bool
            Whether condition is visible
        color : str
            Color name for the condition
        parent : QWidget
            Parent widget
        """
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(10)

        # Checkbox for visibility
        self._checkbox = QCheckBox(name, self)
        self._checkbox.setChecked(visible)
        layout.addWidget(self._checkbox)

        layout.addStretch(1)

        # Color combo box
        self._color_combo = QComboBox(self)
        for color_name in self.COLORS:
            self._color_combo.addItem(color_name)
        self._color_combo.setCurrentText(color)
        layout.addWidget(self._color_combo)

        # Drag handle icon (three horizontal lines)
        drag_handle = QLabel("≡", self)
        drag_handle.setStyleSheet("font-size: 16px;")
        drag_handle.setToolTip("Drag to reorder")
        layout.addWidget(drag_handle)

    def get_name(self) -> str:
        """Return the condition name."""
        return self._checkbox.text()  # type: ignore

    def is_visible(self) -> bool:
        """Return whether the condition is visible."""
        return self._checkbox.isChecked()  # type: ignore

    def get_color(self) -> str:
        """Return the selected color."""
        return self._color_combo.currentText()  # type: ignore


class _ConditionsDialog(QDialog):
    """Dialog for reordering and toggling conditions via drag-and-drop."""

    def __init__(
        self, conditions: dict[str, dict[str, bool | str]], parent: QWidget
    ) -> None:
        """Initialize the conditions dialog.

        Parameters
        ----------
        conditions : dict[str, dict[str, bool | str]]
            Dictionary of condition names to dicts with 'visible' and 'color' keys
        parent : QWidget
            Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Condition Order, Visibility, and Color")
        self.setModal(True)
        self.resize(500, 500)

        # Store original conditions
        self._original_conditions = conditions.copy()

        # Create list widget with drag-and-drop enabled
        self._list_widget = QListWidget(self)
        self._list_widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)

        # Populate list with conditions
        for condition, cond_info in conditions.items():
            enabled = bool(cond_info.get("visible", True))
            color = str(cond_info.get("color", "gray"))

            # Create item and custom widget
            item = QListWidgetItem(self._list_widget)
            widget = _ConditionItemWidget(condition, enabled, color, self._list_widget)

            # Make item draggable
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)

            # Set the custom widget as the item widget
            item.setSizeHint(widget.sizeHint())
            self._list_widget.addItem(item)
            self._list_widget.setItemWidget(item, widget)

        # Instructions label
        instructions = QLabel(
            "Drag conditions to reorder them. Uncheck to hide. Select color for bars.",
            self,
        )
        instructions.setWordWrap(True)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(instructions)
        layout.addWidget(self._list_widget)
        layout.addWidget(button_box)

    def get_conditions(self) -> dict[str, dict[str, bool | str]]:
        """Return the reordered conditions with their enabled state and color.

        Returns
        -------
        dict[str, dict[str, bool | str]]
            Dictionary mapping condition name to dict with 'visible' and 'color' keys
        """
        conditions = {}
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item is not None:
                widget = self._list_widget.itemWidget(item)
                if isinstance(widget, _ConditionItemWidget):
                    condition_name = widget.get_name()
                    is_enabled = widget.is_visible()
                    color = widget.get_color()
                    conditions[condition_name] = {
                        "visible": is_enabled,
                        "color": color,
                    }
        return conditions  # type: ignore


# Human-readable labels for PCA feature columns.
_PCA_FEATURE_LABELS: dict[str, str] = {
    "mean_amplitude": "Mean Amplitude (\u0394F/F\u2080)",
    "mean_frequency": "Mean Frequency (Hz)",
    "mean_iei": "Mean IEI (s)",
    "mean_spike_freq": "Mean Spike Frequency (Hz)",
    "mean_spike_freq_edges": "Mean Spike Freq \u2013 Rising Edges (Hz)",
    "mean_cell_size": "Mean Cell Size (\u00b5m\u00b2)",
    "pct_active": "% Active ROIs",
    "burst_count": "Burst Count",
    "burst_avg_duration_s": "Burst Avg Duration (s)",
    "burst_avg_interval_s": "Burst Avg Interval (s)",
}

# Features that are FOV-level burst stats (excluded for stim-split / evoked).
_BURST_FEATURES = {"burst_count", "burst_avg_duration_s", "burst_avg_interval_s"}


class _PCAFeaturesDialog(QDialog):
    """Dialog for selecting which features to include in PCA."""

    def __init__(
        self,
        current_features: list[str] | None,
        experiment_type: str | None,
        enable_rising_edge: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PCA Feature Selection")
        self.setModal(True)

        from cali._constants import EVOKED
        from cali.plot._multi_wells_plots._dimensionality_reduction import (
            FEATURE_COLUMNS,
        )

        self._feature_columns = list(FEATURE_COLUMNS)
        self._checkboxes: dict[str, QCheckBox] = {}

        # Instructions
        instructions = QLabel(
            "Select the features to include in PCA. "
            "Disabled features are not available for the current experiment.",
            self,
        )
        instructions.setWordWrap(True)

        # Build checkboxes
        cb_layout = QVBoxLayout()
        cb_layout.setSpacing(4)

        is_evoked = experiment_type == EVOKED

        for feat in self._feature_columns:
            label = _PCA_FEATURE_LABELS.get(feat, feat)
            cb = QCheckBox(label, self)

            # Default state: checked if in current_features (or all if None)
            if current_features is None:
                cb.setChecked(True)
            else:
                cb.setChecked(feat in current_features)

            # Disable burst features for evoked experiments
            if feat in _BURST_FEATURES and is_evoked:
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.setToolTip("Burst stats are not available for evoked experiments")

            # Disable rising-edge feature when rising edge analysis is off
            if feat == "mean_spike_freq_edges" and not enable_rising_edge:
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.setToolTip("Enable Rising Edge Analysis to use this feature")

            self._checkboxes[feat] = cb
            cb_layout.addWidget(cb)

        # OK / Cancel
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(instructions)
        layout.addLayout(cb_layout)
        layout.addWidget(button_box)

    # ------------------------------------------------------------------

    def get_features(self) -> list[str] | None:
        """Return list of selected feature names, or None if all are checked."""
        selected = [f for f, cb in self._checkboxes.items() if cb.isChecked()]
        all_enabled = [f for f, cb in self._checkboxes.items() if cb.isEnabled()]
        if set(selected) == set(all_enabled):
            return None  # all enabled features are checked
        return selected
