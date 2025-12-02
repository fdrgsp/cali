from __future__ import annotations

import contextlib
import random
from typing import TYPE_CHECKING

import pyqtgraph as pg
from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIcon, QMouseEvent, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from sqlmodel import Session, col, select
from superqt.fonticon import icon

from cali.plot._main_plot import (
    ANALYSIS_PRODUCTS,
    AnalysisGroup,
    PipelineStage,
    get_available_plots,
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

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(200)

        self._database_path: str | None = None
        self._engine: Engine | None = None
        self._run_id: int | None = None
        self._fov: str = ""
        self._experiment_type: str | None = None

        # ------------------------------------------------------------------ #
        # Top combo + save button
        # ------------------------------------------------------------------ #
        self._combo = QComboBox(self)
        self._update_combo_box()  # Initialize with no experiment type filter

        self._save_btn = QPushButton("Save Image", self)
        self._save_btn.setIcon(QIcon(icon(MDI6.content_save_outline)))
        self._save_btn.clicked.connect(self._on_save)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._save_btn, 0)

        self._choose_dysplayed_traces = _DisplaySingleWellTraces(self)

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

        # Global pg config tweaks (optional)
        pg.setConfigOptions(antialias=False)
        # pg.setConfigOptions(antialias=True, background="w", foreground="k")

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addLayout(top)
        layout.addWidget(self._choose_dysplayed_traces)
        # no toolbar - pan/zoom etc are built-in with mouse in pyqtgraph
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
        self._fov = fov
        self._update_combo_box()  # Update combo box when FOV changes
        self._on_combo_changed(self._combo.currentText())

    @property
    def run_id(self) -> int | None:
        """Return the current run ID (CaliResult.id)."""
        return self._run_id

    @run_id.setter
    def run_id(self, run_id: int | None) -> None:
        """Set the current run ID and refresh the plot."""
        self._run_id = run_id
        self._update_experiment_type()
        self._update_combo_box()
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

    def _update_combo_box(self) -> None:
        """Rebuild combo box based on experiment type and data availability."""
        # Check which pipeline stages are available
        has_detection, has_extraction, has_analysis = (
            self._check_pipeline_stage_availability()
        )

        # Get available plots filtered by experiment type
        combo_options = get_available_plots(
            group=AnalysisGroup.SINGLE_WELL,
            has_detection=True,  # Show all plots, but we'll disable unavailable ones
            has_extraction=True,
            has_analysis=True,
            experiment_type=self._experiment_type,
        )

        # Store current selection
        current_text = self._combo.currentText()

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
                    # Make it visually distinct (grayed out)
                    item.setForeground(Qt.GlobalColor.gray)

                model.appendRow(item)

        # Try to restore previous selection if still valid and enabled
        idx = self._combo.findText(current_text)
        if idx >= 0:
            # Check if the item is enabled
            item = model.item(idx)
            if item and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                self._combo.setCurrentIndex(idx)
            else:
                self._combo.setCurrentIndex(0)  # Default to "None"
        else:
            self._combo.setCurrentIndex(0)  # Default to "None"

    # ------------------------------------------------------------------ #
    # Public helpers used by plot functions
    # ------------------------------------------------------------------ #
    def clear_plot(self) -> None:
        """Completely reset the plot before drawing a new one."""
        plot = self.plot_item
        if plot is None:
            return

        # 1) Disconnect any custom heatmap handlers we attached
        scene = plot.scene()
        for prop_name, signal_name in [
            ("ccorr_hover_handler", "sigMouseMoved"),
            ("ccorr_click_handler", "sigMouseClicked"),
            ("sync_hover_handler", "sigMouseMoved"),
            ("sync_click_handler", "sigMouseClicked"),
        ]:
            handler = plot.property(prop_name)
            if handler is not None:
                with contextlib.suppress(TypeError, RuntimeError):
                    getattr(scene, signal_name).disconnect(handler)
                plot.setProperty(prop_name, None)

        # 2) Clear all items (curves, images, lines, etc.)
        plot.clear()

        # 3) Reset ViewBox transforms and ranges
        vb = plot.getViewBox()
        # Reset any limits that might have been set by raster/heatmap plots
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        # back to normal "math" orientation for traces
        vb.invertY(False)
        # allow non-square aspect by default
        vb.setAspectLocked(False)
        # let pyqtgraph decide ranges next time
        vb.enableAutoRange(x=True, y=True)

        # 4) Reset axes: ticks + value visibility
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            # remove any custom ticks
            axis.setTicks(None)
            # show numeric labels again by default
            axis.setStyle(showValues=True)

        # 5) Reset labels & title
        plot.setTitle("")
        plot.setLabel("left", "")
        plot.setLabel("bottom", "")

        # 6) Hide shared legend if we have one
        if hasattr(self, "legend") and self.legend is not None:
            self.legend.clear()
            self.legend.setVisible(False)

        # 7) Remove colorbar if present
        if hasattr(self, "colorbar") and self.colorbar is not None:
            self.plot_item.layout.removeItem(self.colorbar)
            self.colorbar = None

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
        # Alternatively, use pyqtgraph exporters if you want vector formats.


class _PersistentMenu(QMenu):
    """A QMenu that stays open when checkable actions are triggered."""

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        """Override mouseReleaseEvent to prevent menu closing on checkable actions."""
        if a0 is None:
            super().mouseReleaseEvent(a0)
            return

        action = self.actionAt(a0.pos())
        if action and action.isCheckable():
            # Toggle the action state manually
            action.setChecked(not action.isChecked())
            # Emit the triggered signal manually
            action.triggered.emit(action.isChecked())
            # Don't call the parent implementation to prevent menu closing
            return
        # For non-checkable actions, use default behavior (close menu)
        super().mouseReleaseEvent(a0)


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
