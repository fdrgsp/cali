from __future__ import annotations

import contextlib
import random
from typing import TYPE_CHECKING

from fonticon_mdi6 import MDI6
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIcon, QMouseEvent, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAction,
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
from sqlmodel import Session, col, create_engine, select
from superqt.fonticon import icon

from cali.plot._main_plot import (
    MULTI_WELL_COMBO_OPTIONS_DICT,
    SINGLE_WELL_COMBO_OPTIONS_DICT,
    plot_multi_well_data,
    plot_single_well_data,
    requires_active_rois,
)
from cali.sqlmodel._model import FOV, ROI

if TYPE_CHECKING:
    from pathlib import Path
    from typing import ClassVar

    from ._cali_gui import CaliGui

RED = "#C33"
SECTION_ROLE = Qt.ItemDataRole.UserRole + 1


class _CustomNavigationToolbar(NavigationToolbar2QT):
    """Custom navigation toolbar that excludes the save button."""

    # Override toolitems to exclude 'Save' since we have a custom save button
    # that saves with higher resolution.
    toolitems: ClassVar = [
        item for item in NavigationToolbar2QT.toolitems if item[0] != "Save"
    ]


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

        if rois is None or (db_path := self._graph._database_path) is None:
            return

        plot_single_well_data(self._graph, db_path, self._graph._fov, text, rois=rois)

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
            engine = create_engine(
                f"sqlite:///{self._graph._database_path}", echo=False
            )

            try:
                with Session(engine) as session:
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
            finally:
                engine.dispose(close=True)

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


class _SingleWellGraphWidget(QWidget):
    roiSelected = Signal(object)

    def __init__(self, parent: CaliGui) -> None:
        super().__init__(parent)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(200)

        self._plate_viewer: CaliGui = parent
        self._database_path: str | None = None

        self._fov: str = ""

        self._combo = QComboBox(self)
        model = QStandardItemModel()
        self._combo.setModel(model)

        # add the "None" selectable option to the combo box
        none_item = QStandardItem("None")
        model.appendRow(none_item)

        for key, value in SINGLE_WELL_COMBO_OPTIONS_DICT.items():
            section = QStandardItem(key)
            section.setFlags(Qt.ItemFlag.NoItemFlags)
            section.setData(True, SECTION_ROLE)
            model.appendRow(section)
            for item in value:
                model.appendRow(QStandardItem(item))

        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self._save_btn = QPushButton("Save Image", self)
        self._save_btn.setIcon(QIcon(icon(MDI6.content_save_outline)))
        self._save_btn.clicked.connect(self._on_save)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._save_btn, 0)

        self._choose_dysplayed_traces = _DisplaySingleWellTraces(self)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create navigation toolbar (hidden by default, shown only for specific plots)
        self.toolbar = _CustomNavigationToolbar(self.canvas, self)
        # Make the toolbar more compact
        self.toolbar.setMaximumHeight(32)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.7)
        self.toolbar.show()

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self._choose_dysplayed_traces)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.set_combo_text_red(True)

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
        self._on_combo_changed(self._combo.currentText())

    def clear_plot(self) -> None:
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()

    def set_combo_text_red(self, state: bool) -> None:
        """Set the combo text color to red if state is True or to black otherwise."""
        if state:
            self._combo.setStyleSheet(f"color: {RED};")
        else:
            self._combo.setStyleSheet("")

    def _on_combo_changed(self, text: str) -> None:
        """Update the graph when the combo box is changed."""
        # clear the plot
        self.clear_plot()
        if text == "None" or not self._fov or not self._database_path:
            return

        plot_single_well_data(self, self._database_path, self._fov, text, rois=None)
        if self._choose_dysplayed_traces.isChecked():
            self._choose_dysplayed_traces._update()

    def _on_save(self) -> None:
        """Save the current plot as a .png file."""
        # open a file dialog to select the save location
        name = self._combo.currentText().replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tiff)",
        )
        if not filename:
            return
        self.figure.savefig(filename, dpi=300)


class _MultilWellGraphWidget(QWidget):
    def __init__(self, parent: CaliGui) -> None:
        super().__init__(parent)

        self._plate_viewer: CaliGui = parent
        self._database_path: str | None = None

        self._fov: str = ""

        self._conditions: dict[str, bool] = {}

        self._combo = QComboBox(self)
        model = QStandardItemModel()
        self._combo.setModel(model)

        # add the "None" selectable option to the combo box
        none_item = QStandardItem("None")
        model.appendRow(none_item)

        for key, value in MULTI_WELL_COMBO_OPTIONS_DICT.items():
            section = QStandardItem(key)
            section.setFlags(Qt.ItemFlag.NoItemFlags)
            section.setData(True, SECTION_ROLE)
            model.appendRow(section)
            for item in value:
                model.appendRow(QStandardItem(item))

        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self._conditions_btn = QPushButton("Conditions...", self)
        self._conditions_btn.setEnabled(False)
        self._conditions_btn.clicked.connect(self._show_conditions_menu)

        self._save_btn = QPushButton("Save", self)
        self._save_btn.clicked.connect(self._on_save)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._conditions_btn, 0)
        top.addWidget(self._save_btn, 0)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create navigation toolbar (hidden by default, shown only for specific plots)
        self.toolbar = _CustomNavigationToolbar(self.canvas, self)
        # Make the toolbar more compact
        self.toolbar.setMaximumHeight(32)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.7)
        self.toolbar.show()

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.set_combo_text_red(True)

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
        self._on_combo_changed(self._combo.currentText())

    @property
    def conditions(self) -> dict[str, bool]:
        """Return the list of conditions."""
        return self._conditions

    @conditions.setter
    def conditions(self, conditions: dict[str, bool]) -> None:
        self._conditions = conditions

    def clear_plot(self) -> None:
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()

    def set_combo_text_red(self, state: bool) -> None:
        """Set the combo text color to red if state is True or to black otherwise."""
        if state:
            self._combo.setStyleSheet(f"color: {RED};")
        else:
            self._combo.setStyleSheet("")

    def _on_combo_changed(self, text: str) -> None:
        """Update the graph when the combo box is changed."""
        # clear the plot
        self.clear_plot()
        self._conditions_btn.setEnabled(text != "None")
        if text == "None" or not self._database_path:
            return

        plot_multi_well_data(self, text, self._database_path)

    def _on_save(self) -> None:
        """Save the current plot as a .png file."""
        # open a file dialog to select the save location
        name = self._combo.currentText().replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tiff)",
        )
        if not filename:
            return
        self.figure.savefig(filename, dpi=300)

    def _show_conditions_menu(self) -> None:
        """Show a context menu with condition checkboxes."""
        # Create the persistent context menu
        menu = _PersistentMenu(self)

        for condition, state in self._conditions.items():
            action = QAction(condition, self)
            action.setCheckable(True)
            action.setChecked(state)
            action.triggered.connect(
                lambda checked, text=condition: self._on_condition_toggled(
                    checked, text
                )
            )

            menu.addAction(action)

        # Show the menu at the button position
        button_pos = self._conditions_btn.mapToGlobal(
            self._conditions_btn.rect().bottomLeft()
        )
        menu.exec(button_pos)

    def _on_condition_toggled(self, checked: bool, condition: str) -> None:
        """Handle when a condition checkbox is toggled."""
        self._conditions[condition] = checked
        self._on_combo_changed(self._combo.currentText())
