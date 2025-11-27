"""Widget for creating TiffCollectionReader configurations via a dialog interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fonticon_mdi6 import MDI6
from pymmcore_widgets.useq_widgets import WellPlateWidget
from pymmcore_widgets.useq_widgets._well_plate_widget import (
    DATA_POSITION,
    WellPlateView,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon

if TYPE_CHECKING:
    from collections.abc import Sequence

    from useq import WellPlate


class TiffCollectionWidget(QDialog):
    """Dialog for creating TiffCollectionReader configurations.

    A single dialog with:
    - Top left: WellPlateWidget for plate selection
    - Top right: Two list widgets (available files and assigned files)
    - Middle: Add/Remove buttons
    - Bottom: Metadata inputs (exposure time, pixel size) and OK/Cancel buttons
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        tiff_files: Sequence[Path | str] = [],
    ) -> None:
        """Initialize the TiffCollectionWidget dialog.

        Parameters
        ----------
        tiff_files : Sequence[Path]
            List of TIFF file paths found in the data directory
        parent : QWidget | None
            Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("TIFF Collection Configuration")
        self.setWindowIcon(icon(MDI6.file_image_outline))

        self._tiff_files = sorted(Path(f) for f in tiff_files)
        self._file_map: dict[tuple[int, int], list[Path]] = {}

        # LEFT: Well plate widget
        plate_group = QGroupBox("Plate Selection")
        plate_layout = QVBoxLayout(plate_group)
        plate_layout.setContentsMargins(0, 0, 0, 0)
        self._plate_widget = WellPlateWidget()
        plate_layout.addWidget(self._plate_widget)

        # Make plate view only select a single well at a time
        self._plate_view: WellPlateView | None = None
        for child in self._plate_widget.findChildren(WellPlateView):
            self._plate_view = child
            break
        if self._plate_view:
            self._plate_view.setDragMode(WellPlateView.DragMode.NoDrag)
            self._plate_view.setSelectionMode(
                WellPlateView.SelectionMode.SingleSelection
            )
            self._plate_view.selectionChanged.connect(self._on_well_selection_changed)

        # RIGHT: File lists and buttons
        files_group = QGroupBox("File Assignment")
        files_layout = QVBoxLayout(files_group)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.setSpacing(5)

        # Available files list
        avail_label = QLabel("Available TIFF Files:")
        self._available_list = QListWidget()
        self._available_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        for tiff_file in self._tiff_files:
            self._available_list.addItem(tiff_file.name)
        files_layout.addWidget(avail_label)
        files_layout.addWidget(self._available_list)

        # Add/Remove buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        self._add_btn = QPushButton(icon(MDI6.arrow_down), "Add to Selected Well")
        self._add_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._remove_btn = QPushButton(icon(MDI6.arrow_up), "Remove from Well")
        self._remove_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        button_layout.addWidget(self._add_btn)
        button_layout.addWidget(self._remove_btn)
        files_layout.addLayout(button_layout)

        # Assigned files list
        assigned_label = QLabel("Files Assigned to Selected Well:")
        files_layout.addWidget(assigned_label)
        self._assigned_list = QListWidget()
        self._assigned_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        files_layout.addWidget(self._assigned_list)

        # BOTTOM: Metadata inputs and OK/Cancel buttons
        # Metadata inputs
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QHBoxLayout(metadata_group)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setSpacing(5)

        exp_label = QLabel("Exposure Time:")
        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setRange(0.001, 10000.0)
        self._exposure_spin.setValue(100.0)
        self._exposure_spin.setDecimals(3)
        self._exposure_spin.setSuffix(" ms")

        px_label = QLabel("Pixel Size:")
        self._pixel_size_spin = QDoubleSpinBox()
        self._pixel_size_spin.setRange(0.001, 100.0)
        self._pixel_size_spin.setValue(1.0)
        self._pixel_size_spin.setDecimals(3)
        self._pixel_size_spin.setSuffix(" µm")

        metadata_layout.addWidget(exp_label)
        metadata_layout.addWidget(self._exposure_spin)
        metadata_layout.addWidget(px_label)
        metadata_layout.addWidget(self._pixel_size_spin)
        metadata_layout.addStretch()

        # OK/Cancel buttons
        button_box = QHBoxLayout()
        button_box.setContentsMargins(0, 0, 0, 0)
        button_box.setSpacing(5)
        ok_btn = QPushButton("OK")
        ok_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ok_btn.clicked.connect(self._on_ok_clicked)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel_btn.clicked.connect(self.reject)
        button_box.addWidget(ok_btn)
        button_box.addWidget(cancel_btn)

        # LAYOUTS
        top_layout = QHBoxLayout()
        top_layout.addWidget(plate_group, stretch=3)
        top_layout.addWidget(files_group, stretch=2)

        # Bottom section - metadata and buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(metadata_group)
        bottom_layout.addStretch()
        bottom_layout.addLayout(button_box)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        main_layout.addLayout(top_layout)

        main_layout.addLayout(bottom_layout)

        self.resize(1000, 600)

    # -------------------------PUBLIC METHODS-------------------------

    def set_tiff_files(self, tiff_files: Sequence[Path | str]) -> None:
        """Set or update the list of TIFF files.

        Parameters
        ----------
        tiff_files : Sequence[Path]
            List of TIFF file paths to display in the available files list
        """
        self._tiff_files = sorted(Path(f) for f in tiff_files)
        self._file_map.clear()

        # Update available files list
        self._available_list.clear()
        for tiff_file in self._tiff_files:
            self._available_list.addItem(tiff_file.name)

        # Clear assigned files list
        self._assigned_list.clear()

    def value(
        self,
    ) -> tuple[dict[str, list[Path]], WellPlate, dict[str, float]]:
        """Get the configured TiffCollectionReader parameters.

        Returns
        -------
        tuple[dict[str, list[Path]], useq.WellPlate, dict[str, float]]
            A tuple of (file_map, plate, metadata) where:
            - file_map: Maps well names (e.g., "A1") to lists of TIFF file paths
            - plate: The selected useq.WellPlate
            - metadata: Dictionary with 'exposure_ms' and 'pixel_size_um'
        """
        # Convert from (row, col) tuples to well names like "A1", "B2", etc.
        file_map: dict[str, list[Path]] = {}
        for (row, col), paths in self._file_map.items():
            well_name = f"{chr(ord('A') + row)}{col + 1}"
            file_map[well_name] = paths

        plate_plan = self._plate_widget.value()
        plate = plate_plan.plate

        metadata = {
            "exposure_ms": self._exposure_spin.value(),
            "pixel_size_um": self._pixel_size_spin.value(),
        }

        return file_map, plate, metadata

    # -------------------------PRIVATE METHODS-------------------------

    def _update_available_list_states(self) -> None:
        """Update the enabled/disabled state of items in the available list."""
        # Get all assigned files
        assigned_files = set()
        for files in self._file_map.values():
            assigned_files.update(files)

        # Update each item's state
        for i in range(self._available_list.count()):
            item = self._available_list.item(i)
            if item is None:
                continue
            filename = item.text()
            # Find if this file is assigned
            is_assigned = any(path.name == filename for path in assigned_files)
            # Disable if assigned, enable if not
            if is_assigned:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)

    def _on_well_selection_changed(self) -> None:
        """Update assigned files list when well selection changes."""
        if not self._plate_view:
            return

        # Clear assigned list
        self._assigned_list.clear()

        # Get selected well
        selected = self._plate_view._selected_items
        if not selected or len(selected) != 1:
            return

        # Get well coordinates from well name
        well_item = next(iter(selected))
        position = well_item.data(DATA_POSITION)
        if not position or not hasattr(position, "name") or not position.name:
            return

        # Parse well name like "A1" to get (row, col)
        well_name = position.name
        row = ord(well_name[0]) - ord("A")  # A->0, B->1, etc.
        col = int(well_name[1:]) - 1  # 1->0, 2->1, etc.
        row_col = (row, col)

        # Show files assigned to this well
        if row_col in self._file_map:
            for path in self._file_map[row_col]:
                self._assigned_list.addItem(path.name)

    def _on_add_clicked(self) -> None:
        """Add selected files to the selected well."""
        if not self._plate_view:
            return

        # Get selected well
        selected = self._plate_view._selected_items
        if not selected or len(selected) != 1:
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "No Well Selected",
                "Please select a well before adding files.",
            )
            return

        # Get well coordinates from well name
        well_item = next(iter(selected))
        position = well_item.data(DATA_POSITION)
        well_name = position.name
        row = ord(well_name[0]) - ord("A")
        col = int(well_name[1:]) - 1
        row_col = (row, col)

        # Get selected files
        selected_items = self._available_list.selectedItems()
        if not selected_items:
            return

        # Add files to mapping
        if row_col not in self._file_map:
            self._file_map[row_col] = []

        for item in selected_items:
            filename = item.text()
            # Find the full path
            for path in self._tiff_files:
                if path.name == filename:
                    if path not in self._file_map[row_col]:
                        self._file_map[row_col].append(path)
                    break

        # Update both lists
        self._on_well_selection_changed()
        self._update_available_list_states()

    def _on_remove_clicked(self) -> None:
        """Remove selected files from the selected well."""
        if not self._plate_view:
            return

        # Get selected well
        selected = self._plate_view._selected_items
        if not selected or len(selected) != 1:
            return

        # Get well coordinates from well name
        well_item = next(iter(selected))
        position = well_item.data(DATA_POSITION)
        well_name = position.name
        row = ord(well_name[0]) - ord("A")
        col = int(well_name[1:]) - 1
        row_col = (row, col)

        # Get selected files to remove
        selected_items = self._assigned_list.selectedItems()
        if not selected_items:
            return

        # Remove files from mapping
        if row_col in self._file_map:
            for item in selected_items:
                filename = item.text()
                # Find and remove the file
                for path in self._tiff_files:
                    if path.name == filename and path in self._file_map[row_col]:
                        self._file_map[row_col].remove(path)
                        break

            # Clean up empty wells
            if not self._file_map[row_col]:
                del self._file_map[row_col]

        # Update both lists
        self._on_well_selection_changed()
        self._update_available_list_states()

    def _on_ok_clicked(self) -> None:
        """Validate and accept the dialog."""
        if not self._file_map:
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "No Files Assigned",
                "Please assign at least one TIFF file to a well before continuing.",
            )
            return

        # Check that all wells have the same number of files
        file_counts = {well: len(files) for well, files in self._file_map.items()}
        unique_counts = set(file_counts.values())
        if len(unique_counts) > 1:
            from qtpy.QtWidgets import QMessageBox

            # Convert to well names for display
            well_name_counts = {
                f"{chr(ord('A') + row)}{col + 1}": count
                for (row, col), count in file_counts.items()
            }
            counts_display = "\n".join(
                f"  {well}: {count} file(s)"
                for well, count in sorted(well_name_counts.items())
            )

            QMessageBox.warning(
                self,
                "Inconsistent FOV Count",
                f"All wells must have the same number of TIFF files.\n\n"
                f"Current counts:\n{counts_display}\n\n"
                f"Please ensure each well has the same number of FOVs.",
            )
            return

        self.accept()
