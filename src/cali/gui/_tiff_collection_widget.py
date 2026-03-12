"""Widget for creating TiffCollectionReader configurations via a dialog interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

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
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from cali._constants import natural_sort_key
from cali.gui._util import auto_match_files_grouped
from cali.readers import TiffCollectionReader, TiffCollectionSettings

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        tiff_files: Sequence[Path | str] | Path | str | None = None,
    ) -> None:
        """Initialize the TiffCollectionWidget dialog.

        Parameters
        ----------
        tiff_files : Sequence[Path | str] | Path | str | None
            Either a list of TIFF file paths, a single directory path
            to search for TIFF files (.tif and .tiff extensions),
            or None (default) for empty initialization
        parent : QWidget | None
            Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("TIFF Collection Configuration")
        self.setWindowIcon(QIconifyIcon("mdi:file-image-outline"))

        self._tiff_files: list[Path] = []
        self._file_map: dict[tuple[int, int], list[Path]] = {}

        # LEFT: Well plate widget
        plate_group = QGroupBox("Plate Selection")
        plate_layout = QVBoxLayout(plate_group)
        plate_layout.setContentsMargins(0, 0, 0, 0)
        self._plate_widget = WellPlateWidget()
        self._plate_widget.valueChanged.connect(self._auto_assign_to_wells)
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

        # Create widget to hold available files section
        available_widget = QWidget()
        available_layout = QVBoxLayout(available_widget)
        available_layout.setContentsMargins(5, 5, 5, 5)
        available_layout.addWidget(avail_label)
        available_layout.addWidget(self._available_list)

        # Add/Remove buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        self._add_btn = QPushButton(QIconifyIcon("mdi:arrow-down"), "Add")
        self._add_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._remove_btn = QPushButton(QIconifyIcon("mdi:arrow-up"), "Remove")
        self._remove_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        self._reset_btn = QPushButton(QIconifyIcon("mdi:restart"), "Reset")
        self._reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self._add_btn)
        button_layout.addWidget(self._remove_btn)
        button_layout.addWidget(self._reset_btn)

        # Assigned files list
        assigned_label = QLabel("Files Assigned to Selected Well:")
        self._assigned_list = QListWidget()
        self._assigned_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )

        # Create widget to hold assigned files section
        assigned_widget = QWidget()
        assigned_layout = QVBoxLayout(assigned_widget)
        assigned_layout.setContentsMargins(5, 5, 5, 5)
        assigned_layout.addWidget(assigned_label)
        assigned_layout.addLayout(button_layout)
        assigned_layout.addWidget(self._assigned_list)

        # Create vertical splitter for the two list widgets
        lists_splitter = QSplitter(Qt.Orientation.Vertical)
        lists_splitter.addWidget(available_widget)
        lists_splitter.addWidget(assigned_widget)

        # Add components to files layout
        files_layout.addWidget(lists_splitter)

        # BOTTOM: Metadata inputs and OK/Cancel buttons
        # Metadata inputs
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QHBoxLayout(metadata_group)
        metadata_layout.setContentsMargins(5, 5, 5, 5)
        metadata_layout.setSpacing(5)

        tooltip_exp = (
            "Set the exposure time in milliseconds (ms) for the imaging data.\n"
        )
        exp_label = QLabel("Exposure Time:")
        exp_label.setToolTip(tooltip_exp)
        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setToolTip(tooltip_exp)
        self._exposure_spin.setRange(0.001, 10000.0)
        self._exposure_spin.setValue(100.0)
        self._exposure_spin.setDecimals(3)
        self._exposure_spin.setSuffix(" ms")

        tooltip_px = (
            "Set the pixel size in micrometers (µm) for the imaging data.\n"
            "Set to 0 to keep measurements in pixels."
        )
        px_label = QLabel("Pixel Size:")
        px_label.setToolTip(tooltip_px)
        self._pixel_size_spin = QDoubleSpinBox()
        self._pixel_size_spin.setToolTip(tooltip_px)
        self._pixel_size_spin.setRange(0.0, 100.0)
        self._pixel_size_spin.setValue(0.0)
        self._pixel_size_spin.setDecimals(3)
        self._pixel_size_spin.setSuffix(" µm")

        metadata_layout.addWidget(exp_label)
        metadata_layout.addWidget(self._exposure_spin)
        metadata_layout.addWidget(px_label)
        metadata_layout.addWidget(self._pixel_size_spin)

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
        # Create horizontal splitter between plate and files
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(plate_group)
        top_splitter.addWidget(files_group)
        top_splitter.setStretchFactor(0, 3)  # Plate gets more space
        top_splitter.setStretchFactor(1, 2)  # Files get less space

        # Bottom section - metadata and buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(metadata_group)
        bottom_layout.addStretch()
        bottom_layout.addLayout(button_box)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        main_layout.addWidget(top_splitter)

        main_layout.addLayout(bottom_layout)

        self.resize(1000, 600)

        self.set_tiff_files(tiff_files or [])

    # -------------------------PUBLIC METHODS-------------------------

    def set_tiff_files(self, tiff_files: Sequence[Path | str] | Path | str) -> None:
        """Set or update the list of TIFF files.

        Parameters
        ----------
        tiff_files : Sequence[Path | str] | Path | str
            Either a list of TIFF file paths, or a single directory path
            to search for TIFF files (.tif and .tiff extensions)
        """
        tiff_files_list: list[Path] = []
        # Handle single path or sequence of paths
        if isinstance(tiff_files, (str, Path)):
            tiff_path = Path(tiff_files)
            if tiff_path.is_dir():
                # Find all TIFF files in directory
                tiff_files_list = sorted(
                    list(tiff_path.glob("*.tif")) + list(tiff_path.glob("*.tiff")),
                    key=lambda p: natural_sort_key(p.name),
                )
            else:
                # Single file provided
                tiff_files_list = [tiff_path]
        else:
            tiff_files_list = [Path(f) for f in tiff_files]

        # clear existing data
        self._file_map.clear()
        self._assigned_list.clear()
        self._available_list.clear()
        self._tiff_files.clear()

        if not tiff_files_list:
            return

        # Remove hidden files starting with "." (e.g. on MacOS)
        tiff_files_list = [p for p in tiff_files_list if not p.name.startswith(".")]

        self._tiff_files = sorted(
            (Path(f) for f in tiff_files_list),
            key=lambda p: natural_sort_key(p.name),
        )

        # Update available files list
        for tiff_file in self._tiff_files:
            self._available_list.addItem(tiff_file.name)

        self._auto_assign_to_wells()

    def value(self) -> TiffCollectionReader:
        """Get the configured TiffCollectionReader parameters.

        Returns
        -------
        TiffCollectionSettings
            The configured settings including file map, plate type, and metadata
        """
        if not self._tiff_files:
            raise ValueError("No TIFF files have been set. Use set_tiff_files() first.")

        # Convert from (row, col) tuples to well names like "A1", "B2", etc.
        file_map: dict[str, list[Path | str]] = {}
        for (row, col), paths in self._file_map.items():
            well_name = f"{chr(ord('A') + row)}{col + 1}"
            file_map[well_name] = cast("list[Path | str]", paths)

        plate_plan = self._plate_widget.value()
        plate = plate_plan.plate

        px = self._pixel_size_spin.value()
        metadata = {
            "exposure_ms": self._exposure_spin.value(),
            "pixel_size_um": (None if px == 0.0 else px),
        }

        settings = TiffCollectionSettings(
            file_map=file_map,
            plate=plate,
            metadata=metadata,
            tiff_folder_path=Path(self._tiff_files[0]).parent,
        )
        return TiffCollectionReader(settings)

    # -------------------------PRIVATE METHODS-------------------------

    def _auto_assign_to_wells(self) -> None:
        """Attempt to auto-match TIFF files to wells by filename."""
        # Clear existing assignments first (e.g. when plate type changes)
        self._file_map.clear()
        self._assigned_list.clear()

        if not self._tiff_files:
            self._update_available_list_states()
            return

        plate_plan = self._plate_widget.value()
        plate = plate_plan.plate
        if plate is None:
            self._update_available_list_states()
            return

        # Build list of valid well names and their (row, col)
        well_names: list[str] = []
        well_coords: dict[str, tuple[int, int]] = {}
        for row_idx in range(plate.rows):
            for col_idx in range(plate.columns):
                name = f"{chr(ord('A') + row_idx)}{col_idx + 1}"
                well_names.append(name)
                well_coords[name] = (row_idx, col_idx)

        matches = auto_match_files_grouped(self._tiff_files, well_names)

        for well_name, matched_files in matches.items():
            row_col = well_coords[well_name]
            self._file_map[row_col] = sorted(
                matched_files, key=lambda p: natural_sort_key(p.name)
            )

        self._update_available_list_states()
        self._on_well_selection_changed()

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

    def _on_reset_clicked(self) -> None:
        """Clear all file-to-well assignments."""
        self._file_map.clear()
        self._assigned_list.clear()
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
                for well, count in sorted(
                    well_name_counts.items(),
                    key=lambda x: natural_sort_key(x[0]),
                )
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
