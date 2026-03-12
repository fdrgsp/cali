"""Dialog for importing pre-existing label TIFFs and assigning them to FOVs."""

from __future__ import annotations

from pathlib import Path

from pymmcore_widgets.useq_widgets import WellPlateWidget
from pymmcore_widgets.useq_widgets._well_plate_widget import (
    DATA_POSITION,
    WellPlateView,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon
from superqt.utils import create_worker

from cali._constants import natural_sort_key
from cali.gui._util import (
    _BrowseWidget,
    _ProgressBarWidget,
    auto_match_files,
    show_error_dialog,
)
from cali.logger import cali_logger
from cali.util import import_labels_to_database

# Role for storing FOV data in tree items
_FOV_ID_ROLE = Qt.ItemDataRole.UserRole
_FOV_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
_FOV_POS_IDX_ROLE = Qt.ItemDataRole.UserRole + 2


class _ImportLabelsDialog(QDialog):
    """Dialog for importing label TIFFs and assigning them to specific FOVs.

    The dialog queries the database for the existing well/FOV structure and
    displays a plate map on the left. Users can browse for label TIFF files,
    then assign each label file to a specific FOV. On acceptance, label arrays
    are read, converted to ROI/Mask objects, and committed to the database.
    """

    def __init__(
        self,
        database_path: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Label TIFFs")
        self.setModal(True)

        self._database_path = database_path
        self._label_files: list[Path] = []
        # Maps fov_id -> label TIFF path
        self._label_map: dict[int, Path] = {}
        self._imported_detection_settings_id: int | None = None
        # Maps well_name -> list of (fov_id, fov_name, pos_idx)
        self._well_fovs: dict[str, list[tuple[int, str, int]]] = {}

        # --- Browse widget for labels folder ---
        self._browse_wdg = _BrowseWidget(
            self,
            label="Labels Folder",
            tooltip="Select a folder containing label TIFF files.",
        )
        self._browse_wdg.pathSet.connect(self._on_folder_selected)

        # --- LEFT: Well plate widget (read-only plate type) ---
        plate_group = QGroupBox("Plate")
        plate_layout = QVBoxLayout(plate_group)
        plate_layout.setContentsMargins(0, 0, 0, 0)
        self._plate_widget = WellPlateWidget()
        self._plate_widget.plate_name.setEnabled(False)
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

        # --- RIGHT TOP: Available label files ---
        avail_label = QLabel("Available Label Files:")
        self._available_list = QListWidget()
        self._available_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        available_widget = QWidget()
        available_layout = QVBoxLayout(available_widget)
        available_layout.setContentsMargins(5, 5, 5, 5)
        available_layout.addWidget(avail_label)
        available_layout.addWidget(self._available_list)

        # --- Assign / Unassign / Reset buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(5)
        self._assign_btn = QPushButton(QIconifyIcon("mdi:arrow-down"), "Assign")
        self._assign_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._assign_btn.clicked.connect(self._on_assign)
        self._unassign_btn = QPushButton(QIconifyIcon("mdi:arrow-up"), "Unassign")
        self._unassign_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._unassign_btn.clicked.connect(self._on_unassign)
        self._reset_btn = QPushButton(QIconifyIcon("mdi:restart"), "Reset")
        self._reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self._assign_btn)
        btn_layout.addWidget(self._unassign_btn)
        btn_layout.addWidget(self._reset_btn)

        # --- RIGHT BOTTOM: FOV list for selected well ---
        fov_label = QLabel("Label File assigned the FOVs in Selected Well:")
        self._fov_list = QTreeWidget()
        self._fov_list.setHeaderLabels(["FOV", "Assigned Label"])
        self._fov_list.setColumnCount(2)
        self._fov_list.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._fov_list.setRootIsDecorated(False)

        assigned_widget = QWidget()
        assigned_layout = QVBoxLayout(assigned_widget)
        assigned_layout.setContentsMargins(5, 5, 5, 5)
        assigned_layout.addWidget(fov_label)
        assigned_layout.addLayout(btn_layout)
        assigned_layout.addWidget(self._fov_list)

        # --- Right side: vertical splitter for available files / FOV list ---
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(available_widget)
        right_splitter.addWidget(assigned_widget)

        # --- Horizontal splitter: plate | files ---
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(plate_group)
        top_splitter.addWidget(right_splitter)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)

        # --- OK / Cancel ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)
        bottom_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(ok_btn)
        bottom_layout.addWidget(cancel_btn)

        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        main_layout.addWidget(self._browse_wdg)
        main_layout.addWidget(top_splitter, 1)
        main_layout.addLayout(bottom_layout)

        self.resize(1000, 600)

        # Load plate and FOV data from the database
        self._load_fovs_from_db()

    # --- Public ---

    def value(self) -> dict[str, Path]:
        """Return the current label map as ``{fov_name: label_path}``."""
        id_to_name: dict[int, str] = {
            fov_id: fov_name
            for fovs in self._well_fovs.values()
            for fov_id, fov_name, _ in fovs
        }
        return {
            id_to_name[fov_id]: path
            for fov_id, path in self._label_map.items()
            if fov_id in id_to_name
        }

    def setValue(self, label_map: dict[str, Path | str]) -> None:
        """Populate the label map from ``{fov_name: label_path}``.

        Parameters
        ----------
        label_map : dict[str, Path | str]
            Mapping of FOV names to label TIFF file paths.
            Unknown FOV names are silently ignored.
        """
        name_to_id: dict[str, int] = {
            fov_name: fov_id
            for fovs in self._well_fovs.values()
            for fov_id, fov_name, _ in fovs
        }
        self._label_map = {
            name_to_id[name]: Path(path)
            for name, path in label_map.items()
            if name in name_to_id
        }
        self._on_well_selection_changed()
        self._update_available_list_states()

    # --- Private: UI callbacks ---

    def _on_folder_selected(self, path: str) -> None:
        """Populate the available files list when a folder is selected."""
        folder = Path(path)
        if not folder.is_dir():
            return

        self._label_files = sorted(
            [
                p
                for p in folder.iterdir()
                if p.suffix.lower() in (".tif", ".tiff") and not p.name.startswith(".")
            ],
            key=lambda p: natural_sort_key(p.name),
        )
        self._available_list.clear()
        for f in self._label_files:
            self._available_list.addItem(f.name)

        self._update_available_list_states()
        self._auto_assign_labels()

    def _on_well_selection_changed(self) -> None:  # pragma: no cover
        """Update the FOV list when a well is selected in the plate map."""
        if not self._plate_view:
            return

        self._fov_list.clear()

        selected = self._plate_view._selected_items
        if not selected or len(selected) != 1:
            return

        well_item = next(iter(selected))
        position = well_item.data(DATA_POSITION)
        if not position or not hasattr(position, "name") or not position.name:
            return

        well_name = position.name
        if well_name not in self._well_fovs:
            return

        for fov_id, fov_name, pos_idx in self._well_fovs[well_name]:
            assigned = self._label_map.get(fov_id)
            item = QTreeWidgetItem(
                [f"{fov_name} (pos {pos_idx})", assigned.name if assigned else ""]
            )
            item.setData(0, _FOV_ID_ROLE, fov_id)
            item.setData(0, _FOV_NAME_ROLE, fov_name)
            item.setData(0, _FOV_POS_IDX_ROLE, pos_idx)
            self._fov_list.addTopLevelItem(item)

        self._fov_list.resizeColumnToContents(0)

    def _on_assign(self) -> None:  # pragma: no cover
        """Assign the selected label file to the selected FOV."""
        selected_file_items = self._available_list.selectedItems()
        if not selected_file_items:
            return
        filename = selected_file_items[0].text()

        # Find full path
        label_path: Path | None = None
        for p in self._label_files:
            if p.name == filename:
                label_path = p
                break
        if label_path is None:
            return

        # Get selected FOV
        selected_fov_items = self._fov_list.selectedItems()
        if not selected_fov_items:
            return
        fov_item = selected_fov_items[0]
        fov_id = fov_item.data(0, _FOV_ID_ROLE)
        if fov_id is None:
            return

        self._label_map[fov_id] = label_path
        fov_item.setText(1, label_path.name)
        self._update_available_list_states()

    def _on_unassign(self) -> None:  # pragma: no cover
        """Remove the label assignment from the selected FOV."""
        selected_fov_items = self._fov_list.selectedItems()
        if not selected_fov_items:
            return
        fov_item = selected_fov_items[0]
        fov_id = fov_item.data(0, _FOV_ID_ROLE)
        if fov_id is None:
            return

        if fov_id in self._label_map:
            del self._label_map[fov_id]
            fov_item.setText(1, "")
            self._update_available_list_states()

    def _on_reset(self) -> None:  # pragma: no cover
        """Clear all label-to-FOV assignments."""
        self._label_map.clear()
        # Refresh FOV list for currently selected well
        self._on_well_selection_changed()
        self._update_available_list_states()

    def _on_ok(self) -> None:  # pragma: no cover
        """Validate and import labels to the database."""
        if not self._label_map:
            show_error_dialog(
                self,
                "No label files have been assigned to any FOV.\n"
                "Please assign at least one label file before clicking OK.",
            )
            return

        # Disable UI and show loading bar
        self.setEnabled(False)
        self._loading_bar = _ProgressBarWidget(
            self, text="Importing Labels to the Database..."
        )
        self._loading_bar.show_progress_bar(False)
        self._loading_bar.show()
        # Needed for windows
        QApplication.processEvents()

        self._worker = create_worker(
            import_labels_to_database,
            self._database_path,
            self.value(),
            _start_thread=True,
            _connect={
                "returned": self._on_import_finished,
                "errored": self._on_import_errored,
            },
        )

    def _on_import_finished(self, det_id: int) -> None:  # pragma: no cover
        """Handle successful import."""
        self._loading_bar.hide()
        self.setEnabled(True)
        self._imported_detection_settings_id = det_id
        self.accept()

    def _on_import_errored(self, error: Exception) -> None:  # pragma: no cover
        """Handle import failure."""
        self._loading_bar.hide()
        self.setEnabled(True)
        show_error_dialog(self, f"Failed to import labels:\n{error}")
        cali_logger.error(f"Failed to import labels: {error}")

    # --- Private: auto-assignment ---

    def _auto_assign_labels(self) -> None:
        """Attempt to auto-match label files to FOVs by filename."""
        if not self._label_files:
            return

        # Collect all FOV names -> fov_id from _well_fovs
        fov_info: dict[str, int] = {}
        for fovs in self._well_fovs.values():
            for fov_id, fov_name, _pos_idx in fovs:
                fov_info[fov_name] = fov_id

        matches = auto_match_files(self._label_files, list(fov_info.keys()))

        for fov_name, matched_path in matches.items():
            fov_id = fov_info[fov_name]
            self._label_map[fov_id] = matched_path

        # Refresh FOV list display if a well is selected
        self._on_well_selection_changed()
        self._update_available_list_states()

    # --- Private: data ---

    def _load_fovs_from_db(self) -> None:
        """Query the database for plate info and wells/FOVs."""
        self._well_fovs.clear()

        try:
            from sqlmodel import Session, create_engine, select

            from cali.sqlmodel._model import FOV, Plate, Well

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                echo=False,
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    # Load plate type and set on widget
                    plate_type = session.exec(select(Plate.plate_type)).first()
                    if plate_type:
                        # Set combo box text directly (setValue expects WellPlatePlan)
                        idx = self._plate_widget.plate_name.findText(plate_type)
                        if idx >= 0:
                            self._plate_widget.plate_name.setCurrentIndex(idx)

                    # Load FOVs grouped by well
                    stmt = (
                        select(Well.name, FOV.name, FOV.position_index, FOV.id)
                        .join(FOV, FOV.well_id == Well.id)
                        .order_by(Well.name, FOV.fov_number)
                    )
                    rows = session.exec(stmt).all()
            finally:
                engine.dispose(close=True)

            for well_name, fov_name, pos_idx, fov_id in rows:
                if well_name not in self._well_fovs:
                    self._well_fovs[well_name] = []
                self._well_fovs[well_name].append((fov_id, fov_name, pos_idx))

        except Exception as e:  # pragma: no cover
            cali_logger.error(f"Failed to query FOVs from database: {e}")

    def _update_available_list_states(self) -> None:  # pragma: no cover
        """Grey out files that are already assigned."""
        assigned_names = {p.name for p in self._label_map.values()}
        for i in range(self._available_list.count()):
            item = self._available_list.item(i)
            if item is None:
                continue
            if item.text() in assigned_names:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
