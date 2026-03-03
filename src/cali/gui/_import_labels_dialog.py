"""Dialog for importing pre-existing label TIFFs and assigning them to FOVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
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

from cali.gui._util import _BrowseWidget, show_error_dialog
from cali.logger import cali_logger

# Role for storing FOV data in tree items
_FOV_ID_ROLE = Qt.ItemDataRole.UserRole
_FOV_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
_FOV_POS_IDX_ROLE = Qt.ItemDataRole.UserRole + 2


class _ImportLabelsDialog(QDialog):
    """Dialog for importing label TIFFs and assigning them to specific FOVs.

    The dialog queries the database for the existing well/FOV structure and
    allows the user to browse for label TIFF files, then assign each label
    file to a specific FOV. On acceptance, label arrays are read, converted
    to ROI/Mask objects, and committed to the database immediately.
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

        # --- Browse widget for labels folder ---
        self._browse_wdg = _BrowseWidget(
            self,
            label="Labels Folder",
            tooltip="Select a folder containing label TIFF files.",
        )
        self._browse_wdg.pathSet.connect(self._on_folder_selected)

        # --- Left panel: available label files ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        left_layout.addWidget(QLabel("Available Label Files:"))
        self._available_list = QListWidget()
        self._available_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        left_layout.addWidget(self._available_list)

        # --- Right panel: FOV tree from database ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        right_layout.addWidget(QLabel("Experiment FOVs:"))
        self._fov_tree = QTreeWidget()
        self._fov_tree.setHeaderLabels(["Well / FOV", "Assigned Label"])
        self._fov_tree.setColumnCount(2)
        self._fov_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        right_layout.addWidget(self._fov_tree)

        # --- Assign / Unassign buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(5)
        self._assign_btn = QPushButton("Assign to FOV →")
        self._assign_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._assign_btn.clicked.connect(self._on_assign)
        self._unassign_btn = QPushButton("← Unassign")
        self._unassign_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._unassign_btn.clicked.connect(self._on_unassign)
        btn_layout.addWidget(self._assign_btn)
        btn_layout.addWidget(self._unassign_btn)

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

        # --- Splitter for left/right panels ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(self._browse_wdg)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(splitter, 1)
        main_layout.addLayout(bottom_layout)

        self.resize(800, 500)

        # Populate the FOV tree from the database
        self._populate_fov_tree()

    # --- Public ---

    def value(self) -> int | None:
        """Return the detection_settings_id after successful import."""
        return self._imported_detection_settings_id

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
            ]
        )
        self._available_list.clear()
        for f in self._label_files:
            self._available_list.addItem(f.name)

        self._update_available_list_states()

    def _on_assign(self) -> None:
        """Assign the selected label file to the selected FOV."""
        # Get selected file
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

        # Get selected FOV in tree (must be a child item, not a well group)
        selected_tree_items = self._fov_tree.selectedItems()
        if not selected_tree_items:
            return
        tree_item = selected_tree_items[0]
        fov_id = tree_item.data(0, _FOV_ID_ROLE)
        if fov_id is None:
            # User selected a well group header, not a FOV
            return

        # Assign
        self._label_map[fov_id] = label_path
        tree_item.setText(1, label_path.name)
        self._update_available_list_states()

    def _on_unassign(self) -> None:
        """Remove the label assignment from the selected FOV."""
        selected_tree_items = self._fov_tree.selectedItems()
        if not selected_tree_items:
            return
        tree_item = selected_tree_items[0]
        fov_id = tree_item.data(0, _FOV_ID_ROLE)
        if fov_id is None:
            return

        if fov_id in self._label_map:
            del self._label_map[fov_id]
            tree_item.setText(1, "")
            self._update_available_list_states()

    def _on_ok(self) -> None:
        """Validate and import labels to the database."""
        if not self._label_map:
            show_error_dialog(
                self,
                "No label files have been assigned to any FOV.\n"
                "Please assign at least one label file before clicking OK.",
            )
            return

        try:
            det_id = self._import_labels_to_database()
            self._imported_detection_settings_id = det_id
            self.accept()
        except Exception as e:
            show_error_dialog(self, f"Failed to import labels:\n{e}")
            cali_logger.error(f"Failed to import labels: {e}")

    # --- Private: data ---

    def _populate_fov_tree(self) -> None:
        """Query the database for wells/FOVs and populate the tree widget."""
        self._fov_tree.clear()

        try:
            from sqlmodel import Session, create_engine, select

            from cali.sqlmodel._model import FOV, Well

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                echo=False,
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    stmt = (
                        select(Well.name, FOV.name, FOV.position_index, FOV.id)
                        .join(FOV, FOV.well_id == Well.id)
                        .order_by(Well.name, FOV.fov_number)
                    )
                    rows = session.exec(stmt).all()
            finally:
                engine.dispose(close=True)

            # Group by well name
            well_items: dict[str, QTreeWidgetItem] = {}
            for well_name, fov_name, pos_idx, fov_id in rows:
                if well_name not in well_items:
                    well_item = QTreeWidgetItem(self._fov_tree, [f"Well {well_name}"])
                    well_item.setExpanded(True)
                    well_items[well_name] = well_item

                fov_item = QTreeWidgetItem(
                    well_items[well_name],
                    [f"{fov_name} (pos {pos_idx})", ""],
                )
                fov_item.setData(0, _FOV_ID_ROLE, fov_id)
                fov_item.setData(0, _FOV_NAME_ROLE, fov_name)
                fov_item.setData(0, _FOV_POS_IDX_ROLE, pos_idx)

            self._fov_tree.resizeColumnToContents(0)

        except Exception as e:
            cali_logger.error(f"Failed to query FOVs from database: {e}")

    def _update_available_list_states(self) -> None:
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

    def _import_labels_to_database(self) -> int:
        """Read label TIFFs and commit ROI/Mask objects to the database.

        Returns
        -------
        int
            The detection_settings_id for the imported labels.
        """
        import tifffile
        from sqlmodel import Session, create_engine, select

        from cali.sqlmodel._model import (
            FOV,
            ROI,
            DetectionSettings,
            Experiment,
            Mask,
        )
        from cali.util import commit_fov_result, mask_to_coordinates

        engine = create_engine(
            f"sqlite:///{self._database_path}",
            echo=False,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )

        try:
            with Session(engine) as session:
                # 1. Get or create DetectionSettings(method="imported")
                det_settings = DetectionSettings(method="imported")

                existing = session.exec(
                    select(DetectionSettings).where(
                        DetectionSettings.method == "imported"
                    )
                ).all()
                matched = None
                for candidate in existing:
                    if candidate == det_settings:
                        matched = candidate
                        break

                if matched is not None:
                    det_settings = matched
                else:
                    session.add(det_settings)
                    session.flush()

                det_id = det_settings.id
                assert det_id is not None

                # 2. Load experiment
                experiment_row = session.exec(select(Experiment)).first()
                if experiment_row is None:
                    raise ValueError("No experiment found in database.")

                experiment = Experiment.load_from_database(
                    self._database_path, load_data=False
                )

                # 3. For each assigned label, read TIFF and create FOV with ROIs
                for fov_id, label_path in self._label_map.items():
                    # Get FOV info from DB
                    fov_row = session.get(FOV, fov_id)
                    if fov_row is None:
                        cali_logger.warning(
                            f"FOV id={fov_id} not found in DB, skipping."
                        )
                        continue

                    # Read label TIFF
                    label_array = tifffile.imread(str(label_path))
                    if label_array.ndim != 2:
                        cali_logger.warning(
                            f"Label file {label_path.name} is not 2D "
                            f"(shape={label_array.shape}), skipping."
                        )
                        continue

                    # Get unique labels (excluding background 0)
                    label_values = np.unique(label_array)
                    label_values = label_values[label_values > 0]

                    if len(label_values) == 0:
                        cali_logger.warning(
                            f"No labels found in {label_path.name}, skipping."
                        )
                        continue

                    # Create FOV result with ROIs
                    fov_result = FOV(
                        name=fov_row.name,
                        position_index=fov_row.position_index,
                        fov_number=fov_row.fov_number,
                        rois=[],
                    )

                    for lv in label_values:
                        roi_mask_binary = label_array == lv
                        mask_coords, mask_shape = mask_to_coordinates(roi_mask_binary)
                        mask_obj = Mask(
                            coords_y=mask_coords[0],
                            coords_x=mask_coords[1],
                            height=mask_shape[0],
                            width=mask_shape[1],
                            mask_type="roi",
                        )
                        roi = ROI(
                            label_value=int(lv),
                            active=None,
                            stimulated=None,
                            roi_mask=mask_obj,
                            fov_id=0,  # placeholder, set by commit_fov_result
                        )
                        fov_result.rois.append(roi)

                    commit_fov_result(
                        session=session,
                        experiment=experiment,
                        fov_result=fov_result,
                        detection_settings_id=det_id,
                        commit=False,
                    )

                session.commit()
                cali_logger.info(
                    f"Imported labels for {len(self._label_map)} FOV(s) "
                    f"(detection_settings_id={det_id})"
                )

        finally:
            engine.dispose(close=True)

        return det_id
