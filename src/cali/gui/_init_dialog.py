from __future__ import annotations

import os
from typing import NamedTuple

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cali._constants import DEFAULT_CALI_DB_NAME

from ._settings_tabs import guidance
from ._util import _BrowseWidget


class InputDialogData(NamedTuple):
    data_path: str | None = None
    output_path: str | None = None
    database_path: str | None = None
    database_name: str | None = None


class _InputDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        data_path: str | None = None,
        output_path: str | None = None,
        database_path: str | None = None,
        database_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open experiment")

        # Create tab widget
        self._tab_widget = QTabWidget()

        # ===== First Tab: From Database =====
        database_tab = QWidget()
        database_layout = QGridLayout(database_tab)
        database_layout.setContentsMargins(5, 5, 5, 5)
        database_layout.setSpacing(5)

        # database_path
        self._browse_database = _BrowseWidget(
            database_tab,
            "Saved database",
            database_path,
            "The path to the .cali database file.",
            is_dir=False,
        )

        # data_path for database tab (optional)
        self._browse_data_db = _BrowseWidget(
            database_tab,
            "Imaging data (optional)",
            data_path,
            "Add the original imaging data to detect ROIs or extract new traces.",
        )

        # styling for database tab
        fix_width_db = max(
            self._browse_database._label.minimumSizeHint().width(),
            self._browse_data_db._label.minimumSizeHint().width(),
        )
        self._browse_database._label.setFixedWidth(fix_width_db)
        self._browse_data_db._label.setFixedWidth(fix_width_db)

        # optional legend
        optional_label = guidance(
            "Open a saved experiment to review results, re-analyze stored traces "
            "or export data. Imaging data are needed for new detection or extraction."
        )

        database_layout.addWidget(optional_label, 0, 0)
        database_layout.addWidget(self._browse_database, 1, 0)
        database_layout.addWidget(self._browse_data_db, 2, 0)
        database_layout.setRowStretch(3, 1)

        # ===== Second Tab: From Directories =====
        directories_tab = QWidget()
        directories_layout = QGridLayout(directories_tab)
        directories_layout.setContentsMargins(5, 5, 5, 5)
        directories_layout.setSpacing(5)

        # datastore_path
        self._browse_data = _BrowseWidget(
            directories_tab,
            "Imaging data",
            data_path,
            "The path to the data. It can be a directory containing tiff files or a "
            "zarr datastore.",
        )

        # output_path
        self._browse_output = _BrowseWidget(
            directories_tab,
            "Save results in",
            output_path,
            "The path to the directory where to save the analysis database.",
            is_dir=True,
        )

        # database_name field
        db_name_widget = QWidget(directories_tab)
        db_name_layout = QHBoxLayout(db_name_widget)
        db_name_layout.setContentsMargins(0, 0, 0, 0)
        db_name_layout.setSpacing(5)

        db_name_label = QLabel("Database filename:")
        self._database_name_le = QLineEdit()
        self._database_name_le.setPlaceholderText(DEFAULT_CALI_DB_NAME)
        self._database_name_le.setText(database_name or DEFAULT_CALI_DB_NAME)

        db_name_layout.addWidget(db_name_label)
        db_name_layout.addWidget(self._database_name_le)

        # styling
        fix_width = db_name_label.minimumSizeHint().width()
        self._browse_data._label.setFixedWidth(fix_width)
        self._browse_output._label.setFixedWidth(fix_width)

        directories_layout.addWidget(
            guidance(
                "Start from a folder of TIFF recordings or a Zarr datastore. "
                "Choose where to save the experiment database and its results."
            ),
            0,
            0,
        )
        directories_layout.addWidget(self._browse_data, 1, 0)
        directories_layout.addWidget(self._browse_output, 2, 0)
        directories_layout.addWidget(db_name_widget, 3, 0)
        directories_layout.setRowStretch(4, 1)

        # Add tabs
        self._tab_widget.addTab(database_tab, "Saved experiment")
        self._tab_widget.addTab(directories_tab, "New experiment")

        # Create the button box
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        # Connect the signals
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._tab_widget)
        layout.addWidget(self.buttonBox)

        self._browse_database._path.setFocus()

    def value(self) -> InputDialogData:
        """Return paths based on selected tab.

        Returns
        -------
        InputDialogData
            The output dialog containing the selected paths.
        """
        # from Database
        if self._tab_widget.currentIndex() == 0:
            datastore_path = self._browse_data_db.value()
            database_path = self._browse_database.value()
            return InputDialogData(
                data_path=(
                    os.path.normpath(datastore_path) if datastore_path else None
                ),
                output_path=None,
                database_path=(
                    os.path.normpath(database_path) if database_path else None
                ),
                database_name=None,
            )
        # from Directories
        else:
            datastore_path = self._browse_data.value()
            output_path = self._browse_output.value()
            database_name = (
                self._database_name_le.text().strip() or DEFAULT_CALI_DB_NAME
            )

            return InputDialogData(
                data_path=(
                    os.path.normpath(datastore_path) if datastore_path else None
                ),
                output_path=(os.path.normpath(output_path) if output_path else None),
                database_path=None,
                database_name=database_name,
            )
