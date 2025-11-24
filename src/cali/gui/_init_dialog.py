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
        self.setWindowTitle("Select Data Source")

        # Create tab widget
        self._tab_widget = QTabWidget()

        # ===== First Tab: From Directories =====
        directories_tab = QWidget()
        directories_layout = QGridLayout(directories_tab)
        directories_layout.setContentsMargins(5, 5, 5, 5)
        directories_layout.setSpacing(5)

        # datastore_path
        self._browse_data = _BrowseWidget(
            directories_tab,
            "Data Path",
            data_path,
            "The path to the zarr datastore.",
        )

        # output_path
        self._browse_output = _BrowseWidget(
            directories_tab,
            "Output Path",
            output_path,
            "The path to the directory where to save the analysis database.",
            is_dir=True,
        )

        # database_name field
        db_name_widget = QWidget(directories_tab)
        db_name_layout = QHBoxLayout(db_name_widget)
        db_name_layout.setContentsMargins(0, 0, 0, 0)
        db_name_layout.setSpacing(5)

        db_name_label = QLabel("Database Name:")
        self._database_name_le = QLineEdit()
        self._database_name_le.setPlaceholderText(DEFAULT_CALI_DB_NAME)
        self._database_name_le.setText(database_name or DEFAULT_CALI_DB_NAME)

        db_name_layout.addWidget(db_name_label)
        db_name_layout.addWidget(self._database_name_le)

        # styling
        fix_width = db_name_label.minimumSizeHint().width()
        self._browse_data._label.setFixedWidth(fix_width)
        self._browse_output._label.setFixedWidth(fix_width)

        directories_layout.addWidget(self._browse_data, 0, 0)
        directories_layout.addWidget(self._browse_output, 1, 0)
        directories_layout.addWidget(db_name_widget, 2, 0)
        directories_layout.setRowStretch(3, 1)

        # ===== Second Tab: From Database =====
        database_tab = QWidget()
        database_layout = QGridLayout(database_tab)
        database_layout.setContentsMargins(5, 5, 5, 5)
        database_layout.setSpacing(5)

        # data_path for database tab
        self._browse_data_db = _BrowseWidget(
            database_tab,
            "Data Path",
            data_path,
            "The path to the zarr datastore.",
        )

        # database_path
        self._browse_database = _BrowseWidget(
            database_tab,
            "Database Path",
            database_path,
            "The path to the .cali database file.",
            is_dir=False,
        )

        # styling for database tab
        fix_width_db = self._browse_database._label.minimumSizeHint().width()
        self._browse_data_db._label.setFixedWidth(fix_width_db)

        database_layout.addWidget(self._browse_data_db, 0, 0)
        database_layout.addWidget(self._browse_database, 1, 0)
        database_layout.setRowStretch(2, 1)

        # Add tabs
        self._tab_widget.addTab(directories_tab, "From Directories")
        self._tab_widget.addTab(database_tab, "From Database")

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

    def value(self) -> InputDialogData:
        """Return paths based on selected tab.

        Returns
        -------
        InputDialogData
            The output dialog containing the selected paths.
        """
        # from Directories
        if self._tab_widget.currentIndex() == 0:
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
        # from Database
        else:
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
