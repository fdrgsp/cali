from __future__ import annotations

import os
from typing import NamedTuple

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._util import _BrowseWidget


class InputDialogData(NamedTuple):
    data_path: str | None = None
    labels_path: str | None = None
    analysis_path: str | None = None
    database_path: str | None = None


class _InputDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        data_path: str | None = None,
        labels_path: str | None = None,
        analysis_path: str | None = None,
        database_path: str | None = None,
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

        # analysis_path with json files
        self._browse_analysis = _BrowseWidget(
            directories_tab,
            "Analysis Path",
            analysis_path,
            "The path to the analysis where to save the analysis database.",
            is_dir=True,
        )

        # labels_path with labels images
        self._browse_labels = _BrowseWidget(
            directories_tab,
            "Labels Path",
            labels_path,
            "The path to the labels images. The images should be tif files and "
            "their name should correspond to the data files (e.g. C3_0000_p0.tif).",
        )
        # styling
        fix_width = self._browse_analysis._label.minimumSizeHint().width()
        self._browse_data._label.setFixedWidth(fix_width)
        self._browse_labels._label.setFixedWidth(fix_width)

        directories_layout.addWidget(self._browse_data, 0, 0)
        directories_layout.addWidget(self._browse_analysis, 1, 0)
        directories_layout.addWidget(self._browse_labels, 2, 0)
        directories_layout.setRowStretch(3, 1)

        # ===== Second Tab: From Database =====
        database_tab = QWidget()
        database_layout = QVBoxLayout(database_tab)
        database_layout.setContentsMargins(5, 5, 5, 5)
        database_layout.setSpacing(5)

        # database_path
        self._browse_database = _BrowseWidget(
            database_tab,
            "Database Path",
            database_path,
            "The path to the .cali database file.",
            is_dir=False,
        )

        database_layout.addWidget(self._browse_database)
        database_layout.addStretch()

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
        OutputDialog
            The output dialog containing the selected paths.
        """
        # from Directories
        if self._tab_widget.currentIndex() == 0:
            datastore_path = self._browse_data.value()
            labels_path = self._browse_labels.value()
            analysis_path = self._browse_analysis.value()

            return InputDialogData(
                data_path=(
                    os.path.normpath(datastore_path) if datastore_path else None
                ),
                labels_path=(os.path.normpath(labels_path) if labels_path else None),
                analysis_path=(
                    os.path.normpath(analysis_path) if analysis_path else None
                ),
                database_path=None,
            )
        # from Database
        else:
            database_path = self._browse_database.value()
            return InputDialogData(
                database_path=(
                    os.path.normpath(database_path) if database_path else None
                ),
            )
