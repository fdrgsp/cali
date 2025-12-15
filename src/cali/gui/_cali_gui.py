from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import tifffile
import useq
from ndv import NDViewer
from pymmcore_widgets.useq_widgets._well_plate_widget import (
    DATA_POSITION,
    WellPlateView,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QCloseEvent
from qtpy.QtWidgets import (
    QAbstractGraphicsShapeItem,
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker
from tqdm import tqdm

from cali._constants import (
    DEFAULT_CALI_DB_NAME,
    EVENT_KEY,
    OME_ZARR,
    PYMMCW_METADATA_KEY,
    UNSELECTABLE_COLOR,
    WRITERS,
    ZARR_TESNSORSTORE,
)
from cali.gui._analysis_gui import (
    AnalysisSettingsData,
    CalciumPeaksData,
    ExperimentTypeData,
    SpikeData,
)
from cali.gui._detection_gui import CellposeSettingsData
from cali.gui._extraction_gui import (
    ExtractionSettingsData,
)
from cali.gui._runs_panel import _RunsPanel
from cali.runner._cali_runner import CaliRunner
from cali.sqlmodel import (
    Experiment,
    experiment_to_plate_map_data,
    has_experiment_analysis,
    has_fov_analysis,
    save_experiment_to_database,
)
from cali.sqlmodel._model import AnalysisSettings, CaliResult, DetectionSettings
from cali.util import load_data_from_path

from ._analysis_gui import _AnalysisGUI
from ._detection_gui import _DetectionGUI
from ._extraction_gui import MetadataData, TraceExtractionData, _ExtractionGUI
from ._fov_table import WellInfo, _FOVTable
from ._image_viewer import _ImageViewer
from ._init_dialog import _InputDialog
from ._plate_map import _PlateMapWidget
from ._plate_plan_wizard import PlatePlanWizard
from ._pygraph_plot_widgets import _MultilWellGraphWidget, _SingleWellGraphWidget
from ._run_selection_dialog import RunSelectionDialog
from ._run_widget import CaliRunSettings, _RunCaliWidget
from ._save_as_widgets import _SaveAsCSV, _SaveAsTiff
from ._tiff_collection_widget import TiffCollectionWidget
from ._util import (
    _ElapsedTimer,
    _ProgressBarWidget,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator


from cali.logger import cali_logger
from cali.readers import (
    OMEZarrReader,
    TensorstoreZarrReader,
    TiffCollectionReader,
)


class CaliGui(QMainWindow):
    """A widget for displaying a plate preview."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("cali")

        # ELAPSED TIMER ---------------------------------------------------------------
        self._elapsed_timer = _ElapsedTimer()

        # INTERNAL VARIABLES ---------------------------------------------------------
        self._database_path: str | None = None
        self._data_path: str | None = None
        self._output_path: str | None = None
        self._data: (
            TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader | None
        ) = None

        # RUNNER ----------------------------------------------------------------------
        self._runner = CaliRunner()

        # PROGRESS BAR WIDGET --------------------------------------------------------
        self._loading_bar = _ProgressBarWidget(self)

        # MENU BAR -------------------------------------------------------------------
        self.menu_bar = QMenuBar(self)
        self.file_menu = cast("QMenu", self.menu_bar.addMenu("File"))
        open_action = QAction("Select Data Source...", self)
        open_action.setToolTip(
            "Open a dialog to select zarr datastore and analysis database location."
        )
        open_action.triggered.connect(self._show_data_input_dialog)
        save_as_tiff_action = QAction("Save Data as Tiff...", self)
        save_as_tiff_action.triggered.connect(self._show_save_as_tiff_dialog)
        # save_as_csv_action = QAction("Save Analysis Data as CSV...", self)
        # save_as_csv_action.triggered.connect(self._show_save_as_csv_dialog)
        self.file_menu.addAction(open_action)
        self.file_menu.addAction(save_as_tiff_action)
        # self.file_menu.addAction(save_as_csv_action)
        self.setMenuBar(self.menu_bar)

        # TIFF COLLECTION WIDGET ------------------------------------------------------
        self._tiff_collection_widget = TiffCollectionWidget(parent=self)
        self._tiff_collection_widget.hide()

        # PLATE PLAN WIZARD -----------------------------------------------------------
        self._plate_plan_wizard = PlatePlanWizard(self)
        self._plate_plan_wizard.hide()
        self._default_plate_plan: bool = False

        # PLATE VIEW ------------------------------------------------------------------
        self._plate_view = WellPlateView()
        self._plate_view.setDragMode(WellPlateView.DragMode.NoDrag)
        self._plate_view.setSelectionMode(WellPlateView.SelectionMode.SingleSelection)

        # PLATE MAP WIDGET ------------------------------------------------------------
        self._plate_map_wdg = _PlateMapWidget(self)

        # TABLE FOR THE FIELDS OF VIEW ------------------------------------------------
        self._fov_table = _FOVTable(self)

        # IMAGE VIEWER ----------------------------------------------------------------
        self._image_viewer = _ImageViewer(self)
        self._image_viewer.valueChanged.connect(self._update_graphs_with_roi)

        # LEFT WIDGETS ----------------------------------------------------------------

        # SPLITTER FOR THE PLATE MAP AND THE FOV TABLE --------------------------------
        top_wdg = QWidget()
        top = QVBoxLayout(top_wdg)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._plate_view)
        top.addWidget(self._plate_map_wdg)

        self.splitter_top_left = QSplitter(
            parent=self, orientation=Qt.Orientation.Vertical
        )
        self.splitter_top_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_top_left.setChildrenCollapsible(False)
        self.splitter_top_left.addWidget(top_wdg)
        self.splitter_top_left.addWidget(self._fov_table)
        top_left_group = QGroupBox()
        top_left_layout = QVBoxLayout(top_left_group)
        top_left_layout.setContentsMargins(10, 10, 10, 10)
        top_left_layout.addWidget(self.splitter_top_left)

        # SPLITTER FOR THE PLATE MAP/FOV TABLE AND THE IMAGE VIEWER -------------------
        self.splitter_bottom_left = QSplitter(
            parent=self, orientation=Qt.Orientation.Vertical
        )
        self.splitter_bottom_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_bottom_left.setChildrenCollapsible(False)
        self.splitter_bottom_left.addWidget(top_left_group)
        self.splitter_bottom_left.addWidget(self._image_viewer)

        # RIGHT WIDGETS ---------------------------------------------------------------

        # MAIN TABS: Detection & Analysis | Visualization ----------
        self._main_tab = QTabWidget(self)
        self._main_tab.currentChanged.connect(self._on_tab_changed)

        # DETECTION AND EXTRACTION TAB --------------------------------
        self._detection_extraction_tab = QWidget()
        self._main_tab.addTab(
            self._detection_extraction_tab, "Detection, Extraction and Analysis"
        )
        detection_extraction_layout = QVBoxLayout(self._detection_extraction_tab)
        detection_extraction_layout.setContentsMargins(5, 0, 5, 0)
        detection_extraction_layout.setSpacing(5)

        # SUB-TABS FOR DETECTION AND EXTRACTION ----------------------------------------
        self._sub_tab = QTabWidget(self)
        self._sub_tab.setTabPosition(QTabWidget.TabPosition.North)

        # DETECTION SUB-TAB -----------------------------------------------------------
        self._detection_tab = QWidget()
        self._sub_tab.addTab(self._detection_tab, "Detection")
        detection_tab_layout = QVBoxLayout(self._detection_tab)
        detection_tab_layout.setContentsMargins(5, 5, 5, 5)

        self._detection_wdg = _DetectionGUI(self)
        detection_tab_layout.addWidget(self._detection_wdg)

        # EXTRACTION SUB-TAB ----------------------------------------------------------
        self._extraction_tab = QWidget()
        self._sub_tab.addTab(self._extraction_tab, "Extraction")
        extraction_tab_layout = QVBoxLayout(self._extraction_tab)
        extraction_tab_layout.setContentsMargins(5, 5, 5, 5)

        self._extraction_wdg = _ExtractionGUI(self)
        extraction_tab_layout.addWidget(self._extraction_wdg)

        # ANALYSIS SUB-TAB ------------------------------------------------------------
        self._analysis_tab = QWidget()
        self._sub_tab.addTab(self._analysis_tab, "Analysis")
        analysis_tab_layout = QVBoxLayout(self._analysis_tab)
        analysis_tab_layout.setContentsMargins(5, 5, 5, 5)

        self._analysis_wdg = _AnalysisGUI(self)
        analysis_tab_layout.addWidget(self._analysis_wdg)

        # Add sub-tabs to the Detection & Extraction tab
        detection_extraction_layout.addWidget(self._sub_tab)

        # SHARED RUN WIDGET -----------------------------------------------------------
        # This widget is shared between Detection and Extraction tabs
        self._run_cali_wdg = _RunCaliWidget(self)
        detection_extraction_layout.addWidget(self._run_cali_wdg)

        # VISUALIZATION TAB -----------------------------------------------------------
        self._visualization_tab = QWidget()
        self._main_tab.addTab(self._visualization_tab, "Visualization")
        visualization_layout = QVBoxLayout(self._visualization_tab)
        visualization_layout.setContentsMargins(5, 5, 5, 5)
        visualization_layout.setSpacing(5)

        # Create sub-tabs for single and multi well visualizations
        self._vis_sub_tab = QTabWidget()
        self._vis_sub_tab.setTabPosition(QTabWidget.TabPosition.North)
        visualization_layout.addWidget(self._vis_sub_tab)

        # SINGLE WELL VISUALIZATION TAB -----------------------------------------------
        self._single_well_vis_tab = QWidget()
        self._vis_sub_tab.addTab(self._single_well_vis_tab, "Single Wells")
        single_well_vis_layout = QVBoxLayout(self._single_well_vis_tab)
        single_well_vis_layout.setContentsMargins(5, 5, 5, 5)
        single_well_vis_layout.setSpacing(5)

        self._single_well_graph_1 = _SingleWellGraphWidget(self)
        self._single_well_graph_2 = _SingleWellGraphWidget(self)

        # Create top widget for graphs 1 and 2 side by side
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        top_layout.addWidget(self._single_well_graph_1)

        # Create vertical splitter between top (graphs 1&2) and graph 3
        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.setContentsMargins(0, 0, 0, 0)
        vertical_splitter.setChildrenCollapsible(True)
        vertical_splitter.addWidget(top_widget)
        vertical_splitter.addWidget(self._single_well_graph_2)

        single_well_vis_layout.addWidget(vertical_splitter)
        self.SW_GRAPHS = [
            self._single_well_graph_1,
            self._single_well_graph_2,
        ]

        # MULTI WELL VISUALIZATION TAB ------------------------------------------------
        self._multi_well_vis_tab = QWidget()
        self._vis_sub_tab.addTab(self._multi_well_vis_tab, "Multi Wells")
        multi_well_layout = QGridLayout(self._multi_well_vis_tab)
        multi_well_layout.setContentsMargins(5, 5, 5, 5)
        multi_well_layout.setSpacing(5)

        self._multi_well_graph_1 = _MultilWellGraphWidget(self)
        self._multi_well_graph_1.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._multi_well_graph_1.setMinimumSize(200, 150)
        multi_well_layout.addWidget(self._multi_well_graph_1, 0, 0)

        self.MW_GRAPHS = [self._multi_well_graph_1]

        # RIGHT SPLITTER --------------------------------------------------------------
        # splitter between the tabs and the runs panel
        self.right_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.right_splitter.setContentsMargins(0, 0, 0, 0)
        self.right_splitter.setChildrenCollapsible(True)
        self.right_splitter.addWidget(self._main_tab)

        # RUNS PANEL -------------------------------------------------------------------
        self._runs_panel = _RunsPanel()
        self.right_splitter.addWidget(self._runs_panel)

        # MAIN SPLITTER---------------------------------------------------------------
        # splitter between the plate map/fov table/image viewer and the graphs
        self.main_splitter = QSplitter(self)
        self.main_splitter.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.splitter_bottom_left)

        self.main_splitter.addWidget(self.right_splitter)

        # CENTRAL WIDGET -------------------------------------------------------------
        self._central_widget = QWidget(self)
        self._central_widget_layout = QVBoxLayout(self._central_widget)
        self._central_widget_layout.setContentsMargins(10, 10, 10, 10)
        self._central_widget_layout.addWidget(self.main_splitter)
        self.setCentralWidget(self._central_widget)

        # CONNECT SIGNALS ------------------------------------------------------------
        self._plate_view.selectionChanged.connect(self._on_scene_well_changed)

        self._fov_table.itemSelectionChanged.connect(
            self._on_fov_table_selection_changed
        )
        self._fov_table.doubleClicked.connect(self._on_fov_double_click)

        self._runs_panel.runSelected.connect(self._on_run_item_selected)
        self._runs_panel.settingsDeleted.connect(self._on_settings_deleted)

        # connect the roiSelected signal from the graphs to the image viewer so we can
        # highlight the roi in the image viewer when a roi is selected in the graph
        for graph in self.SW_GRAPHS:
            graph.roiSelected.connect(self._highlight_roi)

        # connect analysis from metadata button
        self._analysis_wdg.from_metadata.connect(self._on_led_info_from_meta_clicked)  # type: ignore
        # connect extraction from metadata button
        self._extraction_wdg.from_metadata.connect(self._on_extraction_meta_clicked)  # type: ignore
        # connect analysis metadata frame rate button
        self._analysis_wdg.from_metadata_frame_rate.connect(  # type: ignore
            self._on_analysis_meta_clicked
        )

        self._extraction_wdg._metadata_wdg._frame_rate_spin.valueChanged.connect(
            self._on_fps_changed
        )
        self._analysis_wdg._metadata_wdg._frame_rate_spin.valueChanged.connect(
            self._on_fps_changed
        )

        # connect the shared run/cancel buttons to appropriate handlers
        self._run_cali_wdg._run_btn.clicked.connect(self._on_cali_run)
        self._run_cali_wdg._cancel_btn.clicked.connect(self._on_cali_cancel)
        self._run_cali_wdg._save_settings_btn.clicked.connect(self._on_save_settings)
        self._run_cali_wdg._load_settings_btn.clicked.connect(self._on_load_settings)

        self._elapsed_timer.elapsed_time_updated.connect(
            self._run_cali_wdg.set_time_label
        )

        # connect plate map widget to save when OK is clicked
        self._plate_map_wdg.plateMapSaved.connect(self._save_plate_map_to_database)

        # FINALIZE WINDOW ------------------------------------------------------------
        self.showMaximized()
        self._set_splitter_sizes()

        # TO REMOVE, IT IS ONLY TO TEST________________________________________________
        # fmt off

        # data_path = "tests/test_data/evoked/evk.tensorstore.zarr"
        # db_path = "tests/test_data/evoked/results.cali"
        # self._initialize_from_database(db_path, data_path)

        # data_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/"
        # "TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"
        # self._initialize_from_database(db_path, data_path)

        # self._data_path = "tests/test_data/spontaneous/spont.tensorstore.zarr"
        # self._database_path = "tests/test_data/spontaneous/results.cali"
        # self._output_path = "tests/test_data/spontaneous/"

        # self._data_path = "/Users/fdrgsp/Desktop/cali_test/tiffs"
        # self._database_path = "/Users/fdrgsp/Desktop/cali_test/from_tiffs.cali"
        # self._initialize_from_database(self._database_path, self._data_path)

        # USED IN TESTS -------------------------------------------------
        # self._data_path = "tests/test_data/evoked/evk.tensorstore.zarr"
        # self._database_path = "tests/test_data/evoked/results.cali"
        # self._output_path = "tests/test_data/evoked/"

        # self._data_path = "tests/test_data/multi_pos/evk.tensorstore.zarr"
        # self._database_path = "/Users/fdrgsp/Desktop/cali_test/exp.cali"
        # self._output_path = "/Users/fdrgsp/Desktop/cali_test/"

        # self._data_path = "/Users/fdrgsp/Desktop/cali_test/tiffs"
        # self._database_path = "/Users/fdrgsp/Desktop/cali_test/from_tiffs.cali"
        # self._output_path = "/Users/fdrgsp/Desktop/cali_test/"

        # self._database_path = "tests/test_data/multi_pos/result_2pos.cali"
        # self._data_path = "tests/test_data/multi_pos/evk.tensorstore.zarr"
        # self._initialize_from_database(self._database_path, self._data_path)

        # self._database_path = "tests/test_data/multi_pos/result_2pos.cali"
        # self._data_path = "tests/test_data/multi_pos/evk.tensorstore.zarr"
        # self._output_path = "tests/test_data/multi_pos/"

        # self._database_path = "tests/test_data/test_for_plot/result_for_plots.cali"
        # self._data_path = "tests/test_data/test_for_plot/evk.tensorstore.zarr"
        # self._output_path = "tests/test_data/test_for_plot/"

        # self._database_path = "/Users/fdrgsp/Desktop/cali_test/phenix.cali"
        # self._data_path = "/Volumes/T7 Shield/Phenix/out"
        # self._output_path = "/Users/fdrgsp/Desktop/cali_test/"
        # self._initialize_from_directories(
        #     self._data_path, self._output_path, "phenix.cali"
        # )

        # ===========================
        # self._data_path = "tests/test_data/multi_pos/evk.tensorstore.zarr"
        # self._database_path = "tests/test_data/multi_pos/result_2pos.cali"
        # self._output_path = "tests/test_data/multi_pos/"

        self._data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
        self._database_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
        self._output_path = "tests/test_data/data_and_db_for_tests/"

        # self._data_path = "/Users/fdrgsp/Desktop/cali_test/tiffs"
        # self._database_path = "/Users/fdrgsp/Desktop/cali_test/new.cali"
        # self._output_path = "/Users/fdrgsp/Desktop/cali_test/")

        # self._database_path = (
        #     "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results_new.cali"
        # )
        # self._data_path = (
        #     "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/"
        #     "TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"
        # )
        # self._initialize_from_database(self._database_path, self._data_path)

        # fmt: on
        # _____________________________________________________________________________

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Override closeEvent to properly dispose of database connections."""
        # Save plate map data before closing
        try:
            self._save_plate_map_to_database()
        except Exception as e:
            cali_logger.debug(f"Error saving plate map: {e}")

        # Close data reader to release file handles (important for external drives)
        if self._data is not None:
            if hasattr(self._data, "close"):
                try:
                    self._data.close()
                    cali_logger.debug("✅ Data reader closed successfully")
                except Exception as e:
                    cali_logger.debug(f"❌ Error closing data reader: {e}")
            self._data = None

        # Dispose of all graph widget engines
        for sw_graph in self.SW_GRAPHS:
            if sw_graph.engine is not None:
                try:
                    sw_graph.engine.dispose(close=True)
                except Exception as e:
                    cali_logger.debug(f"❌ Error disposing graph engine: {e}")
                sw_graph.engine = None

        for mw_graph in self.MW_GRAPHS:
            if mw_graph.engine is not None:
                try:
                    mw_graph.engine.dispose(close=True)
                except Exception as e:
                    cali_logger.debug(f"❌ Error disposing graph engine: {e}")
                mw_graph.engine = None

        # Force garbage collection to release any remaining file handles
        import gc

        gc.collect()

        # Call parent closeEvent
        super().closeEvent(a0)

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_fps_changed(self, fps: float) -> None:
        """Link frame rate changes between analysis and extraction metadata widgets."""
        self._analysis_wdg._metadata_wdg.setValue(fps)
        self._extraction_wdg._metadata_wdg._frame_rate_spin.setValue(fps)

    def _on_save_settings(self) -> None:
        """Handle saving current run settings."""
        from dataclasses import asdict

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Run Settings", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            full_settings = {
                "detection": asdict(self._detection_wdg.value()),
                "extraction": asdict(self._extraction_wdg.value()),
                "analysis": asdict(self._analysis_wdg.value()),
            }
            import json

            with open(path, "w") as f:
                json.dump(full_settings, f, indent=4)
            cali_logger.info(f"💾 Run settings saved to {path}")

    def _on_load_settings(self) -> None:
        """Handle loading run settings."""
        json_file, _ = QFileDialog.getOpenFileName(
            self, "Load Run Settings", "", "JSON Files (*.json);;All Files (*)"
        )
        if not json_file:
            return

        try:
            import json

            with open(json_file) as f:
                settings = json.load(f)

            # detection
            detection = settings.get("detection", {})
            if detection:
                self._detection_wdg.setValue(CellposeSettingsData(**detection))

            # extraction
            extraction = settings.get("extraction", {})
            ext_settings = extraction.get("trace_extraction_data", {})
            metadata_settings = extraction.get("metadata_data", {})
            if ext_settings or metadata_settings:
                from cali.gui._extraction_gui import MetadataData

                self._extraction_wdg.setValue(
                    ExtractionSettingsData(
                        trace_extraction_data=(
                            TraceExtractionData(**ext_settings)
                            if ext_settings
                            else None
                        ),
                        metadata_data=(
                            MetadataData(**metadata_settings)
                            if metadata_settings
                            else None
                        ),
                    )
                )

            # analysis
            analysis = settings.get("analysis", {})
            calcium_peaks_data = analysis.get("calcium_peaks_data", {})
            spikes_data = analysis.get("spikes_data", {})
            experiment_type_data = analysis.get("experiment_type_data", {})
            self._analysis_wdg.setValue(
                AnalysisSettingsData(
                    calcium_peaks_data=(
                        CalciumPeaksData(**calcium_peaks_data)
                        if calcium_peaks_data
                        else None
                    ),
                    spikes_data=(SpikeData(**spikes_data) if spikes_data else None),
                    experiment_type_data=(
                        ExperimentTypeData(**experiment_type_data)
                        if experiment_type_data
                        else None
                    ),
                )
            )

            cali_logger.info(f"📂 Run settings loaded from {json_file}")

        except Exception as e:
            msg = f"❌ Failed to load settings from {json_file}:\n{e}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)

    def _load_data_or_configure_tiff(
        self, data_path: str
    ) -> tuple[
        TiffCollectionReader | Any | None,
        dict[str, Any] | None,
        str | None,
        dict[str, Any] | None,
    ]:
        """Load data from path or configure TIFF collection if needed.

        Returns
        -------
        tuple
            Tuple of (data, tiff_file_map, tiff_plate_type, tiff_metadata)
        """
        tiff_file_map = tiff_plate_type = tiff_metadata = None
        data: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader | None

        data = load_data_from_path(data_path)

        # if data is None and the data_path is a tiff folder, try to create
        # a TiffCollectionReader
        if data is None:
            d_path = Path(data_path)
            tiff_list = list(d_path.glob("*.tif")) + list(d_path.glob("*.tiff"))
            if tiff_list:
                # Show TiffCollectionWidget to configure TIFF files
                self._tiff_collection_widget.set_tiff_files(tiff_list)
                if self._tiff_collection_widget.exec():
                    data = self._tiff_collection_widget.value()
                    tiff_file_map, tiff_plate_type, tiff_metadata = (
                        data.to_experiment_tiff_config()
                    )
                    cali_logger.info("📋 `TiffCollectionReader` Configured.")
                else:
                    return None, None, None, None
            else:
                msg = (
                    f"❌ No valid data found at {data_path}! "
                    "Expected zarr datastore or TIFF file folder."
                )
                show_error_dialog(self, msg)
                cali_logger.error(msg)
                return None, None, None, None

        return data, tiff_file_map, tiff_plate_type, tiff_metadata

    def _validate_data(self, data: TiffCollectionReader | Any | None) -> bool:
        """Validate data and show errors if invalid.

        Returns
        -------
        bool
            True if data is valid, False otherwise.
        """
        if data is None:
            msg = (
                f"❌ Unsupported file format! Currently, Only "
                f"{WRITERS[ZARR_TESNSORSTORE][0]}, {WRITERS[OME_ZARR][0]} and "
                "TiffCollectionReader are supported."
            )
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            self._loading_bar.hide()
            return False

        if data.sequence is None:
            msg = (
                "❌ useq.MDASequence not found! Cannot use the  `CaliGui` without "
                "the useq.MDASequence in the datastore metadata!"
            )
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            self._loading_bar.hide()
            return False

        return True

    def _finalize_initialization(self, experiment: Experiment) -> None:
        """Finalize GUI initialization with experiment data."""
        # UPDATE GUI-------------------------------------------------------------------
        if experiment.plate is not None:
            plate_plan = experiment.plate.plate_plan
            if plate_plan is not None:
                self._draw_plate_with_selection(plate_plan)
            else:
                cali_logger.warning("❌ Plate plan not found in experiment.")
        else:
            cali_logger.warning("❌ Experiment has no plate.")

        # UPDATE GUI SETTINGS ---------------------------------------------------------
        if self._database_path is not None:
            self._update_gui_settings(self._database_path, experiment=experiment)

        # HIDE LOADING BAR ------------------------------------------------------------
        self._loading_bar.hide()

    def _initialize_from_database(
        self, database_path: str | Path, data_path: str | Path
    ) -> None:
        """Initialize the widget with the given database path."""
        # SHOW LOADING BAR ------------------------------------------------------------
        self._init_loading_bar("📚 Initializing cali from database...", False)

        # CLEARING---------------------------------------------------------------------
        self._clear_widget_before_initialization()

        # CHECK IF DATABASE ACTUALLY EXISTS --------------------------------------------
        if not Path(database_path).exists():
            msg = f"❌ Database file not found at:\n{database_path}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            self._loading_bar.hide()
            return

        # OPEN THE DATABASE -----------------------------------------------------------
        cali_logger.info(f"💿 Loading experiment from database at {database_path}")
        # load the first experiment from the database (there should be only one)
        experiment = Experiment.load_from_database(database_path, load_data=False)

        # DATA-------------------------------------------------------------------------
        tiff_settings = experiment.tiff_collection_settings(data_path)
        if tiff_settings is not None:
            self._data = TiffCollectionReader(tiff_settings)
        else:
            self._data = load_data_from_path(data_path)

        if not self._validate_data(self._data):
            return

        # ASSIGN VARIABLES ------------------------------------------------------------
        self._database_path = str(database_path)
        self._data_path = str(data_path)
        self._output_path = str(Path(database_path).parent)

        # PASS DATABASE PATH TO GRAPHS WIDGETS ----------------------------------------
        self._update_graph_properties(self._database_path)

        # FINALIZE---------------------------------------------------------------------
        self._finalize_initialization(experiment)

    def _initialize_from_directories(
        self,
        data_path: str,
        output_path: str,
        database_name: str = DEFAULT_CALI_DB_NAME,
    ) -> None:
        """Initialize the widget with given datastore and analysis path."""
        # SHOW LOADING BAR ------------------------------------------------------------
        self._init_loading_bar("📂 Initializing cali from directories...", False)

        # CLEARING---------------------------------------------------------------------
        self._clear_widget_before_initialization()

        # ASSIGN VARIABLES ------------------------------------------------------------
        self._data_path = data_path
        if not database_name.endswith(".cali"):
            database_name += ".cali"
        self._database_path = str(Path(output_path) / database_name)
        self._output_path = output_path

        # PASS DATABASE PATH TO GRAPHS WIDGETS ----------------------------------------
        self._update_graph_properties(self._database_path)

        # CHECK IF DATABASE EXISTS ----------------------------------------------------
        if Path(self._database_path).exists():
            # Database exists - ask user if they want to overwrite
            reply = QMessageBox.question(
                self,
                "Database Exists",
                f"Database already exists at:\n{self._database_path}\n\n"
                "Do you want to OVERWRITE it? All existing runs will be deleted.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                # User chose not to overwrite - load existing database
                cali_logger.info(
                    f"💿 Loading existing database at {self._database_path}"
                )

                # OPEN THE DATABASE ---------------------------------------------------
                experiment = Experiment.load_from_database(
                    self._database_path, load_data=False
                )

                # DATA-----------------------------------------------------------------
                tiff_settings = experiment.tiff_collection_settings(data_path)
                if tiff_settings is not None:
                    self._data = TiffCollectionReader(tiff_settings)
                else:
                    self._data = load_data_from_path(data_path)

                if not self._validate_data(self._data):
                    return

            else:
                # User chose to overwrite - create new database
                cali_logger.info(f"💾 Overwriting database at {self._database_path}")

                # DATA-----------------------------------------------------------------
                result = self._load_data_or_configure_tiff(data_path)
                self._data, tiff_file_map, tiff_plate_type, tiff_metadata = result

                if self._data is None:
                    self._loading_bar.hide()
                    return

                # if used micromanager-gui without HCS but with list of positions
                pplan = self._get_plate_plan_if_no_hcs()

                # CREATE AND SAVE EXPERIMENT ------------------------------------------
                experiment = Experiment.create_from_data(
                    name="Cali Experiment",
                    data_path=data_path,
                    description=f"Experiment from data at {data_path}.",
                    tiff_file_map=tiff_file_map,
                    tiff_plate_type=tiff_plate_type,
                    tiff_metadata=tiff_metadata,
                    plate_plan=pplan,
                )
                save_experiment_to_database(
                    experiment, output_path, database_name=database_name, overwrite=True
                )

        else:
            # CREATE NEW DATABASE ------------------------------------------------------
            cali_logger.info(f"💾 Creating new database at {self._database_path}")

            # DATA -----------------------------------------------------------------
            result = self._load_data_or_configure_tiff(data_path)
            self._data, tiff_file_map, tiff_plate_type, tiff_metadata = result

            if self._data is None:
                self._loading_bar.hide()
                return

            # if used micromanager-gui without HCS but with list of positions
            pplan = self._get_plate_plan_if_no_hcs()

            # CREATE AND SAVE EXPERIMENT -------------------------------------------
            experiment = Experiment.create_from_data(
                name="Cali Experiment",
                data_path=data_path,
                description=f"Experiment from data at {data_path}.",
                tiff_file_map=tiff_file_map,
                tiff_plate_type=tiff_plate_type,
                tiff_metadata=tiff_metadata,
                plate_plan=pplan,
            )
            save_experiment_to_database(
                experiment, output_path, database_name=database_name, overwrite=True
            )

        # RELOAD DATA IF NEEDED --------------------------------------------------------
        # skip loading data if already loaded as TiffCollectionReader
        if not isinstance(self._data, TiffCollectionReader):
            self._data = load_data_from_path(data_path)

        if not self._validate_data(self._data):
            return

        # FINALIZE---------------------------------------------------------------------
        self._finalize_initialization(experiment)

    def _get_plate_plan_if_no_hcs(self) -> useq.WellPlatePlan | None:
        """Get plate plan using the plate plan wizard if no HCS is present."""
        if self._data is None or self._data.sequence is None:
            return None
        if isinstance(self._data, TiffCollectionReader):
            return None
        pplan = None
        if not isinstance(self._data.sequence.stage_positions, useq.WellPlatePlan):
            self._plate_plan_wizard.dysplay_available_data_positions(
                len(self._data.sequence.stage_positions)
            )
            if self._plate_plan_wizard.exec():
                pplan = self._plate_plan_wizard.value()
        return pplan

    def _update_gui_settings(
        self, database_path: Path | str, experiment: Experiment | None = None
    ) -> None:
        """Update the GUI settings based on the latest analysis result."""
        # set the database path in the runs panel
        self._runs_panel.set_database_path(database_path)
        # select first run if available
        if self._runs_panel._runs_list.count() > 0:
            # select first run
            self._runs_panel._runs_list.setCurrentRow(0)
            # emit runSelected signal for the first run
            if (first_item := self._runs_panel._runs_list.item(0)) is not None:
                self._runs_panel._on_item_clicked(first_item)
        else:
            # populate detection settings combobox in run widget
            self._populate_settings(database_path)
        # load plate plan data
        if experiment is None:
            experiment = Experiment.load_from_database(database_path, load_data=False)

        if experiment.plate is None:
            msg = "❌ Experiment has no plate."
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            return

        if (plate_plan := experiment.plate.plate_plan) is None:
            msg = "❌ Plate plan not found in experiment."
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            return

        plate = plate_plan.plate
        plate_map_data = experiment_to_plate_map_data(experiment)
        if plate_map_data is not None:
            self._plate_map_wdg.setValue(plate, *plate_map_data)

    def _populate_settings(self, database_path: Path | str) -> None:
        """Populate the settings combobox in the run widget.

        Parameters
        ----------
        database_path : Path | str
            Path to the database
        """
        try:
            # Get current selections to preserve them
            current_value = self._run_cali_wdg.value()
            current_run_option = self._get_run_option(current_value)
            preserve_detection_selection = current_value.detection_settings_id
            preserve_extraction_selection = current_value.extraction_settings_id

            # Get all unique detection settings IDs
            detection_ids = self._runs_panel.get_detection_settings_ids()

            if not detection_ids:
                self._run_cali_wdg.populate_detection_settings([])
                return

            settings_list = []

            # Optimize: Load all settings in one query
            from sqlmodel import Session, create_engine, select

            engine = create_engine(
                f"sqlite:///{database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    statement = select(DetectionSettings).where(
                        DetectionSettings.id.in_(detection_ids)  # type: ignore
                    )
                    query_start = time.perf_counter()
                    results = session.exec(statement).all()
                    query_time = time.perf_counter() - query_start
                    cali_logger.debug(
                        f"DB query: populate detection settings took {query_time:.3f}s "
                        f"(found {len(results)} settings)"
                    )
                    for d_settings in results:
                        if d_settings.id is not None:
                            settings_list.append((d_settings.id, d_settings.method))
            finally:
                engine.dispose(close=True)

            # Sort by ID to maintain order
            settings_list.sort(key=lambda x: x[0])

            self._run_cali_wdg.populate_detection_settings(settings_list)

            # Restore run option selection
            combo = self._run_cali_wdg._run_options_combo
            combo.setCurrentIndex(current_run_option)

            # Restore detection selection if it still exists
            if preserve_detection_selection is not None:
                combo = self._run_cali_wdg._detection_settings_combo
                for i in range(combo.count()):
                    if combo.itemData(i) == preserve_detection_selection:
                        combo.setCurrentIndex(i)
                        break

            # Populate extraction settings
            extraction_ids = self._runs_panel.get_extraction_settings_ids()
            self._run_cali_wdg.populate_extraction_settings(extraction_ids)

            # Restore extraction selection if it still exists
            if preserve_extraction_selection is not None:
                combo = self._run_cali_wdg._extraction_settings_combo
                for i in range(combo.count()):
                    if combo.itemData(i) == preserve_extraction_selection:
                        combo.setCurrentIndex(i)
                        break

        except Exception as e:
            cali_logger.error(f"Failed to populate detection settings: {e}")

    def _get_run_option(self, value: CaliRunSettings) -> int:
        """Get the current run option from the run widget."""
        if value.run_detection and value.run_extraction and value.run_analysis:
            return 0  # Detection, Extraction and Analysis
        if value.run_detection and value.run_extraction and not value.run_analysis:
            return 1  # Detection and Extraction
        if value.run_extraction and value.run_analysis and not value.run_detection:
            return 2  # Extraction and Analysis (require detection)
        if value.run_detection and not value.run_extraction and not value.run_analysis:
            return 3  # Detection Only
        if value.run_extraction and not value.run_detection and not value.run_analysis:
            return 4  # Extraction Only (require detection)
        return 5  # Analysis Only (require detection and extraction)

    def _check_positions_missing_detection(
        self, detection_settings_id: int, positions: list[int]
    ) -> list[int]:
        """Check which positions are missing detection data.

        Parameters
        ----------
        detection_settings_id : int
            Detection settings ID to check
        positions : list[int]
            List of position indices to check

        Returns
        -------
        list[int]
            List of position indices missing detection data
        """
        if not self._database_path:
            return []

        from sqlmodel import Session, create_engine, select

        from cali.sqlmodel._model import FOV, ROI

        engine = create_engine(
            f"sqlite:///{self._database_path}",
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )
        try:
            with Session(engine) as session:
                # Find positions that already have ROIs with this detection
                query_start = time.perf_counter()
                existing_positions = session.exec(
                    select(FOV.position_index)
                    .join(ROI)
                    .where(
                        ROI.detection_settings_id == detection_settings_id,
                        FOV.position_index.in_(positions),  # type: ignore
                    )
                    .distinct()
                ).all()
                query_time = time.perf_counter() - query_start
                cali_logger.debug(
                    f"DB query: check missing detection took {query_time:.3f}s "
                    f"(found {len(existing_positions)} existing)"
                )

                existing_set = set(existing_positions)
                return [p for p in positions if p not in existing_set]
        finally:
            engine.dispose(close=True)

    def _check_positions_missing_extraction(
        self,
        detection_settings_id: int,
        extraction_settings_id: int,
        positions: list[int],
    ) -> list[int]:
        """Check which positions are missing extraction data.

        Parameters
        ----------
        detection_settings_id : int
            Detection settings ID to check
        extraction_settings_id : int
            Extraction settings ID to check
        positions : list[int]
            List of position indices to check

        Returns
        -------
        list[int]
            List of position indices missing extraction data
        """
        if not self._database_path:
            return []

        from sqlmodel import Session, create_engine, select

        from cali.sqlmodel._model import FOV, ROI, CaliResult, Traces

        engine = create_engine(
            f"sqlite:///{self._database_path}",
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )
        try:
            with Session(engine) as session:
                # Find positions that have Traces with this combination
                query_start = time.perf_counter()
                existing_positions = session.exec(
                    select(FOV.position_index)
                    .join(ROI)
                    .join(Traces)
                    .where(
                        ROI.detection_settings_id == detection_settings_id,
                        Traces.analysis_result_id.in_(  # type: ignore
                            select(CaliResult.id).where(
                                CaliResult.extraction_settings_id
                                == extraction_settings_id
                            )
                        ),
                        FOV.position_index.in_(positions),  # type: ignore
                    )
                    .distinct()
                ).all()
                query_time = time.perf_counter() - query_start
                cali_logger.debug(
                    f"DB query: check missing extraction took {query_time:.3f}s "
                    f"(found {len(existing_positions)} existing)"
                )

                existing_set = set(existing_positions)
                return [p for p in positions if p not in existing_set]
        finally:
            engine.dispose(close=True)

    # RUNNING DETECTION OR ANALYSIS ---------------------------------------------------

    def _on_cali_run(self) -> None:
        """Handle run button - routes to detection/analysis based on current tab."""
        if (
            self._data is None
            or self._database_path is None
            or self._data.sequence is None
        ):
            return

        try:
            experiment = Experiment.load_from_database(
                self._database_path, load_data=False
            )

            value = self._run_cali_wdg.value()

            # Get positions list early since we need it for validation
            pos = value.positions or list(
                range(len(self._data.sequence.stage_positions))
            )

            # Track if we've already assigned settings (to prevent overwriting)
            detection_settings = None
            extraction_settings = None

            # Get extraction settings - either from GUI or selected ID (analysis-only)
            if value.run_analysis and not value.run_extraction:
                # Analysis-only mode: use existing extraction settings ID
                extraction_settings_id = value.extraction_settings_id
                detection_settings_id_check = value.detection_settings_id
                if (
                    extraction_settings_id is None
                    or detection_settings_id_check is None
                ):
                    missing = []
                    if detection_settings_id_check is None:
                        missing.append("Detection ID")
                    if extraction_settings_id is None:
                        missing.append("Extraction ID")
                    show_error_dialog(
                        self,
                        f"❌ Please select {' and '.join(missing)} to run "
                        f"analysis-only mode.",
                    )
                    return
                extraction_settings = extraction_settings_id

                # Check if selected positions have both detection and extraction data
                missing_detection = self._check_positions_missing_detection(
                    detection_settings_id_check, pos
                )
                missing_extraction = self._check_positions_missing_extraction(
                    detection_settings_id_check, extraction_settings_id, pos
                )

                if missing_detection or missing_extraction:
                    msg = "Data Missing for Analysis\n\n"
                    if missing_detection:
                        msg = msg + f"Missing detection data: {missing_detection}"
                    if missing_extraction:
                        msg = msg + f"Missing extraction data: {missing_extraction}"

                    msg = msg + (
                        "\n\nDo you want to run the full pipeline (detection + "
                        "extraction + analysis) on these positions?\n"
                        "(If you select 'No', only positions with existing "
                        "data will be processed)."
                    )
                    mbox = show_error_dialog(self, msg, type="warning", choice=True)
                    if mbox.exec():  # type: ignore
                        # User wants to run full pipeline - switch modes
                        # Use the selected IDs from combo boxes, not GUI widgets
                        value = CaliRunSettings(
                            positions=value.positions,
                            run_detection=True,
                            run_extraction=True,
                            run_analysis=True,
                            detection_settings_id=None,
                            extraction_settings_id=None,
                        )
                        # Use the IDs that were selected in the combo boxes
                        detection_settings = detection_settings_id_check
                        extraction_settings = extraction_settings_id
            elif value.run_extraction and extraction_settings is None:
                # Extraction or Detection+Extraction mode: get from GUI
                # (only if not already set from dialog above)
                extraction_settings = self._extraction_wdg.to_model_settings()
            elif extraction_settings is None:
                extraction_settings = None

            # Get analysis settings if needed
            analysis_settings = (
                self._analysis_wdg.to_model_settings() if value.run_analysis else None
            )

            if extraction_settings is not None and analysis_settings is not None:
                from cali._constants import EVOKED

                if analysis_settings.experiment_type == EVOKED:
                    missing_fields = []
                    # Check for required evoked experiment fields
                    if not analysis_settings.stimulation_mask_path:
                        missing_fields.append("Stimulation mask (Extraction tab)")
                    if not analysis_settings.led_pulse_duration:
                        missing_fields.append("LED pulse duration (Analysis tab)")
                    if not analysis_settings.led_pulse_powers:
                        missing_fields.append("LED pulse powers (Analysis tab)")
                    if not analysis_settings.led_pulse_on_frames:
                        missing_fields.append("LED pulse on frames (Analysis tab)")
                    if missing_fields:
                        msg = (
                            "❌ Evoked experiment type selected but required fields "
                            "are missing:\n\n"
                            + "\n".join(f"{field}" for field in missing_fields)
                            + "\n\nPlease configure these settings in the "
                            "Extraction tab."
                        )
                        show_error_dialog(self, msg)
                        return

            # Get detection settings - either from GUI or selected ID (extraction-only)
            if value.run_extraction and not value.run_detection:
                # Extraction-only mode: use existing detection settings ID
                detection_settings_id = value.detection_settings_id
                if detection_settings_id is None:
                    show_error_dialog(
                        self,
                        "❌ Please select a Detection ID to run extraction-only mode.",
                    )
                    return
                detection_settings = detection_settings_id

                # Check if selected positions have detection data
                missing_detection = self._check_positions_missing_detection(
                    detection_settings_id, pos
                )
                if missing_detection:
                    msg = "Detection Data Missing\n\n"
                    msg += "The following positions are missing detection data:\n"
                    msg += f"{missing_detection}\n\n"
                    msg += "Do you want to run detection first on these positions?\n"
                    msg += (
                        "(If you select 'No', only positions with existing detection "
                        "will be processed)."
                    )
                    mbox = show_error_dialog(self, msg, type="warning", choice=True)
                    if mbox.exec():  # type: ignore
                        # User wants to run detection first - switch to full pipeline
                        # Use the selected detection ID from combo box, not GUI widget
                        value = CaliRunSettings(
                            positions=value.positions,
                            run_detection=True,
                            run_extraction=True,
                            run_analysis=value.run_analysis,
                            detection_settings_id=None,
                            extraction_settings_id=None,
                        )
                        # Use the detection ID that was selected in the combo box
                        detection_settings = detection_settings_id
                        # For extraction, use GUI widget since we're in
                        # extraction-only mode
                        extraction_settings = self._extraction_wdg.to_model_settings()
            elif detection_settings is None:
                # Detection or Detection+Extraction mode: get from GUI
                # (only if not already set from dialog above)
                detection_settings = self._detection_wdg.to_model_settings()

            pos = value.positions or list(
                range(len(self._data.sequence.stage_positions))
            )

            # Check for ambiguous runs BEFORE starting detection
            # This prevents wasted work if user needs to disambiguate
            if (
                value.run_detection
                and not value.run_extraction
                and not value.run_analysis
                and detection_settings is not None
            ):
                # Detection-only mode: check if multiple runs exist
                from sqlmodel import Session, create_engine, select

                engine = create_engine(
                    f"sqlite:///{self._database_path}",
                    echo=False,
                    connect_args={"timeout": 30.0, "check_same_thread": False},
                    pool_pre_ping=True,
                )

                try:
                    with Session(engine) as session:
                        # Get detection settings ID
                        if isinstance(detection_settings, int):
                            detection_settings_id = detection_settings
                        else:
                            # Need to check if this detection settings exists
                            from cali.runner._cali_runner import CaliRunner

                            runner_temp = CaliRunner()
                            det_settings_obj = (
                                runner_temp._get_or_create_detection_settings(
                                    session, detection_settings
                                )
                            )
                            detection_settings_id = det_settings_obj.id

                        # Check for multiple runs with same detection
                        from cali.sqlmodel._model import CaliResult

                        query_start = time.perf_counter()
                        all_results = list(
                            session.exec(
                                select(CaliResult).where(
                                    CaliResult.experiment == experiment.id,
                                    CaliResult.detection_settings_id
                                    == detection_settings_id,
                                )
                            ).all()
                        )
                        query_time = time.perf_counter() - query_start
                        cali_logger.debug(
                            f"DB query: pre-flight ambiguity check took "
                            f"{query_time:.3f}s (found {len(all_results)} runs)"
                        )

                        if len(all_results) > 1:
                            # Multiple runs exist - show dialog
                            selected_run_id = RunSelectionDialog.select_run(
                                parent=self,
                                runs=all_results,
                                message=(
                                    "Multiple runs exist with the same "
                                    f"detection settings (ID {detection_settings_id})."
                                    "\n\nPlease select which run should receive "
                                    "the new detections:"
                                ),
                            )

                            if selected_run_id is None:
                                # User cancelled
                                return

                            # Get selected run's settings
                            selected_run = next(
                                r for r in all_results if r.id == selected_run_id
                            )

                            # Update settings to match selected run
                            detection_settings = detection_settings_id
                            extraction_settings = selected_run.extraction_settings_id
                            analysis_settings = selected_run.analysis_settings_id
                finally:
                    engine.dispose()

            # Initialize progress bar and timer
            self._run_cali_wdg.reset_progress_bar()
            self._run_cali_wdg.set_progress_bar_text("🚀 Initializing...")
            self._elapsed_timer.start()

            # Save plate map data to database before running
            self._save_plate_map_to_database()

            # Try to run, but catch ambiguity errors
            try:
                # Create a generator function wrapper for create_worker
                def _run_generator() -> Generator[str, None, None]:
                    assert self._data is not None
                    assert self._database_path is not None
                    assert detection_settings is not None  # Ensured by pre-flight check
                    result = self._runner.run(
                        experiment,
                        self._data.path,
                        detection_settings,
                        extraction_settings=extraction_settings,
                        analysis_settings=analysis_settings,
                        global_position_indices=pos,
                        database_name=Path(self._database_path).name,
                        output_path=(
                            Path(self._output_path) if self._output_path else None
                        ),
                        as_generator=True,
                    )
                    assert result is not None
                    yield from result

                # disable gui before running
                self._enable(False)

                create_worker(
                    _run_generator,
                    _start_thread=True,
                    _connect={
                        "errored": self._on_worker_errored,
                        "yielded": self._on_worker_yield,
                        "finished": self._on_worker_finished,
                    },
                )
            except ValueError as e:
                # Check if this is an ambiguity error
                error_msg = str(e)
                if "Multiple runs exist" in error_msg and "same detection" in error_msg:
                    # This is an ambiguity error - show dialog to select run
                    # (Should be rare now due to pre-flight check)
                    assert detection_settings is not None  # Required for run to start
                    self._handle_ambiguous_runs(
                        error_msg,
                        experiment,
                        detection_settings,
                        extraction_settings,
                        analysis_settings,
                        pos,
                    )
                else:
                    # Re-raise other ValueErrors
                    raise e
        except Exception as e:
            self._enable(True)
            msg = f"❌ Failed to run cali:\n{e}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)

    def _handle_ambiguous_runs(
        self,
        error_msg: str,
        experiment: Experiment,
        detection_settings: DetectionSettings | int,
        extraction_settings: Any,
        analysis_settings: Any,
        positions: list[int],
    ) -> None:
        """Handle ambiguous run selection when multiple compatible runs exist.

        Parameters
        ----------
        error_msg : str
            The error message from the runner
        experiment : Experiment
            The experiment being run
        detection_settings : DetectionSettings | int
            Detection settings or ID
        extraction_settings : Any
            Extraction settings, ID, or None
        analysis_settings : Any
            Analysis settings, ID, or None
        positions : list[int]
            Positions to process
        """
        from sqlmodel import Session, create_engine, select

        # Query database for all compatible runs
        assert self._database_path is not None

        engine = create_engine(
            f"sqlite:///{self._database_path}",
            echo=False,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )

        # Get detection settings ID
        if isinstance(detection_settings, int):
            detection_settings_id = detection_settings
        else:
            # Need to get the ID from the database
            with Session(engine) as session:
                # Find or create detection settings
                from cali.runner._cali_runner import CaliRunner

                runner = CaliRunner()
                detection_settings_id = runner._get_or_create_detection_settings(
                    session, detection_settings
                )

        compatible_runs: list[CaliResult] = []
        try:
            with Session(engine) as session:
                # Query all runs with matching detection settings
                stmt = select(CaliResult).where(
                    CaliResult.experiment == experiment.id,
                    CaliResult.detection_settings_id == detection_settings_id,
                )
                query_start = time.perf_counter()
                compatible_runs = list(session.exec(stmt).all())
                query_time = time.perf_counter() - query_start
                cali_logger.debug(
                    f"DB query: post-error ambiguity check took {query_time:.3f}s "
                    f"(found {len(compatible_runs)} runs)"
                )
        finally:
            engine.dispose()

        if not compatible_runs:
            show_error_dialog(
                self,
                "No compatible runs found. Please check your settings.",
            )
            return

        # Show the selection dialog
        selected_run_id = RunSelectionDialog.select_run(
            parent=self,
            runs=compatible_runs,
            message=error_msg,
        )

        if selected_run_id is None:
            # User cancelled - don't run anything
            return

        # Get the selected run's settings
        selected_run = next(r for r in compatible_runs if r.id == selected_run_id)

        # Ensure we have detection settings
        if selected_run.detection_settings_id is None:
            show_error_dialog(
                self,
                "Selected run has no detection settings. This should not happen.",
            )
            return

        # Re-run with the selected run's settings explicitly specified
        # This will tell CaliRunner which run to add the positions to
        assert self._data is not None

        # Initialize progress bar and timer
        self._run_cali_wdg.reset_progress_bar()
        self._run_cali_wdg.set_progress_bar_text("🚀 Initializing...")
        self._elapsed_timer.start()

        # Create generator with explicit settings from selected run
        def _run_generator() -> Generator[str, None, None]:
            assert self._data is not None
            assert self._database_path is not None
            assert selected_run.detection_settings_id is not None
            result = self._runner.run(
                experiment,
                self._data.path,
                detection_settings=selected_run.detection_settings_id,
                extraction_settings=selected_run.extraction_settings_id,
                analysis_settings=selected_run.analysis_settings_id,
                global_position_indices=positions,
                database_name=Path(self._database_path).name,
                output_path=Path(self._output_path) if self._output_path else None,
                as_generator=True,
            )
            assert result is not None
            yield from result

        # Disable GUI before running
        self._enable(False)

        create_worker(
            _run_generator,
            _start_thread=True,
            _connect={
                "errored": self._on_worker_errored,
                "yielded": self._on_worker_yield,
                "finished": self._on_worker_finished,
            },
        )

    def _on_worker_yield(self, progress: str) -> None:
        """Update progress bar with yielded progress information."""
        if progress.startswith("PROGRESS:RESET"):
            try:
                total = int(progress.split(":")[2])
                self._run_cali_wdg.reset_progress_value()
                self._run_cali_wdg.set_progress_bar_range(0, total)
            except ValueError:
                pass
        elif progress.startswith("PROGRESS:UPDATE"):
            self._run_cali_wdg.update_progress_bar_plus_one()
        else:
            self._run_cali_wdg.set_progress_bar_text(progress)

    def _on_worker_errored(self, error: Any) -> None:
        """Handle errors from the runner."""
        import traceback

        self._elapsed_timer.stop()
        self._enable(True)

        # Format the error with full traceback
        if hasattr(error, "__traceback__"):
            tb_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            error_msg = "".join(tb_lines)
        else:
            error_msg = str(error)

        cali_logger.error(
            f"❌ Cali Runner encountered an error during execution:\n{error_msg}"
        )

        # Also show error dialog to user
        show_error_dialog(self, f"❌ Cali Runner Error:\n\n{error_msg}")

    def _on_cali_cancel(self) -> None:
        """Handle cancellation of the runner."""
        self._runner.cancel()
        self._run_cali_wdg.set_progress_bar_text("🚮 Cancel Requested")

    def _on_worker_finished(self) -> None:
        """Handle completion of the runner."""
        self._enable(True)
        self._elapsed_timer.stop()
        self._run_cali_wdg.set_progress_bar_text("🏁 Cali Run Finished")
        # refresh the runs panel
        self._runs_panel.refresh_runs()
        # repopulate detection settings combobox
        if self._database_path:
            self._populate_settings(self._database_path)
            self._update_graph_properties(self._database_path)
        # update GUI with the latest run (latest run is at the end of the list)
        last_idx = self._runs_panel._runs_list.count() - 1
        # select last run (no signal emitted)
        self._runs_panel.select_run_by_index(last_idx, block_signals=True)
        # Update run_id in all graph widgets
        self._update_graph_with_run_id(self._runs_panel.get_run_id_by_index(last_idx))
        # Refresh the image viewer to update labels with the new detection settings
        self._on_fov_table_selection_changed()

    def _save_plate_map_to_database(self) -> None:
        """Save plate map data from GUI to database."""
        if self._database_path is None:
            return

        plate_map_data = self._plate_map_wdg.value()
        _, genotype_data, treatment_data = plate_map_data

        # Load experiment and update it with plate map data
        # Note: We need to update even if both are empty, to clear plate_maps
        from sqlmodel import Session, create_engine, select

        from cali.sqlmodel._model import Condition

        engine = create_engine(
            f"sqlite:///{self._database_path}",
            echo=False,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )
        try:
            with Session(engine) as session:
                # Check if experiment table exists (database is initialized)
                from sqlalchemy import inspect

                inspector = inspect(engine)
                if "experiment" not in inspector.get_table_names():
                    return
                # Load experiment manually without expunge_all
                from sqlalchemy.orm import selectinload

                from cali.sqlmodel._model import Plate, Well

                stmt = select(Experiment).options(
                    selectinload(Experiment.plate)
                    .selectinload(Plate.wells)
                    .selectinload(Well.conditions)
                )
                exp = session.exec(stmt).first()
                if exp is None or exp.plate is None or exp.plate.wells is None:
                    return

                # cache ensures we reuse the same Condition instance in-memory
                condition_cache: dict[tuple[str, str], Condition] = {}

                def get_or_create_condition(
                    name: str, color: str, condition_type: str
                ) -> Condition:
                    key = (name, condition_type)

                    # 1) Check local cache
                    if key in condition_cache:
                        cond = condition_cache[key]
                        cond.color = color  # keep color latest from GUI
                        return cond

                    # 2) Check DB
                    stmt = (
                        select(Condition)
                        .where(Condition.name == name)
                        .where(Condition.condition_type == condition_type)
                    )
                    query_start = time.perf_counter()
                    existing = session.exec(stmt).first()
                    query_time = time.perf_counter() - query_start
                    cali_logger.debug(
                        f"DB query: condition lookup took {query_time:.3f}s"
                    )
                    if existing:
                        existing.color = color
                        condition_cache[key] = existing
                        return existing  # type: ignore
                    # 3) Create new condition and add to session
                    cond = Condition(
                        name=name,
                        color=color,
                        condition_type=condition_type,
                    )
                    session.add(cond)
                    condition_cache[key] = cond
                    return cond

                # Build plate_maps dictionary for hash computation
                plate_maps: dict[str, dict[str, str]] = {}
                if genotype_data:
                    plate_maps["genotype"] = {}
                if treatment_data:
                    plate_maps["treatment"] = {}

                # Assign conditions per well
                for well in exp.plate.wells:
                    # Clear existing conditions
                    well.conditions.clear()

                    # genotype
                    for plate_data in genotype_data:
                        if (well.row, well.column) == plate_data.row_col:
                            condition = get_or_create_condition(
                                name=plate_data.condition[0],
                                color=plate_data.condition[1],
                                condition_type="genotype",
                            )
                            well.conditions.append(condition)
                            # Add to plate_maps dictionary
                            plate_maps["genotype"][well.name] = plate_data.condition[0]
                            break

                    # treatment
                    for plate_data in treatment_data:
                        if (well.row, well.column) == plate_data.row_col:
                            condition = get_or_create_condition(
                                name=plate_data.condition[0],
                                color=plate_data.condition[1],
                                condition_type="treatment",
                            )
                            well.conditions.append(condition)
                            # Add to plate_maps dictionary
                            plate_maps["treatment"][well.name] = plate_data.condition[0]
                            break

                # Save plate_maps to plate for hash computation
                exp.plate.plate_maps = plate_maps if plate_maps else None

                cali_logger.info(f"💾 Saving plate_maps to database: {plate_maps}")

                # Flush to ensure relationship changes are tracked
                session.flush()
                session.commit()
        finally:
            engine.dispose(close=True)

    def _on_settings_deleted(self) -> None:
        """Handle settings deleted signal from runs panel (e.g., after deletion)."""
        if self._database_path:
            # Repopulate detection settings, preserving current selection if possible
            self._populate_settings(self._database_path)

            # Refresh the FOV table selection to update the display
            # This will reload labels if the FOV still exists with remaining data
            self._on_fov_table_selection_changed()

    def _enable(self, state: bool) -> None:
        """Enable or disable the GUI during a run."""
        # Switch to Detection & Analysis tab and prevent tab changes
        self._main_tab.setCurrentIndex(0)
        # Enable/disable tab bar to prevent switching (but keep tab content viewable)
        if tab_bar := self._main_tab.tabBar():
            tab_bar.setEnabled(state)
        # Disable widgets in Detection, Extraction, Analysis and Run Cali
        self._detection_wdg.setEnabled(state)
        self._extraction_wdg.setEnabled(state)
        self._analysis_wdg.setEnabled(state)
        self._run_cali_wdg.enable(state)
        # Disable other GUI components
        self._fov_table.setEnabled(state)
        self._plate_view.setEnabled(state)
        self._plate_map_wdg.setEnabled(state)
        self._image_viewer.setEnabled(state)
        self._runs_panel.setEnabled(state)

    # DATA INITIALIZATION--------------------------------------------------------------

    def _show_data_input_dialog(self) -> None:
        """Show dialog to select zarr datastore, segmentation and analysis path."""
        db_path = Path(self._database_path) if self._database_path else None
        init_dialog = _InputDialog(
            self,
            data_path=self._data_path,
            output_path=self._output_path,
            database_path=self._database_path,
            database_name=(db_path.name if db_path is not None else None),
        )
        init_dialog.resize(700, init_dialog.sizeHint().height())
        if init_dialog.exec():
            value = init_dialog.value()
            # input from database
            if value.database_path is not None and value.data_path is not None:
                try:
                    self._initialize_from_database(value.database_path, value.data_path)
                except Exception as e:
                    msg = f"❌ Failed to initialize from database:\n{e}"
                    show_error_dialog(self, msg)
                    cali_logger.error(msg)
                    self._loading_bar.hide()
                    return

            # input from directories
            elif (data_path := value.data_path) is not None:
                if value.output_path is None:
                    msg = "❌ Output path must be provided to create the cali database!"
                    show_error_dialog(self, msg)
                    cali_logger.error(msg)
                    self._loading_bar.hide()
                    return
                try:
                    self._initialize_from_directories(
                        data_path,
                        value.output_path,
                        value.database_name or DEFAULT_CALI_DB_NAME,
                    )
                except Exception as e:
                    msg = f"❌ Failed to initialize from directories:\n{e}"
                    show_error_dialog(self, msg)
                    cali_logger.error(msg)
                    self._loading_bar.hide()
                    return

    def _clear_widget_before_initialization(self) -> None:
        """Clear the widget before initializing it with new data."""
        # clear paths
        self._database_path = None
        self._data_path = None
        self._output_path = None
        # clear the datastore
        self._data = None
        # clear fov table
        self._fov_table.clear()
        # clear scene
        self._plate_view.clear()
        # clear the image viewer cache
        self._image_viewer._viewer._contour_cache.clear()
        # no plate flag
        self._default_plate_plan = False
        # reset analysis widget gui
        self._extraction_wdg.reset()
        # reset detection widget gui
        self._detection_wdg.reset()
        # reset run cali widget
        self._run_cali_wdg.reset()
        # reset runs panel
        self._runs_panel.clear()

    def _update_graph_properties(self, database_path: Path | str) -> None:
        """Update all graph widgets with the current database path and engine."""
        from sqlmodel import create_engine

        # Create new SQLAlchemy engine for database queries
        engine = create_engine(
            f"sqlite:///{database_path}",
            echo=False,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            pool_pre_ping=True,
        )

        for sw_graph in self.SW_GRAPHS:
            if sw_graph.engine is not None:
                sw_graph.engine.dispose(close=True)
            sw_graph.database_path = database_path
            sw_graph.engine = engine
            sw_graph.clear_plot()

        for mw_graph in self.MW_GRAPHS:
            if mw_graph.engine is not None:
                mw_graph.engine.dispose(close=True)
            mw_graph.database_path = database_path
            mw_graph.engine = engine
            mw_graph.clear_plot()

    def _update_graph_with_run_id(self, run_id: int | None) -> None:
        """Update all graph widgets with the selected run ID.

        Parameters
        ----------
        run_id : int | None
            The CaliResult.id of the selected run, or None to clear
        """
        for sw_graph in self.SW_GRAPHS:
            sw_graph.run_id = run_id
        for mw_graph in self.MW_GRAPHS:
            mw_graph.run_id = run_id

    def _draw_plate_with_selection(self, plate_plan: useq.WellPlatePlan) -> None:
        """Draw the plate and disable non-selected wells."""
        self._plate_view.drawPlate(plate_plan.plate)

        wells = self._plate_view._well_items
        selected_indices = {
            tuple(plate_plan.selected_well_indices[i])
            for i in range(len(plate_plan.selected_well_indices))
        }

        for r, c in wells.keys():
            if (r, c) not in selected_indices:
                self._plate_view.setWellColor(r, c, UNSELECTABLE_COLOR)

    # WIDGETS -------------------------------------------------------------------------

    def _on_run_item_selected(self, run_id: int) -> None:
        """Handle run selection from the runs panel.

        Load the detection and analysis settings for the selected run.

        Parameters
        ----------
        run_id : int
            The ID of the selected CaliResult
        """
        if self._database_path is None:
            return

        self._init_loading_bar(f"💿 Loading Run {run_id}...", False)

        try:
            # Load the selected analysis result
            result = CaliResult.load_from_database(
                self._database_path, id=run_id, load_data=False
            )
            assert isinstance(result, CaliResult)

            # Load and apply detection settings
            if result.detection_settings_id:
                d_settings = DetectionSettings.load_from_database(
                    self._database_path, id=result.detection_settings_id
                )
                assert isinstance(d_settings, DetectionSettings)
                if d_settings.method == "cellpose":
                    self._detection_wdg.setValue(
                        CellposeSettingsData(
                            model_type=d_settings.model_type,
                            model_path=d_settings.custom_model,
                            diameter=d_settings.diameter,
                            cellprob_threshold=d_settings.cellprob_threshold,
                            flow_threshold=d_settings.flow_threshold,
                            min_size=d_settings.min_size,
                            normalize=d_settings.normalize,
                            batch_size=d_settings.batch_size,
                        )
                    )
                else:
                    msg = f"❌ Unknown detection method: {d_settings.method}."
                    show_error_dialog(self, msg)
                    cali_logger.error(msg)
                    return

            # Load and apply extraction settings
            if result.extraction_settings_id:
                from cali.gui._extraction_gui import (
                    ExtractionSettingsData,
                    MetadataData,
                    TraceExtractionData,
                )
                from cali.sqlmodel import ExtractionSettings

                e_settings = ExtractionSettings.load_from_database(
                    self._database_path, id=result.extraction_settings_id
                )
                assert isinstance(e_settings, ExtractionSettings)

                self._extraction_wdg.setValue(
                    ExtractionSettingsData(
                        trace_extraction_data=TraceExtractionData(
                            dff_window_size=e_settings.dff_window,
                            decay_constant=e_settings.decay_constant,
                            neuropil_inner_radius=e_settings.neuropil_inner_radius,
                            neuropil_min_pixels=e_settings.neuropil_min_pixels,
                            neuropil_correction_factor=(
                                e_settings.neuropil_correction_factor
                            ),
                        ),
                        metadata_data=MetadataData(
                            pixel_size=e_settings.pixel_size,
                            frame_rate=e_settings.frame_rate,
                        ),
                        threads=e_settings.threads,
                    )
                )

            # Load and apply analysis settings
            if result.analysis_settings_id:
                from cali.gui._analysis_gui import (
                    CalciumPeaksData,
                    ExperimentTypeData,
                    SpikeData,
                )

                a_settings = AnalysisSettings.load_from_database(
                    self._database_path, id=result.analysis_settings_id
                )
                assert isinstance(a_settings, AnalysisSettings)

                self._analysis_wdg.setValue(
                    AnalysisSettingsData(
                        experiment_type_data=ExperimentTypeData(
                            experiment_type=a_settings.experiment_type,
                            led_power_equation=a_settings.led_power_equation,
                            led_pulse_duration=a_settings.led_pulse_duration,
                            led_pulse_on_frames=a_settings.led_pulse_on_frames,
                            led_pulse_powers=a_settings.led_pulse_powers,
                            stimulation_area_path=a_settings.stimulation_mask_path,
                        ),
                        calcium_peaks_data=CalciumPeaksData(
                            peaks_height=a_settings.peaks_height_value,
                            peaks_height_mode=a_settings.peaks_height_mode,
                            peaks_distance=a_settings.peaks_distance,
                            peaks_prominence_multiplier=(
                                a_settings.peaks_prominence_multiplier
                            ),
                            burst_threshold=a_settings.calcium_burst_threshold,
                            burst_min_duration=a_settings.calcium_burst_min_duration,
                            burst_blur_sigma=a_settings.calcium_burst_gaussian_sigma,
                        ),
                        spikes_data=SpikeData(
                            spike_threshold=a_settings.spike_threshold_value,
                            spike_threshold_mode=a_settings.spike_threshold_mode,
                            burst_threshold=a_settings.burst_threshold,
                            burst_min_duration=a_settings.burst_min_duration,
                            burst_blur_sigma=a_settings.burst_gaussian_sigma,
                            synchrony_lag=a_settings.spikes_sync_cross_corr_lag,
                            synchrony_jitter=a_settings.spikes_sync_jitter_window,
                        ),
                        threads=a_settings.threads,
                    )
                )

            cali_logger.info(f"✅ Loaded settings from Run #{run_id}")

            # Update run_id in all graph widgets
            self._update_graph_with_run_id(run_id)

            # Refresh the image viewer to update labels with the new detection settings
            self._on_fov_table_selection_changed()

            self._loading_bar.hide()

        except Exception as e:
            self._loading_bar.hide()
            show_error_dialog(self, f"Failed to load run settings: {e}")
            cali_logger.error(f"❌ Failed to load run #{run_id}: {e}")

    def _set_splitter_sizes(self) -> None:
        """Set the initial sizes for the splitters."""
        splitter_and_sizes = (
            (self.splitter_top_left, [0.73, 0.27]),
            (self.splitter_bottom_left, [0.50, 0.50]),
            (self.main_splitter, [0.30, 0.70]),
        )
        for splitter, sizes in splitter_and_sizes:
            total_size = splitter.size().width()
            splitter.setSizes([int(size * total_size) for size in sizes])

    def _on_led_info_from_meta_clicked(self) -> None:
        if self._data is None:
            show_error_dialog(self, "❌ Data not loaded! Cannot find metadata!")
            return

        try:
            if (sequence := self._data.sequence) is None:
                msg = "❌ useq.MDASequence not found! Cannot retrieve metadata!"
                show_error_dialog(self, msg)
                cali_logger.error(msg)
                return

            meta = sequence.metadata.get(PYMMCW_METADATA_KEY, {})
            led_meta = cast("dict", meta.get("stimulation", {}))
            if led_meta:
                wdg = self._analysis_wdg._experiment_type_wdg

                # pulse duration
                if led_duration := led_meta.get("led_pulse_duration", None):
                    wdg._led_pulse_duration_spin.setValue(led_duration)

                # led powers and frames
                if pulse_on_frame := led_meta.get("pulse_on_frame", None):
                    wdg._led_powers_le.setText(
                        ", ".join(
                            str(pulse_on_frame[str(frame)])
                            for frame in sorted(int(k) for k in pulse_on_frame.keys())
                        )
                    )
                    wdg._led_pulse_on_frames_le.setText(
                        ", ".join(
                            str(frame)
                            for frame in sorted(int(k) for k in pulse_on_frame.keys())
                        )
                    )
                    cali_logger.info(
                        f"🗒️ Loaded stimulation metadata from datastore: "
                        f"led_pulse_duration={led_duration} "
                        f"led_powers={wdg._led_powers_le.text()}, "
                        f"led_pulse_on_frames={wdg._led_pulse_on_frames_le.text()}"
                    )

            else:
                msg = "❌ No stimulation metadata found in the datastore!"
                show_error_dialog(self, msg)
                cali_logger.warning(msg)

        except Exception as e:
            msg = f"❌ Failed to load metadata from datastore!\n\nError: {e}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            return

    def _on_extraction_meta_clicked(self) -> None:
        """Load pixel size and frame rate from metadata."""
        if self._data is None:
            show_error_dialog(self, "❌ Data not loaded! Cannot find metadata!")
            return

        try:
            if not (meta := self._data.metadata):
                msg = "❌ No metadata found! Cannot retrieve pixel size or frame rate!"
                show_error_dialog(self, msg)
                cali_logger.error(msg)
                return
            if isinstance(meta, dict):
                meta = [meta]
            elif callable(meta):  # ome zarr reader
                meta = meta()
            pixel_size = meta[0].get("pixel_size_um", None)
            exposure_ms = meta[0].get("exposure_ms", None)
            frame_rate = 1000.0 / exposure_ms if exposure_ms else 10
            self._extraction_wdg._metadata_wdg.setValue(
                MetadataData(pixel_size, frame_rate)
            )

            final_msg = ""
            if pixel_size is None:
                final_msg += (
                    "⚠️ No pixel size found in metadata! Using pixels as units.\n"
                )
            else:
                cali_logger.info(f"🗒️ Loaded pixel size from metadata: {pixel_size} µm")

            if exposure_ms is None:
                final_msg += (
                    "⚠️ No exposure time found in metadata! Using default frame rate "
                    "of 10 fps."
                )
            else:
                cali_logger.info(f"🗒️ Loaded frame rate from metadata: {frame_rate} fps")

            if final_msg:
                show_error_dialog(self, final_msg, type="warning")
                cali_logger.warning(final_msg)

        except Exception as e:
            msg = f"❌ Failed to load metadata from datastore!\n\nError: {e}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            return

    def _on_analysis_meta_clicked(self) -> None:
        """Load frame rate from metadata for analysis settings."""
        if self._data is None:
            show_error_dialog(self, "❌ Data not loaded! Cannot find metadata!")
            return

        try:
            if not (meta := self._data.metadata):
                msg = "❌ No metadata found! Cannot retrieve frame rate!"
                show_error_dialog(self, msg)
                cali_logger.error(msg)
                return
            if isinstance(meta, dict):
                meta = [meta]
            elif callable(meta):  # ome zarr reader
                meta = meta()
            exposure_ms = meta[0].get("exposure_ms", None)
            frame_rate = 1000.0 / exposure_ms if exposure_ms else 10
            self._analysis_wdg._metadata_wdg.setValue(frame_rate)

            if exposure_ms is None:
                msg = (
                    "⚠️ No exposure time found in metadata! Using default frame rate "
                    "of 10 fps.\n"
                )
                cali_logger.warning(msg)
                show_error_dialog(self, msg, type="warning")
            else:
                cali_logger.info(f"🗒️ Loaded frame rate from metadata: {frame_rate} fps")

        except Exception as e:
            msg = f"❌ Failed to load metadata from datastore!\n\nError: {e}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            return

    def _init_loading_bar(self, text: str, show_progress_bar: bool = True) -> None:
        """Reset the loading bar."""
        self._loading_bar.setEnabled(True)
        self._loading_bar.setText(text)
        self._loading_bar.setValue(0)
        self._loading_bar.showPercentage(show_progress_bar)
        self._loading_bar.show_progress_bar(show_progress_bar)
        self._loading_bar.show()
        # force update of the loading bar (windows requires this)
        QApplication.processEvents()

    def _update_graphs_with_roi(self, roi: int) -> None:
        """Update the graphs with the given roi.

        This function is called when a roi is selected in the image viewer and will
        update the graphs with the traces of the selected roi.
        """
        # get the current main tab index (0=Detection & Analysis, 1=Visualization)
        idx = self._main_tab.currentIndex()
        if idx == 0:  # Detection & Analysis tab
            return
        for graph in self.SW_GRAPHS:
            if graph._combo.currentText() == "None":
                continue
            graph._choose_dysplayed_traces.setChecked(True)
            graph._choose_dysplayed_traces._roi_le.setText(str(roi))
            graph._choose_dysplayed_traces._update()

    def _on_tab_changed(self, idx: int) -> None:
        """Update the graph combo boxes when the tab is changed."""
        # skip if the tab is the Detection & Analysis tab
        if idx == 0:
            return
        # if visualization tab is selected (main tab index 1)
        if idx == 1:
            # get the current fov
            value = self._fov_table.value() if self._fov_table.selectedItems() else None
            if value is None:
                return
            # update the graphs combo boxes
            self._update_single_wells_graphs_combo()

    def _highlight_roi(self, roi: str | list[str]) -> None:
        """Highlight the selected roi in the image viewer.

        Parameters
        ----------
        roi : str | list[str]
            Single ROI as string, or list where first element is the selected ROI
            and remaining elements are connected ROIs (e.g., from connectivity plot).
        """
        if isinstance(roi, list):
            if len(roi) == 0:
                return
            # First ROI is the selected one (green), rest are connected (yellow)
            selected = roi[0]
            connected = [int(r) for r in roi[1:]] if len(roi) > 1 else None
            self._image_viewer._roi_number_le.setText(selected)
            self._image_viewer._highlight_rois(int(selected), connected_rois=connected)
        else:
            self._image_viewer._roi_number_le.setText(roi)
            self._image_viewer._highlight_rois(int(roi))

    def _on_scene_well_changed(self) -> None:
        """Update the FOV table when a well is selected."""
        self._fov_table.clear()
        self._image_viewer._clear_highlight()

        # Clear plots and reset FOV to ensure reload when selecting same well again
        for sw_graph in self.SW_GRAPHS:
            sw_graph.clear_plot()
            sw_graph.fov = ""

        if self._data is None:
            return

        if self._data.sequence is None:
            show_error_dialog(
                self,
                "❌ useq.MDASequence not found! Cannot retrieve the Well data without "
                "the tensorstore useq.MDASequence!",
            )
            return

        well_dict: set[QAbstractGraphicsShapeItem] = self._plate_view._selected_items
        if not well_dict or len(well_dict) != 1:
            return
        well_name = next(iter(well_dict)).data(DATA_POSITION).name

        # Get FOVs from database for this well to handle wizard-created mappings
        # where original position names don't match the well name
        well_fov_positions: set[int] = set()
        if self._database_path:
            from sqlmodel import Session, create_engine, select

            from cali.sqlmodel._model import FOV, Well

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                connect_args={"timeout": 30.0, "check_same_thread": False},
            )
            try:
                with Session(engine) as session:
                    # Get FOVs for this well from database
                    stmt = (
                        select(FOV.position_index)
                        .join(Well)
                        .where(Well.name == well_name)
                    )
                    results = session.exec(stmt).all()
                    well_fov_positions = set(results)
            finally:
                engine.dispose(close=True)

        # Add the FOV per position to the table
        for idx, pos in enumerate(self._data.sequence.stage_positions):
            # Match by position name OR by position index from database
            if (
                self._default_plate_plan
                or (pos.name and well_name in pos.name)
                or (idx in well_fov_positions)
            ):
                self._fov_table.add_position(WellInfo(idx, pos))

        if self._fov_table.rowCount() > 0:
            self._fov_table.selectRow(0)

    def _on_fov_table_selection_changed(self) -> None:
        """Update the image viewer with the first frame of the selected FOV."""
        try:
            self._image_viewer._clear_highlight()
            value = self._fov_table.value() if self._fov_table.selectedItems() else None

            if value is None:
                self._image_viewer.setData(None, None)
                self._update_single_wells_graphs_combo(clear=True)
                return

            if self._data is None:
                return

            if not self._data.sequence:
                return

            # get a single frame for the selected FOV (at 2/3 of the time points)
            t = int(len(self._data.sequence.stage_positions) / 3 * 2)
            data = cast("np.ndarray", self._data.isel(p=value.pos_idx, t=t, c=0))
            # get labels and neuropil masks if they exist
            roi_labels, neuropil_labels = self._get_labels(value)
            # flip data and labels or will look different from the StackViewer
            data = np.flip(data, axis=0)
            roi_labels = np.flip(roi_labels, axis=0) if roi_labels is not None else None
            neuropil_labels = (
                np.flip(neuropil_labels, axis=0)
                if neuropil_labels is not None
                else None
            )
            self._image_viewer.setData(data, roi_labels, neuropil_labels)
            # Update graph widgets with new FOV - this will trigger plot reload
            # Use the FOV name directly - it already contains the full identifier
            # (e.g., "B5_0000" for well B5, position 0000)
            title = value.fov.name or f"Position {value.pos_idx}"
            self._update_single_wells_graphs_combo(set_fov=title)
            self._loading_bar.hide()
        except Exception as e:
            msg = f"❌ Failed to load FOV:\n{e}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)

    def _has_fov_analysis(self, value: WellInfo) -> bool:
        """Check if the given FOV has been analyzed (has ROIs with data).

        This efficiently queries the database directly to check if the FOV has
        analyzed ROIs, without loading the entire experiment object.

        Parameters
        ----------
        value : WellInfo
            FOV information from the table

        Returns
        -------
        bool
            True if the FOV has been analyzed, False otherwise
        """
        if self._database_path is None:
            return False

        # Use the FOV name from the value
        if not (fov_name := value.fov.name):
            return False

        return has_fov_analysis(self._database_path, fov_name)

    def _get_labels(
        self, value: WellInfo
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get the labels (ROI and neuropil masks) for the given FOV from the database.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            Tuple of (roi_mask, neuropil_mask) arrays, or (None, None) if not available
        """
        if self._database_path is None:
            return None, None

        # Get FOV name
        fov_name = value.fov.name
        if not fov_name:
            return None, None

        # Get selected run ID and detection settings ID from currently selected run
        run_id = self._runs_panel.get_selected_run_id()
        detection_settings_id = self._runs_panel.get_selected_detection_settings_id()

        try:
            from sqlalchemy.orm import selectinload
            from sqlmodel import Session, create_engine, select

            from cali.sqlmodel._model import FOV, ROI, Traces
            from cali.util import coordinates_to_mask

            engine = create_engine(
                f"sqlite:///{self._database_path}",
                echo=False,
                connect_args={"timeout": 30.0, "check_same_thread": False},
                pool_pre_ping=True,
            )
            try:
                with Session(engine) as session:
                    # Query ROIs with detection_settings_id filter
                    stmt = (
                        select(ROI)
                        .join(FOV)
                        .where(FOV.name == fov_name)
                        .options(selectinload(ROI.roi_mask))
                    )

                    # Add detection_settings_id filter if available
                    if detection_settings_id is not None:
                        stmt = stmt.where(
                            ROI.detection_settings_id == detection_settings_id
                        )

                    query_start = time.perf_counter()
                    rois = session.exec(stmt).all()
                    query_time = time.perf_counter() - query_start
                    cali_logger.debug(
                        f"DB query: visualize mask ROIs took {query_time:.3f}s "
                        f"(found {len(rois)} ROIs)"
                    )

                    if not rois:
                        return None, None

                    # Get the shape from the first ROI mask
                    first_mask = rois[0].roi_mask
                    if (
                        not first_mask
                        or first_mask.height is None
                        or first_mask.width is None
                    ):
                        return None, None

                    shape = (first_mask.height, first_mask.width)

                    # Create combined label masks
                    roi_mask = np.zeros(shape, dtype=np.uint16)
                    neuropil_mask = np.zeros(shape, dtype=np.uint16)

                    # Build ROI mask
                    for roi in rois:
                        if (
                            roi.roi_mask
                            and roi.roi_mask.coords_y
                            and roi.roi_mask.coords_x
                        ):
                            coords = (roi.roi_mask.coords_y, roi.roi_mask.coords_x)
                            roi_binary_mask = coordinates_to_mask(coords, shape)
                            roi_mask[roi_binary_mask] = roi.label_value

                    # Query neuropil masks from Traces for the selected run
                    # Neuropil masks are now stored per-Trace (per analysis run)
                    if run_id is not None:
                        traces_stmt = (
                            select(Traces)
                            .join(ROI)
                            .join(FOV)
                            .where(
                                Traces.analysis_result_id == run_id,
                                FOV.name == fov_name,
                            )
                            .options(
                                selectinload(Traces.roi),
                                selectinload(Traces.neuropil_mask),
                            )
                        )
                        query_start = time.perf_counter()
                        traces = session.exec(traces_stmt).all()
                        query_time = time.perf_counter() - query_start
                        cali_logger.debug(
                            f"DB query: visualize mask traces took {query_time:.3f}s "
                            f"(found {len(traces)} traces)"
                        )

                        # Build neuropil mask from Traces
                        for trace in traces:
                            # Include traces from ROIs matching detection_settings_id
                            if (
                                trace.roi
                                and trace.roi.detection_settings_id
                                == detection_settings_id
                                and trace.neuropil_mask
                                and trace.neuropil_mask.coords_y
                                and trace.neuropil_mask.coords_x
                            ):
                                coords = (
                                    trace.neuropil_mask.coords_y,
                                    trace.neuropil_mask.coords_x,
                                )
                                neuropil_binary_mask = coordinates_to_mask(
                                    coords, shape
                                )
                                # Use the ROI's label_value
                                neuropil_mask[neuropil_binary_mask] = (
                                    trace.roi.label_value
                                )

                    # Return masks (None if empty)
                    roi_result = roi_mask if roi_mask.max() > 0 else None
                    neuropil_result = neuropil_mask if neuropil_mask.max() > 0 else None

                    return roi_result, neuropil_result
            finally:
                engine.dispose(close=True)

        except Exception as e:
            cali_logger.warning(f"❌ Failed to load ROI masks from database: {e}")
            return None, None

    def _on_fov_double_click(self) -> None:
        """Open the selected FOV in a new StackViewer window."""
        value = self._fov_table.value() if self._fov_table.selectedItems() else None
        if value is None or self._data is None:
            return

        data = self._data.isel(p=value.pos_idx)
        viewer = NDViewer(data, parent=self)
        viewer._ndims_btn.hide()
        viewer.setWindowTitle(value.fov.name or f"Position {value.pos_idx}")
        viewer.setWindowFlag(Qt.WindowType.Dialog)
        viewer.show()

    def _update_single_wells_graphs_combo(
        self,
        set_fov: str | None = None,
        clear: bool = False,
    ) -> None:
        """Update single-well graph widgets.

        Parameters
        ----------
        set_fov : str | None
            If provided, set the FOV and update combo availability + reload plot.
            If None, don't change FOV.
        clear : bool
            If True, clear the plot (used when deselecting FOV). Combo selection
            persists.
        """
        for sw_graph in self.SW_GRAPHS:
            # Set FOV if provided - this will update combo availability and reload plot
            if set_fov is not None:
                sw_graph.fov = set_fov
            # Just clear plot if requested (combo selection stays)
            elif clear:
                sw_graph.clear_plot()

    def _update_multi_wells_graphs_combo(self) -> None:
        # Combo boxes now automatically disable unavailable items
        # based on pipeline stage availability
        pass

    # MENU SAVE ACTIONS----------------------------------------------------------------

    def _show_save_as_tiff_dialog(self) -> None:
        """Show the save as tiff dialog."""
        if self._data is None or (sequence := self._data.sequence) is None:
            show_error_dialog(
                self,
                "❌ No data to save or useq.MDASequence not found! "
                "Cannot save the data.",
            )
            return

        dialog = _SaveAsTiff(self)

        if dialog.exec():
            path, positions = dialog.value()

            if not Path(path).is_dir():
                show_error_dialog(
                    self,
                    f"❌ The path {path} is not a directory! Cannot save the data.",
                )
                return

            # If no positions specified, use all positions
            if not positions:
                positions = list(range(len(sequence.stage_positions)))

            # start the waiting progress bar
            self._init_loading_bar("Saving as tiff...")
            self._loading_bar.setRange(0, len(positions))

            create_worker(
                self._save_as_tiff,
                path=path,
                positions=positions,
                sequence=sequence,
                _start_thread=True,
                _connect={
                    "yielded": self._update_progress,
                    "finished": self._on_loading_finished,
                },
            )

    def _update_progress(self, value: int | str) -> None:
        """Update the progress bar value."""
        if isinstance(value, str):
            show_error_dialog(self, value)
        else:
            self._loading_bar.setValue(value)

    def _on_loading_finished(self) -> None:
        """Called when the loading of the analysis data is finished."""
        self._loading_bar.hide()

    def _save_as_tiff(
        self, path: str, positions: list[int], sequence: useq.MDASequence
    ) -> Generator[int, None, None]:
        """Save the selected positions as tiff files."""
        # TODO: multithreading or multiprocessing
        # TODO: also save metadata
        if not self._data:
            return
        if not positions:
            positions = list(range(len(sequence.stage_positions)))
        for pos in tqdm(positions, desc="Saving as tiff"):
            data, meta = self._data.isel(p=pos, metadata=True)
            # get the well name from metadata
            pos_name = (
                meta[0].get(EVENT_KEY, {}).get("pos_name", f"pos_{str(pos).zfill(4)}")
            )
            # save the data as tiff
            tifffile.imwrite(Path(path) / f"{pos_name}.tiff", data)
            yield pos + 1

    def _show_save_as_csv_dialog(self) -> None:
        """Show the save as csv dialog."""
        # Check if experiment has analysis data
        if self._database_path is None:
            show_error_dialog(
                self, "❌ No data to save! Run or load analysis data first."
            )
            return

        if not has_experiment_analysis(self._database_path):
            show_error_dialog(
                self, "❌ No data to save! Run or load analysis data first."
            )
            return

        dialog = _SaveAsCSV(self)
        dialog.resize(500, dialog.sizeHint().height())

        if dialog.exec():
            path = dialog.value()
            if not Path(path).is_dir():
                show_error_dialog(
                    self,
                    f"❌ The path {path} is not a directory! Cannot save the data.",
                )
                return

            # TODO: Update these functions to work with SQLModel Experiment
            # save_trace_data_to_csv(path, self._experiment)
            # save_analysis_data_to_csv(path, self._experiment)
            show_error_dialog(
                self, "❌ CSV export is not yet implemented for SQLModel."
            )
