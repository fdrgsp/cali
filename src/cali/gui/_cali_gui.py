from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
import useq
from fonticon_mdi6 import MDI6
from ndv import NDViewer
from pymmcore_widgets.useq_widgets._well_plate_widget import (
    DATA_POSITION,
    WellPlateView,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QIcon
from qtpy.QtWidgets import (
    QAbstractGraphicsShapeItem,
    QGridLayout,
    QGroupBox,
    QMainWindow,
    QMenuBar,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from cali._constants import (
    EVENT_KEY,
    EVOKED,
    OME_ZARR,
    PYMMCW_METADATA_KEY,
    SPONTANEOUS,
    UNSELECTABLE_COLOR,
    WRITERS,
    ZARR_TESNSORSTORE,
)
from cali.analysis import AnalysisRunner
from cali.detection._detection_runner import DetectionRunner
from cali.logger import cali_logger
from cali.sqlmodel import (
    Experiment,
    experiment_to_plate_map_data,
    experiment_to_useq_plate_plan,
    has_experiment_analysis,
    has_fov_analysis,
    save_experiment_to_database,
)
from cali.sqlmodel._db_to_useq_plate import experiment_to_useq_plate
from cali.sqlmodel._model import AnalysisResult, AnalysisSettings, DetectionSettings
from cali.util import load_data

from ._analysis_gui import (
    AnalysisSettingsData,
    CalciumPeaksData,
    ExperimentTypeData,
    SpikeData,
    TraceExtractionData,
    _AnalysisGUI,
)
from ._detection_gui import CaimanSettings, CellposeSettings, _DetectionGUI
from ._fov_table import WellInfo, _FOVTable
from ._graph_widgets import _MultilWellGraphWidget, _SingleWellGraphWidget
from ._image_viewer import _ImageViewer
from ._init_dialog import _InputDialog
from ._plate_plan_wizard import PlatePlanWizard
from ._save_as_widgets import _SaveAsCSV, _SaveAsTiff
from ._util import (
    _ElapsedTimer,
    _ProgressBarWidget,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from cali.readers import OMEZarrReader, TensorstoreZarrReader


DEFAULT_PLATE_PLAN = useq.WellPlatePlan(
    plate=useq.WellPlate.from_str("coverslip-18mm-square"),
    a1_center_xy=(0.0, 0.0),
    selected_wells=((0,), (0,)),
)


class CaliGui(QMainWindow):
    """A widget for displaying a plate preview."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Plate Viewer")
        self.setWindowIcon(QIcon(icon(MDI6.view_comfy, color="#00FF00")))

        # ELAPSED TIMER ---------------------------------------------------------------
        self._elapsed_timer = _ElapsedTimer()
        # TODO: FIX ME...should update detection or analysis based on context
        # self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        # INTERNAL VARIABLES ---------------------------------------------------------
        self._database_path: Path | None = None
        self._data_path: str | None = None
        self._analysis_path: str | None = None
        self._data: TensorstoreZarrReader | OMEZarrReader | None = None

        # RUNNERS --------------------------------------------------------------------
        self._analysis_runner = AnalysisRunner()
        self._detection_runner = DetectionRunner()

        # PROGRESS BAR WIDGET --------------------------------------------------------
        self._loading_bar = _ProgressBarWidget(self)

        # MENU BAR -------------------------------------------------------------------
        self.menu_bar = QMenuBar(self)
        self.file_menu = self.menu_bar.addMenu("File")
        open_action = QAction("Load Data and Set Directories...", self)
        open_action.setToolTip(
            "Load a zarr datastore and directories for labels and analysis data."
        )
        open_action.triggered.connect(self._show_data_input_dialog)
        save_as_tiff_action = QAction("Save Data as Tiff...", self)
        save_as_tiff_action.triggered.connect(self._show_save_as_tiff_dialog)
        save_as_csv_action = QAction("Save Analysis Data as CSV...", self)
        save_as_csv_action.triggered.connect(self._show_save_as_csv_dialog)
        self.file_menu.addAction(open_action)
        self.file_menu.addAction(save_as_tiff_action)
        self.file_menu.addAction(save_as_csv_action)
        self.setMenuBar(self.menu_bar)

        # PLATE PLAN WIZARD -----------------------------------------------------------
        self._plate_plan_wizard = PlatePlanWizard(self)
        self._plate_plan_wizard.hide()
        self._default_plate_plan: bool = False

        # PLATE VIEW ------------------------------------------------------------------
        self._plate_view = WellPlateView()
        self._plate_view.setDragMode(WellPlateView.DragMode.NoDrag)
        self._plate_view.setSelectionMode(WellPlateView.SelectionMode.SingleSelection)

        # TABLE FOR THE FIELDS OF VIEW ------------------------------------------------
        self._fov_table = _FOVTable(self)
        self._fov_table.itemSelectionChanged.connect(
            self._on_fov_table_selection_changed
        )
        self._fov_table.doubleClicked.connect(self._on_fov_double_click)

        # IMAGE VIEWER ----------------------------------------------------------------
        self._image_viewer = _ImageViewer(self)
        self._image_viewer.valueChanged.connect(self._update_graphs_with_roi)

        # LEFT WIDGETS ----------------------------------------------------------------

        # SPLITTER FOR THE PLATE MAP AND THE FOV TABLE --------------------------------
        self.splitter_top_left = QSplitter(
            parent=self, orientation=Qt.Orientation.Vertical
        )
        self.splitter_top_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_top_left.setChildrenCollapsible(False)
        self.splitter_top_left.addWidget(self._plate_view)
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

        # TABS FOR ANALYSIS AND VISUALIZATION -----------------------------------------
        self._tab = QTabWidget(self)
        self._tab.currentChanged.connect(self._on_tab_changed)

        # DETECTION TAB ---------------------------------------------------------------
        self._detection_tab = QWidget()
        self._tab.addTab(self._detection_tab, "Detection Tab")
        detection_tab_layout = QVBoxLayout(self._detection_tab)
        detection_tab_layout.setContentsMargins(0, 0, 0, 0)

        # DETECTION WIDGET ------------------------------------------------------------
        self._detection_wdg = _DetectionGUI(self)
        detection_tab_layout.addWidget(self._detection_wdg)

        # ANALYSIS TAB ----------------------------------------------------------------
        self._analysis_tab = QWidget()
        self._tab.addTab(self._analysis_tab, "Analysis Tab")
        analysis_tab_layout = QVBoxLayout(self._analysis_tab)
        analysis_tab_layout.setContentsMargins(0, 0, 0, 0)

        # ANALYSIS WIDGET -------------------------------------------------------------
        self._analysis_wdg = _AnalysisGUI(self)
        analysis_tab_layout.addWidget(self._analysis_wdg)

        # SINGLE WELL VISUALIZATION TAB -----------------------------------------------
        self._single_well_vis_tab = QWidget()
        self._tab.addTab(self._single_well_vis_tab, "Single Wells Visualization Tab")
        single_well_vis_layout = QGridLayout(self._single_well_vis_tab)
        single_well_vis_layout.setContentsMargins(5, 5, 5, 5)
        single_well_vis_layout.setSpacing(5)

        self._single_well_graph_1 = _SingleWellGraphWidget(self)
        self._single_well_graph_2 = _SingleWellGraphWidget(self)
        self._single_well_graph_3 = _SingleWellGraphWidget(self)
        # self._single_well_graph_4 = _SingleWellGraphWidget(self)

        single_well_vis_layout.addWidget(self._single_well_graph_1, 0, 0)
        single_well_vis_layout.addWidget(self._single_well_graph_2, 0, 1)
        single_well_vis_layout.addWidget(self._single_well_graph_3, 1, 0, 1, 2)
        # single_well_vis_layout.addWidget(self._single_well_graph_4, 1, 1)
        self.SW_GRAPHS = [
            self._single_well_graph_1,
            self._single_well_graph_2,
            self._single_well_graph_3,
            # self._single_well_graph_4,
        ]

        # MULTI WELL VISUALIZATION TAB ------------------------------------------------
        self._multi_well_vis_tab = QWidget()
        self._tab.addTab(self._multi_well_vis_tab, "Multi Wells Visualization Tab")
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

        # MAIN SPLITTER-------------------------------------------------------------
        # splitter between the plate map/fov table/image viewer and the graphs
        self.main_splitter = QSplitter(self)
        self.main_splitter.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.splitter_bottom_left)
        self.main_splitter.addWidget(self._tab)

        # CENTRAL WIDGET -------------------------------------------------------------
        self._central_widget = QWidget(self)
        self._central_widget_layout = QVBoxLayout(self._central_widget)
        self._central_widget_layout.setContentsMargins(10, 10, 10, 10)
        self._central_widget_layout.addWidget(self.main_splitter)
        self.setCentralWidget(self._central_widget)

        # CONNECT SIGNALS ------------------------------------------------------------
        self._plate_view.selectionChanged.connect(self._on_scene_well_changed)

        # connect the roiSelected signal from the graphs to the image viewer so we can
        # highlight the roi in the image viewer when a roi is selected in the graph
        for graph in self.SW_GRAPHS:
            graph.roiSelected.connect(self._highlight_roi)

        # connect analysis from metadata button
        self._analysis_wdg.from_metadata.connect(self._on_led_info_from_meta_clicked)
        # connect the run analysis button
        self._analysis_wdg.run.connect(self._on_run_analysis_clicked)
        self._analysis_wdg.cancel.connect(self._analysis_runner.cancel)
        # connect analysis runner signal
        # self._analysis_runner.analysisInfo.connect(self._on_analysis_info)

        # connect the run detection button
        self._detection_wdg.run.connect(self._on_run_detection_clicked)
        self._detection_wdg.cancel.connect(self._detection_runner.cancel)

        # TODO: FIX ME FOR NEW GUI
        # self._segmentation_wdg.segmentationFinished.connect(
        #     self._on_fov_table_selection_changed
        #

        # self._analysis_wdg._frame_rate_wdg._from_meta_btn.clicked.connect(
        #     self._on_frame_rate_info_from_meta_clicked
        # )

        # FINALIZE WINDOW ------------------------------------------------------------
        self.showMaximized()
        self._set_splitter_sizes()

        # TO REMOVE, IT IS ONLY TO TEST________________________________________________
        # fmt off

        # data = "tests/test_data/evoked/evk.tensorstore.zarr"
        # self._analysis_path = "/Users/fdrgsp/Desktop/cali_test"
        # self.initialize_widget_from_directories(data, self._analysis_path)

        # data = "tests/test_data/spontaneous/spont.tensorstore.zarr"
        # self._analysis_path = "/Users/fdrgsp/Desktop/cali_test"
        # self.initialize_widget_from_directories(data self._analysis_path)

        # data = "tests/test_data/spontaneous/spont.tensorstore.zarr"
        # self._analysis_path = "/Users/fdrgsp/Desktop/cali_test"
        # self.initialize_widget_from_directories(data, self._analysis_path)

        data = "tests/test_data/evoked/database/cali.db"
        self.initialize_widget_from_database(data)

        # data = "tests/test_data/spontaneous/spont_analysis/spont.tensorstore.zarr.db"
        # self.initialize_widget_from_database(data)

        # fmt: on
        # ____________________________________________________________________________

    # PUBLIC METHODS-------------------------------------------------------------------
    def initialize_widget_from_database(self, database_path: str | Path) -> None:
        """Initialize the widget with the given database path."""
        # SHOW LOADING BAR ------------------------------------------------------------
        self._init_loading_bar("Initializing cali from database...", False)

        # CLEARING---------------------------------------------------------------------
        self._clear_widget_before_initialization()

        # OPEN THE DATABASE -----------------------------------------------------------
        cali_logger.info(f"💿 Loading experiment from database at {database_path}")
        # load the first experiment from the database (there should be only one)
        exp = Experiment.load_from_db(database_path)

        # DATA-------------------------------------------------------------------------
        self._data = load_data(exp.data_path)
        if self._data is None:
            msg = (
                f"Unsupported file format! Only {WRITERS[ZARR_TESNSORSTORE][0]} and"
                f" {WRITERS[OME_ZARR][0]} are supported."
            )
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            self._loading_bar.hide()
            return

        if self._data.sequence is None:
            msg = (
                "useq.MDASequence not found! Cannot use the  `CaliGui` without "
                "the useq.MDASequence in the datastore metadata!"
            )
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            self._loading_bar.hide()
            return

        # ASSIGN VARIABLES ------------------------------------------------------------
        self._database_path = Path(exp.analysis_path) / exp.database_name
        self._analysis_path = exp.analysis_path

        # PASS DATABASE PATH TO GRAPHS WIDGETS ----------------------------------------
        self._update_graph_with_database_path(self._database_path)

        # PLATE------------------------------------------------------------------------
        plate_plan = experiment_to_useq_plate_plan(exp)
        if plate_plan is not None:
            self._draw_plate_with_selection(plate_plan)
        else:
            cali_logger.warning("❌ Plate plan not found in experiment.")

        # UPDATE GUI-------------------------------------------------------------------
        self._update_gui_settings(self._database_path)

        # HIDE LOADING BAR ------------------------------------------------------------
        self._loading_bar.hide()

    def _update_gui_settings(self, database_path: Path) -> None:
        """Update the GUI settings based on the latest analysis result."""
        # load the latest analysis result
        latest_result = AnalysisResult.load_from_database(database_path)
        if isinstance(latest_result, list):
            latest_result = latest_result[0]

        # get and set the latest analysis settings
        d_id = latest_result.detection_settings
        d_settings = DetectionSettings.load_from_database(database_path, id=d_id)
        assert isinstance(d_settings, DetectionSettings)  # it cannot be a list here
        if d_settings.method == "cellpose":
            model_options = [
                self._detection_wdg._cellpose_wdg._models_combo.itemText(i)
                for i in range(self._detection_wdg._cellpose_wdg._models_combo.count())
            ]
            model_path = (
                d_settings.model_type
                if d_settings.model_type not in model_options
                else None
            )
            self._detection_wdg.setValue(
                CellposeSettings(
                    model_type=d_settings.model_type,
                    model_path=model_path,
                    diameter=d_settings.diameter,
                    cellprob_threshold=d_settings.cellprob_threshold,
                    flow_threshold=d_settings.flow_threshold,
                    min_size=d_settings.min_size,
                    normalize=d_settings.normalize,
                    batch_size=d_settings.batch_size,
                )
            )
        elif d_settings.method == "caiman":
            self._detection_wdg.setValue(CaimanSettings())
        else:
            raise ValueError(f"Unknown detection method: {d_settings.method}.")

        # get and set the latest analysis settings
        a_id = latest_result.analysis_settings
        a_settings = AnalysisSettings.load_from_database(database_path, id=a_id)
        assert isinstance(a_settings, AnalysisSettings)  # it cannot be a list here
        self._analysis_wdg.setValue(
            AnalysisSettingsData(
                experiment_type_data=ExperimentTypeData(
                    experiment_type=(
                        EVOKED
                        if a_settings.stimulated_mask_area() is not None
                        else SPONTANEOUS
                    ),
                    led_power_equation=a_settings.led_power_equation,
                    led_pulse_duration=a_settings.led_pulse_duration,
                    led_pulse_on_frames=a_settings.led_pulse_on_frames,
                    led_pulse_powers=a_settings.led_pulse_powers,
                    stimulation_area_path=a_settings.stimulation_mask_path,
                ),
                trace_extraction_data=TraceExtractionData(
                    dff_window_size=a_settings.dff_window,
                    decay_constant=a_settings.decay_constant,
                    neuropil_inner_radius=a_settings.neuropil_inner_radius,
                    neuropil_min_pixels=a_settings.neuropil_min_pixels,
                    neuropil_correction_factor=a_settings.neuropil_correction_factor,
                ),
                calcium_peaks_data=CalciumPeaksData(
                    peaks_height=a_settings.peaks_height_value,
                    peaks_height_mode=a_settings.peaks_height_mode,
                    peaks_distance=a_settings.peaks_distance,
                    peaks_prominence_multiplier=a_settings.peaks_prominence_multiplier,
                    calcium_synchrony_jitter=a_settings.calcium_sync_jitter_window,
                    calcium_network_threshold=a_settings.calcium_network_threshold,
                ),
                spikes_data=SpikeData(
                    spike_threshold=a_settings.spike_threshold_value,
                    spike_threshold_mode=a_settings.spike_threshold_mode,
                    burst_threshold=a_settings.burst_threshold,
                    burst_min_duration=a_settings.burst_min_duration,
                    burst_blur_sigma=a_settings.burst_gaussian_sigma,
                    synchrony_lag=a_settings.spikes_sync_cross_corr_lag,
                )
            )
        )
        # load plate plan data
        exp = Experiment.load_from_db(database_path)
        plate = experiment_to_useq_plate(exp)
        plate_map_data = experiment_to_plate_map_data(exp)
        if plate_map_data is not None and plate is not None:
            self._analysis_wdg._plate_map_wdg.setValue(plate, *plate_map_data)

    def initialize_widget_from_directories(
        self, data_path: str, analysis_path: str
    ) -> None:
        """Initialize the widget with given datastore and analysis path."""
        # SHOW LOADING BAR ------------------------------------------------------------
        self._init_loading_bar("Initializing cali from directories...", False)

        # CLEARING---------------------------------------------------------------------
        self._clear_widget_before_initialization()

        # DATASTORE--------------------------------------------------------------------
        self._data = load_data(data_path)
        if self._data is None:
            msg = (
                f"Unsupported file format! Only {WRITERS[ZARR_TESNSORSTORE][0]} and"
                f" {WRITERS[OME_ZARR][0]} are supported."
            )
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            self._loading_bar.hide()
            return

        if self._data.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot use the  `CaliGui` without "
                "the useq.MDASequence in the datastore metadata!",
            )
            self._loading_bar.hide()
            return

        # ASSIGN VARIABLES ------------------------------------------------------------
        self._data_path = data_path
        self._analysis_path = analysis_path
        database_name = f"{Path(data_path).name}.db"
        self._database_path = Path(analysis_path) / database_name

        # PASS DATABASE PATH TO GRAPHS WIDGETS ----------------------------------------
        self._update_graph_with_database_path(self._database_path)

        # CREATE THE EXPERIMENT BASED ON DATA -----------------------------------------
        experiment = Experiment.create_from_data(
            name="Cali Experiment",
            data_path=data_path,
            analysis_path=analysis_path,
            database_name=database_name,
        )

        # SAVE THE EXPERIMENT TO A NEW DATABASE----------------------------------------
        # TODO: ask the user to overwrite if the database already exists
        cali_logger.info(f"💾 Creating new database at {self._database_path}")
        save_experiment_to_database(experiment, overwrite=True)

        # UPDATE GUI-------------------------------------------------------------------
        self._update_gui_plate_plan(self._data.sequence.stage_positions)

        # HIDE LOADING BAR ------------------------------------------------------------
        self._loading_bar.hide()

    # RUNNING THE DETECTION------------------------------------------------------------
    def _on_run_detection_clicked(self) -> None: ...

    # RUNNING THE ANALYSIS-------------------------------------------------------------
    def _on_run_analysis_clicked(self) -> None:

        create_worker(
            self._analysis_runner.run,
            _start_thread=True,
            _connect={"errored": self._on_worker_errored},
        )
        ...
        # exp = self.experiment()
        # if exp is None:
        #     return

        # # Check for settings consistency before running analysis
        # if not self._check_analysis_settings_before_run(exp):
        #     return

        # # update the experiment analysis settings
        # self._update_experiment_analysis_settings()

    def _check_analysis_settings_before_run(self, experiment: Experiment) -> bool:
        """Check if current GUI settings differ from experiment's analysis settings.

        Returns
        -------
        bool
            True if it's safe to proceed with analysis, False if user cancelled
        """
        return True
        # from qtpy.QtWidgets import QMessageBox

        # if experiment is None:
        #     return False

        # # If no existing analysis settings, safe to proceed
        # if experiment.analysis_settings is None:
        #     return True

        # # Get current GUI settings
        # new_settings = self._analysis_wdg.to_model_settings(experiment.id or 0)[1]
        # # Exclude 'id' and 'created_at' from comparison since it's database-specific
        # new_settings_dict = new_settings.model_dump(exclude={"id", "created_at"})

        # # Compare experiment settings with current GUI settings
        # # Exclude 'id'  and 'created_at' since it's database-specific
        # existing_settings = experiment.analysis_settings
        # existing_settings_dict = existing_settings.model_dump(
        #     exclude={"id", "created_at"}
        # )

        # if existing_settings_dict != new_settings_dict:
        #     msg_box = QMessageBox(self)
        #     msg_box.setIcon(QMessageBox.Icon.Warning)
        #     msg_box.setWindowTitle("Different Analysis Settings Detected!")
        #     msg_box.setText(
        #         "The settings stored in the current database during previous analysis "
        #         "run are different from the ones in the GUI.\n\n"
        #         "If you continue and click ok 'OK', the previous analysis results will "
        #         "be deleted from the database and update it with the new settings and "
        #         "newly analyzed positions.\n\n"
        #         "Options:\n"
        #         "• Ok: Delete previous results and run new analysis\n"
        #         "• Cancel: Keep existing data and do not run analysis\n\n"
        #         "NOTE: To keep the old database, please set a new analysis path from "
        #         "the 'Load Data and Set Directories...' menu so that the analysis will "
        #         "be saved in a new database while keeping the old one intact."
        #     )
        #     msg_box.setStandardButtons(
        #         QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        #     )
        #     msg_box.setDefaultButton(QMessageBox.StandardButton.Cancel)

        #     result = msg_box.exec()
        #     if result == QMessageBox.StandardButton.Cancel:
        #         return False

        #     self._delete_all_analysis_data()

        # return True

    # def _delete_all_analysis_data(self) -> None:
    #     """Delete all existing analysis data (ROIs) from the experiment."""
    #     # Delegate the clearing logic to the analysis runner
    #     self._analysis_runner.clear_analysis_results()

    # def _update_experiment_analysis_settings(self) -> None:
    #     exp = self.experiment()
    #     if exp is None or self._data is None:
    #         return

    #     # Ensure experiment has an ID (should be set if loaded from DB)
    #     if exp.id is None:
    #         cali_logger.warning("Experiment has no ID, cannot update analysis settings")
    #         return

    #     # Update or set the experiment's type based on gui state
    #     exp_type = self._analysis_wdg._experiment_type_wdg.value()
    #     exp.experiment_type = exp_type.experiment_type or SPONTANEOUS

    #     # Get positions to analyze and new settings from GUI
    #     pos, new_settings = self._analysis_wdg.to_model_settings(exp.id)

    #     # Update positions to analyze based on selected wells in the plate view
    #     # If no position selected, analyze all positions from the data
    #     if len(pos) == 0:
    #         if self._data.sequence is None:
    #             show_error_dialog(
    #                 self,
    #                 "No MDASequence found in the datastore! Cannot determine "
    #                 "positions to analyze.",
    #             )
    #             return
    #         pos = list(range(len(self._data.sequence.stage_positions)))
    #     exp.positions_analyzed = pos

    #     # Update existing settings or create new one
    #     if exp.analysis_settings is not None:
    #         exp.analysis_settings.sqlmodel_update(
    #             new_settings.model_dump(exclude={"id"})
    #         )
    #     else:
    #         # Create new settings
    #         exp.analysis_settings = new_settings

    #     # Update the analysis runner with the current data, experiment and settings
    #     # This will also save the experiment to the database before running analysis
    #     self._analysis_runner.set_experiment(exp)

    #     create_worker(
    #         self._analysis_runner.run,
    #         _start_thread=True,
    #         _connect={"errored": self._on_worker_errored},
    #     )

    def _on_worker_errored(self) -> None:
        cali_logger.error("Analysis runner encountered an error during execution.")

    def _on_analysis_info(self, msg: str, type: str) -> None:
        """Handle analysis info messages from the analysis runner."""
        print(f"ANALYSIS INFO: {msg}")
        # cannot do that...I need to accumulate and show when the work is done!
        # if type == "error":
        #     show_error_dialog(self, msg)

    # DATA INITIALIZATION--------------------------------------------------------------

    def _show_data_input_dialog(self) -> None:
        """Show dialog to select zarr datastore, segmentation and analysis path."""
        init_dialog = _InputDialog(
            self,
            data_path=(str(self._data.path) if self._data is not None else None),
            analysis_path=self._analysis_path,
        )
        init_dialog.resize(700, init_dialog.sizeHint().height())
        if init_dialog.exec():
            value = init_dialog.value()
            # input from database
            if value.database_path is not None:
                self.initialize_widget_from_database(value.database_path)
            # input from directories
            elif (data_path := value.data_path) is not None:
                if value.analysis_path is None:
                    msg = (
                        "Analysis path must be provided to create the analysis "
                        "database!"
                    )
                    show_error_dialog(self, msg)
                    cali_logger.error(msg)
                    return
                self.initialize_widget_from_directories(data_path, value.analysis_path)

    def _clear_widget_before_initialization(self) -> None:
        """Clear the widget before initializing it with new data."""
        # clear paths
        self._database_path = None
        self._data_path = None
        self._analysis_path = None
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
        self._analysis_wdg.reset()
        # reset detection widget gui
        self._detection_wdg.reset()

        # clear the segmentation widget - TO REMOVE
        # self._segmentation_wdg.experiment = None
        # self._segmentation_wdg.data = None
        # self._segmentation_wdg.labels_path = None

    def _update_graph_with_database_path(self, database_path: Path) -> None:
        """Update all graph widgets with the current database path."""
        for sw_graph in self.SW_GRAPHS:
            sw_graph.database_path = database_path
        for mw_graph in self.MW_GRAPHS:
            mw_graph.database_path = database_path

    def _update_gui_plate_plan(
        self, plate_plan: useq.WellPlatePlan | tuple[useq.Position, ...] | None = None
    ) -> None:
        """Update the gui based on the specified plate plan."""
        if self._data is None or plate_plan is None:
            return None

        final_plate_plan: useq.WellPlatePlan | None = None

        # if already a WellPlatePlan, use it directly
        if isinstance(plate_plan, useq.WellPlatePlan):
            final_plate_plan = plate_plan
        else:
            # plate_plan is a tuple of positions - need to create a plate plan
            # try to use the plate plan wizard first
            final_plate_plan = self._resolve_plate_plan()

            # set the flag if using default plate plan
            if final_plate_plan == DEFAULT_PLATE_PLAN:
                self._default_plate_plan = True

        if final_plate_plan is None:
            return None

        self._draw_plate_with_selection(final_plate_plan)

    def _resolve_plate_plan(self) -> useq.WellPlatePlan | None:
        """Resolve plate plan from various sources in order of preference."""
        # try using the wizard
        if self._plate_plan_wizard.exec():
            return self._plate_plan_wizard.value()
        # if no HCSWizard was used but single position list was created,
        # fallback to a default square coverslip plate plan
        return DEFAULT_PLATE_PLAN

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

    def _update_gui(self, plate: useq.WellPlate | None = None) -> None:
        """Update the analysis widgets gui."""
        # analysis widget
        self._update_analysis_gui_settings(plate)

        # # segmentation widget - TO REMOVE
        # self._segmentation_wdg.data = self._data
        # self._segmentation_wdg.labels_path = self._labels_path

    def _update_analysis_gui_settings(
        self, plate: useq.WellPlate | None = None
    ) -> None:
        """Update the analysis widgets settings."""
        exp = self.experiment()
        if exp is None:
            self._analysis_wdg.reset()
            return

        settings = exp.analysis_settings
        if settings is None:
            self._analysis_wdg.reset()
            return

        plate_map_data = None
        if plate is not None:
            plate_map_data = (plate, *experiment_to_plate_map_data(exp))

        value = AnalysisSettingsData(
            plate_map_data=plate_map_data,
            experiment_type_data=ExperimentTypeData(
                experiment_type=exp.experiment_type,
                led_power_equation=settings.led_power_equation,
                stimulation_area_path=settings.stimulation_mask_path,
                led_pulse_duration=settings.led_pulse_duration,
                led_pulse_powers=settings.led_pulse_powers,
                led_pulse_on_frames=settings.led_pulse_on_frames,
            ),
            trace_extraction_data=TraceExtractionData(
                dff_window_size=settings.dff_window,
                decay_constant=settings.decay_constant,
                neuropil_inner_radius=settings.neuropil_inner_radius,
                neuropil_min_pixels=settings.neuropil_min_pixels,
                neuropil_correction_factor=settings.neuropil_correction_factor,
            ),
            calcium_peaks_data=CalciumPeaksData(
                peaks_height=settings.peaks_height_value,
                peaks_height_mode=settings.peaks_height_mode,
                peaks_distance=settings.peaks_distance,
                peaks_prominence_multiplier=settings.peaks_prominence_multiplier,
                calcium_synchrony_jitter=settings.calcium_sync_jitter_window,
                calcium_network_threshold=settings.calcium_network_threshold,
            ),
            spikes_data=SpikeData(
                spike_threshold=settings.spike_threshold_value,
                spike_threshold_mode=settings.spike_threshold_mode,
                burst_threshold=settings.burst_threshold,
                burst_min_duration=settings.burst_min_duration,
                burst_blur_sigma=settings.burst_gaussian_sigma,
                synchrony_lag=settings.spikes_sync_cross_corr_lag,
            ),
        )
        self.set_analysis_settings(value)

    # ---------------------WIDGETS------------------------------------

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
            show_error_dialog(
                self, "Data not loaded! Cannot load metadata from datastore!"
            )
            return

        try:
            if (sequence := self._data.sequence) is None:
                msg = "useq.MDASequence not found! Cannot retrieve metadata!"
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
                        f"Loaded stimulation metadata from datastore: "
                        f"led_pulse_duration={led_duration}"
                        f"led_powers={wdg._led_powers_le.text()}, "
                        f"led_pulse_on_frames={wdg._led_pulse_on_frames_le.text()}"
                    )

            else:
                msg = "No stimulation metadata found in the datastore!"
                show_error_dialog(self, msg)
                cali_logger.warning(msg)

        except Exception as e:
            msg = f"Failed to load metadata from datastore!\n\nError: {e}"
            show_error_dialog(self, msg)
            cali_logger.error(msg)
            return

    def _init_loading_bar(self, text: str, show_progress_bar: bool = True) -> None:
        """Reset the loading bar."""
        self._loading_bar.setEnabled(True)
        self._loading_bar.setText(text)
        self._loading_bar.setValue(0)
        self._loading_bar.showPercentage(True)
        self._loading_bar.show_progress_bar(show_progress_bar)
        self._loading_bar.show()

    def _update_graphs_with_roi(self, roi: int) -> None:
        """Update the graphs with the given roi.

        This function is called when a roi is selected in the image viewer and will
        update the graphs with the traces of the selected roi.
        """
        # get the current tab index
        idx = self._tab.currentIndex()
        if idx == 0 or idx == 1:
            return
        for graph in self.SW_GRAPHS:
            if graph._combo.currentText() == "None":
                continue
            graph._choose_dysplayed_traces.setChecked(True)
            graph._choose_dysplayed_traces._roi_le.setText(str(roi))
            graph._choose_dysplayed_traces._update()

    def _on_tab_changed(self, idx: int) -> None:
        """Update the graph combo boxes when the tab is changed."""
        # skip if the tab is the segmentation tab or analysis tab
        if idx == 0 or idx == 1:
            return

        # if single wells tab is selected
        if idx == 2:
            # get the current fov
            value = self._fov_table.value() if self._fov_table.selectedItems() else None
            if value is None:
                return

            from rich import print

            print(f"ON TAB CHANGED - Selected FOV value: {value}")

            # check if the FOV has been analyzed (has ROIs with data)
            has_analysis = self._has_fov_analysis(value)

            # update the graphs combo boxes
            self._update_single_wells_graphs_combo(combo_red=(not has_analysis))

        # if multi wells tab is selected
        elif idx == 3:
            self._update_multi_wells_graphs_combo()

    def _highlight_roi(self, roi: str | list[str]) -> None:
        """Highlight the selected roi in the image viewer."""
        if isinstance(roi, list):
            roi = ",".join(roi)
        self._image_viewer._roi_number_le.setText(roi)
        self._image_viewer._highlight_rois()

    def _on_scene_well_changed(self) -> None:
        """Update the FOV table when a well is selected."""
        self._fov_table.clear()
        self._image_viewer._clear_highlight()

        if self._data is None:
            return

        if self._data.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot retrieve the Well data without "
                "the tensorstore useq.MDASequence!",
            )
            return

        well_dict: set[QAbstractGraphicsShapeItem] = self._plate_view._selected_items
        if not well_dict or len(well_dict) != 1:
            return
        well_name = next(iter(well_dict)).data(DATA_POSITION).name

        # add the fov per position to the table
        for idx, pos in enumerate(self._data.sequence.stage_positions):
            if self._default_plate_plan or (pos.name and well_name in pos.name):
                self._fov_table.add_position(WellInfo(idx, pos))

        if self._fov_table.rowCount() > 0:
            self._fov_table.selectRow(0)

    def _on_fov_table_selection_changed(self) -> None:
        """Update the image viewer with the first frame of the selected FOV."""
        self._image_viewer._clear_highlight()
        value = self._fov_table.value() if self._fov_table.selectedItems() else None

        if value is None:
            self._image_viewer.setData(None, None)
            self._update_single_wells_graphs_combo(combo_red=True, clear=True)
            return

        if self._data is None:
            return

        if not self._data.sequence:
            return

        # get a single frame for the selected FOV (at 2/3 of the time points)
        t = int(len(self._data.sequence.stage_positions) / 3 * 2)
        data = cast("np.ndarray", self._data.isel(p=value.pos_idx, t=t, c=0))
        # get labels if they exist
        labels = self._get_labels(value)
        # flip data and labels vertically or will look different from the StackViewer
        data = np.flip(data, axis=0)
        labels = np.flip(labels, axis=0) if labels is not None else None
        self._image_viewer.setData(data, labels)
        self._set_graphs_fov(value)

        # Check if the FOV has been analyzed (has ROIs with data)
        has_analysis = self._has_fov_analysis(value)
        self._update_single_wells_graphs_combo(
            combo_red=(not has_analysis), clear=(not has_analysis)
        )

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

        # the FOV name in the database includes the position index suffix
        fov_name = f"{fov_name}_p{value.pos_idx}"

        return has_fov_analysis(self._database_path, fov_name)

    def _set_graphs_fov(self, value: WellInfo | None) -> None:
        """Set the FOV title for the graphs."""
        if value is None:
            return
        title = value.fov.name or f"Position {value.pos_idx}"
        title = f"{title}_p{value.pos_idx}"
        self._update_single_wells_graphs_combo(set_title=title)

    def _get_labels(self, value: WellInfo) -> np.ndarray | None:
        """Get the labels for the given FOV."""
        if self._labels_path is None:
            return None

        if not Path(self._labels_path).is_dir():
            show_error_dialog(
                self,
                f"Error while loading the labels. Path {self._labels_path} is not a "
                "directory!",
            )
            return None
        # the labels tif file should have the same name as the position
        # and should end with _pn where n is the position number (e.g. C3_0000_p0.tif)
        pos_idx = f"p{value.pos_idx}"
        pos_name = value.fov.name
        for f in Path(self._labels_path).iterdir():
            name = f.name.replace(f.suffix, "")
            if pos_name and pos_name in f.name and name.endswith(f"_{pos_idx}"):
                return tifffile.imread(f)  # type: ignore
        return None

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
        set_title: str | None = None,
        combo_red: bool = False,
        clear: bool = False,
    ) -> None:
        for sw_graph in self.SW_GRAPHS:
            if set_title is not None:
                sw_graph.fov = set_title

            if clear:
                sw_graph.clear_plot()

            sw_graph.set_combo_text_red(combo_red)

    def _update_multi_wells_graphs_combo(self) -> None:
        if self._database_path is None:
            has_analysis = False
        else:
            has_analysis = has_experiment_analysis(self._database_path)

        for mw_graph in self.MW_GRAPHS:
            mw_graph.set_combo_text_red(not has_analysis)

    # MENU SAVE ACTIONS----------------------------------------------------------------

    def _show_save_as_tiff_dialog(self) -> None:
        """Show the save as tiff dialog."""
        if self._data is None or (sequence := self._data.sequence) is None:
            show_error_dialog(
                self,
                "No data to save or useq.MDASequence not found! Cannot save the data.",
            )
            return

        dialog = _SaveAsTiff(self)

        if dialog.exec():
            path, positions = dialog.value()

            if not Path(path).is_dir():
                show_error_dialog(
                    self, f"The path {path} is not a directory! Cannot save the data."
                )
                return

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
            show_error_dialog(self, "No data to save! Run or load analysis data first.")
            return

        if not has_experiment_analysis(self._database_path):
            show_error_dialog(self, "No data to save! Run or load analysis data first.")
            return

        dialog = _SaveAsCSV(self)
        dialog.resize(500, dialog.sizeHint().height())

        if dialog.exec():
            path = dialog.value()
            if not Path(path).is_dir():
                show_error_dialog(
                    self, f"The path {path} is not a directory! Cannot save the data."
                )
                return

            # TODO: Update these functions to work with SQLModel Experiment
            # save_trace_data_to_csv(path, self._experiment)
            # save_analysis_data_to_csv(path, self._experiment)
            show_error_dialog(
                self,
                "CSV export is not yet implemented for SQLModel. "
                "Please use the database export instead.",
            )

    # def _on_frame_rate_info_from_meta_clicked(self) -> None:
    #     if self._data is None:
    #         show_error_dialog(
    #             self, "Data not loaded! Cannot load metadata from datastore!"
    #         )
    #         return

    #     try:
    #         # Get metadata directly from the data reader
    #         if not (meta := cast("list[dict]", self._data.metadata)):
    #             msg = "No metadata found in the datastore!"
    #             show_error_dialog(self, msg)
    #             cali_logger.error(msg)
    #             return

    #         if (sequence := self._data.sequence) is None:
    #             msg = "useq.MDASequence not found! Cannot retrieve metadata!"
    #             show_error_dialog(self, msg)
    #             cali_logger.error(msg)
    #             return

    #         # Get exposure time from metadata (first frame)
    #         exp_time = meta[0].get("mda_event", {}).get("exposure", 0.0)
    #         if exp_time <= 0:
    #             msg = "Invalid exposure time found in metadata!"
    #             show_error_dialog(self, msg)
    #             cali_logger.error(msg)
    #             return

    #         # Get timepoints
    #         timepoints = sequence.sizes.get("t", 0)
    #         if timepoints == 0:
    #             msg = "No timepoints found in the sequence!"
    #             show_error_dialog(self, msg)
    #             cali_logger.error(msg)
    #             return

    #         frame_rate = (timepoints - 1) / ((timepoints * exp_time) / 1000)
    #         self._analysis_wdg._frame_rate_wdg._frame_rate_spin.setValue(frame_rate)

    #         cali_logger.info(f"Frame rate set to: {frame_rate:.2f} fps.")

    #     except Exception as e:
    #         msg = f"Failed to load frame rate from datastore!\n\nError: {e}"
    #         show_error_dialog(self, msg)
    #         cali_logger.error(msg)
    #         return
