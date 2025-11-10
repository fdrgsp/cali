from __future__ import annotations

import datetime
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
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from cali._plate_viewer._analysis_with_sqlmodel import AnalysisRunner
from cali._util import OME_ZARR, WRITERS, ZARR_TESNSORSTORE
from cali.cali_logger import LOGGER
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.sqlmodel import (
    Experiment,
    load_experiment_from_database,
    save_experiment_to_database,
    useq_plate_plan_to_db,
)
from cali.sqlmodel._db_to_plate_map import experiment_to_plate_map_data
from cali.sqlmodel._db_to_useq_plate import (
    experiment_to_useq_plate_plan,
)

from ._analysis_gui import (
    AnalysisSettingsData,
    CalciumPeaksData,
    ExperimentTypeData,
    SpikeData,
    TraceExtractionData,
    _CalciumAnalysisGUI,
)
from ._fov_table import WellInfo, _FOVTable
from ._graph_widgets import _MultilWellGraphWidget, _SingleWellGraphWidget
from ._image_viewer import _ImageViewer
from ._init_dialog import _InputDialog
from ._plate_plan_wizard import PlatePlanWizard
from ._save_as_widgets import _SaveAsCSV, _SaveAsTiff

# from ._segmentation import _CellposeSegmentation
from ._segmentation import _CellposeSegmentation
from ._to_csv import save_analysis_data_to_csv, save_trace_data_to_csv
from ._util import (
    EVENT_KEY,
    SPONTANEOUS,
    ROIData,
    _ProgressBarWidget,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator


HCS = "hcs"
UNSELECTABLE_COLOR = "#404040"
DEFAULT_PLATE_PLAN = useq.WellPlatePlan(
    plate=useq.WellPlate.from_str("coverslip-18mm-square"),
    a1_center_xy=(0.0, 0.0),
    selected_wells=((0,), (0,)),
)
PYMMCW_METADATA_KEY = "pymmcore_widgets"
TS = WRITERS[ZARR_TESNSORSTORE][0]
ZR = WRITERS[OME_ZARR][0]


class PlateViewer(QMainWindow):
    """A widget for displaying a plate preview."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Plate Viewer")
        self.setWindowIcon(QIcon(icon(MDI6.view_comfy, color="#00FF00")))

        # INTERNAL VARIABLES ---------------------------------------------------------
        self._database_path: Path | None = None
        self._labels_path: str | None = None
        self._analysis_path: str | None = None
        self._experiment: Experiment | None = None
        self._data: TensorstoreZarrReader | OMEZarrReader | None = None
        self._analysis_data: dict[str, dict[str, ROIData]] = {}

        # RUNNER ---------------------------------------------------------------------
        self._analysis_runner: AnalysisRunner = AnalysisRunner()

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

        # SEGMENTATION TAB ------------------------------------------------------------
        self._segmentation_tab = QWidget()
        self._tab.addTab(self._segmentation_tab, "Segmentation Tab")
        segmentation_tab_layout = QVBoxLayout(self._segmentation_tab)
        segmentation_tab_layout.setContentsMargins(0, 0, 0, 0)

        # SEGMENTATION TAB SCROLL AREA ------------------------------------------------
        segmentation_scroll_area = QScrollArea()
        segmentation_scroll_area.setWidgetResizable(True)
        segmentation_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        segmentation_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # SEGMENTATION WIDGET ---------------------------------------------------------
        self._segmentation_wdg = _CellposeSegmentation(self)
        segmentation_content_widget = QWidget()
        segmentation_layout = QVBoxLayout(segmentation_content_widget)
        segmentation_layout.setContentsMargins(10, 10, 10, 10)
        segmentation_layout.setSpacing(15)
        segmentation_layout.addWidget(self._segmentation_wdg)
        segmentation_layout.addStretch(1)
        segmentation_scroll_area.setWidget(segmentation_content_widget)
        segmentation_tab_layout.addWidget(segmentation_scroll_area)

        # ANALYSIS TAB ----------------------------------------------------------------
        self._analysis_tab = QWidget()
        self._tab.addTab(self._analysis_tab, "Analysis Tab")
        analysis_tab_layout = QVBoxLayout(self._analysis_tab)
        analysis_tab_layout.setContentsMargins(0, 0, 0, 0)

        # ANALYSIS TAB SCROLL AREA ----------------------------------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # ANALYSIS WIDGET -------------------------------------------------------------
        self._analysis_wdg = _CalciumAnalysisGUI(self)
        analysis_content_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_content_widget)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(15)
        analysis_layout.addWidget(self._analysis_wdg)
        analysis_layout.addStretch(1)
        scroll_area.setWidget(analysis_content_widget)
        analysis_tab_layout.addWidget(scroll_area)

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
        self._segmentation_wdg.segmentationFinished.connect(
            self._on_fov_table_selection_changed
        )
        # connect the roiSelected signal from the graphs to the image viewer so we can
        # highlight the roi in the image viewer when a roi is selected in the graph
        for graph in self.SW_GRAPHS:
            graph.roiSelected.connect(self._highlight_roi)
        # connect meta button
        self._analysis_wdg._experiment_type_wdg._from_meta_btn.clicked.connect(
            self._on_led_info_from_meta_clicked
        )
        # connect analysis runner signal
        self._analysis_runner.analysisInfo.connect(self._on_analysis_info)
        # connect the run analysis button
        self._analysis_wdg._run_analysis_wdg._run_btn.clicked.connect(
            self._on_run_analysis_clicked
        )
        self._analysis_wdg._run_analysis_wdg._cancel_btn.clicked.connect(
            self._analysis_runner.cancel
        )
        # self._analysis_wdg._frame_rate_wdg._from_meta_btn.clicked.connect(
        #     self._on_frame_rate_info_from_meta_clicked
        # )

        # FINALIZE WINDOW ------------------------------------------------------------
        self.showMaximized()
        self._set_splitter_sizes()

        # TO REMOVE, IT IS ONLY TO TEST________________________________________________
        # fmt off

        data = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"  # noqa: E501
        self._pv_labels_path = (
            "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"
        )
        self._pv_analysis_path = (
            "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis"
        )
        self.initialize_widget_from_directories(
            data, self._pv_labels_path, self._pv_analysis_path
        )

        # data = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont.tensorstore.zarr"  # noqa: E501
        # self._labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_labels"  # noqa: E501
        # self._analysis_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_analysis"  # noqa: E501
        # self.initialize_widget_from_directories(data, self._labels_path, self._analysis_path)  # noqa: E501

        # data = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis/cali.db"  # noqa: E501
        # self.initialize_widget_from_database(data)

        # data = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont.tensorstore.zarr"  # noqa: E501
        # self._labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_labels"  # noqa: E501
        # self._analysis_path = "/Users/fdrgsp/Desktop/cali_test"
        # self.initialize_widget_from_directories(data, self._analysis_path, self._labels_path)  # noqa: E501

        # fmt: on
        # ____________________________________________________________________________

    # PUBLIC METHODS-------------------------------------------------------------------
    def initialize_widget_from_database(self, database_path: str | Path) -> None:
        """Initialize the widget with the given database path."""
        # CLEARING---------------------------------------------------------------------
        self._clear_widget_before_initialization()

        # SHOW LOADING BAR ------------------------------------------------------------
        self._init_loading_bar("Initializing cali from database...", False)

        # OPEN THE DATABASE -----------------------------------------------------------
        LOGGER.info(f"💿 Loading experiment from database at {database_path}")
        self._experiment = load_experiment_from_database(database_path)
        if self._experiment is None:
            msg = f"Could not load experiment from database at {database_path}!"
            show_error_dialog(self, msg)
            LOGGER.error(msg)
            self._loading_bar.hide()  # Close entire dialog on error
            return

        self._database_path = Path(database_path)

        data_path = self._experiment.data_path
        if data_path is None:
            msg = "Data path not found in the database! Cannot initialize the "
            "PlateViewer without a valid data path."
            show_error_dialog(self, msg)
            LOGGER.error(msg)
            self._loading_bar.hide()  # Close entire dialog on error
            return

        self._analysis_path = self._experiment.analysis_path
        self._labels_path = self._experiment.labels_path

        # DATA-------------------------------------------------------------------------

        # select which reader to use for the datastore
        if data_path.endswith(TS):
            # read tensorstore
            self._data = TensorstoreZarrReader(data_path)
        elif data_path.endswith(ZR):
            # read ome zarr
            self._data = OMEZarrReader(data_path)
        else:
            self._data = None
            msg = (
                f"Unsupported file format! Only {WRITERS[ZARR_TESNSORSTORE][0]} and"
                f" {WRITERS[OME_ZARR][0]} are supported."
            )
            show_error_dialog(self, msg)
            LOGGER.error(msg)
            self._loading_bar.hide()  # Close entire dialog on error
            return

        if self._data.sequence is None:
            msg = (
                "useq.MDASequence not found! Cannot use the  `PlateViewer` without "
                "the useq.MDASequence in the datastore metadata!"
            )
            show_error_dialog(self, msg)
            LOGGER.error(msg)
            self._loading_bar.hide()  # Close entire dialog on error
            return

        # PLATE------------------------------------------------------------------------
        plate_plan = experiment_to_useq_plate_plan(self._experiment)
        if plate_plan is not None:
            self._draw_plate_with_selection(plate_plan)

        # UPDATE WIDGETS---------------------------------------------------------------
        self._update_gui(plate_plan.plate if plate_plan is not None else None)

        # HIDE LOADING BAR ------------------------------------------------------------
        self._loading_bar.hide()  # Close entire dialog when done

    def initialize_widget_from_directories(
        self, datastore_path: str, analysis_path: str, labels_path: str | None
    ) -> None:
        """Initialize the widget with given datastore, labels and analysis path."""
        # CLEARING---------------------------------------------------------------------

        self._clear_widget_before_initialization()

        # SHOW LOADING BAR ------------------------------------------------------------
        self._init_loading_bar("Initializing cali from directories...", False)

        # DATASTORE--------------------------------------------------------------------

        # select which reader to use for the datastore
        if datastore_path.endswith(TS):
            # read tensorstore
            self._data = TensorstoreZarrReader(datastore_path)
        elif datastore_path.endswith(ZR):
            # read ome zarr
            self._data = OMEZarrReader(datastore_path)
        else:
            self._data = None
            show_error_dialog(
                self,
                f"Unsupported file format! Only {WRITERS[ZARR_TESNSORSTORE][0]} and"
                f" {WRITERS[OME_ZARR][0]} are supported.",
            )
            self._loading_bar.hide()  # Close entire dialog on error
            return

        if self._data.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot use the  `PlateViewer` without "
                "the useq.MDASequence in the datastore metadata!",
            )
            self._loading_bar.hide()  # Close entire dialog on error
            return

        # CREATE THE DATABASE ---------------------------------------------------------
        self._experiment = Experiment(
            # temporary ID, will be updated when saved to db. Needed for relationships.
            id=0,
            name="Experiment",
            description="A test experiment.",
            created_at=datetime.datetime.now(),
            data_path=datastore_path,
            labels_path=labels_path,
            analysis_path=analysis_path,
        )

        # LOAD ANALYSIS DATA-----------------------------------------------------------
        self._analysis_path = analysis_path
        self._labels_path = labels_path

        # LOAD PLATE-------------------------------------------------------------------
        plate_plan = self._load_plate_plan(self._data.sequence.stage_positions)
        if plate_plan is not None:
            self._experiment.plate = useq_plate_plan_to_db(plate_plan, self._experiment)

        # UPDATE SEGMENTATION AND ANALYSIS WIDGETS-----------------------------------
        self._update_gui(plate_plan.plate if plate_plan is not None else None)

        # SAVE THE EXPERIMENT TO A NEW DATABASE----------------------------------------
        # TODO: ask the user to overwrite if the database already exists
        self._database_path = Path(analysis_path) / "cali.db"
        LOGGER.info(f"💾 Creating new database at {self._database_path}")
        save_experiment_to_database(
            self._experiment, self._database_path, overwrite=True
        )

        # HIDE LOADING BAR ------------------------------------------------------------
        self._loading_bar.hide()  # Close entire dialog when done

    def get_analysis_settings(self) -> AnalysisSettingsData | None:
        """Get the current analysis settings from the analysis widget."""
        return self._analysis_wdg.value()

    def set_analysis_settings(self, value: AnalysisSettingsData) -> None:
        """Set the current analysis settings in the analysis widget."""
        self._analysis_wdg.setValue(value)
        self._analysis_wdg._run_analysis_wdg.reset()

    # RUNNING THE ANALYSIS-------------------------------------------------------------
    def _on_run_analysis_clicked(self) -> None:
        if self._experiment is None:
            return
        # update the experiment analysis settings
        self._update_experiment_analysis_settings()

    def _update_experiment_analysis_settings(self) -> None:
        if self._experiment is None or self._data is None:
            return

        # Ensure experiment has an ID (should be set if loaded from DB)
        if self._experiment.id is None:
            LOGGER.warning("Experiment has no ID, cannot update analysis settings")
            return

        # Update or set the experiment's type based on gui state
        exp_type = self._analysis_wdg._experiment_type_wdg.value()
        self._experiment.experiment_type = exp_type.experiment_type or SPONTANEOUS

        # Get positions to analyze and new settings from GUI
        pos, new_settings = self._analysis_wdg.to_model_settings(self._experiment.id)

        # Update positions to analyze based on selected wells in the plate view
        # If no position selected, analyze all positions from the data
        if len(pos) == 0:
            if self._data.sequence is None:
                show_error_dialog(
                    self,
                    "No MDASequence found in the datastore! Cannot determine "
                    "positions to analyze.",
                )
                return
            pos = list(range(len(self._data.sequence.stage_positions)))
        self._experiment.positions_analyzed = pos

        # Update existing settings or create new one
        if self._experiment.analysis_settings is not None:
            self._experiment.analysis_settings.sqlmodel_update(
                new_settings.model_dump(exclude={"id"})
            )
        else:
            # Create new settings
            self._experiment.analysis_settings = new_settings

        # Update the analysis runner with the current data and experiment
        self._analysis_runner.set_data(self._data)
        self._analysis_runner.set_experiment(self._experiment)

        create_worker(
            self._analysis_runner.run,
            _start_thread=True,
            _connect={"errored": self._on_worker_errored},
        )

    def _on_worker_errored(self) -> None:
        LOGGER.error("Analysis runner encountered an error during execution.")

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
            labels_path=self._labels_path,
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
                    LOGGER.error(msg)
                    return
                self.initialize_widget_from_directories(
                    data_path, value.analysis_path, value.labels_path
                )

    def _clear_widget_before_initialization(self) -> None:
        """Clear the widget before initializing it with new data."""
        # clear paths
        self._database_path = None
        self._analysis_path = None
        self._labels_path = None
        # clear experiment
        self._experiment = None
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

        # clear the segmentation widget - TO REMOVE
        self._segmentation_wdg.experiment = None
        self._segmentation_wdg.data = None
        self._segmentation_wdg.labels_path = None

    def _load_plate_plan(
        self, plate_plan: useq.WellPlatePlan | tuple[useq.Position, ...] | None = None
    ) -> useq.WellPlatePlan | None:
        """Load the plate from the datastore."""
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
        return final_plate_plan

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
        """Update the segmentation and analysis widgets gui."""
        # analysis widget
        self._update_analysis_gui_settings(plate)

        # segmentation widget - TO REMOVE
        self._segmentation_wdg.data = self._data
        self._segmentation_wdg.labels_path = self._labels_path

    def _update_analysis_gui_settings(
        self, plate: useq.WellPlate | None = None
    ) -> None:
        """Update the analysis widgets settings."""
        if self._experiment is None:
            self._analysis_wdg.reset()
            return

        settings = self._experiment.analysis_settings
        if settings is None:
            self._analysis_wdg.reset()
            return

        plate_map_data = None
        if plate is not None:
            plate_map_data = (plate, *experiment_to_plate_map_data(self._experiment))

        value = AnalysisSettingsData(
            plate_map_data=plate_map_data,
            experiment_type_data=ExperimentTypeData(
                experiment_type=self._experiment.experiment_type,
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
                LOGGER.error(msg)
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
                    LOGGER.info(
                        f"Loaded stimulation metadata from datastore: "
                        f"led_pulse_duration={led_duration}"
                        f"led_powers={wdg._led_powers_le.text()}, "
                        f"led_pulse_on_frames={wdg._led_pulse_on_frames_le.text()}"
                    )

            else:
                msg = "No stimulation metadata found in the datastore!"
                show_error_dialog(self, msg)
                LOGGER.warning(msg)

        except Exception as e:
            msg = f"Failed to load metadata from datastore!\n\nError: {e}"
            show_error_dialog(self, msg)
            LOGGER.error(msg)
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
            fov_data = self._get_fov_data(value)
            # update the graphs combo boxes
            self._update_single_wells_graphs_combo(combo_red=(fov_data is None))

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
        # get the analysis data for the current fov if it exists
        fov_data = self._get_fov_data(value)
        # flip data and labels vertically or will look different from the StackViewer
        data = np.flip(data, axis=0)
        labels = np.flip(labels, axis=0) if labels is not None else None
        self._image_viewer.setData(data, labels)
        self._set_graphs_fov(value)

        self._update_single_wells_graphs_combo(
            combo_red=(fov_data is None), clear=(fov_data is None)
        )

    def _get_fov_data(self, value: WellInfo) -> dict[str, ROIData] | None:
        """Get the analysis data for the given FOV."""
        fov_name = f"{value.fov.name}_p{value.pos_idx}"
        fov_data = self._analysis_data.get(str(value.fov.name), None)
        # use the old name we used to save the data (without position index. e.g. "_p0")
        if fov_data is None:
            fov_data = self._analysis_data.get(fov_name, None)
        return fov_data

    def _set_graphs_fov(self, value: WellInfo | None) -> None:
        """Set the FOV title for the graphs."""
        if value is None:
            return
        title = value.fov.name or f"Position {value.pos_idx}"
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
        for mw_graph in self.MW_GRAPHS:
            mw_graph.set_combo_text_red(not self._analysis_data)

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
        if not self._analysis_data:
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

            save_trace_data_to_csv(path, self._analysis_data)
            save_analysis_data_to_csv(path, self._analysis_data)

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
    #             LOGGER.error(msg)
    #             return

    #         if (sequence := self._data.sequence) is None:
    #             msg = "useq.MDASequence not found! Cannot retrieve metadata!"
    #             show_error_dialog(self, msg)
    #             LOGGER.error(msg)
    #             return

    #         # Get exposure time from metadata (first frame)
    #         exp_time = meta[0].get("mda_event", {}).get("exposure", 0.0)
    #         if exp_time <= 0:
    #             msg = "Invalid exposure time found in metadata!"
    #             show_error_dialog(self, msg)
    #             LOGGER.error(msg)
    #             return

    #         # Get timepoints
    #         timepoints = sequence.sizes.get("t", 0)
    #         if timepoints == 0:
    #             msg = "No timepoints found in the sequence!"
    #             show_error_dialog(self, msg)
    #             LOGGER.error(msg)
    #             return

    #         frame_rate = (timepoints - 1) / ((timepoints * exp_time) / 1000)
    #         self._analysis_wdg._frame_rate_wdg._frame_rate_spin.setValue(frame_rate)

    #         LOGGER.info(f"Frame rate set to: {frame_rate:.2f} fps.")

    #     except Exception as e:
    #         msg = f"Failed to load frame rate from datastore!\n\nError: {e}"
    #         show_error_dialog(self, msg)
    #         LOGGER.error(msg)
    #         return
