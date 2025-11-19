from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import tifffile
from cellpose import core
from cellpose.models import CellposeModel
from cellpose.utils import fill_holes_and_remove_small_masks
from fonticon_mdi6 import MDI6
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from cali._constants import EVENT_KEY, GREEN, RED
from cali.logger import cali_logger

from ._util import (
    _BrowseWidget,
    _ElapsedTimer,
    create_divider_line,
    parse_lineedit_text,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from cali.readers import OMEZarrReader, TensorstoreZarrReader
    from cali.sqlmodel._model import Experiment

    from ._plate_viewer import PlateViewer


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

CUSTOM_MODEL_PATH = (
    Path(__file__).parent.parent
    / "_batch_cellpose"
    / "cellpose_models"
    / "cp3_img8_epoch7000_py"
)


class _SelectModelPath(_BrowseWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "Custom Model",
        tooltip: str = "Choose the path to the custom Cellpose model.",
    ) -> None:
        super().__init__(parent, label, "", tooltip, is_dir=False)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select the {self._label_text}.",
            "",
            "",
        )
        if path:
            self._path.setText(path)


class _CellposeSegmentation(QWidget):
    """Widget to perform Cellpose segmentation on a PlateViewer data."""

    segmentationFinished = Signal()

    def __init__(
        self,
        parent: PlateViewer | None = None,
        *,
        data: TensorstoreZarrReader | OMEZarrReader | None = None,
    ) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer | None = parent
        self._data: TensorstoreZarrReader | OMEZarrReader | None = data
        self._labels_path: str | None = None
        self._labels: dict[str, np.ndarray] = {}
        self._worker: GeneratorWorker | None = None

        # ELAPSED TIMER ---------------------------------------------------------
        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        # MODEL WIDGET ----------------------------------------------------------
        self._model_wdg = QWidget(self)
        model_wdg_layout = QHBoxLayout(self._model_wdg)
        model_wdg_layout.setContentsMargins(0, 0, 0, 0)
        model_wdg_layout.setSpacing(5)
        models_combo_label = QLabel("Model Type:")
        models_combo_label.setSizePolicy(*FIXED)
        self._models_combo = QComboBox()
        self._models_combo.addItems(["cpsam", "cyto3", "custom"])
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)
        model_wdg_layout.addWidget(models_combo_label)
        model_wdg_layout.addWidget(self._models_combo, 1)

        self._browse_custom_model = _SelectModelPath(self)
        self._browse_custom_model.setValue(CUSTOM_MODEL_PATH)
        self._browse_custom_model.hide()

        # DIAMETER WIDGETS ------------------------------------------------------
        self._diameter_wdg = QWidget(self)
        self._diameter_wdg.setToolTip(
            "Set the diameter of the cells. Leave 0 for automatic detection."
        )
        diameter_layout = QHBoxLayout(self._diameter_wdg)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        diameter_layout.setSpacing(5)
        diameter_label = QLabel("Diameter:")
        diameter_label.setSizePolicy(*FIXED)
        self._diameter_spin = QDoubleSpinBox(self)
        self._diameter_spin.setSpecialValueText("Auto")
        self._diameter_spin.setRange(0, 1000)
        self._diameter_spin.setValue(0)
        diameter_layout.addWidget(diameter_label)
        diameter_layout.addWidget(self._diameter_spin)

        # CELLPOSE THRESHOLDS ---------------------------------------------------
        self._cellprob_wdg = QWidget(self)
        self._cellprob_wdg.setToolTip(
            "Cell probability threshold (all pixels > threshold are used "
            "for dynamics). Lower values detect more masks. Default is 0.0 ("
            "cellpose default)."
        )
        prob_layout = QHBoxLayout(self._cellprob_wdg)
        prob_layout.setContentsMargins(0, 0, 0, 0)
        prob_layout.setSpacing(5)
        cellprob_label = QLabel("Cell Prob Threshold:")
        cellprob_label.setSizePolicy(*FIXED)
        self._cellprob_threshold_spin = QDoubleSpinBox(self)
        self._cellprob_threshold_spin.setRange(-6.0, 6.0)
        self._cellprob_threshold_spin.setValue(0.0)
        self._cellprob_threshold_spin.setSingleStep(0.1)
        prob_layout.addWidget(cellprob_label)
        prob_layout.addWidget(self._cellprob_threshold_spin)

        self._flow_wdg = QWidget(self)
        self._flow_wdg.setToolTip(
            "Flow error threshold (all cells with errors below threshold are kept). "
            "Higher values detect more masks."
        )
        flow_layout = QHBoxLayout(self._flow_wdg)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setSpacing(5)
        flow_label = QLabel("Flow Threshold:")
        flow_label.setSizePolicy(*FIXED)
        self._flow_threshold_spin = QDoubleSpinBox(self)
        self._flow_threshold_spin.setRange(0.0, 3.0)
        self._flow_threshold_spin.setValue(0.4)
        self._flow_threshold_spin.setSingleStep(0.1)
        self._flow_threshold_spin.setToolTip(
            "Flow error threshold (all cells with errors below threshold are kept). "
            "Higher values detect more masks. Default is 0.4 (cellpose default)."
        )
        flow_layout.addWidget(flow_label)
        flow_layout.addWidget(self._flow_threshold_spin)

        # MIN SIZE WIDGET -------------------------------------------------------
        self._min_size_wdg = QWidget(self)
        self._min_size_wdg.setToolTip(
            "Minimum number of pixels for a mask to be kept. Masks smaller than "
            "this will be removed as they are likely artifacts or debris. "
            "Default is 15 pixels. Set to 1 to keep all masks."
        )
        min_size_layout = QHBoxLayout(self._min_size_wdg)
        min_size_layout.setContentsMargins(0, 0, 0, 0)
        min_size_layout.setSpacing(5)
        min_size_label = QLabel("Min Mask Size (px):")
        min_size_label.setSizePolicy(*FIXED)
        self._min_size_spin = QSpinBox()
        self._min_size_spin.setRange(1, 10000)
        self._min_size_spin.setValue(15)
        min_size_layout.addWidget(min_size_label)
        min_size_layout.addWidget(self._min_size_spin)

        # NORMALIZE CHECKBOX ----------------------------------------------------
        self._normalize_wdg = QWidget(self)
        self._normalize_wdg.setToolTip(
            "Normalize images before segmentation. "
            "This rescales pixel values to 0-1 range using 1st and 99th percentiles.\n"
            "By default, this is enabled (cellpose default)."
        )
        normalize_layout = QHBoxLayout(self._normalize_wdg)
        normalize_layout.setContentsMargins(0, 0, 0, 0)
        normalize_layout.setSpacing(5)
        normalize_label = QLabel("Normalize Images:")
        normalize_label.setSizePolicy(*FIXED)
        self._normalize_checkbox = QCheckBox()
        self._normalize_checkbox.setChecked(True)
        normalize_layout.addWidget(normalize_label)
        normalize_layout.addWidget(self._normalize_checkbox)
        normalize_layout.addStretch(1)

        # BATCH SIZE WIDGET -----------------------------------------------------
        self._batch_wdg = QWidget(self)
        self._batch_wdg.setToolTip(
            "Number of images to process per batch. Higher values are faster "
            "but use more memory."
        )
        batch_layout = QHBoxLayout(self._batch_wdg)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(5)
        batch_label = QLabel("Batch Size:")
        batch_label.setSizePolicy(*FIXED)
        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 32)
        self._batch_size_spin.setValue(8)
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self._batch_size_spin)

        # POSITIONS WIDGET ------------------------------------------------------
        self._pos_wdg = QWidget(self)
        self._pos_wdg.setToolTip(
            "Select the Positions to segment. Leave blank to segment all Positions.\n"
            "You can input single Positions (e.g. 30, 33) a range (e.g. 1-10), or a "
            "mix of single Positions and ranges (e.g. 1-10, 30, 50-65).\n"
            "NOTE: The Positions are 0-indexed."
        )
        pos_wdg_layout = QHBoxLayout(self._pos_wdg)
        pos_wdg_layout.setContentsMargins(0, 0, 0, 0)
        pos_wdg_layout.setSpacing(5)
        pos_lbl = QLabel("Segment Positions:")
        pos_lbl.setSizePolicy(*FIXED)
        self._pos_le = QLineEdit()
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33. Leave empty for all.")
        pos_wdg_layout.addWidget(pos_lbl)
        pos_wdg_layout.addWidget(self._pos_le)

        # PROGRESS BAR WIDGET ---------------------------------------------------
        progress_wdg = QWidget(self)
        progress_layout = QHBoxLayout(progress_wdg)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(5)

        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(icon(MDI6.stop, color=RED))
        self._cancel_btn.clicked.connect(self.cancel)

        self._progress_label = QLabel("[0/0]")
        self._progress_bar = QProgressBar(self)
        self._elapsed_time_label = QLabel("00:00:00")

        # STYLING ---------------------------------------------------------------
        fixed_lbl_width = cellprob_label.sizeHint().width()
        pos_lbl.setMinimumWidth(fixed_lbl_width)
        models_combo_label.setMinimumWidth(fixed_lbl_width)
        self._browse_custom_model._label.setMinimumWidth(fixed_lbl_width)
        diameter_label.setMinimumWidth(fixed_lbl_width)
        cellprob_label.setMinimumWidth(fixed_lbl_width)
        flow_label.setMinimumWidth(fixed_lbl_width)
        min_size_label.setMinimumWidth(fixed_lbl_width)
        batch_label.setMinimumWidth(fixed_lbl_width)
        normalize_label.setMinimumWidth(fixed_lbl_width)

        # LAYOUT ----------------------------------------------------------------
        progress_layout.addWidget(self._run_btn)
        progress_layout.addWidget(self._cancel_btn)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._elapsed_time_label)

        self.groupbox = QGroupBox("Cellpose Segmentation", self)
        settings_groupbox_layout = QVBoxLayout(self.groupbox)
        settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
        settings_groupbox_layout.setSpacing(5)
        settings_groupbox_layout.addWidget(create_divider_line("Select Cellpose Model"))
        settings_groupbox_layout.addWidget(self._model_wdg)
        settings_groupbox_layout.addWidget(self._browse_custom_model)
        settings_groupbox_layout.addWidget(create_divider_line("Cellpose Parameters"))
        settings_groupbox_layout.addWidget(self._diameter_wdg)
        settings_groupbox_layout.addWidget(self._cellprob_wdg)
        settings_groupbox_layout.addWidget(self._flow_wdg)
        settings_groupbox_layout.addWidget(self._min_size_wdg)
        settings_groupbox_layout.addWidget(self._normalize_wdg)
        settings_groupbox_layout.addWidget(create_divider_line("Batch Processing"))
        settings_groupbox_layout.addWidget(self._batch_wdg)
        settings_groupbox_layout.addWidget(create_divider_line("Positions to Segment"))
        settings_groupbox_layout.addWidget(self._pos_wdg)
        settings_groupbox_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

    @property
    def experiment(self) -> Experiment | None:
        return self._experiment

    @experiment.setter
    def experiment(self, experiment: Experiment | None) -> None:
        self._experiment = experiment

    @property
    def data(self) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @data.setter
    def data(self, data: TensorstoreZarrReader | OMEZarrReader | None) -> None:
        self._data = data

    @property
    def labels(self) -> dict[str, np.ndarray]:
        return self._labels

    @property
    def labels_path(self) -> str | None:
        return self._labels_path

    @labels_path.setter
    def labels_path(self, labels_path: str | None) -> None:
        self._labels_path = labels_path

    # PUBLIC METHODS ------------------------------------------------------------------

    def run(self) -> None:
        """Perform the Cellpose segmentation in a separate thread."""
        if self._worker is not None and self._worker.is_running:
            return

        self._reset_progress_bar()

        if not self._validate_segmentation_setup():
            return

        positions = self._get_positions()
        if positions is None:
            return

        if not self._handle_existing_labels():
            return

        self._start_segmentation_thread(positions)

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is not None:
            self._worker.quit()
        self._elapsed_timer.stop()
        self._reset_progress_bar()
        self._enable(True)
        cali_logger.info("Cellpose segmentation canceled.")

    # PRIVATE METHODS -----------------------------------------------------------------

    # PREPARE FOR RUN -----------------------------------------------------------------
    def _validate_segmentation_setup(self) -> bool:
        """Check if the necessary data is available before segmentation."""
        if self._data is None:
            show_error_dialog(
                self,
                "No data loaded!\n"
                "Please load a PlateViewer data in "
                "File > Load Data and Set Directories....",
            )
            return False

        if not self._labels_path:
            cali_logger.error("No Segmentation Path selected.")
            show_error_dialog(
                self,
                "Please select a Segmentation Path.\n"
                "You can do this in File > Load Data and Set Directories...' "
                "and set the Segmentation Path'.",
            )
            return False

        if not Path(self._labels_path).is_dir():
            cali_logger.error("Invalid Segmentation Path.")
            show_error_dialog(
                self,
                "The Segmentation Path is not a valid directory!\n"
                "Please select a valid directory "
                "in File > Load Data and Set Directories....",
            )
            return False

        sequence = self._data.sequence
        if sequence is None:
            msg = "No useq.MDAsequence found!"
            cali_logger.error(msg)
            show_error_dialog(self, msg)
            return False

        # if cpsam is chosen, make sure cellpose4 is installed
        if self._models_combo.currentText() == "cpsam":
            # get cellpose version
            try:
                cellpose_version = version("cellpose")
                if not cellpose_version.startswith("4."):
                    msg = (
                        "Cellpose version 4.x is required for the cpsam model!\n"
                        f"Current version: {cellpose_version}"
                    )
                    cali_logger.error(msg)
                    show_error_dialog(self, msg)
                    return False

            except PackageNotFoundError:
                print("Cellpose is not installed!")

        return True

    def _get_positions(self) -> list[int] | None:
        """Retrieve and validate the positions for segmentation."""
        if self._data is None or (sequence := self._data.sequence) is None:
            return None

        if not self._pos_le.text():
            return list(range(len(sequence.stage_positions)))

        positions = parse_lineedit_text(self._pos_le.text())
        if not positions or max(positions) >= len(sequence.stage_positions):
            msg = "Invalid or out-of-range Positions provided!"
            cali_logger.error(msg)
            show_error_dialog(self, msg)
            return None

        return positions

    def _handle_existing_labels(self) -> bool:
        """Check if label files exist and ask the user for overwrite confirmation."""
        if not (path := self._labels_path):
            return False
        if list(Path(path).glob("*.tif")):
            from qtpy.QtWidgets import QMessageBox

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setText(
                "The Labels directory already contains some files!\n\n"
                "Do you want to overwrite them?"
            )
            msg.setWindowTitle("Overwrite Labels")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg.setDefaultButton(QMessageBox.StandardButton.No)
            response = msg.exec()
            if response == QMessageBox.StandardButton.No:
                return False
        return True

    # RUN THE SEGMENTATION ------------------------------------------------------------

    def _start_segmentation_thread(self, positions: list[int]) -> None:
        """Prepare segmentation and start it in a separate thread."""
        model = self._initialize_model()
        if model is None:
            return

        self._progress_bar.setRange(0, len(positions))
        self._enable(False)
        self._elapsed_timer.start()

        # Gather all parameters
        params = {
            "path": self._labels_path,
            "positions": positions,
            "model": model,
            "diameter": self._diameter_spin.value() or None,
            "cellprob_threshold": self._cellprob_threshold_spin.value(),
            "flow_threshold": self._flow_threshold_spin.value(),
            "batch_size": self._batch_size_spin.value(),
            "min_size": self._min_size_spin.value(),
            "normalize": self._normalize_checkbox.isChecked(),
        }

        self._worker = create_worker(
            self._segment,
            **params,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress_bar,
                "finished": self._on_worker_finished,
                "errored": self._on_worker_finished,
            },
        )

    def _segment(
        self,
        path: str,
        positions: list[int],
        model: CellposeModel,
        diameter: float | None,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        batch_size: int = 8,
        min_size: int = 15,
        normalize: bool = True,
    ) -> Generator[str | int, None, None]:
        """Perform the segmentation using Cellpose."""
        cali_logger.info("Starting Cellpose segmentation.")

        if self._data is None:
            return

        # Load all data first for batch processing
        all_images = []
        all_pos_names = []

        for p in tqdm(positions, desc="Loading data"):
            if self._worker is not None and self._worker.abort_requested:
                return

            # Get the data
            data, meta = self._data.isel(p=p, metadata=True)

            # Get position name from metadata
            pos_name = (
                meta[0].get(EVENT_KEY, {}).get("pos_name", f"pos_{str(p).zfill(4)}")
            )
            all_pos_names.append((pos_name, p))

            # Preprocess data: max projection from half to end of stack
            data_half_to_end = data[data.shape[0] // 2 :, :, :]

            # Max projection
            cyto_frame = data_half_to_end.max(axis=0)

            all_images.append(cyto_frame)

        # Process in batches
        cali_logger.info(
            f"Processing {len(all_images)} images in batches of {batch_size}"
        )
        all_masks = []
        for batch_masks, progress_msg in self._batch_process(
            model=model,
            images=all_images,
            pos_names=all_pos_names,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            min_size=min_size,
            normalize=normalize,
        ):
            all_masks.extend(batch_masks)
            yield progress_msg

        # Save all masks
        for i, ((pos_name, p), masks) in enumerate(zip(all_pos_names, all_masks)):
            if self._worker is not None and self._worker.abort_requested:
                return

            yield f"[Saving {pos_name} ({i + 1}/{len(all_masks)})]"

            # Store the masks in the labels dict
            self._labels[f"{pos_name}_p{p}"] = masks

            # Save to disk
            tifffile.imwrite(Path(path) / f"{pos_name}_p{p}.tif", masks)

            yield i + 1

    def _batch_process(
        self,
        model: CellposeModel,
        images: list[np.ndarray],
        pos_names: list[tuple[str, int]],
        diameter: float | None,
        cellprob_threshold: float,
        flow_threshold: float,
        batch_size: int,
        min_size: int = 15,
        normalize: bool = True,
    ) -> Generator[tuple[list[np.ndarray], str], None, None]:
        """Process images in batches.

        Yields batches of masks along with progress messages.
        """
        n_images = len(images)
        batch_masks = []

        for batch_start in range(0, n_images, batch_size):
            if self._worker is not None and self._worker.abort_requested:
                break

            batch_end = min(batch_start + batch_size, n_images)
            batch_images = images[batch_start:batch_end]
            current_batch_masks = []

            # Process each image in the batch
            for i, img in enumerate(batch_images):
                img_idx = batch_start + i
                pos_name = pos_names[img_idx][0]

                # Yield progress message
                yield (
                    current_batch_masks,
                    f"[Processing {pos_name} ({img_idx + 1}/{n_images})]",
                )

                # Run cellpose
                masks = model.eval(
                    img,
                    diameter=diameter,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    normalize=normalize,
                )[0]

                # Mask cleaning with min_size
                masks = fill_holes_and_remove_small_masks(masks, min_size=min_size)

                current_batch_masks.append(masks)
                batch_masks.append(masks)

            # Yield the completed batch
            yield (current_batch_masks, f"[Batch complete: {batch_end}/{n_images}]")

    def _on_worker_finished(self) -> None:
        """Enable the widgets when the segmentation is finished."""
        cali_logger.info("Cellpose segmentation finished.")
        self._enable(True)
        self._elapsed_timer.stop()
        self._progress_bar.setValue(self._progress_bar.maximum())
        self.segmentationFinished.emit()

    # WIDGET---------------------------------------------------------------------------

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._model_wdg.setEnabled(enable)
        self._browse_custom_model.setEnabled(enable)
        self._diameter_wdg.setEnabled(enable)
        self._cellprob_wdg.setEnabled(enable)
        self._batch_wdg.setEnabled(enable)
        self._min_size_wdg.setEnabled(enable)
        self._cellprob_wdg.setEnabled(enable)
        self._flow_wdg.setEnabled(enable)
        self._normalize_wdg.setEnabled(enable)
        self._pos_wdg.setEnabled(enable)
        self._run_btn.setEnabled(enable)
        if self._plate_viewer is None:
            return
        self._plate_viewer._analysis_wdg.setEnabled(enable)
        # disable graphs tabs
        self._plate_viewer._tab.setTabEnabled(1, enable)
        self._plate_viewer._tab.setTabEnabled(2, enable)

    def _reset_progress_bar(self) -> None:
        """Reset and initialize progress bar."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def _initialize_model(self) -> CellposeModel | None:
        """Initialize the Cellpose model based on user selection."""
        use_gpu = core.use_gpu()
        cali_logger.info(f"Use GPU: {use_gpu}")

        if self._models_combo.currentText() == "custom":
            custom_model_path = self._browse_custom_model.value()
            if not custom_model_path:
                show_error_dialog(self, "Please select a custom model path.")
                cali_logger.error("No custom model path selected.")
                return None
            cali_logger.info(f"Loading custom model from {custom_model_path}")
            return CellposeModel(pretrained_model=str(custom_model_path), gpu=use_gpu)

        model_type = self._models_combo.currentText()
        cali_logger.info(f"Loading cellpose model: {model_type}")
        return CellposeModel(model_type=model_type, gpu=use_gpu)

    def _on_model_combo_changed(self, text: str) -> None:
        """Show or hide the custom model path widget."""
        if text == "custom":
            self._browse_custom_model.show()
        else:
            self._browse_custom_model.hide()

    def _update_progress_bar(self, value: str | int) -> None:
        # update only the progress label if the value is a string
        if isinstance(value, str):
            self._progress_label.setText(value)
            return
        # update the progress bar value if the value is an integer
        self._progress_bar.setValue(value)

    def _update_progress_label(self, time_str: str) -> None:
        """Update the progress label with elapsed time."""
        self._elapsed_time_label.setText(time_str)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(a0)
