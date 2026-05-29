"""Detection runner for executing segmentation and saving to database."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from cali._constants import EVENT_KEY
from cali.logger import cali_logger
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.readers._tiff_collection_reader import TiffCollectionReader
from cali.sqlmodel._model import FOV, ROI, DetectionSettings, Mask
from cali.util import mask_to_coordinates

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from cellpose.models import CellposeModel


class DetectionRunner:
    """Runner for neuron detection that saves masks directly to database.

    Similar to ExtractionRunner but for the detection/segmentation phase.
    Supports both Cellpose and CaImAn detection methods.
    Creates FOV and ROI objects with masks in the database.
    """

    def __init__(self) -> None:
        super().__init__()
        # Use threading.Event for cancellation control
        self._cancellation_event = threading.Event()

    # ---------------------PUBLIC METHODS --------------------- #

    def cancel(self) -> None:
        """Request cancellation of the detection process."""
        self._cancellation_event.set()
        cali_logger.info("🚮 Cancellation requested...")

    def run(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
        as_generator: bool = False,
    ) -> Generator[FOV, None, None] | list[FOV]:
        """Run detection and yield FOV results with ROIs and masks.

        Automatically selects the appropriate detection method based on
        detection_settings.method (e.g. "cellpose").

        This method performs pure computation and does not interact with the database.
        Database operations (checking for duplicates, saving results) should be
        handled by the caller (typically CaliRunner).

        Parameters
        ----------
        dataset : TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader
            Data reader instance for imaging data
        detection_settings : DetectionSettings
            Detection parameters (method field determines which algorithm to use)
        global_position_indices : Sequence[int]
            Position indices to process
        as_generator : bool
            If True, returns a Generator that yields FOVs.
            If False (default), returns a list of FOVs.

        Returns
        -------
        Generator[FOV, None, None] | list[FOV]
            FOV objects with ROIs and masks, ready to be saved to database
        """
        generator = self._run_generator(
            dataset=dataset,
            detection_settings=detection_settings,
            global_position_indices=global_position_indices,
        )

        return generator if as_generator else list(generator)

    def _run_generator(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
    ) -> Generator[FOV, None, None]:
        """Internal generator for detection process."""
        # Reset cancellation event
        self._cancellation_event.clear()

        assert isinstance(
            dataset, (TensorstoreZarrReader, OMEZarrReader, TiffCollectionReader)
        ), (
            "Data must be a TensorstoreZarrReader, OMEZarrReader, or "
            "TiffCollectionReader instance."
        )

        if detection_settings.method == "cellpose":
            yield from self._run_cellpose(
                dataset=dataset,
                detection_settings=detection_settings,
                position_indices=global_position_indices,
            )
        elif detection_settings.method == "imported_labels":
            cali_logger.info("🏷️ Using imported labels - no detection to run.")
            return
        else:
            msg = (
                f"❌ Unknown detection method: {detection_settings.method}. "
                "Supported methods: 'cellpose', 'imported_labels'."
            )
            cali_logger.error(msg)
            raise ValueError(msg)

    # ---------------------PRIVATE METHODS --------------------- #

    def _run_cellpose(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        detection_settings: DetectionSettings,
        position_indices: Sequence[int],
    ) -> Generator[FOV, None, None]:
        """Run Cellpose segmentation and yield FOV results with ROIs and masks.

        Parameters
        ----------
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader
            Data reader instance for imaging data
        detection_settings : DetectionSettings
            Detection parameters (method should be "cellpose")
        position_indices : Sequence[int]
            Position indices to process
        cellpose_debug : bool
            Enable Cellpose debug logging

        Yields
        ------
        FOV
            FOV objects with ROIs and masks, ready to be saved to database
        """
        try:
            from cellpose import core
            from cellpose.models import CellposeModel
        except ModuleNotFoundError as e:
            cali_logger.error("Cellpose is not installed!")
            raise ModuleNotFoundError(
                "Cellpose detection requires the cellpose package.\n\n"
                "To install Cellpose:\n"
                "• For Cellpose 4: uv sync --extra cp4\n"
                "• For Cellpose 3: uv sync --extra cp3\n"
                "• Or via pip: pip install cellpose"
            ) from e

        cali_logger.info("🔍 Running Cellpose Detection...")

        # Check GPU availability and user preference
        gpu_available = core.use_gpu()
        use_gpu = detection_settings.use_gpu and gpu_available

        if detection_settings.use_gpu and not gpu_available:
            cali_logger.warning(
                "⚠️ GPU requested but not available. Falling back to CPU processing."
            )

        cali_logger.info(f"🖥️ Use GPU: {use_gpu}")

        # Use custom_model path if model_type is "custom", otherwise use model_type
        model_path = (
            detection_settings.custom_model
            if detection_settings.model_type == "custom"
            else detection_settings.model_type
        )
        cali_logger.info(f"💿 Loading model from `{model_path}`.")
        model = CellposeModel(pretrained_model=str(model_path), gpu=use_gpu)
        # Run detection and yield FOV results
        yield from self._run_cellpose_detection(
            dataset=dataset,
            position_indices=position_indices,
            model=model,
            diameter=detection_settings.diameter,
            cellprob_threshold=detection_settings.cellprob_threshold,
            flow_threshold=detection_settings.flow_threshold,
            batch_size=detection_settings.batch_size,
            min_size=detection_settings.min_size,
            normalize=detection_settings.normalize,
        )

        if self._cancellation_event.is_set():
            cali_logger.info("🛑 Detection Cancelled!")
        else:
            cali_logger.info("✅ Detection complete!")

    def _run_cellpose_detection(
        self,
        dataset: TensorstoreZarrReader | OMEZarrReader | TiffCollectionReader,
        position_indices: Sequence[int],
        model: CellposeModel,
        diameter: float | None,
        cellprob_threshold: float,
        flow_threshold: float,
        batch_size: int,
        min_size: int,
        normalize: bool,
    ) -> Generator[FOV, None, None]:
        """Internal method to run Cellpose detection and yield FOV objects.

        Yields
        ------
        FOV
            FOV objects with ROIs and Masks, ready to be committed
        """
        assert isinstance(
            dataset, (TensorstoreZarrReader, OMEZarrReader, TiffCollectionReader)
        ), (
            "Data must be a TensorstoreZarrReader, OMEZarrReader, or "
            "TiffCollectionReader instance."
        )

        n_positions = len(position_indices)
        cali_logger.info(f"Processing {n_positions} positions")
        msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - "
            "cali_logger - INFO - 🔍 Running Cellpose"
        )
        for pos_idx in tqdm(position_indices, desc=msg):
            if self._check_for_abort_requested():
                return

            data, meta = dataset.isel(p=pos_idx, metadata=True)

            # Preprocess data: max projection from half to end of stack
            if data.ndim == 3:  # (t, y, x)
                image = data[data.shape[0] // 2 :, :, :].max(axis=0)
            else:  # already 2D
                image = data

            if self._check_for_abort_requested():
                return

            mask = self._run_cellpose_on_image(
                model=model,
                image=image,
                diameter=diameter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                batch_size=batch_size,
                min_size=min_size,
                normalize=normalize,
            )

            if self._check_for_abort_requested():
                return

            fov_result = self._create_fov_with_rois(pos_idx, meta, mask)
            if fov_result:
                yield fov_result

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        if self._cancellation_event.is_set():
            cali_logger.info("🚮 Cancellation requested")
            return True
        return False

    def _run_cellpose_on_image(
        self,
        model: CellposeModel,
        image: np.ndarray,
        diameter: float | None,
        cellprob_threshold: float,
        flow_threshold: float,
        batch_size: int,
        min_size: int,
        normalize: bool,
    ) -> np.ndarray:
        """Run Cellpose on a single image and return the processed mask."""
        from cellpose.utils import fill_holes_and_remove_small_masks

        mask, _, _ = model.eval(
            image,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            normalize=normalize,
            batch_size=batch_size,
        )

        return fill_holes_and_remove_small_masks(mask, min_size=min_size)

    def _create_fov_with_rois(
        self,
        global_pos_idx: int,
        meta: list[dict],
        masks_2d: np.ndarray,
    ) -> FOV | None:
        """Create FOV with ROIs and Masks from segmentation results.

        Parameters
        ----------
        experiment : Experiment
            Parent experiment
        global_pos_idx : int
            Position index
        meta : list[dict]
            Metadata for this position
        masks_2d : np.ndarray
            2D label mask from Cellpose

        Returns
        -------
        FOV | None
            FOV object with ROIs and Masks, ready to commit
        """
        # Get FOV name from metadata
        fov_name = self._get_fov_name(EVENT_KEY, meta, global_pos_idx)

        # Get unique label values (excluding background 0)
        label_values = np.unique(masks_2d)
        label_values = label_values[label_values > 0]

        if len(label_values) == 0:
            cali_logger.warning(f"⚠️ No cells detected in {fov_name}")
            return None

        # Extract fov_number from name (e.g., "B2_0001" -> 1)
        # FOV number is the FOV index within the well, not the global position
        try:
            fov_number = int(fov_name.split("_")[1])
        except (IndexError, ValueError):
            # Fallback if name format is unexpected
            fov_number = global_pos_idx

        # Create FOV (well association will be handled by commit_detection_result)
        fov = FOV(
            name=fov_name,
            position_index=global_pos_idx,
            fov_number=fov_number,
            rois=[],
        )

        # Create ROIs with masks
        for label_value in label_values:
            if self._check_for_abort_requested():
                return None

            # Create binary mask for this ROI
            roi_mask_binary = masks_2d == label_value

            # Convert to coordinates
            mask_coords, mask_shape = mask_to_coordinates(roi_mask_binary)

            # Create Mask object
            mask_obj = Mask(
                coords_y=mask_coords[0],
                coords_x=mask_coords[1],
                height=mask_shape[0],
                width=mask_shape[1],
                mask_type="roi",
            )

            # Create ROI with mask
            # Note: fov_id will be set when the FOV is persisted
            # The relationship will handle the connection
            roi = ROI(
                label_value=int(label_value),
                active=None,  # Will be determined during analysis
                stimulated=None,  # Will be determined during analysis
                roi_mask=mask_obj,
                fov_id=0,  # Placeholder - will be set by relationship
            )

            fov.rois.append(roi)
        return fov

    def _get_fov_name(
        self, event_key: str, meta: list[dict], global_pos_idx: int
    ) -> str:
        """Get the FOV name from metadata."""
        try:
            # Try to get pos_name first (e.g., "B5_0000")
            pos_name = meta[0][event_key].get("pos_name")
            if pos_name:
                return pos_name  # type: ignore
        except (KeyError, IndexError, AttributeError):
            pass

        # Fallback to constructing from axes
        try:
            well = meta[0][event_key]["axes"]["p"]
            return f"{well}_{global_pos_idx:04d}"
        except (KeyError, IndexError):
            pass

        # Final fallback
        return f"pos_{global_pos_idx}"
