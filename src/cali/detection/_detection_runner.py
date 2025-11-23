"""Detection runner for executing segmentation and saving to database."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from cali._constants import EVENT_KEY
from cali.logger import cali_logger
from cali.readers import OMEZarrReader, TensorstoreZarrReader
from cali.sqlmodel._model import FOV, ROI, DetectionSettings, Mask
from cali.util import load_data, mask_to_coordinates

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from cellpose.models import CellposeModel



def _get_fov_name(event_key: str, meta: list[dict], global_pos_idx: int) -> str:
    """Get the FOV name from metadata."""
    try:
        # Try to get pos_name first (e.g., "B5_0000")
        pos_name = meta[0][event_key].get("pos_name")
        if pos_name:
            return pos_name
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


class DetectionRunner:
    """Runner for neuron detection that saves masks directly to database.

    Similar to AnalysisRunner but for the detection/segmentation phase.
    Supports both Cellpose and CaImAn detection methods.
    Creates FOV and ROI objects with masks in the database.
    """

    def __init__(self) -> None:
        super().__init__()
        # Use threading.Event for cancellation control
        self._cancellation_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of the detection process."""
        self._cancellation_event.set()
        cali_logger.info("Cancellation requested...")

    def run(
        self,
        dataset: str | Path | TensorstoreZarrReader | OMEZarrReader,
        detection_settings: DetectionSettings,
        global_position_indices: Sequence[int],
    ) -> Generator[FOV, None, None]:
        """Run detection and yield FOV results with ROIs and masks.

        Automatically selects the appropriate detection method based on
        detection_settings.method ("cellpose" or "caiman").

        This method performs pure computation and does not interact with the database.
        Database operations (checking for duplicates, saving results) should be
        handled by the caller (typically CaliRunner).

        Parameters
        ----------
        dataset : str | Path | TensorstoreZarrReader | OMEZarrReader
            Path to imaging data (zarr store) or a data reader instance
        detection_settings : DetectionSettings
            Detection parameters (method field determines which algorithm to use)
        global_position_indices : Sequence[int]
            Position indices to process

        Yields
        ------
        FOV
            FOV objects with ROIs and masks, ready to be saved to database
        """
        # Load data
        if isinstance(dataset, (str, Path)):
            dataset = load_data(dataset)
        else:
            assert isinstance(
                dataset, (TensorstoreZarrReader, OMEZarrReader)
            ), "Data must be a TensorstoreZarrReader or OMEZarrReader instance."

        if detection_settings.method == "cellpose":
            yield from self._run_cellpose(
                dataset=dataset,
                detection_settings=detection_settings,
                position_indices=global_position_indices,
            )
        elif detection_settings.method == "caiman":
            yield from self._run_caiman(
                dataset=dataset,
                detection_settings=detection_settings,
                position_indices=global_position_indices,
            )
        else:
            msg = (
                f"Unknown detection method: {detection_settings.method}. "
                "Supported methods: 'cellpose', 'caiman'"
            )
            cali_logger.error(msg)
            raise ValueError(msg)

    def _run_caiman(
        self,
        dataset: str | Path | TensorstoreZarrReader | OMEZarrReader,
        detection_settings: DetectionSettings,
        position_indices: Sequence[int],
    ) -> list[FOV]:
        """Run CaImAn detection and return FOV results with ROIs and masks.

        Parameters
        ----------
        dataset: str | Path | TensorstoreZarrReader | OMEZarrReader
            Path to imaging data (zarr store) or a data reader instance
        detection_settings : DetectionSettings
            Detection parameters (method should be "caiman")
        position_indices : Sequence[int]
            Position indices to process

        Returns
        -------
        list[FOV]
            List of FOV objects with ROIs and masks
        """
        # TODO: Implement CaImAn detection
        cali_logger.warning("CaImAn detection not yet implemented")
        raise NotImplementedError("CaImAn detection coming soon!")

    def _run_cellpose(
        self,
        dataset: str | Path |TensorstoreZarrReader | OMEZarrReader,
        detection_settings: DetectionSettings,
        position_indices: Sequence[int],
        cellpose_debug: bool = False,
    ) -> Generator[FOV, None, None]:
        """Run Cellpose segmentation and yield FOV results with ROIs and masks.

        Parameters
        ----------
        dataset: str | Path | TensorstoreZarrReader | OMEZarrReader
            Path to imaging data (zarr store) or a data reader instance
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
            from cellpose import core, io
            from cellpose.models import CellposeModel
        except ImportError as e:
            cali_logger.error(
                "Cellpose is not installed. Please install Cellpose to use "
                "Cellpose detection: `uv sync --extra cp4`."
            )
            raise e

        if cellpose_debug:
            io.logger_setup()

        use_gpu = core.use_gpu()
        cali_logger.info(f"Use GPU: {use_gpu}")

        cali_logger.info(f"Loading model from `{detection_settings.model_type}`.")
        model = CellposeModel(
            pretrained_model=str(detection_settings.model_type), gpu=use_gpu
        )

        # Run detection and yield FOV results
        total_fovs = 0
        total_rois = 0
        for fov in self._run_cellpose_detection(
            dataset=dataset,
            position_indices=position_indices,
            model=model,
            diameter=detection_settings.diameter,
            cellprob_threshold=detection_settings.cellprob_threshold,
            flow_threshold=detection_settings.flow_threshold,
            batch_size=detection_settings.batch_size,
            min_size=detection_settings.min_size,
            normalize=detection_settings.normalize,
        ):
            total_fovs += 1
            total_rois += len(fov.rois)
            yield fov

        cali_logger.info("✅ Detection complete!")

    def _run_cellpose_detection(
        self,
        dataset: str | Path | TensorstoreZarrReader | OMEZarrReader,
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
        # Load data
        if isinstance(dataset, (str, Path)):
            dataset = load_data(dataset)
        else:
            assert isinstance(
                dataset, (TensorstoreZarrReader, OMEZarrReader)
            ), "Data must be a TensorstoreZarrReader or OMEZarrReader instance."

        # Load all images for batch processing
        all_images = []
        all_metadata = []
        all_pos_indices = []

        cali_logger.info("Loading images for batch processing...")
        for pos_idx in position_indices:
            if self._check_for_abort_requested():
                cali_logger.info("Detection cancelled during image loading")
                return

            data, meta = dataset.isel(p=pos_idx, metadata=True)

            # Preprocess data: max projection from half to end of stack
            if data.ndim == 3:  # (t, y, x)
                data_half_to_end = data[data.shape[0] // 2 :, :, :]
                image = data_half_to_end.max(axis=0)
            else:  # already 2D
                image = data
            all_images.append(image)
            all_metadata.append(meta)
            all_pos_indices.append(pos_idx)

        if self._check_for_abort_requested():
            return

        # Process in batches
        cali_logger.info(
            f"Processing {len(all_images)} images in batches of {batch_size}"
        )
        all_masks = self._batch_process(
            model=model,
            images=all_images,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            min_size=min_size,
            normalize=normalize,
        )

        if self._check_for_abort_requested():
            return

        # Yield FOV objects one at a time
        for pos_idx, meta, masks_2d in zip(all_pos_indices, all_metadata, all_masks):
            if self._check_for_abort_requested():
                cali_logger.info("Detection cancelled during FOV creation")
                return

            fov_result = self._create_fov_with_rois(pos_idx, meta, masks_2d)

            if fov_result:
                yield fov_result

    def _check_for_abort_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_event.is_set()

    def _batch_process(
        self,
        model: CellposeModel,
        images: list[np.ndarray],
        diameter: float | None,
        cellprob_threshold: float,
        flow_threshold: float,
        batch_size: int,
        min_size: int,
        normalize: bool,
    ) -> list[np.ndarray]:
        """Process images in batches using Cellpose.

        Returns
        -------
        list[np.ndarray]
            List of 2D label masks, one per image
        """
        from cellpose.utils import fill_holes_and_remove_small_masks

        all_masks = []
        n_batches = (len(images) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Running Cellpose"):
            if self._check_for_abort_requested():
                break

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images))
            batch_images = images[start_idx:end_idx]

            # Run Cellpose on batch
            masks, _, _ = model.eval(
                batch_images,
                diameter=diameter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                normalize=normalize,
                batch_size=len(batch_images),
            )

            # Post-process masks
            for mask in masks:
                mask = fill_holes_and_remove_small_masks(mask, min_size=min_size)
                all_masks.append(mask)

        return all_masks

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
        fov_name = _get_fov_name(EVENT_KEY, meta, global_pos_idx)

        # Get unique label values (excluding background 0)
        label_values = np.unique(masks_2d)
        label_values = label_values[label_values > 0]

        if len(label_values) == 0:
            cali_logger.warning(f"No cells detected in {fov_name}")
            return None

        # Create FOV (well association will be handled by commit_detection_result)
        fov = FOV(
            name=fov_name,
            position_index=global_pos_idx,
            fov_number=global_pos_idx,
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
