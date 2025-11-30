from __future__ import annotations

import numpy as np
import tifffile
from skimage import filters, morphology


def create_stimulation_mask(stimulation_file: str) -> np.ndarray:
    """Create a binary mask from an input image.

    We use this to create a mask of the stimulated area. If the input image is a
    mask image already, simply return it.

    Parameters
    ----------
    stimulation_file : str
        Path to the stimulation image.
    """
    # load grayscale image
    blue_img = tifffile.imread(stimulation_file)

    # check if the image is already a binary mask
    unique = np.unique(blue_img)
    # if only pne values which is 1 (full fov illumination)
    if unique.size == 1 and unique[0] == 1:
        return blue_img
    # if only two values which are 0 and 1 (binary mask)
    elif unique.size == 2:
        # if the image is already a binary mask, return it
        if unique[0] == 0 and unique[1] == 1:
            return blue_img
    # apply Gaussian Blur to reduce noise
    blur = filters.gaussian(blue_img, sigma=2)

    # set the threshold to otsu's threshold and apply thresholding
    th = blur > filters.threshold_otsu(blur)

    # morphological operations
    selem_small = morphology.disk(2)
    selem_large = morphology.disk(5)

    # closing operation (removes small holes)
    closed = morphology.closing(th, selem_small)

    # erosion (removes small noise)
    eroded = morphology.erosion(closed, selem_small)

    # final closing with a larger structuring element
    final_mask = morphology.closing(eroded, selem_large)

    return final_mask.astype(np.uint8)


def get_overlap_roi_with_stimulated_area(
    stimulation_mask: np.ndarray, roi_mask: np.ndarray
) -> float:
    """Compute the fraction of the ROI that overlaps with the stimulated area."""
    if roi_mask.shape != stimulation_mask.shape:
        raise ValueError("roi_mask and st_area must have the same dimensions.")

    # count nonzero pixels in the ROI mask
    cell_pixels = np.count_nonzero(roi_mask)

    # if the ROI mask has no pixels, return 0
    if cell_pixels == 0:
        return 0.0

    # count overlapping pixels (logical AND operation)
    overlapping_pixels = np.count_nonzero(roi_mask & stimulation_mask)

    return float(overlapping_pixels / cell_pixels)
