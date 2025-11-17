from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage import filters, morphology


def calculate_dff(
    data: np.ndarray, window: int = 100, percentile: int = 10, plot: bool = False
) -> np.ndarray:
    """Calculate the delta F/F using a sliding window and a percentile.

    Parameters
    ----------
    data : np.ndarray
        Array representing the fluorescence trace.
    window : int
        Size of the moving window for the background calculation. Default is 100.
    percentile : int
        Percentile to use for the background calculation. Default is 10.
    plot : bool
        Whether to show a plot of the background and trace. Default is False.

    Returns
    -------
    np.ndarray
        Array representing the delta F/F.
    """
    dff: np.ndarray = np.array([])
    bg: np.ndarray = _calculate_bg(data, window, percentile)
    dff = (data - bg) / bg

    # plot background and trace
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(bg, label="background", color="black")
        plt.plot(data, label="trace", color="green")
        plt.legend()
        plt.show()

    return dff


def _calculate_bg(data: np.ndarray, window: int, percentile: int = 10) -> np.ndarray:
    """
    Calculate the background using a moving window and a specified percentile.

    Parameters
    ----------
    data : np.ndarray
        Array representing the fluorescence trace.
    window : int
        Size of the moving window.
    percentile : int
        Percentile to use for the background calculation. Default is 10.

    Returns
    -------
    np.ndarray
        Array representing the background.
    """
    # Initialize background array
    background: np.ndarray = np.zeros_like(data)

    # Use a centered sliding window to calculate background from percentile
    # This provides symmetric context around each point and reduces edge artifacts
    for y in range(len(data)):
        start = max(0, y - window // 2)
        end = min(len(data), y + window // 2 + 1)
        lower_percentile = np.percentile(data[start:end], percentile)
        background[y] = lower_percentile

    return background


def get_iei(peaks: np.ndarray, elapsed_time_list_ms: list[float]) -> list[float] | None:
    """Calculate the interevent interval."""
    # if less than 2 peaks or framerate is negative
    if len(peaks) < 2 or len(elapsed_time_list_ms) <= 1:
        return None

    peaks_time_stamps = [elapsed_time_list_ms[i] for i in peaks]  # ms

    # calculate the difference in time between two consecutive peaks
    iei_ms = np.diff(np.array(peaks_time_stamps))  # ms

    return [float(iei_peak / 1000) for iei_peak in iei_ms]


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
        return blue_img  # type: ignore
    # if only two values which are 0 and 1 (binary mask)
    elif unique.size == 2:
        # if the image is already a binary mask, return it
        if unique[0] == 0 and unique[1] == 1:
            return blue_img  # type: ignore

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

    return final_mask.astype(np.uint8)  # type: ignore


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

def _extendROI(
    ypix: np.ndarray,
    xpix: np.ndarray,
    Ly: int,
    Lx: int,
    niter: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extend ypix and xpix by niter pixel(s) on each side.

    This matches suite2p's extendROI function from sparsedetect.py.

    Parameters
    ----------
    ypix : np.ndarray
        Y-coordinates of pixels
    xpix : np.ndarray
        X-coordinates of pixels
    Ly : int
        Image height
    Lx : int
        Image width
    niter : int
        Number of iterations to extend

    Returns
    -------
    ypix : np.ndarray
        Extended y-coordinates
    xpix : np.ndarray
        Extended x-coordinates
    """
    for _ in range(niter):
        # Extend in 4 cardinal directions: same, right, left, up, down
        yx_tuple = (
            (ypix, ypix, ypix, ypix - 1, ypix + 1),
            (xpix, xpix + 1, xpix - 1, xpix, xpix),
        )
        yx = np.array(yx_tuple)
        yx = yx.reshape((2, -1))
        # Get unique pixels
        yu = np.unique(yx, axis=1)
        # Keep only valid pixels within bounds
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix


def create_neuropil_from_dilation(
    cell_masks: list[np.ndarray],
    height: int,
    width: int,
    inner_neuropil_radius: int = 2,
    min_neuropil_pixels: int = 350,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create neuropil masks using iterative ROI extension, matching Suite2p.

    This function creates a "donut-shaped" neuropil region around each cell by:
    1. Extending the ROI pixels outward from the cell boundary
    2. Excluding pixels too close to the cell (inner_neuropil_radius)
    3. Excluding pixels belonging to other cells
    4. Ensuring minimum pixel count by continued expansion

    The implementation follows suite2p's create_neuropil_masks function from
    suite2p/extraction/masks.py, using iterative pixel-by-pixel extension
    rather than morphological dilation.

    Parameters
    ----------
    cell_masks : list[np.ndarray]
        List of binary masks, one per cell
    height : int
        Image height (Ly)
    width : int
        Image width (Lx)
    inner_neuropil_radius : int, default=2
        Number of pixels to keep between ROI and neuropil donut. Creates a
        "forbidden zone" around the cell to avoid contamination from optical
        blur/diffraction. The neuropil region starts BEYOND this distance.
    min_neuropil_pixels : int, default=350
        Minimum number of pixels required in the neuropil mask. The algorithm
        will continue expanding the ROI until this threshold is met (up to 100
        iterations).

    Returns
    -------
    cell_masks_eroded : list[np.ndarray]
        List of eroded cell masks (actual cell regions used for trace
        extraction). Each mask is eroded by 1 pixel to avoid edge effects.
    neuropil_masks : list[np.ndarray]
        List of binary masks for neuropil regions, one per cell.

    Notes
    -----
    The algorithm matches suite2p's behavior:
    1. Create cell_pix array marking all pixels occupied by any cell
    2. For each cell:
       a. Get initial pixels (ypix, xpix) from mask
       b. Extend by inner_neuropil_radius to create forbidden zone
       c. Iteratively extend outward using _extendROI (5 pixels at a time)
       d. Keep only pixels where cell_pix < 0.5 (not occupied by cells)
       e. Continue until min_neuropil_pixels threshold met or 100 iterations
    """
    from scipy import ndimage

    # Ensure cell_masks are boolean
    cell_masks = [mask.astype(bool) for mask in cell_masks]

    # Create cell_pix array (pixels belonging to any cell)
    # In suite2p, this is a float array where values > 0.5 indicate cell
    # occupancy
    cell_pix = np.zeros((height, width), dtype=np.float32)
    for mask in cell_masks:
        cell_pix[mask] = 1.0

    # Valid pixels check: pixels are valid if not occupied by cells
    def valid_pixels(ypix: np.ndarray, xpix: np.ndarray) -> np.ndarray:
        return cell_pix[ypix, xpix] < 0.5  # type: ignore[no-any-return]

    # Extension step size (suite2p default: 5 pixels at a time)
    extend_by = 5

    cell_masks_eroded = []
    neuropil_masks = []

    for cell_mask in cell_masks:
        # Get pixel coordinates for this cell
        ypix, xpix = np.nonzero(cell_mask)

        # Create neuropil mask array
        neuropil_mask = np.zeros((height, width), dtype=bool)

        # Step 1: Extend to get ring of dis-allowed pixels (forbidden zone)
        ypix_forbidden, xpix_forbidden = _extendROI(
            ypix, xpix, height, width, niter=inner_neuropil_radius
        )
        nring = np.sum(valid_pixels(ypix_forbidden, xpix_forbidden))

        # Step 2: Iteratively extend to build neuropil region
        nreps = 0
        ypix1, xpix1 = ypix.copy(), xpix.copy()

        # Continue until we have enough valid neuropil pixels (up to 100
        # iterations)
        while nreps < 100:
            # Extend the ROI
            ypix1, xpix1 = _extendROI(ypix1, xpix1, height, width, niter=extend_by)

            # Count valid pixels (not in cells, beyond forbidden zone)
            n_valid = int(np.sum(valid_pixels(ypix1, xpix1))) - int(nring)

            # Check if we have enough pixels
            if n_valid >= min_neuropil_pixels:
                break

            nreps += 1

        # Step 3: Mark valid neuropil pixels
        ix = valid_pixels(ypix1, xpix1)
        neuropil_mask[ypix1[ix], xpix1[ix]] = True

        # Remove the original cell pixels and forbidden zone
        neuropil_mask[ypix, xpix] = False
        neuropil_mask[ypix_forbidden, xpix_forbidden] = False

        # Step 4: Erode the original cell mask slightly for actual cell region
        cell_mask_eroded = ndimage.binary_erosion(cell_mask, iterations=1).astype(bool)

        cell_masks_eroded.append(cell_mask_eroded)
        neuropil_masks.append(neuropil_mask)

    return cell_masks_eroded, neuropil_masks
