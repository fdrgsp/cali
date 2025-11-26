from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, percentile_filter

# TODO: implement/use also Caimans neuropil creation method


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


def _create_cell_pix_from_masks(
    cell_masks: list[np.ndarray],
    height: int,
    width: int,
    lam_percentile: float = 50.0,
) -> np.ndarray:
    """
    Approximate Suite2p's create_cell_pix using binary masks.

    - Builds a lammap where all ROI pixels have lam = 1.
    - Estimates a typical ROI radius from the ROI areas.
    - Applies a local percentile filter (radius * 5) to lammap.
    - Pixels with lam >= local percentile are marked as cell_pix.

    Returns
    -------
    cell_pix : np.ndarray (float32)
        Array where values > 0.5 indicate cell occupancy, matching Suite2p's
        usage in create_neuropil_masks.
    """
    lammap = np.zeros((height, width), dtype=np.float32)
    radii: list[float] = []

    for mask in cell_masks:
        ypix, xpix = np.nonzero(mask)
        if ypix.size == 0:
            continue

        # fake lam = 1 for all ROI pixels (we don't have Suite2p's lam weights)
        lam = np.ones_like(ypix, dtype=np.float32)
        lammap[ypix, xpix] = np.maximum(lammap[ypix, xpix], lam)

        # approximate radius from area
        area = ypix.size
        r = float(np.sqrt(area / np.pi))
        radii.append(r)

    if not radii:
        return np.zeros((height, width), dtype=np.float32)

    radius = float(np.median(radii))

    if lam_percentile > 0.0 and radius > 0:
        size = max(1, int(radius * 5))
        filt = percentile_filter(lammap, percentile=lam_percentile, size=size)
        # logical mask as in Suite2p: core pixels only
        cell_core = np.logical_and(lammap >= filt, lammap > 0.0)
    else:
        cell_core = lammap > 0.0

    # Suite2p's create_cell_pix returns float array, later used with < 0.5
    return cell_core.astype(np.float32)


def create_neuropil_from_dilation(
    cell_masks: list[np.ndarray],
    height: int,
    width: int,
    inner_neuropil_radius: int = 2,
    min_neuropil_pixels: int = 350,
    lam_percentile: float = 50.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create neuropil masks using iterative ROI extension, matching Suite2p.

    This function creates a "donut-shaped" neuropil region around each cell by:
    1. Extending the ROI pixels outward from the cell boundary
    2. Excluding pixels too close to the cell (inner_neuropil_radius)
    3. Excluding pixels belonging to other cells (via cell_pix)
    4. Ensuring minimum pixel count by continued expansion

    Parameters
    ----------
    cell_masks : list[np.ndarray]
        List of binary masks, one per cell.
    height : int
        Image height (Ly).
    width : int
        Image width (Lx).
    inner_neuropil_radius : int, default=2
        Number of pixels to keep between ROI and neuropil donut. Creates a
        "forbidden zone" around the cell to avoid contamination from optical
        blur/diffraction. The neuropil region starts BEYOND this distance.
    min_neuropil_pixels : int, default=350
        Minimum number of pixels required in the neuropil mask. The algorithm
        will continue expanding the ROI until this threshold is met (up to 100
        iterations).
    lam_percentile : float, default=50.0
        Percentile used in the Suite2p-like cell_pix construction.

    Returns
    -------
    cell_masks_eroded : list[np.ndarray]
        List of eroded cell masks (actual cell regions used for trace
        extraction). Each mask is eroded by 1 pixel to avoid edge effects.
    neuropil_masks : list[np.ndarray]
        List of binary masks for neuropil regions, one per cell.
    """
    # Ensure cell_masks are boolean
    cell_masks = [mask.astype(bool) for mask in cell_masks]

    # Suite2p-style cell_pix
    cell_pix = _create_cell_pix_from_masks(
        cell_masks,
        height,
        width,
        lam_percentile=lam_percentile,
    )

    # Valid pixels check: pixels are valid if not occupied by cells
    def valid_pixels(ypix: np.ndarray, xpix: np.ndarray) -> np.ndarray:
        return cell_pix[ypix, xpix] < 0.5  # type: ignore[no-any-return]

    # Extension step size (Suite2p default: 5 pixels at a time)
    extend_by = 5

    cell_masks_eroded: list[np.ndarray] = []
    neuropil_masks: list[np.ndarray] = []

    for cell_mask in cell_masks:
        # Get pixel coordinates for this cell
        ypix, xpix = np.nonzero(cell_mask)
        if ypix.size == 0:
            # empty ROI, keep empty
            cell_masks_eroded.append(cell_mask.copy())
            neuropil_masks.append(np.zeros((height, width), dtype=bool))
            continue

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

        # Continue until we have enough valid neuropil pixels (up to 100 iterations)
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
        cell_mask_eroded = binary_erosion(cell_mask, iterations=1).astype(bool)

        cell_masks_eroded.append(cell_mask_eroded)
        neuropil_masks.append(neuropil_mask)

    return cell_masks_eroded, neuropil_masks
