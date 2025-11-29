"""Test script to actually visualize the stimulated area mask from database."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import create_engine
from sqlmodel import Session, select, col

from cali.sqlmodel._model import FOV, ROI, Mask, CaliResult, AnalysisSettings
from cali.util import coordinates_to_mask

# Path to test database
db_path = Path("tests/test_data/evoked/results.cali")
engine = create_engine(f"sqlite:///{db_path}")

# Query data
fov_name = "B5_0000"
run_id = 5  # The actual run ID in the test database

with Session(engine) as session:
    # Get stimulation mask
    result = session.get(CaliResult, run_id)
    stim_mask = None
    if result and result.analysis_settings:
        analysis_settings = session.get(AnalysisSettings, result.analysis_settings)
        if analysis_settings and analysis_settings.stimulation_mask_id:
            mask_obj = session.get(Mask, analysis_settings.stimulation_mask_id)
            if mask_obj and mask_obj.coords_y and mask_obj.coords_x:
                coords = (mask_obj.coords_y, mask_obj.coords_x)
                shape = (mask_obj.height, mask_obj.width)
                stim_mask = coordinates_to_mask(coords, shape)
                print(f"✓ Loaded stimulation mask: shape={shape}, pixels={stim_mask.sum()}")

    # Query ROIs with their masks
    stmt = (
        select(ROI, Mask)
        .join(FOV, ROI.fov_id == FOV.id)
        .outerjoin(Mask, ROI.roi_mask_id == Mask.id)
        .where(col(FOV.name) == fov_name)
        .order_by(col(ROI.label_value))
    )
    
    results = session.exec(stmt).all()
    print(f"\n✓ Found {len(results)} ROIs")
    
    # Reconstruct label image from ROI masks
    if results and stim_mask is not None:
        import numpy as np
        labels = np.zeros_like(stim_mask, dtype=np.int32)
        
        for roi, mask in results:
            if mask and mask.coords_y and mask.coords_x:
                roi_mask = coordinates_to_mask(
                    (mask.coords_y, mask.coords_x),
                    (mask.height, mask.width)
                )
                labels[roi_mask] = roi.label_value
                status = "STIMULATED" if roi.stimulated else "non-stimulated"
                print(f"  ROI {roi.label_value}: {status}, pixels={roi_mask.sum()}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Stimulation mask
        axes[0].imshow(stim_mask, cmap='gray')
        axes[0].set_title("Stimulation Mask")
        axes[0].axis('off')
        
        # Plot 2: ROI labels
        axes[1].imshow(labels, cmap='tab20')
        axes[1].set_title(f"ROI Labels (n={len(results)})")
        axes[1].axis('off')
        
        # Plot 3: Combined
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch
        
        color_mapping = {0: "black"}
        for roi, _ in results:
            if roi.stimulated:
                color_mapping[roi.label_value] = "green"
            else:
                color_mapping[roi.label_value] = "magenta"
        
        unique_labels = np.unique(labels)
        colors = [color_mapping.get(lbl, "gray") for lbl in unique_labels]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            boundaries=np.append(unique_labels, unique_labels[-1] + 1),
            ncolors=len(colors)
        )
        
        axes[2].imshow(labels, cmap=cmap, norm=norm)
        
        # Add stimulation area contours
        from skimage.measure import find_contours
        contours = find_contours(stim_mask.astype(float), level=0.5)
        for contour in contours:
            axes[2].plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=2)
        
        axes[2].set_title("Combined (ROIs + Stim Area)")
        axes[2].axis('off')
        
        # Add legend
        legend_patches = [
            Patch(color="green", label="Stimulated ROIs"),
            Patch(color="magenta", label="Non-Stimulated ROIs"),
            Patch(color="yellow", label="Stimulation Area")
        ]
        axes[2].legend(handles=legend_patches, loc='upper right', fontsize=8)
        
        # Save
        output_path = Path("_dev/test_mask_visualization.png")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization to: {output_path}")
        print(f"   Unique ROI labels: {sorted(unique_labels[unique_labels > 0])}")
        
        # Count stimulated vs non-stimulated
        stim_count = sum(1 for roi, _ in results if roi.stimulated)
        non_stim_count = len(results) - stim_count
        print(f"   Stimulated ROIs: {stim_count}")
        print(f"   Non-stimulated ROIs: {non_stim_count}")
    else:
        print("❌ No data found!")
