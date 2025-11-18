"""Get ROI masks grouped by DetectionSettings.

This script retrieves all ROIs from the database and groups them by the
DetectionSettings that were used to create them, allowing you to compare
detection results from different methods or parameters.
"""

from sqlmodel import Session, create_engine, select

from cali.sqlmodel._model import FOV, ROI, DetectionSettings
from cali.util import coordinates_to_mask

# Update this path to your actual database
DB_PATH = "/Users/fdrgsp/Desktop/cali_test/evk.tensorstore.zarr.db"

engine = create_engine(f"sqlite:///{DB_PATH}")

with Session(engine) as session:
    # Get all DetectionSettings
    detection_settings_list = session.exec(select(DetectionSettings)).all()

    print("=" * 80)
    print("ROIs grouped by DetectionSettings")
    print("=" * 80)

    if not detection_settings_list:
        print("\n⚠️  No DetectionSettings found in database.")
        print("   Run detection first to generate ROIs.")
    else:
        print(f"\nFound {len(detection_settings_list)} DetectionSettings:\n")

    # Group ROIs by detection_settings_id
    rois_by_detection = {}

    for detection in detection_settings_list:
        print(f"\nDetectionSettings ID: {detection.id}")
        print(f"  Method: {detection.method}")
        print(f"  Model: {detection.model_type}")
        print(f"  Diameter: {detection.diameter}")
        print(f"  Cell prob threshold: {detection.cellprob_threshold}")
        print(f"  Flow threshold: {detection.flow_threshold}")
        print(f"  Min size: {detection.min_size}")
        print(f"  Created: {detection.created_at}")

        # Get all ROIs created with this DetectionSettings
        rois = session.exec(
            select(ROI).where(ROI.detection_settings_id == detection.id)
        ).all()

        print(f"  Total ROIs: {len(rois)}")

        # Group by FOV for better organization
        rois_by_fov = {}
        for roi in rois:
            if roi.fov_id not in rois_by_fov:
                rois_by_fov[roi.fov_id] = []
            rois_by_fov[roi.fov_id].append(roi)

        print(f"  Across {len(rois_by_fov)} FOVs")

        # Store for later access
        rois_by_detection[detection.id] = {
            "settings": detection,
            "rois": rois,
            "rois_by_fov": rois_by_fov,
        }

    # Print detailed breakdown
    print("\n" + "=" * 80)
    print("Detailed ROI breakdown by FOV")
    print("=" * 80)

    for det_id, data in rois_by_detection.items():
        print(f"\n DetectionSettings ID {det_id}:")
        for fov_id, fov_rois in data["rois_by_fov"].items():
            fov = session.exec(select(FOV).where(FOV.id == fov_id)).first()
            fov_name = fov.name if fov else f"FOV_{fov_id}"
            print(f"    {fov_name}: {len(fov_rois)} ROIs", end="")

            # Count ROIs with masks
            rois_with_mask = [roi for roi in fov_rois if roi.roi_mask_id is not None]
            print(f" ({len(rois_with_mask)} with masks)")

    # Example: Get masks for a specific DetectionSettings
    print("\n" + "=" * 80)
    print("Example: Accessing ROI masks by DetectionSettings")
    print("=" * 80)

    if rois_by_detection:
        # Get first detection settings
        first_det_id = next(iter(rois_by_detection.keys()))
        first_det_data = rois_by_detection[first_det_id]

        print(f"\nExample with DetectionSettings ID {first_det_id}:")
        print(
            f"  Method: {first_det_data['settings'].method}, "
            f"Model: {first_det_data['settings'].model_type}"
        )

        # Get masks for first 3 ROIs
        sample_rois = first_det_data["rois"][:3]
        for roi in sample_rois:
            print(f"\n  ROI ID {roi.id} (label={roi.label_value}):")
            if roi.roi_mask and roi.roi_mask.coords_y and roi.roi_mask.coords_x:
                # Convert coordinates to mask array
                mask = coordinates_to_mask(
                    (roi.roi_mask.coords_y, roi.roi_mask.coords_x),
                    (roi.roi_mask.height, roi.roi_mask.width),
                )
                print(f"    Mask shape: {mask.shape}")
                print(f"    Mask pixels: {mask.sum()}")
                print(f"    Active: {roi.active}")
                print(f"    Stimulated: {roi.stimulated}")
            else:
                print("    No mask data available")

    # Compare different detection settings
    print("\n" + "=" * 80)
    print("Comparison across DetectionSettings")
    print("=" * 80)

    if len(rois_by_detection) > 1:
        print("\nComparing detection methods:\n")
        for det_id, data in rois_by_detection.items():
            settings = data["settings"]
            total_rois = len(data["rois"])
            active_rois = len([roi for roi in data["rois"] if roi.active])
            print(
                f"  DetectionSettings {det_id} ({settings.method}, "
                f"{settings.model_type}):"
            )
            print(f"    Total ROIs: {total_rois}")
            print(f"    Active ROIs: {active_rois}")
            if total_rois > 0:
                print(f"    Active %: {100 * active_rois / total_rois:.1f}%")
    else:
        print("\nOnly one DetectionSettings found.")
        print("Run detection with different parameters to compare results.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_rois = sum(len(data["rois"]) for data in rois_by_detection.values())
    print(f"\nTotal DetectionSettings: {len(rois_by_detection)}")
    print(f"Total ROIs across all settings: {total_rois}")

    if rois_by_detection:
        print("\n✅ ROIs are properly linked to DetectionSettings!")
        print("   You can now:")
        print("   - Compare detection results from different methods")
        print("   - Access ROI masks for specific detection settings")
        print("   - Track which detection parameters produced which ROIs")
    else:
        print("\n⚠️  No DetectionSettings or ROIs found.")

        print("   Run detection on your experiment first.")

    # Get all ROIs for DetectionSettings ID = 1
    rois1 = session.exec(select(ROI).where(ROI.detection_settings_id == 1)).all()
    rois2 = session.exec(select(ROI).where(ROI.detection_settings_id == 2)).all()

    # Convert ROI masks to numpy arrays
    from cali.util import coordinates_to_mask

    masks = []
    for roi in rois1:
        if roi.roi_mask and roi.roi_mask.coords_y and roi.roi_mask.coords_x:
            mask = coordinates_to_mask(
                (roi.roi_mask.coords_y, roi.roi_mask.coords_x),
                (roi.roi_mask.height, roi.roi_mask.width),
            )
            masks.append(mask)
    for roi in rois2:
        if roi.roi_mask and roi.roi_mask.coords_y and roi.roi_mask.coords_x:
            mask = coordinates_to_mask(
                (roi.roi_mask.coords_y, roi.roi_mask.coords_x),
                (roi.roi_mask.height, roi.roi_mask.width),
            )
            masks.append(mask)

    print(f"Loaded {len(masks)} masks")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for i, mask in enumerate(masks):
        plt.subplot(1, len(masks), i + 1)
        plt.imshow(mask, cmap="gray")
        plt.title(f"ROI {i + 1}")
        plt.axis("off")
    plt.show()


#     # Code example for accessing masks programmatically
#     print("\n" + "=" * 80)
#     print("Code Example: Access masks for a specific DetectionSettings")
#     print("=" * 80)
#     print(
#         """
# # Get all ROIs for DetectionSettings ID = 1
# rois = session.exec(
#     select(ROI).where(ROI.detection_settings_id == 1)
# ).all()

# # Convert ROI masks to numpy arrays
# from cali.util import coordinates_to_mask

# masks = []
# for roi in rois:
#     if roi.roi_mask and roi.roi_mask.coords_y and roi.roi_mask.coords_x:
#         mask = coordinates_to_mask(
#             (roi.roi_mask.coords_y, roi.roi_mask.coords_x),
#             (roi.roi_mask.height, roi.roi_mask.width),
#         )
#         masks.append(mask)

# print(f"Loaded {len(masks)} masks")
# """
#     )
