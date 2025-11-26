"""Clean up orphaned masks and verify database integrity."""

from pathlib import Path

from sqlmodel import Session, create_engine, delete, select

from cali.sqlmodel._model import FOV, ROI, Mask, Traces

DB_PATH = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
)


def cleanup_orphaned_masks(db_path: Path) -> None:
    """Remove masks that have no corresponding ROI."""
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        # Find all mask IDs that are referenced by ROIs
        stmt = select(ROI.roi_mask_id).where(ROI.roi_mask_id != None)  # noqa: E711
        referenced_mask_ids = set(session.exec(stmt).all())

        # Find all mask IDs in the Mask table
        stmt = select(Mask.id)
        all_mask_ids = set(session.exec(stmt).all())

        # Orphaned masks = masks not referenced by any ROI
        orphaned_ids = all_mask_ids - referenced_mask_ids

        if orphaned_ids:
            print(f"Found {len(orphaned_ids)} orphaned masks")
            # Delete orphaned masks
            stmt = delete(Mask).where(Mask.id.in_(orphaned_ids))
            result = session.exec(stmt)
            session.commit()
            print(f"Deleted {result.rowcount} orphaned masks")
        else:
            print("✅ No orphaned masks found")


def cleanup_orphaned_neuropil_masks(db_path: Path) -> None:
    """Remove neuropil masks that have no corresponding Trace."""
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        # Find all neuropil mask IDs referenced by Traces
        stmt = select(Traces.neuropil_mask_id).where(Traces.neuropil_mask_id != None)  # noqa: E711
        referenced_ids = set(session.exec(stmt).all())

        # Find all mask IDs that are neuropil type
        stmt = select(Mask.id).where(Mask.mask_type == "neuropil")
        all_neuropil_ids = set(session.exec(stmt).all())

        # Orphaned neuropil masks
        orphaned_ids = all_neuropil_ids - referenced_ids

        if orphaned_ids:
            print(f"Found {len(orphaned_ids)} orphaned neuropil masks")
            stmt = delete(Mask).where(Mask.id.in_(orphaned_ids))
            result = session.exec(stmt)
            session.commit()
            print(f"Deleted {result.rowcount} orphaned neuropil masks")
        else:
            print("✅ No orphaned neuropil masks found")


def verify_roi_masks(db_path: Path) -> None:
    """Verify all ROIs have valid mask references."""
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        # Find ROIs with missing masks
        all_rois = session.exec(select(ROI)).all()
        missing_masks = []

        for roi in all_rois:
            if roi.roi_mask_id is not None:
                mask = session.get(Mask, roi.roi_mask_id)
                if mask is None:
                    missing_masks.append((roi.id, roi.label_value, roi.roi_mask_id))

        if missing_masks:
            print(f"\n❌ Found {len(missing_masks)} ROIs with missing masks:")
            for roi_id, label, mask_id in missing_masks[:10]:
                print(f"  ROI id={roi_id}, label={label}, missing mask_id={mask_id}")
            if len(missing_masks) > 10:
                print(f"  ... and {len(missing_masks) - 10} more")
        else:
            print("\n✅ All ROIs have valid mask references")


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        exit(1)

    print(f"🧹 Cleaning up database: {DB_PATH}")
    print()

    verify_roi_masks(DB_PATH)
    cleanup_orphaned_masks(DB_PATH)
    cleanup_orphaned_neuropil_masks(DB_PATH)
    verify_roi_masks(DB_PATH)

    print("\n✅ Cleanup complete!")
