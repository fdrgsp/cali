"""Tests for multi-well bar plot functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlmodel import Session

from cali._constants import EVK_NON_STIM, EVK_STIM
from cali.plot._multi_wells_plots._multi_well_bar_plot import _get_condition_label
from cali.sqlmodel import ROI, Condition, Experiment, Plate, Well

if TYPE_CHECKING:
    from tests.conftest import TempDB


def test_condition_label_ordering(temp_db: TempDB) -> None:
    """Test that condition labels are consistently ordered by condition_type."""
    engine, _ = temp_db

    # Create experiment and plate
    exp = Experiment(name="TestExperiment")
    plate = Plate(experiment=exp, name="TestPlate", plate_type="96-well")

    # Create conditions with different types
    genotype_wt = Condition(name="WT", condition_type="genotype", color="blue")
    treatment_drug = Condition(name="Drug", condition_type="treatment", color="green")

    # Test 1: Well with genotype then treatment (alphabetically sorted order)
    well1 = Well(
        plate=plate,
        name="A1",
        row=0,
        column=0,
        conditions=[genotype_wt, treatment_drug],
    )

    # Test 2: Well with treatment then genotype (reverse alphabetical order)
    # This simulates how conditions might be stored inconsistently in database
    well2 = Well(
        plate=plate,
        name="A2",
        row=0,
        column=1,
        conditions=[treatment_drug, genotype_wt],
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(exp)
        session.refresh(well1)
        session.refresh(well2)

        # Both should produce the same label (genotype before treatment)
        label1 = _get_condition_label(well1)
        label2 = _get_condition_label(well2)

        # Should always be genotype_treatment regardless of storage order
        assert label1 == "WT_Drug"
        assert label2 == "WT_Drug"
        assert label1 == label2


def test_condition_label_single_condition(temp_db: TempDB) -> None:
    """Test condition label with single condition."""
    engine, _ = temp_db

    exp = Experiment(name="TestExperiment")
    plate = Plate(experiment=exp, name="TestPlate", plate_type="96-well")
    genotype = Condition(name="WT", condition_type="genotype", color="blue")
    well = Well(plate=plate, name="A1", row=0, column=0, conditions=[genotype])

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(well)

        label = _get_condition_label(well)
        assert label == "WT"


def test_condition_label_no_conditions(temp_db: TempDB) -> None:
    """Test condition label with no conditions."""
    engine, _ = temp_db

    exp = Experiment(name="TestExperiment")
    plate = Plate(experiment=exp, name="TestPlate", plate_type="96-well")
    well = Well(plate=plate, name="B5", row=1, column=4, conditions=[])

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(well)

        label = _get_condition_label(well)
        assert label == "Well_B5"


def test_condition_label_multiple_condition_types(temp_db: TempDB) -> None:
    """Test condition label with three different condition types."""
    engine, _ = temp_db

    exp = Experiment(name="TestExperiment")
    plate = Plate(experiment=exp, name="TestPlate", plate_type="96-well")

    # Three different condition types
    genotype = Condition(name="KO", condition_type="genotype", color="red")
    treatment = Condition(name="Drug", condition_type="treatment", color="green")
    other = Condition(name="Special", condition_type="other", color="yellow")

    # Add conditions in non-alphabetical order
    well = Well(
        plate=plate,
        name="A1",
        row=0,
        column=0,
        conditions=[treatment, other, genotype],  # Deliberately out of order
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(well)

        label = _get_condition_label(well)

        # Should be sorted alphabetically by condition_type:
        # genotype, other, treatment
        assert label == "KO_Special_Drug"


def test_condition_label_consistency_across_wells(temp_db: TempDB) -> None:
    """Test that multiple wells with same conditions produce same labels."""
    engine, _ = temp_db

    exp = Experiment(name="TestExperiment")
    plate = Plate(experiment=exp, name="TestPlate", plate_type="96-well")

    genotype = Condition(name="WT", condition_type="genotype", color="blue")
    treatment = Condition(name="Control", condition_type="treatment", color="gray")

    # Create multiple wells with same conditions but potentially different order
    wells = [
        Well(
            plate=plate,
            name=f"A{i}",
            row=0,
            column=i - 1,
            conditions=[genotype, treatment] if i % 2 == 0 else [treatment, genotype],
        )
        for i in range(1, 5)
    ]

    with Session(engine) as session:
        session.add(exp)
        session.commit()

        for well in wells:
            session.refresh(well)

        # All wells should produce the same label
        labels = [_get_condition_label(well) for well in wells]
        assert all(label == "WT_Control" for label in labels)
        assert len(set(labels)) == 1  # All labels are identical


def test_condition_label_with_stimulation_status(temp_db: TempDB) -> None:
    """Test that stimulation status is appended to condition labels.

    For evoked experiments.
    """
    engine, _ = temp_db

    exp = Experiment(name="TestExperiment")
    plate = Plate(experiment=exp, name="TestPlate", plate_type="96-well")

    genotype = Condition(name="WT", condition_type="genotype", color="blue")
    treatment = Condition(name="Drug", condition_type="treatment", color="green")

    well = Well(
        plate=plate,
        name="A1",
        row=0,
        column=0,
        conditions=[genotype, treatment],
    )

    with Session(engine) as session:
        session.add(exp)
        session.commit()
        session.refresh(well)

        # Create ROIs with different stimulation statuses
        roi_stimulated = ROI(
            label_value=1,
            fov_id=None,  # Not important for this test
            stimulated=True,
        )
        roi_non_stimulated = ROI(
            label_value=2,
            fov_id=None,
            stimulated=False,
        )
        roi_no_stim_info = ROI(
            label_value=3,
            fov_id=None,
            stimulated=None,
        )

        # Test stimulated ROI
        label_stim = _get_condition_label(well, roi_stimulated)
        assert label_stim == f"WT_Drug_{EVK_STIM}"

        # Test non-stimulated ROI
        label_non_stim = _get_condition_label(well, roi_non_stimulated)
        assert label_non_stim == f"WT_Drug_{EVK_NON_STIM}"

        # Test ROI without stimulation info (spontaneous experiment)
        label_no_stim = _get_condition_label(well, roi_no_stim_info)
        assert label_no_stim == "WT_Drug"  # No suffix

        # Test without passing ROI (backward compatibility)
        label_no_roi = _get_condition_label(well)
        assert label_no_roi == "WT_Drug"  # No suffix
