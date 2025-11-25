"""Example script to load an experiment from a database and print its tree structure."""

import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlmodel import Session, select

from cali.sqlmodel import Traces
from cali.sqlmodel._model import CaliResult
from cali.sqlmodel._visualize_experiment import print_cali_results

database_path = "tests/test_data/evoked/results.cali"
engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine, show_settings=False)


with Session(engine) as session:
    detection_id = 2  # Filter by detection settings ID

    statement = (
        select(Traces)
        .join(CaliResult)
        .where(CaliResult.detection_settings == detection_id)
    )

    traces_list = session.exec(statement).all()

    # Plot all dec_dff traces
    fig, ax = plt.subplots(figsize=(12, 6))

    for trace in traces_list:
        if trace.dec_dff is not None:
            # Use x_axis if available, otherwise use frame numbers
            x_data = (
                trace.x_axis
                if trace.x_axis is not None
                else list(range(len(trace.dec_dff)))
            )
            ax.plot(
                x_data,
                trace.dec_dff,
                label=f"ROI {trace.roi.label_value}",
                alpha=0.7,
            )

    x_label = "Time (ms)" if traces_list and traces_list[0].x_axis else "Frame"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Deconvolved ΔF/F")
    ax.set_title("dec_dff Traces - Position 0")
    # ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
