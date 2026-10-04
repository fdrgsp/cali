"""Export known noise summaries with estimator, units and selected run/model scope."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from sqlmodel import Session, col, select

from cali.sqlmodel import FOV, FOVAnalysis, ensure_schema_current

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def export_noise_qc_to_csv(
    engine: Engine,
    output_path: str | Path,
    *,
    run_id: int,
    fov_name: str | None = None,
    position_indices: list[int] | None = None,
) -> bool:
    """Write separate calcium/CASCADE rows, returning False if no QC is stored.

    Counts include known finite estimates from inactive ROIs. Unknown historical
    summaries produce no rows; missing individual estimates never become zeros.
    An empty selection removes a previous export at this path to avoid stale QC.
    """
    ensure_schema_current(engine)
    rows: list[dict] = []
    with Session(engine) as session:
        statement = (
            select(FOVAnalysis, FOV)
            .join(FOV, col(FOVAnalysis.fov_id) == col(FOV.id))
            .where(FOVAnalysis.analysis_result_id == run_id)
            .order_by(col(FOV.position_index), col(FOVAnalysis.id))
        )
        if fov_name is not None:
            statement = statement.where(FOV.name == fov_name)
        if position_indices is not None:
            statement = statement.where(col(FOV.position_index).in_(position_indices))
        for parent, fov in session.exec(statement).all():
            base = {
                "analysis_result_id": run_id,
                "fov_name": fov.name,
                "position_index": fov.position_index,
            }
            if parent.calcium_noise_roi_count is not None:
                rows.append(
                    {
                        **base,
                        "method": "oasis_denoising",
                        "noise_estimator": "GetSn PSD median [0.25, 0.5]",
                        "noise_units": "dF/F",
                        "noise_median": parent.calcium_noise_median,
                        "noise_iqr": parent.calcium_noise_iqr,
                        "known_noise_roi_count": parent.calcium_noise_roi_count,
                        "model_name": None,
                        "model_sampling_rate_hz": None,
                        "weights_manifest_sha256": None,
                    }
                )
            child = parent.get_spike_analysis("cascade")
            if child is not None and child.model_noise_roi_count is not None:
                run = child.inference_run
                rows.append(
                    {
                        **base,
                        "method": "cascade",
                        "noise_estimator": (
                            "median(abs(diff(dff))) / sqrt(model Hz) * 100"
                        ),
                        "noise_units": "CASCADE model noise",
                        "noise_median": child.model_noise_median,
                        "noise_iqr": child.model_noise_iqr,
                        "known_noise_roi_count": child.model_noise_roi_count,
                        "model_name": run.resolved_model if run else None,
                        "model_sampling_rate_hz": run.model_sampling_rate_hz
                        if run
                        else None,
                        "weights_manifest_sha256": run.weights_manifest_sha256
                        if run
                        else None,
                    }
                )
    if not rows:
        Path(output_path).unlink(missing_ok=True)
        return False
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return True
