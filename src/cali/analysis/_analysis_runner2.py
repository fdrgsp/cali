import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from sqlmodel import Session

from cali.sqlmodel._model import AnalysisSettings, Experiment

from ._analysis_with_sqlmodel import AnalysisRunner as OriginalAnalysisRunner

cali_logger = logging.getLogger("cali_logger")


def exec_(
    analyze: Callable,
    cancel_event: threading.Event,
    positions: Sequence[int],
    experiment: Experiment,
    max_workers: int | None = None,
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Check for cancellation before submitting futures
        if cancel_event.is_set():
            cali_logger.info("Cancellation requested before starting thread pool")
            return

        futures = [executor.submit(analyze, p, experiment) for p in positions]

        for future in as_completed(futures):
            # Check for cancellation at the start of each iteration
            if cancel_event.is_set():
                cali_logger.info("Cancellation requested, shutting down executor...")
                # Cancel pending futures and shutdown executor
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                break

            try:
                future.result()

                # Check for cancellation after each completed position
                if cancel_event.is_set():
                    cali_logger.info("Cancellation requested after position")
                    break

            except Exception as e:
                cali_logger.error(f"An error occurred: {e}", "error")
                break

    # Check if cancelled before finishing
    if cancel_event.is_set():
        cali_logger.info("Run Cancelled")


class AnalysisRunner:
    def run(
        self,
        experiment: Experiment,
        settings: AnalysisSettings,
        positions: Sequence[int],
    ) -> None:
        self._runner = runner = OriginalAnalysisRunner()
        experiment.positions_analyzed = list(positions)
        experiment.analysis_settings = settings
        settings.experiment_id = experiment.id

        runner.set_experiment(experiment)

        runner._stimulated_area_mask = settings.stimulated_mask_area()
        cancellation_event = threading.Event()
        exec_(
            analyze=self._analyze_position,
            cancel_event=cancellation_event,
            positions=positions,
            experiment=experiment,
            max_workers=settings.threads,
        )

    def _analyze_position(_self, p: int, experiment: Experiment) -> None:
        """Extract the roi traces for the given position.

        This method works with database session to ensure all changes are persisted.
        """
        self = _self._runner
        if self._engine is None or self._data is None:
            return
        if self._check_for_abort_requested():
            return

        with Session(self._engine) as session:
            # Fetch the experiment from the database using its ID
            # This avoids trying to INSERT the entire detached object graph
            exp = session.get(Experiment, experiment.id)
            if exp is None:
                return

            self._extract_trace_data_per_position(session, exp, p)
