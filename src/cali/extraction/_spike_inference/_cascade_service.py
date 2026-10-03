"""One device-owning CASCADE worker with bounded submission and cancellation."""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import CancelledError, Future, TimeoutError
from dataclasses import dataclass, field
from functools import partial
from queue import Full, Queue
from typing import TYPE_CHECKING

from ._base import InferenceCancelled
from ._cascade_cached import CachedCascadePredictor, CascadeCacheStats

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from cali._cascade_models import CascadeModel
    from cali.extraction._frame_window import TimingDescriptor

    from ._cascade_reference import CascadeResult


@dataclass
class _Request:
    future: Future[CascadeResult | None]
    dff: np.ndarray | None = None
    frame_rate: float = 0
    timing: TimingDescriptor | None = None
    cancel: Callable[[], bool] | None = None
    aborted: threading.Event = field(default_factory=threading.Event)
    prepare: bool = False


class CascadeInferenceService:
    """Synchronous FOV facade whose sole worker owns all Torch/model operations."""

    def __init__(
        self,
        model_name: str,
        model_dir: str | Path | None = None,
        *,
        expected_manifest: str | None = None,
        device: str = "auto",
        max_windows: int = 1024,
        max_cache_bytes: int = 128 * 1024 * 1024,
        queue_capacity: int = 1,
    ) -> None:
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError("CASCADE queue capacity must be a positive integer.")
        # Constructor reads metadata only. Package/device initialization and every
        # cache operation remain on the worker, including cleanup before first use.
        self._predictor = CachedCascadePredictor(
            model_name,
            model_dir,
            expected_manifest=expected_manifest,
            device=device,
            max_windows=max_windows,
            max_cache_bytes=max_cache_bytes,
        )
        self._queue: Queue[_Request | None] = Queue(maxsize=queue_capacity)
        self._closing = threading.Event()
        self._close_lock = threading.Lock()
        self._stats = self._predictor.stats
        self._worker = threading.Thread(
            target=self._run, name="cali-cascade-inference", daemon=True
        )
        self._worker.start()

    @property
    def minimum_frames(self) -> int:
        """Expose model preflight without involving the worker or Torch."""
        return self._predictor.minimum_frames

    @property
    def model(self) -> CascadeModel:
        """Expose immutable verified metadata without touching Torch."""
        return self._predictor.model

    def prepare(self, frame_rate: float) -> None:
        """Preflight package/device on the worker before Phase A starts."""
        self._submit(_Request(Future(), frame_rate=frame_rate, prepare=True))

    @property
    def stats(self) -> CascadeCacheStats:
        """Return the last worker-published immutable cache diagnostics."""
        return self._stats

    @property
    def pending_count(self) -> int:
        """Count queued requests; blocked callers retain their own input arrays."""
        return self._queue.qsize()

    def _cancelled(self, request: _Request) -> bool:
        return (
            self._closing.is_set()
            or request.aborted.is_set()
            or (request.cancel is not None and request.cancel())
        )

    def _submit(self, request: _Request) -> CascadeResult | None:
        if threading.current_thread() is self._worker:
            raise RuntimeError("CASCADE service cannot synchronously submit to itself.")
        while True:
            if self._closing.is_set():
                raise RuntimeError("CASCADE inference service is closed.")
            if self._cancelled(request):
                raise InferenceCancelled("CASCADE submission cancelled.")
            # Coordinate acceptance with close. Never block while holding the
            # lifecycle lock: waiting callers must still cancel or close promptly.
            with self._close_lock:
                if self._closing.is_set():
                    raise RuntimeError("CASCADE inference service is closed.")
                try:
                    self._queue.put_nowait(request)
                    break
                except Full:
                    pass
            request.aborted.wait(0.05)
        while True:
            try:
                return request.future.result(timeout=0.05)
            except CancelledError:
                raise InferenceCancelled(
                    "CASCADE queued submission cancelled."
                ) from None
            except TimeoutError:
                if request.future.done():
                    # A backend TimeoutError is a result, not a polling timeout.
                    return request.future.result()
                if self._cancelled(request):
                    request.aborted.set()
                    if request.future.cancel():
                        # The worker cannot start a cancelled Future. Release its
                        # input now; cancellation need not wait behind another FOV.
                        request.dff = None
                        request.timing = None
                        request.cancel = None
                        raise InferenceCancelled(
                            "CASCADE queued submission cancelled."
                        ) from None
                    # The worker observes this between chunks. Wait for ownership
                    # to finish rather than returning a still-mutating result.

    def infer_all(
        self,
        dff: np.ndarray,
        frame_rate: float,
        *,
        timing: TimingDescriptor,
        cancel: Callable[[], bool] | None = None,
    ) -> CascadeResult:
        """Submit a complete FOV, checking cancellation even while the queue is full."""
        result = self._submit(_Request(Future(), dff, frame_rate, timing, cancel))
        assert result is not None
        return result

    def clear_cache(self) -> None:
        """Schedule deterministic cache release on the owning worker."""
        self._submit(_Request(Future()))

    def _run(self) -> None:
        try:
            while True:
                request = self._queue.get()
                result = None
                try:
                    if request is None:
                        return
                    if not request.future.set_running_or_notify_cancel():
                        continue
                    try:
                        if self._cancelled(request):
                            raise InferenceCancelled("CASCADE queued batch cancelled.")
                        if request.prepare:
                            self._predictor.prepare(request.frame_rate)
                        elif request.dff is None:
                            self._predictor.clear_cache()
                            result = None
                        else:
                            assert request.timing is not None
                            result = self._predictor.infer_all(
                                request.dff,
                                request.frame_rate,
                                timing=request.timing,
                                cancel=partial(self._cancelled, request),
                            )
                        self._stats = self._predictor.stats
                        if self._cancelled(request):
                            raise InferenceCancelled("CASCADE service batch cancelled.")
                        request.future.set_result(result)
                    except BaseException as error:
                        self._stats = self._predictor.stats
                        # Preserve traceback locations, freeing failed inference
                        # frames' potentially large DFF/windows/Torch locals.
                        traceback.clear_frames(error.__traceback__)
                        request.future.set_exception(error)
                finally:
                    self._queue.task_done()
                    if request is not None:
                        request.dff = None
                        request.timing = None
                        request.cancel = None
                    # A blocked get must not keep the last FOV's DFF alive.
                    del result, request
        finally:
            self._predictor.close()
            self._stats = self._predictor.stats

    def close(self) -> None:
        """Cancel work, finish the current chunk, and join the owning worker."""
        if threading.current_thread() is self._worker:
            raise RuntimeError("CASCADE service cannot close from its own worker.")
        with self._close_lock:
            if not self._closing.is_set():
                self._closing.set()
                # Worker cancellation empties the bounded queue. A sentinel must
                # follow every accepted request so no caller's Future is abandoned.
                self._queue.put(None)
        self._worker.join()

    def __enter__(self) -> CascadeInferenceService:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
