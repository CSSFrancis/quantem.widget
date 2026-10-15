"""The live parts of ShowBragg: per-pattern detection off the kernel thread, the teach
marks, and the detector catalog the model menu shows."""

from __future__ import annotations

import asyncio
import base64
import io
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

#: Mark labels, as quantem.diffraction.neural_adapt defines them.
DISK, NOT_DISK = 1, 0

ICON_DIRECTORY = Path(__file__).parent / "showbragg_icons"
#: The built-in DE SpotNet detectors, as the model menu lists them.
BUILT_IN_DETECTORS = (
    ("spotnet", "DE SpotNet", "scale-robust all-rounder (default)"),
    ("dynamical", "DE SpotNet Dynamical", "centers on the disk edge; dynamical fills"),
    ("production", "DE SpotNet Production", "sharpest at the trained disk size"),
    ("base16", "DE SpotNet Base16", "earlier edge-trained detector"),
)


class LatestOnlyWorker:
    """Runs one job at a time on a background thread, always the newest.

    A request replaces any request still waiting, so a fast drag never queues stale
    work: whatever was asked last runs once the current job finishes. Results are
    handed to ``apply`` on the event loop that was running when the worker was made
    (the kernel's), because widget state must change on that thread. Without a
    running loop (a script, a test) requests run synchronously on the caller.
    """

    def __init__(self, apply: Callable[[Any], None], name: str = "showbragg-live"):
        self._apply = apply
        self._name = name
        self._condition = threading.Condition()
        self._pending: Callable[[], Any] | None = None
        self._thread: threading.Thread | None = None
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None
        self.synchronous = self._loop is None or not self._loop.is_running()

    def submit(self, job: Callable[[], Any]) -> None:
        """Run ``job`` soon, replacing any job that has not started yet."""
        if self.synchronous:
            self._apply(self._run(job))
            return
        with self._condition:
            self._pending = job
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop_jobs, name=self._name, daemon=True)
                self._thread.start()
            self._condition.notify()

    def _loop_jobs(self) -> None:
        while True:
            with self._condition:
                if self._pending is None:
                    self._condition.wait(timeout=5.0)
                    if self._pending is None:
                        self._thread = None
                        return
                job, self._pending = self._pending, None
            result = self._run(job)
            self._loop.call_soon_threadsafe(self._apply, result)

    @staticmethod
    def _run(job: Callable[[], Any]) -> Any:
        try:
            return job()
        except Exception as error:  # surfaced in the widget's status line
            return error


class TeachMarks:
    """The user's marks, per scan position: ``[row, col, label, confirmed]`` in detector
    pixels, where ``confirmed`` tells a disk mark on a detection the user agreed with
    from one on a disk the detector missed.

    One click cycles a place through the marks SpyDE uses. On a detection: wrong
    (:data:`NOT_DISK`), then confirmed (:data:`DISK`), then no mark. On empty pattern:
    a missed disk (:data:`DISK`), then no mark.
    """

    def __init__(self) -> None:
        self._marks: dict[tuple[int, int], list[list[float]]] = {}

    def click(
        self,
        position: tuple[int, int],
        row: float,
        col: float,
        detections: np.ndarray,
        reach: float,
    ) -> str:
        """Apply one click at ``(row, col)`` and say what it did.

        Parameters
        ----------
        position : tuple of int
            Scan position of the pattern.
        row, col : float
            The click, in detector pixels.
        detections : np.ndarray
            ``(N, >=2)`` detections shown on this pattern.
        reach : float
            How close (detector pixels) a click must be to hit a mark or detection.
        """
        position = (int(position[0]), int(position[1]))
        marks = self._marks.setdefault(position, [])
        nearest_mark = self._nearest(marks, row, col)
        if nearest_mark is not None and nearest_mark[1] <= reach:
            index = nearest_mark[0]
            mark = marks[index]
            if int(mark[2]) == NOT_DISK:
                mark[2], mark[3] = float(DISK), 1.0
                result = "confirmed"
            else:
                marks.pop(index)
                result = "removed"
        else:
            detections = (
                np.asarray(detections, np.float64).reshape(len(detections), -1)
                if len(detections)
                else np.zeros((0, 2))
            )
            if len(detections):
                distance = np.hypot(detections[:, 0] - row, detections[:, 1] - col)
                nearest = int(np.argmin(distance))
            if len(detections) and distance[nearest] <= reach:
                marks.append(
                    [float(detections[nearest, 0]), float(detections[nearest, 1]), float(NOT_DISK), 0.0]
                )
                result = "wrong"
            else:
                marks.append([float(row), float(col), float(DISK), 0.0])
                result = "missed"
        if not marks:
            self._marks.pop(position, None)
        return result

    @staticmethod
    def _nearest(marks, row, col):
        if not marks:
            return None
        distance = [np.hypot(m[0] - row, m[1] - col) for m in marks]
        index = int(np.argmin(distance))
        return index, float(distance[index])

    def at(self, position) -> np.ndarray:
        """``(n, 3)`` ``[row, col, label]`` marks at one position, for teaching."""
        marks = self._marks.get((int(position[0]), int(position[1])), [])
        return np.asarray(marks, np.float64).reshape(-1, 4)[:, :3]

    def positions(self) -> list[tuple[int, int]]:
        return sorted(self._marks)

    def counts(self) -> dict[str, int]:
        """How many marks of each kind: wrong, missed and confirmed."""
        kinds = [self.kind(mark) for marks in self._marks.values() for mark in marks]
        return {kind: kinds.count(kind) for kind in ("wrong", "missed", "confirmed")}

    @staticmethod
    def kind(mark) -> str:
        if int(mark[2]) == NOT_DISK:
            return "wrong"
        return "confirmed" if mark[3] else "missed"

    def clear(self) -> None:
        self._marks.clear()

    def __len__(self) -> int:
        return sum(len(m) for m in self._marks.values())

    def to_trait(self) -> dict[str, list[list[float]]]:
        return {f"{r},{c}": [list(m) for m in marks] for (r, c), marks in self._marks.items()}


def icon_data_url(name: str) -> str:
    """A built-in detector's icon (its preferred input) as a PNG data URL."""
    path = ICON_DIRECTORY / ("de_spotnet.png" if name == "spotnet" else f"de_spotnet_{name}.png")
    if not path.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def preferred_input_icon(model) -> str:
    """A taught model's icon: its preferred input on a diverging scale, as SpyDE draws
    the built-in icons."""
    from matplotlib import colormaps
    from PIL import Image
    from quantem.diffraction.neural_adapt import preferred_input

    image = preferred_input(model)
    margin = image.shape[0] // 8
    image = image[margin:-margin, margin:-margin]
    scale = float(np.percentile(np.abs(image), 99.5)) or 1.0
    rgba = colormaps["RdBu_r"](np.clip(0.5 + 0.5 * image / scale, 0, 1))
    picture = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).resize((64, 64), Image.BICUBIC)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
