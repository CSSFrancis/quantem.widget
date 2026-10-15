"""showbragg: front end for the BraggVectors disk-detection workflow"""

import base64
import json
import pathlib
import re
import threading
import time
from typing import Any

import anywidget
import numpy as np
import torch
import traitlets
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.vector import Vector
from quantem.diffraction.bragg_vectors import BraggVectors
from quantem.diffraction.disk_detection import SUBPIXEL_MODES
from quantem.diffraction.strain import StrainMap

try:
    from quantem.diffraction.neural_disk_detection import (
        counts_radius,
        load_spot_net,
        neural_heatmap,
        save_spot_net,
    )
    from quantem.diffraction.spot_detection import (
        detect_beamstop,
        difference_of_gaussians_snr,
    )
except ImportError:  # a quantem without the template-free detectors
    load_spot_net = neural_heatmap = detect_beamstop = difference_of_gaussians_snr = None
    counts_radius = save_spot_net = None

from quantem.widget.render import frame_to_rgb, rgb_to_png_bytes
from quantem.widget.showbragg_live import (
    BUILT_IN_DETECTORS,
    LatestOnlyWorker,
    TeachMarks,
    icon_data_url,
    preferred_input_icon,
)
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.static_fallback import StaticFallbackMixin
from quantem.widget.utils.traits import reject_unknown_kwargs
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

TEMPLATE_SOURCES = ("synthetic", "data", "probe")
STAGE_STATES = ("idle", "preview", "running", "done", "error")
NAVIGATOR_KINDS = ("sum", "adf")
DETECTION_METHODS = ("correlation", "dog", "neural")
# Switching method resets the controls whose meaning changes with it: the
# Difference-of-Gaussians threshold is a signal-to-noise ratio, the neural one a
# heatmap confidence, and correlation thresholds on min_abs_intensity instead.
METHOD_DEFAULTS = {
    "correlation": {"min_spacing": 0.0, "subpixel": "upsample"},
    "dog": {"threshold": 10.0, "min_spacing": 5.0, "subpixel": "parabolic"},
    "neural": {"threshold": 0.3, "min_spacing": 4.0},
}


class ShowBragg(StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive driver for the BraggVectors disk-detection and lattice-fitting workflow.

    Parameters
    ----------
    data : Dataset4dstem or BraggVectors
        A 4D-STEM dataset, or an existing BraggVectors to reopen with its
        detection kept.
    probe : array_like, optional
        Measured vacuum probe, required for template_source="probe".
    model : str, Path or SpotNet, optional
        DE SpotNet detector for detection_method="neural": "spotnet" (the default),
        "dynamical", "production", "base16", a local checkpoint (a taught model saved
        earlier, say), or a model in memory. Named detectors download from
        em-database.
    replay : str or Path, optional
        The replay buffer teaching mixes in (de_spotnet_replay.npz). ``None`` uses the
        published one when it can be fetched and otherwise teaches without it.
    refine_model, friedel_model : str or Path, optional
        Local checkpoints of DE CenterNet and DE FriedelNet for the two center steps
        (refine_centers and friedel_partner), instead of the em-database downloads.
    device : str, default="cpu"
        Torch device passed through to BraggVectors.
    cmap : str, default="inferno"
        Matplotlib colormap for every image panel.
    title : str, default=""
        Title above the panels. Falls back to the dataset name.
    log_scale : bool, default=False
        Log-stretch every image panel.
    ui_mode : {"interactive", "presentation", "report", "minimal"}, default="interactive"
        Display-chrome preset; explicit keywords below override it.
    show_title : bool, default=True
        Show the title row.
    show_controls : bool, default=True
        Expose the parameter controls at all.
    controls_collapsed : bool, default=False
        Start with the parameter controls hidden.
    save_state : bool, default=False
        Embed the full interactive state in the notebook.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default=None
        Static preview format used when save_state=False. None matches
        ChooseLattice, whose live widget does not hide the fallback sibling.
    notebook_preview_quality : int, default=88
        Lossy preview quality for JPEG/WebP, from 1 to 100.
    notebook_preview_max_px : int, default=512
        Longest image side for the saved-notebook preview.

    Notes
    -----
    The top row is the live view: drag the crosshair on the navigator (or step it with
    the arrow keys) and the pattern under it is detected again on a background thread,
    the newest position always winning, so a fast drag never queues stale work. Any
    detection setting re-detects the same pattern.

    With the neural method, clicking the pattern teaches the detector: a click on a
    detection marks it wrong, a second click confirms it, a third clears the mark;
    a click on empty pattern marks a disk the detector missed. Teach fine-tunes a copy
    of the detector on every mark so far (quantem.diffraction.neural_adapt) and
    switches to it; Revert goes back; save_model writes it for model= later.

    template_radius, max_peak_shift and num_candidates use 0 as the sentinel for
    "let the data choose"; none of the three is meaningful at zero. num_candidates
    resolves to however many peaks the busiest scan position detected.

    Examples
    --------
    >>> widget = ShowBragg(dataset)
    >>> widget
    >>> widget.detect()
    >>> widget.fit()
    >>> strain = widget.strain_map()
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showbragg.js"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    scan_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    q_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)

    # Stage 1, template
    template_source = traitlets.Enum(TEMPLATE_SOURCES, default_value="synthetic").tag(sync=True)
    template_radius = traitlets.Float(0.0).tag(sync=True)
    template_edge = traitlets.Float(1.0).tag(sync=True)
    template_center = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    template_subtract_mean = traitlets.Bool(True).tag(sync=True)
    template_roi = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    template_png = traitlets.Bytes(b"").tag(sync=True)
    template_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    has_probe = traitlets.Bool(False).tag(sync=True)

    # Stage 2, correlation probe
    probe_position = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    probe_diffraction_png = traitlets.Bytes(b"").tag(sync=True)
    probe_correlation_png = traitlets.Bytes(b"").tag(sync=True)

    # The live view: navigator, the pattern under the crosshair and its detections
    navigator_kind = traitlets.Enum(NAVIGATOR_KINDS, default_value="sum").tag(sync=True)
    navigator_png = traitlets.Bytes(b"").tag(sync=True)
    pattern_png = traitlets.Bytes(b"").tag(sync=True)
    pattern_log = traitlets.Bool(True).tag(sync=True)
    show_response = traitlets.Bool(False).tag(sync=True)
    spots = traitlets.List(default_value=[]).tag(sync=True)
    spots_position = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    spot_display_radius = traitlets.Float(4.0).tag(sync=True)
    detect_ms = traitlets.Float(0.0).tag(sync=True)
    live_status = traitlets.Unicode("").tag(sync=True)

    # Teaching the neural detector
    models = traitlets.List(default_value=[]).tag(sync=True)
    neural_model = traitlets.Unicode("spotnet").tag(sync=True)
    teach_marks = traitlets.Dict(default_value={}).tag(sync=True)
    teach_state = traitlets.Dict(default_value={}).tag(sync=True)

    # Stages 3 and 4, detection
    detection_method = traitlets.Enum(DETECTION_METHODS, default_value="correlation").tag(
        sync=True
    )
    threshold = traitlets.Float(0.3).tag(sync=True)
    dog_sigma1 = traitlets.Float(0.8).tag(sync=True)
    dog_sigma2 = traitlets.Float(2.0).tag(sync=True)
    spot_radius = traitlets.Float(0.0).tag(sync=True)
    mask_beamstop = traitlets.Bool(False).tag(sync=True)
    refine_centers = traitlets.Bool(True).tag(sync=True)
    friedel_partner = traitlets.Bool(True).tag(sync=True)
    neighbor_refine = traitlets.Bool(False).tag(sync=True)
    min_abs_intensity = traitlets.Float(0.0).tag(sync=True)
    min_spacing = traitlets.Float(0.0).tag(sync=True)
    edge_boundary = traitlets.Int(1).tag(sync=True)
    subpixel = traitlets.Enum(SUBPIXEL_MODES, default_value="upsample").tag(sync=True)
    upsample_factor = traitlets.Int(16).tag(sync=True)
    max_num_peaks = traitlets.Int(1000).tag(sync=True)
    preview_grid = traitlets.Int(8).tag(sync=True)
    preview_peaks = traitlets.Unicode("").tag(sync=True)
    detection_state = traitlets.Enum(STAGE_STATES, default_value="idle").tag(sync=True)

    # Stages 5 and 6, Bragg vector map and basis
    bvm_sampling = traitlets.Float(1.0).tag(sync=True)
    bvm_png = traitlets.Bytes(b"").tag(sync=True)
    num_candidates = traitlets.Int(0).tag(sync=True)
    candidate_min_spacing = traitlets.Float(2.0).tag(sync=True)
    candidate_min_abs_intensity = traitlets.Float(0.0).tag(sync=True)
    candidates = traitlets.List(default_value=[]).tag(sync=True)
    origin_index = traitlets.Int(-1).tag(sync=True)
    g1_index = traitlets.Int(-1).tag(sync=True)
    g2_index = traitlets.Int(-1).tag(sync=True)
    origin_rc = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    g1_rc = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    g2_rc = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)

    # Stage 7, lattice fit
    min_num_peaks = traitlets.Int(5).tag(sync=True)
    max_peak_shift = traitlets.Float(0.0).tag(sync=True)
    fit_state = traitlets.Enum(STAGE_STATES, default_value="idle").tag(sync=True)
    mask_weight_png = traitlets.Bytes(b"").tag(sync=True)
    fit_error_png = traitlets.Bytes(b"").tag(sync=True)

    _TEMPLATE_TRAITS = (
        "template_source",
        "template_radius",
        "template_edge",
        "template_center",
        "template_subtract_mean",
        "template_roi",
    )
    _LIVE_TRAITS = (
        "probe_position",
        "dog_sigma1",
        "dog_sigma2",
        "spot_radius",
        "mask_beamstop",
        "threshold",
        "min_abs_intensity",
        "min_spacing",
        "edge_boundary",
        "subpixel",
        "upsample_factor",
        "max_num_peaks",
        "refine_centers",
        "friedel_partner",
        "neural_model",
        "show_response",
        "pattern_log",
    )
    _CANDIDATE_TRAITS = (
        "num_candidates",
        "candidate_min_spacing",
        "candidate_min_abs_intensity",
        "origin_index",
        "g1_index",
        "g2_index",
        "origin_rc",
        "g1_rc",
        "g2_rc",
    )

    def __init__(
        self,
        data,
        *,
        probe=None,
        model=None,
        refine_model=None,
        friedel_model=None,
        replay=None,
        device: str = "cpu",
        cmap: str = "inferno",
        log_scale: bool = False,
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        save_state: bool = False,
        notebook_preview_format: str | None = None,
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        **kwargs,
    ) -> None:
        reject_unknown_kwargs(type(self), kwargs)
        method = kwargs.get("detection_method")
        if method in METHOD_DEFAULTS:
            # The method's own defaults, unless the caller set them too.
            kwargs = {**METHOD_DEFAULTS[method], **kwargs}
        super().__init__(**kwargs)

        if isinstance(data, BraggVectors):
            self._bragg = data
        elif isinstance(data, Dataset4dstem):
            self._bragg = BraggVectors.from_dataset(data, device=device)
        else:
            raise TypeError(
                "ShowBragg expects a Dataset4dstem or a BraggVectors instance, got "
                f"{type(data).__name__}."
            )

        dataset = self._bragg.dataset
        self._probe = None if probe is None else to_numpy(probe, dtype=np.float32)
        self._refine_model = refine_model
        self._friedel_model = friedel_model
        self._replay = replay
        self._custom_model = None
        if model is not None and str(model) in {name for name, _, _ in BUILT_IN_DETECTORS}:
            self.neural_model = str(model)
        elif model is not None:
            self._custom_model = model
            self.neural_model = "custom"
        self._taught = None
        self._taught_icon = ""
        self._teach_base = None
        self._marks = TeachMarks()
        self._spot_nets: dict[str, Any] = {}
        self._detect_lock = threading.Lock()
        self._estimated_radius: float | None = None
        self._live = LatestOnlyWorker(self._apply_live)
        self._teacher = LatestOnlyWorker(self._apply_teach, name="showbragg-teach")

        if self._bragg.template is None:
            self._build_template()

        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        self._save_state = bool(save_state)

        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_controls": True,
                "controls_collapsed": False,
            },
            overrides={
                "show_title": show_title,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
            },
        )

        with self.hold_sync():
            self.show_title = bool(ui["show_title"])
            self.show_controls = bool(ui["show_controls"])
            self.controls_collapsed = bool(ui["controls_collapsed"])
            self.scan_shape = [int(dataset.shape[0]), int(dataset.shape[1])]
            self.q_shape = [int(dataset.shape[-2]), int(dataset.shape[-1])]
            self.title = str(title or getattr(dataset, "name", "") or "")
            self.cmap = str(cmap)
            self.log_scale = bool(log_scale)
            self.has_probe = self._probe is not None
            self._render_template()
            self._render_navigator()
            self._publish_models()
            if self._bragg.peaks is not None:
                self.detection_state = "done"
                self._update_bvm()
                self._choose_basis()
            if self._bragg.g1_array is not None:
                self.fit_state = "done"
                self._render_fit()

        self.observe(self._on_template_change, names=self._TEMPLATE_TRAITS)
        self.observe(self._on_method_change, names="detection_method")
        self.observe(self._on_live_change, names=self._LIVE_TRAITS)
        self.observe(self._on_navigator_change, names="navigator_kind")
        self.observe(self._on_basis_change, names=self._CANDIDATE_TRAITS)
        self.observe(self._on_display_change, names=["cmap", "log_scale"])
        self.on_msg(self._on_panel_message)
        self._request_live()

        try:
            from importlib.metadata import version

            self.widget_version = version("quantem-widget")
        except Exception:
            pass

    # Validators

    @traitlets.validate("probe_position")
    def _validate_probe_position(self, proposal):
        row, col = proposal["value"]
        rows, cols = self.scan_shape if self.scan_shape else (1, 1)
        return [
            int(np.clip(row, 0, max(0, rows - 1))),
            int(np.clip(col, 0, max(0, cols - 1))),
        ]

    @traitlets.validate("template_center", "origin_rc", "g1_rc", "g2_rc")
    def _validate_pair(self, proposal):
        value = [float(v) for v in proposal["value"]]
        if value and len(value) != 2:
            raise traitlets.TraitError(
                f"{proposal['trait'].name} must be empty or a (row, col) pair, "
                f"got {len(value)} values."
            )
        return value

    @traitlets.validate("template_roi")
    def _validate_roi(self, proposal):
        value = [float(v) for v in proposal["value"]]
        if value and len(value) != 4:
            raise traitlets.TraitError(
                f"template_roi must be empty or [r0, c0, r1, c1], got {len(value)} values."
            )
        return value

    # Public API

    @property
    def bragg(self) -> BraggVectors:
        """The wrapped workflow object, for anything the panels do not expose."""
        return self._bragg

    @property
    def peaks(self) -> Vector | None:
        """Detected peaks, or None before a full detection run."""
        return self._bragg.peaks

    @property
    def bvm(self) -> Dataset2d | None:
        """Bragg vector map, or None before a full detection run."""
        return self._bragg.bvm

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(origin, g1, g2), with origin absolute and g1/g2 offsets from it."""
        if self._bragg.origin is None:
            return None
        return (self._bragg.origin, self._bragg.g1, self._bragg.g2)

    @property
    def taught_model(self):
        """The detector taught on the marks, or None before Teach."""
        return self._taught

    @property
    def marks(self) -> TeachMarks:
        """The teach marks, per scan position."""
        return self._marks

    def mark(self, row: float, col: float, position: tuple[int, int] | None = None) -> str:
        """Click the pattern at ``(row, col)`` detector pixels, as the panel does.

        On a detection the click marks it wrong, then confirmed, then clears the mark;
        on empty pattern it marks a missed disk, then clears it.

        Returns
        -------
        str
            "wrong", "confirmed", "missed" or "removed".
        """
        position = tuple(position or self.probe_position)
        shown = np.asarray(self.spots, np.float64).reshape(-1, 3) if self.spots else np.zeros((0, 3))
        if list(position) != list(self.spots_position):
            shown = np.zeros((0, 3))
        reach = max(3.0, float(self.spot_display_radius))
        result = self._marks.click(position, float(row), float(col), shown[:, :2], reach)
        if self._teach_base is None and self.neural_model != "taught":
            self._teach_base = self.neural_model
        self._publish_marks(f"Marked {result}.")
        return result

    def teach(self) -> None:
        """Fine-tune a copy of the detector on every mark so far, then switch to it.

        Runs on a background thread in a live kernel; the status line reports when it
        is done.
        """
        if not len(self._marks):
            raise ValueError("Mark a wrong detection or a missed disk first.")
        if str(self.detection_method) != "neural":
            raise ValueError("Teaching applies to the neural detector.")
        base_name = self._teach_base or self.neural_model
        settings = {
            "spot_radius": float(self.spot_radius) or None,
            "min_spacing": int(round(float(self.min_spacing))),
            "threshold": float(self.threshold),
        }
        positions = self._marks.positions()
        marks = [self._marks.at(p) for p in positions]
        patterns = [np.asarray(self._bragg.dataset.array[r, c], np.float32) for r, c in positions]
        base = self._spot_net(base_name)
        replay = self._replay
        self._set_teach_state(busy=True)
        self.live_status = "Teaching..."

        def job():
            from quantem.diffraction.neural_adapt import teach_spot_net

            bank = replay
            note = ""
            if bank is None:
                try:
                    from quantem.diffraction.neural_weights import (
                        REPLAY_ENTRY,
                        entry_path,
                    )

                    bank = entry_path(REPLAY_ENTRY)
                except Exception:
                    bank = None
                    note = " (no replay buffer: pass ShowBragg(replay=...) to keep it general)"
            model, report = teach_spot_net(base, patterns, marks, replay=bank, **settings)
            return {"model": model, "report": report, "icon": preferred_input_icon(model),
                    "base": base_name, "note": note}

        self._teacher.submit(job)

    def revert_teaching(self) -> None:
        """Back to the detector the marks were made against; the marks are cleared."""
        if self._teach_base is not None:
            self.neural_model = self._teach_base
        self._taught = None
        self._teach_base = None
        self._marks.clear()
        self._set_teach_state(taught=False, summary="", original_f1=None, original_f1_base=None)
        self._publish_models()
        self._publish_marks("Back to the original detector.")

    def save_model(self, path: str | pathlib.Path | None = None) -> pathlib.Path:
        """Write the taught detector as a checkpoint ``model=`` can load later.

        Parameters
        ----------
        path : str or Path, optional
            The ``.pt`` file; by default ``<title>_spotnet_taught.pt`` here.

        Returns
        -------
        Path
            The written file.
        """
        if self._taught is None:
            raise ValueError("Teach first: there is no taught detector to save.")
        if path is None:
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", self.title or "dataset").strip("_") or "dataset"
            path = pathlib.Path(f"{stem}_spotnet_taught.pt")
        written = save_spot_net(self._taught, path)
        self._set_teach_state(saved=str(written))
        self.live_status = f"Saved the taught detector to {written}."
        return written

    def detect(self, positions: list[tuple[int, int]] | None = None, **kwargs) -> Vector:
        """Run disk detection with the current detection traits.

        Parameters
        ----------
        positions : list of tuple of int, optional
            (row, col) positions to test on. None runs the full scan and
            populates peaks; a subset leaves the workflow state untouched.
        **kwargs
            Overrides for the detection traits, forwarded to detect_disks.

        Returns
        -------
        Vector
            Detected peaks.
        """
        detect_kwargs = {**self._method_kwargs(), **kwargs}
        preview = positions is not None
        if not preview:
            detect_kwargs.setdefault("neighbor_refine", bool(self.neighbor_refine))

        self.detection_state = "preview" if preview else "running"
        self.status = ""
        try:
            with self._detect_lock:
                found = self._bragg.detect_disks(
                    positions=positions, progressbar=False, **detect_kwargs
                )
        except Exception as exc:
            self.detection_state = "error"
            self.status = f"Detection failed: {exc}"
            raise

        if preview:
            self.preview_peaks = json.dumps(
                {
                    "positions": [[int(r), int(c)] for r, c in positions],
                    "counts": [int(found[i].numpy().shape[0]) for i in range(len(positions))],
                    "peaks": [[float(v) for v in row] for row in found[0].numpy()],
                }
            )
            self.detection_state = "done" if self._bragg.peaks is not None else "idle"
            return found

        with self.hold_sync():
            self.detection_state = "done"
            self._update_bvm()
            self._choose_basis()
        return found

    def fit(self, **kwargs) -> "ShowBragg":
        """Fit the lattice at every scan position with the current fit traits.

        Parameters
        ----------
        **kwargs
            Overrides forwarded to fit_lattice.

        Returns
        -------
        ShowBragg
            self, for method chaining.
        """
        fit_kwargs = {
            "min_num_peaks": int(self.min_num_peaks),
            "max_peak_shift": float(self.max_peak_shift) or None,
            **kwargs,
        }

        self.fit_state = "running"
        self.status = ""
        try:
            self._bragg.fit_lattice(progressbar=False, plot=False, **fit_kwargs)
        except Exception as exc:
            self.fit_state = "error"
            self.status = f"Lattice fit failed: {exc}"
            raise

        with self.hold_sync():
            self.fit_state = "done"
            self._render_fit()
        return self

    def collapse_controls(self) -> "ShowBragg":
        """Collapse the parameter controls."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> "ShowBragg":
        """Expand the parameter controls."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> "ShowBragg":
        """Toggle whether the parameter controls are collapsed."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def strain_map(self, **kwargs) -> StrainMap:
        """Hand the fitted lattice vectors to a StrainMap.

        Parameters
        ----------
        **kwargs
            u_ref, v_ref and mask, forwarded to calculate_strain_map.

        Returns
        -------
        StrainMap
            Strain map built from the per-position lattice vectors.
        """
        return self._bragg.calculate_strain_map(**kwargs)

    # Stage handlers

    def _method_kwargs(self) -> dict[str, Any]:
        """detect_disks keyword arguments for the selected detection method."""
        method = str(self.detection_method)
        common = {
            "method": method,
            "min_spacing": float(self.min_spacing),
            "max_num_peaks": int(self.max_num_peaks),
        }
        if method == "correlation":
            return {
                **common,
                "min_abs_intensity": float(self.min_abs_intensity),
                "edge_boundary": int(self.edge_boundary),
                "subpixel": str(self.subpixel),
                "upsample_factor": int(self.upsample_factor),
            }
        mask = "auto" if self.mask_beamstop else None
        if method == "dog":
            return {
                **common,
                "threshold": float(self.threshold),
                "sigma1": float(self.dog_sigma1),
                "sigma2": float(self.dog_sigma2),
                "edge_boundary": int(self.edge_boundary),
                "subpixel": str(self.subpixel),
                "mask": mask,
            }
        return {
            **common,
            "threshold": float(self.threshold),
            "spot_diameter": 2.0 * float(self.spot_radius) or None,
            "mask": mask,
            "model": self._model_argument(self.neural_model),
            "refine_centers": (self._refine_model or True) if self.refine_centers else False,
            "friedel_partner": (self._friedel_model or True) if self.friedel_partner else False,
        }

    def _model_argument(self, name: str):
        """What detect_disks(model=...) takes for a model-menu entry."""
        if name == "taught" and self._taught is not None:
            return self._taught
        if name == "custom" and self._custom_model is not None:
            return self._custom_model
        return name

    def _spot_net(self, name: str):
        """The DE SpotNet for a model-menu entry, loaded on first use."""
        if name == "taught" and self._taught is not None:
            return self._taught
        if name not in self._spot_nets:
            self._spot_nets[name] = load_spot_net(
                self._model_argument(name), device=self._bragg.device
            )
        return self._spot_nets[name]

    def _neural_model(self):
        """The DE SpotNet detector in use for detection_method="neural"."""
        return self._spot_net(self.neural_model)

    def _response_map(self, row: int, col: int) -> np.ndarray:
        """What the selected method thresholds, for the pattern at one scan position."""
        method = str(self.detection_method)
        if method == "correlation":
            return self._bragg.correlation_map(row, col)
        if difference_of_gaussians_snr is None:
            raise ImportError(
                f"detection_method={method!r} needs a quantem with "
                "quantem.diffraction.spot_detection."
            )
        pattern = torch.as_tensor(
            np.asarray(self._bragg.dataset.array[row, col]),
            dtype=torch.float32,
            device=self._bragg.device,
        )[None]
        if method == "dog":
            mask = self._beamstop()
            mask_t = None if mask is None else torch.as_tensor(mask, device=pattern.device)
            snr = difference_of_gaussians_snr(
                pattern, float(self.dog_sigma1), float(self.dog_sigma2), mask_t
            )
            return snr[0].clamp_min(0).cpu().numpy()
        heatmap = neural_heatmap(
            pattern, self._neural_model(), spot_diameter=2.0 * float(self.spot_radius) or None
        )
        return heatmap[0].cpu().numpy()

    def _beamstop(self) -> np.ndarray | None:
        """Beam-stop mask from the mean pattern when mask_beamstop is on."""
        if not self.mask_beamstop:
            return None
        return detect_beamstop(np.asarray(self._bragg.dataset.dp_mean.array))

    # The live view

    def _display_radius(self) -> float:
        """Radius the detections are drawn at: the spot size when set, else the disk
        size estimated once from the mean pattern."""
        if float(self.spot_radius) > 0:
            return float(self.spot_radius)
        if self._estimated_radius is None:
            mean = np.asarray(self._bragg.dataset.dp_mean.array, np.float32)
            self._estimated_radius = (
                float(counts_radius(mean, None)) if counts_radius is not None else 4.0
            )
        return self._estimated_radius

    def _request_live(self) -> None:
        """Detect the pattern under the crosshair again, off the kernel thread."""
        position = (int(self.probe_position[0]), int(self.probe_position[1]))
        try:
            kwargs = self._method_kwargs()
        except Exception as exc:
            self.live_status = str(exc)
            return
        show_response = bool(self.show_response)
        pattern_log = bool(self.pattern_log)
        method = str(self.detection_method)

        def job():
            started = time.perf_counter()
            with self._detect_lock:
                found = self._bragg.detect_disks(
                    positions=[position], progressbar=False, **kwargs
                )[0].numpy()
            milliseconds = 1000.0 * (time.perf_counter() - started)
            pattern = np.asarray(self._bragg.dataset.array[position], dtype=float)
            if show_response:
                image = self._render_frame(self._response_map(*position), log=False)
            else:
                image = self._render_frame(pattern, log=pattern_log)
            columns = [0, 1, 3] if found.shape[1] >= 4 else [0, 1, 2]
            return {
                "position": list(position),
                "method": method,
                "spots": found[:, columns].tolist(),
                "milliseconds": milliseconds,
                "image": image,
                "response": show_response,
            }

        self._live.submit(job)

    def _apply_live(self, result) -> None:
        if isinstance(result, Exception):
            self.live_status = f"Detection failed: {result}"
            return
        if result["position"] != list(self.probe_position):
            return  # superseded: a newer position is already queued
        with self.hold_sync():
            self.spots = result["spots"]
            self.spots_position = result["position"]
            self.spot_display_radius = self._display_radius()
            self.detect_ms = float(result["milliseconds"])
            self.pattern_png = result["image"]
            if result["response"]:
                self.probe_correlation_png = result["image"]
            else:
                self.probe_diffraction_png = result["image"]
            row, col = result["position"]
            self.live_status = (
                f"{len(result['spots'])} spots at ({row}, {col}) · "
                f"{result['milliseconds']:.0f} ms"
            )

    def _apply_teach(self, result) -> None:
        if isinstance(result, Exception):
            self._set_teach_state(busy=False)
            self.live_status = f"Teaching failed: {result}"
            return
        report = result["report"]
        self._taught = result["model"]
        self._taught_icon = result["icon"]
        self._teach_base = result["base"]
        learned, total = report["marks_learned"], report["marks_total"]
        verdict = "all marks learned" if learned == total else f"{learned} of {total} marks learned"
        general = ""
        if report["original_f1"] is not None:
            general = (
                f" · general F1 {report['original_f1']:.2f}"
                f" (was {report['original_f1_base']:.2f})"
            )
        self._set_teach_state(
            busy=False,
            taught=True,
            learned=learned,
            total=total,
            original_f1=report["original_f1"],
            original_f1_base=report["original_f1_base"],
        )
        summary = (
            f"Taught in {report['seconds']:.1f} s ({report['steps']} steps): "
            f"{verdict}{general}{result['note']}"
        )
        self._set_teach_state(summary=summary)
        self._publish_models()
        self.live_status = summary
        self.neural_model = "taught"

    def _set_teach_state(self, **changes) -> None:
        state = {**self.teach_state, **changes}
        state.update(self._marks.counts())
        self.teach_state = state

    def _publish_marks(self, message: str) -> None:
        with self.hold_sync():
            self.teach_marks = self._marks.to_trait()
            self._set_teach_state()
            self.live_status = message

    def _publish_models(self) -> None:
        """The model menu: the built-in detectors, then a custom or taught one."""
        entries = [
            {"id": name, "label": label, "subtitle": subtitle, "icon": icon_data_url(name)}
            for name, label, subtitle in BUILT_IN_DETECTORS
        ]
        if self._custom_model is not None:
            stem = pathlib.Path(str(self._custom_model)).stem
            built_in = {
                ("de_spotnet" if name == "spotnet" else f"de_spotnet_{name}"): (name, label)
                for name, label, _ in BUILT_IN_DETECTORS
            }
            if stem in built_in:
                name, label = built_in[stem]
                entry = {"label": f"{label} (local file)", "icon": icon_data_url(name)}
            else:
                entry = {"label": stem, "icon": ""}
            entries.append({"id": "custom", "subtitle": str(self._custom_model), **entry})
        if self._taught is not None:
            entries.append(
                {
                    "id": "taught",
                    "label": f"Taught: {self.title or 'this dataset'}",
                    "subtitle": f"from {self._teach_base} on {len(self._marks)} marks",
                    "icon": self._taught_icon,
                    "taught": True,
                }
            )
        self.models = entries

    def _navigator_image(self) -> np.ndarray:
        """Virtual image of the scan: the summed pattern, or an annular dark field from
        twice the central beam radius outward."""
        data = np.asarray(self._bragg.dataset.array)
        if self.navigator_kind == "sum":
            return data.sum(axis=(-2, -1))
        mean = data.mean(axis=(0, 1))
        rows, cols = np.indices(mean.shape)
        peak = np.unravel_index(np.argmax(mean), mean.shape)
        radius = 2.0 * self._display_radius()
        annulus = np.hypot(rows - peak[0], cols - peak[1]) > radius
        return (data * annulus).sum(axis=(-2, -1))

    def _render_navigator(self) -> None:
        self.navigator_png = self._render_frame(self._navigator_image(), log=False)

    def _render_frame(self, frame, *, log: bool) -> bytes:
        return rgb_to_png_bytes(
            frame_to_rgb(np.asarray(frame, dtype=float), cmap=self.cmap, log_scale=log)
        )

    def _build_template(self) -> None:
        """Rebuild the correlation template from the current template traits."""
        center = tuple(self.template_center) if self.template_center else None
        subtract_mean = bool(self.template_subtract_mean)

        if self.template_source == "synthetic":
            self._bragg.make_template_synthetic(
                radius=float(self.template_radius) or None,
                edge=float(self.template_edge),
                center=center,
                subtract_mean=subtract_mean,
            )
        elif self.template_source == "data":
            self._bragg.make_template_from_data(
                roi=self._roi_mask(),
                subtract_mean=subtract_mean,
                center=center,
            )
        else:
            if self._probe is None:
                raise ValueError(
                    "template_source='probe' needs a probe image; pass "
                    "ShowBragg(..., probe=vacuum_probe)."
                )
            self._bragg.make_template_from_probe(
                self._probe, center=center, subtract_mean=subtract_mean
            )

    def _roi_mask(self) -> np.ndarray | None:
        """Boolean scan mask from template_roi, or None for the whole scan."""
        if not self.template_roi:
            return None
        rows, cols = self.scan_shape
        r0, c0, r1, c1 = self.template_roi
        mask = np.zeros((rows, cols), dtype=bool)
        mask[
            int(np.clip(min(r0, r1), 0, rows)) : int(np.clip(max(r0, r1), 0, rows)) + 1,
            int(np.clip(min(c0, c1), 0, cols)) : int(np.clip(max(c0, c1), 0, cols)) + 1,
        ] = True
        return mask

    def _preview_positions(self) -> list[tuple[int, int]]:
        """Probe position first, then an evenly spaced grid across the scan."""
        rows, cols = self.scan_shape
        n = max(1, int(self.preview_grid))
        grid_rows = np.unique(np.linspace(0, rows - 1, min(n, rows)).astype(int))
        grid_cols = np.unique(np.linspace(0, cols - 1, min(n, cols)).astype(int))

        probe = (int(self.probe_position[0]), int(self.probe_position[1]))
        positions = [probe]
        positions += [
            (int(r), int(c)) for r in grid_rows for c in grid_cols if (int(r), int(c)) != probe
        ]
        return positions

    def _candidate_count(self) -> int:
        """How many candidate peaks to number, from the data unless set explicitly."""
        if self.num_candidates > 0:
            return int(self.num_candidates)

        peaks = self._bragg.peaks
        counts = peaks.row_counts() if peaks is not None else []
        return max(4, int(max(counts))) if len(counts) else 100

    def _choose_basis(self) -> None:
        """Re-derive the candidate peaks and resolve the basis from the picks."""
        if self._bragg.bvm is None:
            return

        try:
            self._bragg.choose_basis_vectors(
                origin=self._basis_pick("origin"),
                g1=self._basis_pick("g1"),
                g2=self._basis_pick("g2"),
                num_candidates=self._candidate_count(),
                min_spacing=float(self.candidate_min_spacing),
                min_abs_intensity=float(self.candidate_min_abs_intensity),
                plot=False,
            )
            self._bragg.index_peaks(plot=False)
        except Exception as exc:
            self.status = f"Basis selection failed: {exc}"
            return

        cand_rc = self._bragg.candidates_rc
        cand_int = self._bragg.candidates_intensity

        self._resolving_basis = True
        try:
            with self.hold_sync():
                self.status = ""
                self.candidates = [
                    [float(rc[0]), float(rc[1]), float(i)]
                    for rc, i in zip(cand_rc, cand_int)
                ]
                self.origin_rc = [float(v) for v in self._bragg.origin]
                self.g1_rc = [float(v) for v in self._bragg.g1]
                self.g2_rc = [float(v) for v in self._bragg.g2]
        finally:
            self._resolving_basis = False

    def _basis_pick(self, name: str) -> int | list[float] | None:
        """One basis argument: candidate index, explicit vector, or None for auto."""
        index = int(getattr(self, f"{name}_index"))
        if index >= 0:
            return index
        vector = getattr(self, f"{name}_rc")
        return list(vector) if vector else None

    # Rendering

    def _render(self, frame) -> bytes:
        """Colormap a 2D array into PNG bytes, or empty bytes when absent."""
        if frame is None:
            return b""
        return rgb_to_png_bytes(
            frame_to_rgb(
                np.asarray(frame, dtype=float),
                cmap=self.cmap,
                log_scale=bool(self.log_scale),
            )
        )

    def _render_template(self) -> None:
        """Render the template cropped to the disk, which is tiny on a large detector."""
        template = self._bragg.template
        if template is None:
            self.template_png = b""
            return

        frame = np.asarray(template, dtype=float)
        rows, cols = frame.shape
        centre = (rows // 2, cols // 2)

        strong = np.argwhere(frame > 0.25 * frame.max())
        reach = 8.0 if strong.size == 0 else np.abs(strong - np.array(centre)).max()
        half = int(max(12, min(min(rows, cols) // 2, round(3 * reach))))

        crop = frame[
            centre[0] - half : centre[0] + half + 1,
            centre[1] - half : centre[1] + half + 1,
        ]
        with self.hold_sync():
            self.template_shape = [int(crop.shape[0]), int(crop.shape[1])]
            self.template_png = self._render(crop)

    def _update_bvm(self) -> None:
        bvm = self._bragg.compute_bvm(float(self.bvm_sampling))
        self.bvm_png = self._render(bvm.array)

    def _render_fit(self) -> None:
        with self.hold_sync():
            self.mask_weight_png = self._render(self._bragg.mask_weight)
            self.fit_error_png = self._render(self._bragg.fit_error)

    # Observers and messages

    def _on_template_change(self, _change) -> None:
        try:
            self._build_template()
        except Exception as exc:
            self.status = str(exc)
            return
        with self.hold_sync():
            self.status = ""
            self._render_template()
        self._request_live()

    def _on_live_change(self, _change) -> None:
        self._request_live()

    def _on_navigator_change(self, _change) -> None:
        self._render_navigator()

    def _on_method_change(self, change) -> None:
        with self.hold_sync():
            for name, value in METHOD_DEFAULTS[change["new"]].items():
                setattr(self, name, value)
            self.preview_peaks = ""
        self._request_live()

    def _on_basis_change(self, _change) -> None:
        if getattr(self, "_resolving_basis", False):
            return
        self._choose_basis()

    def _on_display_change(self, _change) -> None:
        self._request_live()
        with self.hold_sync():
            self._render_template()
            self._render_navigator()
            if self._bragg.bvm is not None:
                self.bvm_png = self._render(self._bragg.bvm.array)
            if self._bragg.g1_array is not None:
                self._render_fit()

    def _on_panel_message(
        self, _widget: Any, content: dict[str, Any], _buffers: list[Any]
    ) -> None:
        """Panel actions. Errors land in status instead of the kernel."""
        action = content.get("type")
        try:
            if action == "preview_detect":
                self.detect(positions=self._preview_positions())
            elif action == "run_detect":
                self.detect()
            elif action == "compute_bvm":
                self._update_bvm()
                self._choose_basis()
            elif action == "run_fit":
                self.fit()
            elif action == "mark":
                self.mark(float(content["row"]), float(content["col"]))
            elif action == "teach":
                self.teach()
            elif action == "revert_teaching":
                self.revert_teaching()
            elif action == "save_model":
                self.save_model(content.get("path") or None)
        except Exception as exc:
            if action in ("mark", "teach", "revert_teaching", "save_model"):
                self.live_status = str(exc)
            elif not self.status:
                self.status = str(exc)

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        preview = self.bvm_png or self.pattern_png or self.template_png
        if not preview:
            return None
        return base64.b64encode(bytes(preview)).decode("ascii")
