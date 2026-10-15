"""ShowBragg's live view, the template-free detectors, and teaching the neural one."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from quantem.widget.showbragg import METHOD_DEFAULTS, ShowBragg
from quantem.widget.showbragg_live import DISK, NOT_DISK, LatestOnlyWorker, TeachMarks

pytest.importorskip("quantem.diffraction.spot_detection")

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.ml.spot_net import SpotNet

SPOTS = [(9.3, 9.6), (9.1, 22.4), (22.7, 9.2), (22.6, 22.5), (16.2, 15.8)]


def small_spot_dataset(scan=(4, 3), detector=(32, 32)) -> Dataset4dstem:
    """Small Gaussian spots at subpixel positions, shifted one pixel per scan column."""
    rr, cc = np.mgrid[0 : detector[0], 0 : detector[1]]
    frames = []
    for _ in range(scan[0]):
        for column in range(scan[1]):
            pattern = sum(
                200.0 * np.exp(-((rr - r) ** 2 + (cc - c - column) ** 2) / (2 * 1.2**2))
                for r, c in SPOTS
            )
            frames.append(pattern + 5.0)
    data = np.stack(frames).reshape(*scan, *detector).astype(np.float32)
    return Dataset4dstem.from_array(data, name="small spots")


def tiny_checkpoint(path) -> None:
    torch.manual_seed(0)
    model = SpotNet(base=4, levels=2)
    torch.save({"state_dict": model.state_dict(), "base": 4, "in_ch": 1, "levels": 2}, path)


@pytest.fixture
def widget() -> ShowBragg:
    return ShowBragg(small_spot_dataset())


def test_switching_method_resets_threshold_and_spacing(widget: ShowBragg) -> None:
    assert widget.detection_method == "correlation"
    for method in ("dog", "neural", "correlation"):
        widget.detection_method = method
        for name, value in METHOD_DEFAULTS[method].items():
            assert getattr(widget, name) == value


def test_live_view_detects_the_pattern_under_the_crosshair(widget: ShowBragg) -> None:
    widget.detection_method = "dog"
    widget.threshold = 8.0
    assert len(widget.navigator_png) > 0 and len(widget.pattern_png) > 0
    for column in (0, 2):
        widget.probe_position = [1, column]
        assert widget.spots_position == [1, column]
        spots = np.asarray(widget.spots)
        assert len(spots) == len(SPOTS)
        # the spots follow the data: each column shifts them one pixel
        for r, c in SPOTS:
            assert np.hypot(spots[:, 0] - r, spots[:, 1] - c - column).min() < 0.3
    assert widget.detect_ms > 0 and "spots at (1, 2)" in widget.live_status


def test_show_response_and_navigator_kind(widget: ShowBragg) -> None:
    widget.detection_method = "dog"
    pattern = bytes(widget.pattern_png)
    widget.show_response = True
    assert len(widget.probe_correlation_png) > 0 and bytes(widget.pattern_png) != pattern
    navigator = bytes(widget.navigator_png)
    widget.navigator_kind = "adf"
    assert bytes(widget.navigator_png) != navigator


def test_dog_preview_and_full_detection(widget: ShowBragg) -> None:
    widget.detection_method = "dog"
    widget.threshold = 8.0
    widget._on_panel_message(widget, {"type": "preview_detect"}, [])
    preview = json.loads(widget.preview_peaks)
    assert len(preview["peaks"]) == len(SPOTS)
    assert set(preview["counts"]) == {len(SPOTS)}

    peaks = widget.detect()
    assert peaks.shape == (4, 3)
    assert widget.detection_state == "done"
    assert widget.bragg.metadata["detect"]["method"] == "dog"
    assert widget.bragg.metadata["detect"]["neighbor_refine"] is False


def test_latest_only_worker_runs_synchronously_without_a_loop() -> None:
    results = []
    worker = LatestOnlyWorker(results.append)
    assert worker.synchronous
    worker.submit(lambda: 1)
    worker.submit(lambda: 1 / 0)
    assert results[0] == 1 and isinstance(results[1], ZeroDivisionError)


def test_marks_cycle_wrong_confirmed_cleared_and_missed() -> None:
    marks = TeachMarks()
    detections = np.array([[10.0, 10.0], [20.0, 20.0]])
    assert marks.click((0, 1), 10.4, 9.8, detections, reach=3) == "wrong"
    assert marks.at((0, 1)).tolist() == [[10.0, 10.0, NOT_DISK]]
    assert marks.click((0, 1), 10.2, 10.1, detections, reach=3) == "confirmed"
    assert marks.at((0, 1)).tolist() == [[10.0, 10.0, DISK]]
    assert marks.click((0, 1), 9.9, 10.0, detections, reach=3) == "removed"
    assert len(marks) == 0 and marks.positions() == []

    assert marks.click((2, 0), 5.0, 25.0, detections, reach=3) == "missed"
    assert marks.click((0, 1), 20.0, 20.5, detections, reach=3) == "wrong"
    assert marks.counts() == {"wrong": 1, "missed": 1, "confirmed": 0}
    assert marks.positions() == [(0, 1), (2, 0)]
    assert marks.to_trait()["2,0"] == [[5.0, 25.0, float(DISK), 0.0]]
    assert marks.click((2, 0), 5.5, 24.6, detections, reach=3) == "removed"


def test_widget_marks_persist_per_scan_position(widget: ShowBragg) -> None:
    widget.detection_method = "dog"
    widget.threshold = 8.0
    r, c = SPOTS[0]
    assert widget.mark(r, c) == "wrong"
    widget._on_panel_message(widget, {"type": "mark", "row": 3.0, "col": 28.0}, [])
    widget.probe_position = [2, 1]
    widget.mark(r, c + 1)
    assert widget.teach_marks.keys() == {"0,0", "2,1"}
    assert len(widget.teach_marks["0,0"]) == 2
    assert widget.teach_state["wrong"] == 2 and widget.teach_state["missed"] == 1


def test_teach_switches_to_the_taught_detector_and_saves_it(tmp_path) -> None:
    path = tmp_path / "tiny.pt"
    tiny_checkpoint(path)
    widget = ShowBragg(small_spot_dataset(), model=path, refine_model=False)
    widget.detection_method = "neural"
    widget.refine_centers = False
    widget.threshold = 0.0  # a random network fires everywhere: something to mark
    shown = np.asarray(widget.spots)
    assert len(shown) > 0
    widget.mark(*shown[0, :2])
    widget.mark(5.0, 5.0)
    widget._replay = None
    widget.teach()
    assert widget.neural_model == "taught" and widget.taught_model is not None
    assert widget.teach_state["taught"] is True and widget.teach_state["busy"] is False
    assert [m["id"] for m in widget.models][-1] == "taught"
    assert widget.models[-1]["icon"].startswith("data:image/png;base64,")
    assert widget.teach_state["summary"].startswith("Taught in")

    saved = widget.save_model(tmp_path / "taught.pt")
    reopened = ShowBragg(small_spot_dataset(), model=saved)
    assert reopened.neural_model == "custom"
    assert [m["id"] for m in reopened.models][-1] == "custom"

    widget.revert_teaching()
    assert widget.neural_model == "custom" and widget.taught_model is None
    assert widget.teach_marks == {}


def test_neural_method_uses_a_local_checkpoint(tmp_path) -> None:
    path = tmp_path / "spot_net.pt"
    tiny_checkpoint(path)
    widget = ShowBragg(small_spot_dataset(), model=path)
    widget.detection_method = "neural"
    widget.threshold = 0.0
    widget.refine_centers = False
    found = widget.detect(positions=[(0, 0)])
    assert found[0].numpy().shape[1] == 5  # [q_row, q_col, intensity, confidence, sigma]
    assert widget.bragg.peaks is None
    assert np.asarray(widget.spots).shape[1] == 3  # [row, col, confidence] for display


def test_center_step_toggles_reach_detect_disks(tmp_path) -> None:
    widget = ShowBragg(small_spot_dataset(), model=tmp_path / "d.pt", refine_model="centernet.pt")
    widget.detection_method = "neural"
    kwargs = widget._method_kwargs()
    assert kwargs["refine_centers"] == "centernet.pt" and kwargs["friedel_partner"] is True
    widget.refine_centers = False
    widget.friedel_partner = False
    kwargs = widget._method_kwargs()
    assert kwargs["refine_centers"] is False and kwargs["friedel_partner"] is False


def test_neural_without_weights_reports_in_the_status_line(widget: ShowBragg, monkeypatch) -> None:
    from emdatabase import data

    monkeypatch.delattr(data, "DE_SpotNet_Torch", raising=False)
    widget.detection_method = "neural"
    assert "DE_SpotNet_Torch" in widget.live_status and "model=" in widget.live_status


def test_frontend_updates_reach_the_traits(widget: ShowBragg) -> None:
    # ipywidgets routes frontend state through Widget._handle_msg; a panel handler
    # with that name would swallow every update from the browser.
    state = {"probe_position": [2, 1]}
    data = {"method": "update", "state": state, "buffer_paths": []}
    widget._handle_msg({"content": {"data": data}, "buffers": []})
    assert widget.probe_position == [2, 1] and widget.spots_position == [2, 1]
