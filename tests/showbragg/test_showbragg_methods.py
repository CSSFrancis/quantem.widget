"""ShowBragg with the template-free detectors: Difference-of-Gaussians and neural."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from quantem.widget.showbragg import METHOD_DEFAULTS, ShowBragg

pytest.importorskip("quantem.diffraction.spot_detection")

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.ml.spot_unet import SpotUNet

SPOTS = [(9.3, 9.6), (9.1, 22.4), (22.7, 9.2), (22.6, 22.5), (16.2, 15.8)]


def small_spot_dataset(scan=(4, 3), detector=(32, 32)) -> Dataset4dstem:
    """Small Gaussian spots at subpixel positions, identical at every scan position."""
    rr, cc = np.mgrid[0 : detector[0], 0 : detector[1]]
    pattern = sum(
        200.0 * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * 1.2**2)) for r, c in SPOTS
    )
    pattern = (pattern + 5.0).astype(np.float32)
    data = np.broadcast_to(pattern, (*scan, *detector)).copy()
    return Dataset4dstem.from_array(data, name="small spots")


@pytest.fixture
def widget() -> ShowBragg:
    return ShowBragg(small_spot_dataset())


def test_switching_method_resets_threshold_and_spacing(widget: ShowBragg) -> None:
    assert widget.detection_method == "correlation"
    for method in ("dog", "neural", "correlation"):
        widget.detection_method = method
        for name, value in METHOD_DEFAULTS[method].items():
            assert getattr(widget, name) == value


def test_dog_preview_and_full_detection(widget: ShowBragg) -> None:
    widget.detection_method = "dog"
    widget.threshold = 8.0
    assert len(widget.probe_correlation_png) > 0  # the band-pass map, not a correlation

    widget._handle_msg(widget, {"type": "preview_detect"}, [])
    preview = json.loads(widget.preview_peaks)
    assert len(preview["peaks"]) == len(SPOTS)
    assert set(preview["counts"]) == {len(SPOTS)}

    peaks = widget.detect()
    assert peaks.shape == (4, 3)
    assert widget.detection_state == "done"
    assert widget.bragg.metadata["detect"]["method"] == "dog"
    found = peaks[0, 0].numpy()
    for r, c in SPOTS:
        assert np.hypot(found[:, 0] - r, found[:, 1] - c).min() < 0.3


def test_neural_method_uses_a_local_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    path = tmp_path / "spot_unet.pt"
    model = SpotUNet(base=4, levels=2)
    torch.save({"state_dict": model.state_dict(), "base": 4, "in_ch": 1, "levels": 2}, path)

    widget = ShowBragg(small_spot_dataset(), model_path=path)
    widget.detection_method = "neural"
    assert widget.status == ""
    assert len(widget.probe_correlation_png) > 0  # the network's confidence map

    widget.threshold = 0.0  # a random network carries no meaning; exercise every stage
    found = widget.detect(positions=[(0, 0)])
    assert found[0].numpy().shape[1] == 3
    assert widget.bragg.peaks is None


def test_neural_without_weights_reports_in_status(widget: ShowBragg, monkeypatch) -> None:
    from emdatabase import data

    monkeypatch.delattr(data, "SpotUNetDiffraction", raising=False)
    widget.detection_method = "neural"
    assert "model_path" in widget.status
