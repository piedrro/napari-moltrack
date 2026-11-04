import threading
from concurrent.futures import CancelledError

import numpy as np
import pandas as pd
import pytest
from molcore import Acquisition, ArraySource, Dataset, StaleResultError

from moltrack.session import ProjectSession


def make_session():
    model = Dataset()
    model.add_acquisition(
        Acquisition(ArraySource(np.zeros((2, 15, 15))), id="camera")
    )
    session = ProjectSession(model)
    session.bind("a", "image", "camera", "image")
    session.replace_localizations(
        "a",
        "image",
        pd.DataFrame({"frame": [0, 1], "x": [5.0, 6.0], "y": [5.0, 6.0]}),
    )
    return session


def request(frame=None):
    from moltrack.model_processing import DetectionSettings, ProcessingRequest

    return ProcessingRequest(
        "detect", (("a", "image"),), DetectionSettings(frame_index=frame)
    )


def test_empty_preview_clears_only_selected_frame(monkeypatch):
    from moltrack import model_processing as proc

    session = make_session()
    monkeypatch.setattr(
        proc,
        "_detect_partition",
        lambda *a, **k: pd.DataFrame(columns=["frame", "x", "y"]),
    )
    pending = proc.compute(proc.prepare(session, request(1)))
    proc.publish(session, pending)
    assert session.query_localizations("a", "image").frame.tolist() == [0]


def test_changed_rows_reject_pending(monkeypatch):
    from moltrack import model_processing as proc

    session = make_session()
    monkeypatch.setattr(
        proc,
        "_detect_partition",
        lambda *a, **k: pd.DataFrame(columns=["frame", "x", "y"]),
    )
    pending = proc.compute(proc.prepare(session, request()))
    session.replace_localizations(
        "a", "image", pd.DataFrame({"frame": [0], "x": [2.0], "y": [2.0]})
    )
    with pytest.raises(StaleResultError):
        proc.publish(session, pending)
    assert session.query_localizations("a", "image").x.tolist() == [2.0]


def test_cancelled_batch_preserves_model(monkeypatch):
    from moltrack import model_processing as proc

    session = make_session()
    event = threading.Event()
    monkeypatch.setattr(
        proc,
        "_detect_partition",
        lambda *a, **k: pd.DataFrame(columns=["frame", "x", "y"]),
    )
    pending = proc.compute(proc.prepare(session, request()))
    event.set()
    with pytest.raises(CancelledError):
        proc.publish(session, pending, cancel_event=event)
    assert len(session.query_localizations("a", "image")) == 2


def test_failed_second_partition_preserves_first(monkeypatch):
    from moltrack import model_processing as proc

    session = make_session()
    session.model.add_acquisition(
        Acquisition(ArraySource(np.zeros((2, 15, 15))), id="other")
    )
    session.bind("a", "other", "other", "image")

    def detect(session, dataset, channel, *args):
        if channel == "other":
            raise ValueError("failed")
        return pd.DataFrame(columns=["frame", "x", "y"])

    monkeypatch.setattr(proc, "_detect_partition", detect)
    req = proc.ProcessingRequest(
        "detect", (("a", "image"), ("a", "other")), proc.DetectionSettings()
    )
    with pytest.raises(ValueError, match="failed"):
        proc.compute(proc.prepare(session, req))
    assert len(session.query_localizations("a", "image")) == 2


def test_deleted_binding_rejects_pending(monkeypatch):
    from moltrack import model_processing as proc

    session = make_session()
    monkeypatch.setattr(
        proc,
        "_detect_partition",
        lambda *a, **k: pd.DataFrame(columns=["frame", "x", "y"]),
    )
    pending = proc.compute(proc.prepare(session, request()))
    session._partitions["a"].pop("image")
    with pytest.raises(StaleResultError):
        proc.publish(session, pending)


def test_gui_detection_captures_settings_and_publishes_on_result(monkeypatch):
    from types import SimpleNamespace

    from moltrack import model_processing as proc
    from moltrack.funcs.picasso_detect_utils import _picasso_detect_utils

    session = make_session()
    host = _picasso_detect_utils()
    host.session = session
    host.stop_event = threading.Event()

    def value(v):
        return SimpleNamespace(value=lambda: v)

    def check(v):
        return SimpleNamespace(isChecked=lambda: v)

    def combo(v):
        return SimpleNamespace(currentText=lambda: v)

    button = SimpleNamespace(setEnabled=lambda v: None)
    host.gui = SimpleNamespace(
        smlm_detect_mode=combo("Picasso"),
        smlm_fit_mode=combo("Picasso"),
        picasso_dataset=combo("a"),
        picasso_channel=combo("image"),
        picasso_frame_mode=combo("active"),
        picasso_minimise_ram=check(False),
        picasso_progressbar=SimpleNamespace(setValue=lambda v: None),
        picasso_detect=button,
        picasso_fit=button,
        picasso_detectfit=button,
        picasso_box_size=value(7),
        smlm_threshold=value(100),
        picasso_remove_overlapping=check(False),
        moltrack_kernel_size=SimpleNamespace(text=lambda: "3"),
    )
    host.viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(1,)))
    host.generate_roi = dict
    host.update_ui = lambda **kw: None
    host.moltrack_progress = lambda *a, **kw: None
    host._picasso_wrapper_finished = lambda: None
    host.threadpool = SimpleNamespace(start=lambda worker: None)
    monkeypatch.setattr(
        proc,
        "_detect_partition",
        lambda *a, **k: pd.DataFrame(columns=["frame", "x", "y"]),
    )
    host.init_picasso(detect=True)
    assert len(session.query_localizations("a", "image")) == 2
    host.worker.run()
    assert session.query_localizations("a", "image").frame.tolist() == [0]


def test_fit_retains_candidate_identity_and_metadata():
    from moltrack.model_processing import (
        DetectionSettings,
        ProcessingRequest,
        compute,
        prepare,
        publish,
    )

    y, x = np.indices((23, 23))
    image = 100 + 1000 * np.exp(-((x - 11.2) ** 2 + (y - 12.1) ** 2) / 2)
    model = Dataset()
    model.add_acquisition(
        Acquisition(
            ArraySource(np.stack([image, image]).astype(np.float32)),
            id="camera",
        )
    )
    session = ProjectSession(model)
    session.bind("a", "image", "camera", "image")
    ids = session.replace_localizations(
        "a",
        "image",
        pd.DataFrame(
            {"frame": [1], "x": [11.0], "y": [12.0], "label": ["molecule A"]}
        ),
    )
    req = ProcessingRequest(
        "detect",
        (("a", "image"),),
        DetectionSettings(detect=False, fit=True, frame_index=1),
    )
    publish(session, compute(prepare(session, req)))
    result = session.query_localizations("a", "image", space="raw")
    assert result.id.tolist() == ids.tolist()
    assert result.label.tolist() == ["molecule A"]
    np.testing.assert_allclose(
        result[["x", "y"]].to_numpy()[0], [11.2, 12.1], atol=0.1
    )


def test_changed_calibration_rejects_pending(monkeypatch):
    from moltrack import model_processing as proc

    session = make_session()
    monkeypatch.setattr(
        proc,
        "_detect_partition",
        lambda *a, **k: pd.DataFrame(columns=["frame", "x", "y"]),
    )
    pending = proc.compute(proc.prepare(session, request()))
    session.update_metadata("a", pixel_size=120.0)
    with pytest.raises(StaleResultError):
        proc.publish(session, pending)
    assert len(session.query_localizations("a", "image")) == 2


def test_real_detection_uses_persistent_source():
    from moltrack.model_processing import (
        DetectionSettings,
        ProcessingRequest,
        compute,
        prepare,
        publish,
    )

    y, x = np.indices((23, 23))
    image = 100 + 1000 * np.exp(-((x - 11.0) ** 2 + (y - 12.0) ** 2) / 2)
    model = Dataset()
    model.add_acquisition(
        Acquisition(
            ArraySource(np.stack([image, image]).astype(np.float32)),
            id="camera",
        )
    )
    session = ProjectSession(model)
    session.bind("a", "image", "camera", "image")
    req = ProcessingRequest(
        "detect",
        (("a", "image"),),
        DetectionSettings(threshold=50, frame_index=1),
    )
    publish(session, compute(prepare(session, req)))
    result = session.query_localizations("a", "image", space="raw")
    assert result.frame.tolist() == [1]
    np.testing.assert_allclose(
        result[["x", "y"]].to_numpy()[0], [11.0, 12.0], atol=0.1
    )


@pytest.mark.parametrize(
    "matrix,displacement,expected_frames",
    [
        (((1.0, 0.0, 5.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), None, [0, 1]),
        (((1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), None, [0, 1]),
        (
            ((1.0, 0.0, 5.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            [[4.0, 0.0], [0.0, 0.0]],
            [1],
        ),
    ],
)
def test_detection_roi_uses_corrected_candidates_and_retains_raw_positions(
    matrix, displacement, expected_frames
):
    from molcore import Calibration

    from moltrack import model_processing as proc

    y, x = np.indices((23, 23))
    image = 100 + 1000 * np.exp(-((x - 11.0) ** 2 + (y - 12.0) ** 2) / 2)
    model = Dataset()
    model.add_acquisition(
        Acquisition(
            ArraySource(np.stack([image, image]).astype(np.float32)),
            id="camera",
        )
    )
    session = ProjectSession(model)
    session.bind("a", "image", "camera", "image")
    model.transforms.add_calibration(
        Calibration("registration", np.asarray(matrix))
    )
    model.transforms.select("camera", "image", "registration")
    if displacement is not None:
        model.transforms.set_drift("camera", np.asarray(displacement))
    raw_pixels = session.image("a", "image", space="raw")[:].copy()
    roi = ((("a", "image"), ((10.0, 14.0), (15.0, 19.0))),)
    request = proc.ProcessingRequest(
        "detect",
        (("a", "image"),),
        proc.DetectionSettings(threshold=50.0, roi=roi),
    )
    proc.publish(session, proc.compute(proc.prepare(session, request)))
    result = session.query_localizations("a", "image", space="raw")
    assert result.frame.tolist() == expected_frames
    np.testing.assert_allclose(result.x, 11.0)
    np.testing.assert_allclose(result.y, 12.0)
    displayed = session.query_localizations("a", "image")
    assert displayed.x.between(14.0, 19.0, inclusive="left").all()
    assert displayed.y.between(10.0, 15.0, inclusive="left").all()
    np.testing.assert_array_equal(
        session.image("a", "image", space="raw")[:], raw_pixels
    )


def test_display_roi_keeps_half_open_bounds_and_raw_overlap_distances():
    from molcore import Calibration

    from moltrack import model_processing as proc

    session = make_session()
    session.model.transforms.add_calibration(
        Calibration("scaled", np.diag([10.0, 1.0, 1.0]))
    )
    session.model.transforms.select("camera", "image", "scaled")
    columns = {
        "frame": np.array([0, 0]),
        "x": np.array([5.0, 6.0]),
        "y": np.array([5.0, 5.0]),
        "n_id": np.array([1, 2]),
    }
    # Candidate 2 lies on the exclusive high edge in displayed space.
    roi = ((0.0, 50.0), (10.0, 60.0))
    result = proc._filter_detection_candidates(
        session.model, "camera", "image", columns, roi, False, 7
    )
    assert result["n_id"].tolist() == [1]
    np.testing.assert_array_equal(result["x"], [5.0])
    # The outside-ROI candidate still overlaps the first raw fitting box, despite
    # the calibration expanding their displayed separation from 1 to 10 pixels.
    result = proc._filter_detection_candidates(
        session.model, "camera", "image", columns, roi, True, 7
    )
    assert not len(result["frame"])
