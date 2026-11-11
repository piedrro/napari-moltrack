from types import SimpleNamespace

import numpy as np
import pytest


def selector(text):
    return SimpleNamespace(currentText=lambda: text)


def fiducials(points):
    return [SimpleNamespace(x=x, y=y) for x, y in points]


def example(model):
    source = np.array([[4., 4.], [13., 5.], [5., 14.], [16., 16.]])
    rotation = np.array([[np.cos(.03), -np.sin(.03)], [np.sin(.03), np.cos(.03)]])
    matrix = np.eye(3)
    matrix[:2, :2] = (1.04 if model == "Similarity" else 1.) * rotation
    matrix[:2, 2] = [1., -1.]
    return source, source @ matrix[:2, :2].T + matrix[:2, 2], matrix


@pytest.mark.parametrize("model", ["Rigid", "Similarity", "Auto"])
def test_channel_model_is_used(model, monkeypatch):
    from moltrack.funcs.transform_utils import _transform_utils

    source, target, expected = example(model)
    tool = _transform_utils()
    tool.gui = SimpleNamespace(
        tform_compute_dataset=selector("round"),
        tform_compute_channel=selector("donor"),
        tform_compute_target_channel=selector("acceptor"),
        tform_compute_model=selector(model),
        save_tform=SimpleNamespace(isChecked=lambda: False),
    )
    tool.session = SimpleNamespace(ordinary_datasets=lambda: ["round"])
    tool.get_locs = lambda _, channel: fiducials(source if channel == "acceptor" else target)
    tool.compute_fret_transform_matrix()
    np.testing.assert_allclose(tool.transform_matrix, expected, atol=1e-5)

    from moltrack.model_processing import RegistrationSettings
    from test_model_processing import make_session
    from test_model_workflows import run

    session = make_session()
    raw = session.query_localizations("a", "image", space="raw")[["x", "y"]].to_numpy()
    run(session, "registration", RegistrationSettings(tuple(map(tuple, tool.transform_matrix))))
    registered = session.query_localizations("a", "image")[["x", "y"]].to_numpy()
    np.testing.assert_allclose(registered, raw @ expected[:2, :2].T + expected[:2, 2], atol=1e-5)

    messages = []
    monkeypatch.setattr("moltrack.funcs.transform_utils.show_info", messages.append)
    tool.gui.tform_compute_model = selector("Rigid")
    tool.get_locs = lambda _, channel: fiducials((source if channel == "acceptor" else target)[:1])
    tool.compute_fret_transform_matrix()
    assert tool.transform_matrix is None
    assert any("fiducial pairs" in message for message in messages)

