from copy import deepcopy

import numpy as np
import pytest
from molcore import Segmentation
from test_model_persistence import populated

from moltrack import model_io


@pytest.fixture
def widget(qtbot, monkeypatch):
    import napari

    from moltrack._widget import MolTrack

    monkeypatch.setattr(MolTrack, "import_dev_data", lambda self: None)
    viewer = napari.Viewer(show=False)
    w = MolTrack(viewer)
    qtbot.addWidget(w)
    yield w
    w.session.close()
    viewer.dims.events.current_step.disconnect()
    viewer.close()


def test_actual_project_action_image_free_preserves_masks_and_reconnects_new_session(
    widget, tmp_path, monkeypatch
):
    import moltrack.funcs.events_utils as events

    original = populated()
    a, c = original.partition("test", "DA")
    original.model.add_segmentation(
        Segmentation(
            id="saved-mask",
            acquisition_id=a,
            channel=c,
            frame=1,
            labels=np.ones((4, 4), dtype=np.int32),
        )
    )
    polygon = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]])
    original.annotations["segmentations"] = {
        "data": [polygon],
        "shape_types": ["polygon"],
        "properties": {},
        "scale": [117.0, 117.0],
        "mask_ids": ["saved-mask"],
    }
    path = tmp_path / "gui.moltrack"
    model_io.save_project(original, path)
    widget.session = model_io.load_project(path)
    old = widget.session
    from pathlib import Path

    old_resource = Path(old._project_resources.name)
    widget.initialise_segLayer([polygon], 100.0)
    old_annotations = deepcopy(old.annotations)
    widget.gui.import_project_images.setChecked(False)

    def forbidden(*args, **kwargs):
        pytest.fail("image-free project load tried to compute autocontrast")

    monkeypatch.setattr(events, "autocontrast", forbidden)
    widget.import_moltrack_project(path=str(path))
    assert widget.session is not old
    assert not old_resource.exists()
    assert widget.session.partition("test", "DA") == original.partition(
        "test", "DA"
    )
    assert "saved-mask" in widget.session.model.segmentations
    assert widget.session.annotations["segmentations"]["mask_ids"] == [
        "saved-mask"
    ]
    np.testing.assert_array_equal(
        widget.cellLayer.data[0], original.annotations["cells"]["data"][0]
    )
    assert len(widget.session.query_tracks("test", "DA")) == 4
    widget.segLayer.data = [polygon + 0.5]
    assert "saved-mask" not in widget.session.model.segmentations
    np.testing.assert_array_equal(
        widget.session.annotations["segmentations"]["data"][0], polygon + 0.5
    )
    np.testing.assert_array_equal(
        old.annotations["segmentations"]["data"][0],
        old_annotations["segmentations"]["data"][0],
    )


@pytest.mark.parametrize("kind", ["invalid-file", "invalid-legacy"])
def test_failed_candidate_load_does_not_replace_live_project(
    widget, tmp_path, kind
):
    widget.session = populated()
    before = widget.session
    path = tmp_path / "broken.moltrack"
    if kind == "invalid-file":
        path.write_bytes(b"invalid")
    else:
        import pickle

        from test_model_persistence import legacy

        project = legacy()
        project["localisation_dict"]["old"]["cropped"]["localisations"][
            "frame"
        ][0] = 99
        path.write_bytes(pickle.dumps(project))
    widget.import_moltrack_project(path=str(path))
    assert widget.session is before
    assert len(widget.session.query_tracks("test", "DA")) == 4


def test_actual_project_save_captures_live_contrast_and_shape_annotations(
    widget, tmp_path
):
    from molcore.staging import state_token

    widget.session = populated()
    polygon = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]])
    widget.initialise_segLayer([polygon], 117.0)
    ids = widget.session.query_localizations("test", "DA").id.to_numpy()
    widget.session.update_measurements(ids, {"dstar": [10.0, 20.0]})
    token = state_token(widget.session.model)
    widget.contrast_dict = {
        "test": {"DA": {"contrast_limits": [2.0, 100.0], "gamma": 1.4}}
    }
    path = tmp_path / "saved.moltrack"
    widget.export_moltrack_project(path=str(path))
    assert state_token(widget.session.model) == token
    with model_io.load_project(path) as loaded:
        assert loaded.annotations["contrast"] == widget.contrast_dict
        np.testing.assert_array_equal(
            loaded.annotations["segmentations"]["data"][0], polygon
        )
        assert loaded.partition("test", "DA") == widget.session.partition(
            "test", "DA"
        )
        np.testing.assert_array_equal(
            loaded.query_localizations("test", "DA").dstar, [10.0, 20.0]
        )


@pytest.mark.parametrize(
    "malformation", ["scale-dimensions", "mixed-dimensions"]
)
def test_malformed_shape_candidate_preserves_live_session_payload_and_layers(
    widget, tmp_path, malformation
):
    import json
    import zipfile
    from pathlib import Path

    from moltrack.project_codec import AnnotationCodec

    original = populated()
    good = tmp_path / "good.moltrack"
    model_io.save_project(original, good)
    widget.session = model_io.load_project(good)
    polygon = original.annotations["cells"]["data"][0]
    widget.initialise_segLayer([polygon], 117.0)
    previous = widget.session
    previous_layers = tuple(widget.viewer.layers)
    previous_payload = Path(previous._project_resources.name)
    annotations = deepcopy(original.annotations)
    if malformation == "scale-dimensions":
        annotations["cells"]["scale"] = [117.0, 117.0, 117.0]
    else:
        annotations["cells"]["data"].append(
            np.column_stack([np.zeros(len(polygon)), polygon])
        )
        annotations["cells"]["shape_types"].append("polygon")
        annotations["cells"]["properties"] = {}
    bad = tmp_path / "bad.moltrack"
    with zipfile.ZipFile(good) as archive:
        payload = archive.read("model.h5")
        manifest = json.loads(archive.read("manifest.json"))
    with zipfile.ZipFile(bad, "w") as archive:
        codec = AnnotationCodec(archive)
        manifest["annotations"] = codec.encode(annotations)
        manifest["arrays"] = codec.members
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("model.h5", payload)
    widget.import_moltrack_project(path=str(bad))
    assert widget.session is previous
    assert tuple(widget.viewer.layers) == previous_layers
    assert previous_payload.exists()
    np.testing.assert_array_equal(
        widget.session.image("test", "DA")[:], original.image("test", "DA")[:]
    )
