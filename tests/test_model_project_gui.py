from copy import deepcopy

import numpy as np
import pytest
from molcore import (
    Acquisition,
    ArraySource,
    BactFitCell,
    Dataset,
    Segmentation,
)
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


def clear_import_fixture(widget, tmp_path, *, loaded):
    from moltrack.model_processing import (
        ProcessingRequest,
        RegistrationSettings,
        compute,
        prepare,
    )

    image = np.arange(128, dtype=np.float32).reshape(2, 8, 8) + 100.0
    image[:, 3, 3] = 1000.0
    descriptor = {
        "source_images": image,
        "pixel_size": 100.0,
        "exposure_time": 20.0,
        "multichannel_mode": "None",
        "channel_name": "image",
        "path": "ordinary.tif",
        "dataset_name": "ordinary",
    }
    widget.publish_import_images([descriptor], clear=False, concatenate=False)
    widget.publish_image_import(
        {
            "import_mode": "Segmentation Image",
            "jobs": [dict(descriptor, frame_averaging=True)],
        }
    )
    session = widget.session
    binding = session.annotations["segmentation_image_binding"]
    session.translate_image(*binding, dx=1.0, dy=0.0)
    session.annotations["notes"] = {"keep": "editable application annotation"}
    ordinary = session.partition("ordinary", "image")
    ids = session.replace_localizations(
        "ordinary",
        "image",
        {
            "frame": [0, 1],
            "x": [3.0, 3.0],
            "y": [3.0, 3.0],
        },
    )
    session.model.tracks.add(ids)
    for mask_id, partition in (
        ("old-mask", ordinary),
        ("seg-mask", session.partition(*binding)),
    ):
        session.model.add_segmentation(
            Segmentation(
                id=mask_id,
                acquisition_id=partition[0],
                channel=partition[1],
                frame=0,
                labels=np.ones((8, 8), dtype=np.int32),
            )
        )
    polygon = np.array([[1.0, 1.0], [1.0, 4.0], [4.0, 4.0], [4.0, 1.0]])
    widget.initialise_segLayer(
        [polygon], 100.0, model_mask_ids=["old-mask", "seg-mask"]
    )
    widget.initialise_cellLayer(
        [polygon], ["polygon"], {"name": ["cell"], "radius": [2.0]}
    )
    if loaded:
        path = tmp_path / "clear-source.moltrack"
        model_io.save_project(session, path)
        widget.import_moltrack_project(path=str(path))
        session = widget.session
    pending = compute(
        prepare(
            session,
            ProcessingRequest(
                "registration",
                (("ordinary", "image"),),
                RegistrationSettings(
                    ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                ),
            ),
        )
    )
    return descriptor, session, binding, polygon, pending


@pytest.mark.parametrize("loaded", [False, True])
def test_clear_import_preserves_segmentation_source_annotations_and_listeners(
    widget, tmp_path, loaded
):
    from pathlib import Path
    from weakref import ref

    from molcore import StaleResultError
    from molcore.staging import state_token

    from moltrack.model_processing import publish

    descriptor, session, binding, polygon, pending = clear_import_fixture(
        widget, tmp_path, loaded=loaded
    )
    model = session.model
    resource = (
        None
        if session._project_resources is None
        else Path(session._project_resources.name)
    )
    before_pixels = session.image(*binding)[:].copy()
    before_partition = session.partition(*binding)
    provenance = deepcopy(model.acquisitions[before_partition[0]].metadata)
    original_source_id = provenance["source_acquisition_id"]
    original_pixels = model.acquisitions[original_source_id].source[:].copy()
    old_source = session.partition("ordinary", "image")[0]
    old_layer = widget.segLayer
    old_cell_layer = widget.cellLayer
    queued_import = {
        "import_mode": "data",
        "jobs": [dict(descriptor, dataset_name="stale")],
        "publication": {
            "owner": ref(model),
            "expected": state_token(model),
            "clear": False,
            "concatenate": False,
        },
    }
    before_seg_token = session.model.transforms.revision
    widget.gui.import_clear.setChecked(True)
    widget.publish_image_import({"import_mode": "data", "jobs": [descriptor]})
    widget.import_data_finished()

    assert widget.session is session
    assert session.model is model
    assert session.partition(*binding) == before_partition
    assert model.acquisitions[before_partition[0]].metadata == provenance
    np.testing.assert_array_equal(
        model.acquisitions[original_source_id].source[:], original_pixels
    )
    assert old_source not in model.acquisitions
    assert session.query_localizations("ordinary", "image").empty
    assert session.query_tracks("ordinary", "image").empty
    assert "old-mask" not in model.segmentations
    assert "seg-mask" in model.segmentations
    assert session.annotations["segmentations"]["mask_ids"] == ["seg-mask"]
    assert session.annotations["notes"] == {
        "keep": "editable application annotation"
    }
    assert session.model.transforms.revision >= before_seg_token
    np.testing.assert_array_equal(session.image(*binding)[:], before_pixels)
    assert session.image(*binding)[0][3, 4] == 1000.0
    if resource is not None:
        assert resource.exists()
        assert session._project_resources.name == str(resource)
    with pytest.raises(StaleResultError):
        publish(session, pending)
    after_clear = state_token(model)
    widget.publish_image_import(queued_import)
    assert state_token(model) == after_clear
    assert "stale" not in session.datasets()
    widget.gui.cellpose_dataset.setCurrentText("Segmentation Image")
    prepared = widget._prepare_model_segmentation("active")
    assert prepared.bindings == (before_partition,)
    np.testing.assert_array_equal(
        prepared.session.image(*binding)[:], before_pixels
    )

    assert widget.segLayer is old_layer
    assert widget.cellLayer is old_cell_layer
    widget.segLayer.data = [polygon + 0.5]
    np.testing.assert_array_equal(
        session.annotations["segmentations"]["data"][0], polygon + 0.5
    )
    widget.cellLayer.properties = {
        "name": np.array(["changed"]),
        "radius": np.array([3.0]),
    }
    assert session.annotations["cells"]["properties"]["name"].tolist() == [
        "changed"
    ]


@pytest.mark.parametrize("failure", ["descriptor", "publication"])
def test_failed_clear_import_keeps_loaded_sources_annotations_and_layers(
    widget, tmp_path, monkeypatch, failure
):
    from pathlib import Path

    from molcore import Dataset
    from molcore.staging import state_token

    descriptor, session, binding, _, _ = clear_import_fixture(
        widget, tmp_path, loaded=True
    )
    before_token = state_token(session.model)
    annotations = deepcopy(session.annotations)
    before_pixels = session.image(*binding)[:].copy()
    layer = widget.segLayer
    resource = Path(session._project_resources.name)
    descriptor = dict(
        descriptor, path="broken", source_images=np.ones((2, 8, 8))
    )
    if failure == "descriptor":
        descriptor["source_images"] = np.ones((2, 8, 7))
        descriptor["multichannel_mode"] = "FRET"
    else:
        original_add = Dataset.add_acquisition

        def add(model, acquisition):
            if acquisition.metadata.get("path") == "broken":
                raise ValueError("failed import publication")
            return original_add(model, acquisition)

        monkeypatch.setattr(Dataset, "add_acquisition", add)
    with pytest.raises(ValueError):
        widget.publish_import_images(
            [descriptor], clear=True, concatenate=False
        )
    assert widget.session is session
    assert state_token(session.model) == before_token
    assert widget.segLayer is layer
    assert resource.exists()
    np.testing.assert_array_equal(session.image(*binding)[:], before_pixels)
    assert session.annotations["notes"] == annotations["notes"]
    assert (
        session.annotations["segmentations"]["mask_ids"]
        == annotations["segmentations"]["mask_ids"]
    )
    np.testing.assert_array_equal(
        session.annotations["segmentations"]["data"][0],
        annotations["segmentations"]["data"][0],
    )


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


def test_native_fits_are_editable_and_export_dialog_appends_compound_suffix(
    widget, tmp_path, monkeypatch
):
    from moltrack.funcs.export_utils import QFileDialog

    native = Dataset()
    native.add_acquisition(
        Acquisition(ArraySource(np.zeros((2, 8, 8))), id="uncalibrated")
    )
    native.add_bactfit_cell(
        BactFitCell(
            "native-fit",
            [(1, 1), (4, 1), (4, 3), (1, 3)],
            [(1, 2), (4, 2)],
            [(1, 2), (4, 2)],
            (0, 2),
            1,
            frame=1,
            pixel_size=110,
        )
    )
    source = tmp_path / "native.h5"
    native.save(source, image_policy="full")
    widget.import_moltrack_project(path=str(source))
    assert widget.image_layer.data.shape == (2, 8, 8)
    assert widget.session.metadata("uncalibrated")["pixel_size"] is None
    assert widget.cellLayer.shape_type == ["polygon", "path"]
    cell = widget.get_cell("native-fit", bactfit=True)
    assert cell.frame_index == 1
    np.testing.assert_array_equal(cell.cell_midline.coords, [[1, 2], [4, 2]])
    widget.cellLayer.data = [
        shape + [0, 1, 0] for shape in widget.cellLayer.data
    ]

    def choose_path(parent, title, suggested, file_filter):
        assert suggested.endswith(".moltrack.h5")
        assert "*.moltrack.h5" in file_filter
        return str(tmp_path / "edited"), file_filter

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_path)
    widget.export_moltrack_project()
    with Dataset.open(tmp_path / "edited.moltrack.h5") as exported:
        fit = exported.bactfit_cells["native-fit"]
        assert fit.frame == 1
        assert fit.midline[0] == (1, 3)


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
    good = tmp_path / "good.moltrack.h5"
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
    payload = tmp_path / "model.h5"
    original.model.save(payload, image_policy="full")
    manifest = {
        "format": "moltrack-project",
        "version": 1,
        "bindings": [
            [dataset, channel, *original.partition(dataset, channel)]
            for dataset in original.datasets()
            for channel in original.channels(dataset)
        ],
    }
    with zipfile.ZipFile(bad, "w") as archive:
        codec = AnnotationCodec(archive)
        manifest["annotations"] = codec.encode(annotations)
        manifest["arrays"] = codec.members
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.write(payload, "model.h5")
    widget.import_moltrack_project(path=str(bad))
    assert widget.session is previous
    assert tuple(widget.viewer.layers) == previous_layers
    assert previous_payload.exists()
    np.testing.assert_array_equal(
        widget.session.image("test", "DA")[:], original.image("test", "DA")[:]
    )
