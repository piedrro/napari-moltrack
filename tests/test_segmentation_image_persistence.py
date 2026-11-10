"""The standalone segmentation image is required project data."""

import json
import pickle
import zipfile

import numpy as np
import pytest
from molcore import Dataset, MissingSource
from test_model_import import job
from test_model_persistence import legacy

from moltrack import model_io
from moltrack.project_codec import AnnotationCodec
from moltrack.session import ProjectSession


def segmentation_session(*, averaged=False, role=True):
    session = ProjectSession()
    model_io.import_images(session, [job("None")])
    descriptor = dict(
        job("None"),
        dataset_name="Segmentation Image",
        channel_name="Segmentation Image",
        source_images=np.arange(32, dtype=np.uint16).reshape(2, 4, 4),
        frame_averaging=averaged,
    )
    if role:
        descriptor["role"] = "segmentation_image"
    model_io.import_images(session, [descriptor])
    binding = ("Segmentation Image", "Segmentation Image")
    session.annotations["segmentation_image_binding"] = binding
    session.translate_image(*binding, dx=1.0, dy=0.0)
    return session, binding


@pytest.mark.parametrize("save_images", [False, True])
@pytest.mark.parametrize("load_images", [False, True])
@pytest.mark.parametrize(
    "averaged,role", [(False, True), (True, True), (False, False)]
)
def test_segmentation_pixels_embedded_independently_of_image_options(
    tmp_path, save_images, load_images, averaged, role
):
    session, binding = segmentation_session(averaged=averaged, role=role)
    acquisition_id = session.partition(*binding)[0]
    raw = session.model.acquisitions[acquisition_id].source[:].copy()
    corrected = session.image(*binding)[:].copy()
    path = tmp_path / "segmentation.moltrack.h5"
    model_io.save_project(session, path, include_images=save_images)
    # Native readers must see actual embedded pixels, not just an app annotation.
    with Dataset.open(path) as native:
        np.testing.assert_array_equal(
            native.acquisitions[acquisition_id].source[:], raw
        )
        ordinary = native.acquisitions[session.partition("test", "image")[0]]
        assert isinstance(ordinary.source, MissingSource) == (not save_images)
    # Saving must leave the live ordinary and segmentation sources available.
    assert session.image_available("test", "image")
    np.testing.assert_array_equal(session.image(*binding)[:], corrected)
    with model_io.load_project(path, include_images=load_images) as loaded:
        assert loaded.annotations["segmentation_image_binding"] == binding
        assert loaded.partition(*binding) == session.partition(*binding)
        assert loaded.image_available(*binding)
        np.testing.assert_array_equal(loaded.image(*binding)[:], corrected)
        assert loaded.image_available("test", "image") == (
            save_images and load_images
        )
        # Re-exporting an image-free import must retain its segmentation pixels.
        model_io.save_project(loaded, path, include_images=False)
        np.testing.assert_array_equal(loaded.image(*binding)[:], corrected)
    with Dataset.open(path) as native:
        np.testing.assert_array_equal(
            native.acquisitions[acquisition_id].source[:], raw
        )


@pytest.mark.parametrize("container", ["zip", "native"])
def test_image_free_import_retains_segmentation_from_older_containers(
    tmp_path, container
):
    session, binding = segmentation_session()
    pixels = session.image(*binding)[:].copy()
    payload = tmp_path / "native.h5"
    session.model.save(payload, image_policy="full")
    if container == "zip":
        path = tmp_path / "historical.moltrack"
        with zipfile.ZipFile(path, "w") as archive:
            codec = AnnotationCodec(archive)
            annotations = codec.encode(session.annotations)
            archive.write(payload, "model.h5")
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "format": "moltrack-project",
                        "version": 1,
                        "bindings": [
                            [d, c, *session.partition(d, c)]
                            for d in session.datasets()
                            for c in session.channels(d)
                        ],
                        "annotations": annotations,
                        "arrays": codec.members,
                    }
                ),
            )
    else:
        path = payload
    with model_io.load_project(path, include_images=False) as loaded:
        restored_binding = loaded.annotations["segmentation_image_binding"]
        assert loaded.image_available(*restored_binding)
        np.testing.assert_array_equal(
            loaded.image(*restored_binding)[:], pixels
        )
        ordinary_id = session.partition("test", "image")[0]
        assert isinstance(
            loaded.model.acquisitions[ordinary_id].source, MissingSource
        )


def test_legacy_segmentation_pixels_retained_when_ordinary_images_omitted(
    tmp_path,
):
    project = legacy()
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
    project["segmentation_image"] = {"data": pixels, "pixel_size": 110}
    path = tmp_path / "legacy.moltrack"
    path.write_bytes(pickle.dumps(project))
    with model_io.load_project(path, include_images=False) as loaded:
        binding = loaded.annotations["segmentation_image_binding"]
        assert loaded.image_available(*binding)
        np.testing.assert_array_equal(loaded.image(*binding)[:], pixels[None])
        assert not loaded.image_available("old", "cropped")


def test_omitting_stacks_never_reads_ordinary_pixels(tmp_path, monkeypatch):
    session, binding = segmentation_session()
    ordinary = session.model.acquisitions[
        session.partition("test", "image")[0]
    ]

    def forbidden(index):
        pytest.fail("Saving without movie stacks read ordinary image pixels")

    monkeypatch.setattr(
        type(ordinary.source),
        "_frame",
        lambda source, index: (
            forbidden(index)
            if source is ordinary.source
            else source.array[index]
        ),
    )
    path = tmp_path / "without-stacks.h5"
    model_io.save_project(session, path, include_images=False)
    with model_io.load_project(path, include_images=False) as loaded:
        np.testing.assert_array_equal(
            loaded.image(*binding)[:], session.image(*binding)[:]
        )


def test_unavailable_segmentation_pixels_remain_explicitly_unavailable(
    tmp_path,
):
    session, binding = segmentation_session()
    acquisition = session.model.acquisitions[session.partition(*binding)[0]]
    session.model.update_acquisition(
        acquisition.id,
        source=MissingSource(
            acquisition.source.shape, acquisition.source.dtype
        ),
    )
    path = tmp_path / "already-missing.h5"
    model_io.save_project(session, path, include_images=False)
    with model_io.load_project(path, include_images=False) as loaded:
        assert not loaded.image_available(*binding)
        assert loaded.image(*binding).shape == (2, 4, 4)
