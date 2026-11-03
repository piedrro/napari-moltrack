import json
import pickle
import zipfile

import numpy as np
import pytest
from bactfit.cell import Cell, CellList
from molcore import MissingSource
from shapely.geometry import LineString, Polygon
from test_model_import import job

from moltrack import model_io
from moltrack.session import ProjectSession


def populated():
    s = ProjectSession()
    model_io.import_images(s, [job("ALEX")])
    ids = s.replace_localizations(
        "test",
        "DA",
        {
            "frame": [0, 1],
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            "unusual": [7.0, 8.0],
        },
    )
    s.model.tracks.add(ids, metadata={"particle": "original"})
    s.import_tracks(
        "test",
        "DA",
        {
            "frame": [0, 1],
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            "particle": ["independent"] * 2,
        },
    )
    s.translate_image("test", "DA", dx=0.5, dy=-0.25)
    polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])
    cell = Cell(
        {
            "name": "known",
            "cell_polygon": polygon,
            "cell_midline": LineString([(1, 0), (1, 3)]),
        }
    )
    cell.data = {"extra": np.arange(3), "fit": {"radius": 2.5}}
    cells = CellList([cell])
    s.annotations = {
        "cells": {
            "data": [np.asarray(polygon.exterior.coords)],
            "shape_types": ["polygon"],
            "scale": [117.0, 117.0],
            "mask_ids": [],
            "properties": {
                "cell": np.array([cell], dtype=object),
                "parameters": np.array(
                    [{"radius": 2.0, "poly_params": np.array([1.0, 2.0])}],
                    dtype=object,
                ),
            },
        },
        "celllist": cells,
        "contrast": {
            "test": {"DA": {"contrast_limits": (1.0, 80.0), "gamma": 1.2}}
        },
    }
    return s


def test_container_preserves_native_identity_transforms_annotations_and_lazy_lifetime(
    tmp_path,
):
    original = populated()
    path = tmp_path / "project.moltrack"
    model_io.save_project(original, path)
    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "moltrack-project"
        assert manifest["version"] == 1
        assert "model.h5" in archive.namelist()
        assert all(
            not name.endswith(("pickle", "pkl")) for name in archive.namelist()
        )
    loaded = model_io.load_project(path)
    try:
        assert loaded._partitions == original._partitions
        assert set(loaded.model.acquisitions) == set(
            original.model.acquisitions
        )
        for query in ("query_localizations", "query_tracks"):
            a, b = (
                getattr(original, query)("test", "DA"),
                getattr(loaded, query)("test", "DA"),
            )
            assert list(a.id) == list(b.id)
            np.testing.assert_array_equal(a.x, b.x)
            np.testing.assert_array_equal(a.unusual, b.unusual)
        np.testing.assert_array_equal(
            loaded.image("test", "DA")[:], original.image("test", "DA")[:]
        )
        a = loaded.model.acquisitions[loaded.partition("test", "DA")[0]]
        np.testing.assert_array_equal(a.channels["DA"].source_frames, [1, 3])
        np.testing.assert_array_equal(a.timestamps, np.arange(5) * 0.02)
        cell = loaded.annotations["cells"]["properties"]["cell"][0]
        assert isinstance(cell, Cell)
        assert cell.cell_polygon.equals(
            original.annotations["celllist"].data[0].cell_polygon
        )
        np.testing.assert_array_equal(cell.data["extra"], np.arange(3))
        assert isinstance(loaded.annotations["celllist"], CellList)
        assert (
            loaded.annotations["contrast"] == original.annotations["contrast"]
        )
        resource_path = loaded._project_resources.name
    finally:
        loaded.close()
    from pathlib import Path

    assert not Path(resource_path).exists()


@pytest.mark.parametrize(
    "save_images,load_images", [(False, True), (True, False)]
)
def test_image_free_preserves_geometry_tables_and_explicit_unavailability(
    tmp_path, save_images, load_images
):
    s = populated()
    path = tmp_path / "without-images.moltrack"
    model_io.save_project(s, path, include_images=save_images)
    loaded = model_io.load_project(path, include_images=load_images)
    try:
        acq = loaded.model.acquisitions[loaded.partition("test", "DA")[0]]
        assert isinstance(acq.source, MissingSource)
        assert loaded.image("test", "DA").shape == (2, 4, 4)
        assert len(loaded.query_tracks("test", "DA")) == 4
        with pytest.raises(
            Exception, match="(?i)(not saved|unavailable|missing)"
        ):
            loaded.image("test", "DA")[0]
    finally:
        loaded.close()


def test_resave_mixed_missing_and_available_sources_keeps_geometry_and_pixels(
    tmp_path,
):
    from molcore import Acquisition

    s = populated()
    s.model.add_acquisition(
        Acquisition(
            MissingSource((4, 8, 9), "u2"),
            id="missing",
            metadata={"path": "absent.tif"},
        )
    )
    s.bind("absent", "image", "missing", "image")
    path = tmp_path / "mixed.moltrack"
    model_io.save_project(s, path, include_images=True)
    with model_io.load_project(path) as loaded:
        assert loaded.image("absent", "image").shape == (4, 8, 9)
        assert not loaded.image_available("absent", "image")
        assert loaded.image_available("test", "DA")
        np.testing.assert_array_equal(
            loaded.image("test", "DA")[:], s.image("test", "DA")[:]
        )
    # A wholly image-free project can also be re-saved with the usual checkbox.
    empty = ProjectSession()
    empty.model.add_acquisition(
        Acquisition(MissingSource((1, 3, 4), "f4"), id="table")
    )
    empty.bind("table", "image", "table", "image")
    model_io.save_project(empty, path)
    with model_io.load_project(path) as loaded:
        assert not loaded.image_available("table", "image")


def test_unsupported_annotation_and_native_failure_leave_destination_unchanged(
    tmp_path, monkeypatch
):
    s = populated()
    path = tmp_path / "atomic.moltrack"
    path.write_bytes(b"previous")
    s.annotations["bad"] = object()
    with pytest.raises(TypeError, match="Unsupported annotation"):
        model_io.save_project(s, path)
    assert path.read_bytes() == b"previous"
    del s.annotations["bad"]

    def fail(*args, **kwargs):
        raise RuntimeError("disk failure")

    monkeypatch.setattr(type(s.model), "save", fail)
    with pytest.raises(RuntimeError, match="disk failure"):
        model_io.save_project(s, path)
    assert path.read_bytes() == b"previous"


def test_unknown_version_and_unexpected_zip_members_are_rejected(tmp_path):
    path = tmp_path / "invalid.moltrack"
    for manifest, extra in [
        ({"format": "moltrack-project", "version": 99}, ""),
        ({"format": "moltrack-project", "version": 1}, "../escape"),
    ]:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("model.h5", b"invalid")
            if extra:
                archive.writestr(extra, b"invalid")
        with pytest.raises(ValueError):
            model_io.load_project(path)


@pytest.mark.parametrize(
    "change", ["vertices", "properties", "shape-type", "scale"]
)
def test_invalid_annotation_schema_cannot_publish_a_project(tmp_path, change):
    s = populated()
    value = s.annotations["cells"]
    if change == "vertices":
        value["data"] = [np.zeros((2, 5))]
    elif change == "properties":
        value["properties"]["cell"] = np.array([], dtype=object)
    elif change == "shape-type":
        value["shape_types"] = ["unknown"]
    else:
        value["scale"] = [0.0, 117.0]
    path = tmp_path / "existing.moltrack"
    path.write_bytes(b"previous")
    with pytest.raises(
        ValueError, match="(?i)(annotation|shape|scale|properties)"
    ):
        model_io.save_project(s, path)
    assert path.read_bytes() == b"previous"


def test_legacy_unsupported_cell_attribute_fails_before_candidate_adoption():
    project = legacy()
    project["cells"]["properties"]["cell"][0].unknown = object()
    with pytest.raises(TypeError, match="Unsupported annotation"):
        model_io.from_legacy(project)


def legacy():
    images = np.arange(3 * 6 * 7).reshape(3, 6, 7)
    rows = np.array(
        [(0, 2.0, 3.0), (1, 3.0, 4.0)],
        dtype=[("frame", "i4"), ("x", "f8"), ("y", "f8")],
    )
    tracks = np.array(
        [(0, 2.0, 3.0, 8), (1, 3.0, 4.0, 8)],
        dtype=rows.dtype.descr + [("particle", "i4")],
    )
    cell = Cell(
        {"name": "legacy", "cell_polygon": Polygon([(0, 0), (0, 3), (3, 3)])}
    )
    return {
        "dataset_dict": {
            "old": {
                "images": {"cropped": images},
                "path": "missing.tif",
                "pixel_size": 100.0,
                "exposure_time": 20.0,
                "drift": np.ones((3, 2)),
            }
        },
        "localisation_dict": {
            "old": {
                "cropped": {
                    "localisations": rows,
                    "image_shape": [3, 6, 7],
                    "fitted": True,
                }
            }
        },
        "tracking_dict": {"old": {"cropped": {"tracks": tracks}}},
        "cells": {
            "shapes": [np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0]])],
            "shape_types": ["polygon"],
            "properties": {"cell": np.array([cell], dtype=object)},
            "pixel_size": 100.0,
        },
    }


def test_legacy_corrected_baseline_independent_tracks_and_real_cells(tmp_path):
    project = legacy()
    path = tmp_path / "old.moltrack"
    path.write_bytes(pickle.dumps(project))
    s = model_io.load_project(path)
    try:
        np.testing.assert_array_equal(
            s.image("old", "cropped")[:],
            project["dataset_dict"]["old"]["images"]["cropped"],
        )
        assert s.query_localizations("old", "cropped").x.tolist() == [2.0, 3.0]
        assert s.query_tracks("old", "cropped").x.tolist() == [2.0, 3.0]
        assert not set(s.query_localizations("old", "cropped").id) & set(
            s.query_tracks("old", "cropped").id
        )
        assert isinstance(
            s.annotations["cells"]["properties"]["cell"][0], Cell
        )
        assert s.model.metadata["migration_report"]
    finally:
        s.close()


def test_legacy_image_free_uses_saved_geometry_and_reports_unrecoverable_bounds():
    project = legacy()
    del project["dataset_dict"]["old"]["images"]
    s = model_io.from_legacy(project)
    assert s.image("old", "cropped").shape == (3, 6, 7)
    assert len(s.query_tracks("old", "cropped")) == 2
    del project["localisation_dict"]["old"]["cropped"]["image_shape"]
    s = model_io.from_legacy(project)
    acq = s.model.acquisitions[s.partition("old", "cropped")[0]]
    assert acq.metadata["legacy_geometry_status"] == "inferred_table_bounds"
    assert any(
        "inferred" in line.lower()
        for line in s.model.metadata["migration_report"]
    )


@pytest.mark.parametrize(
    "mode,crop,channel,width,expected_width",
    [("None", "Right", "image", 9, 5), ("ALEX", "None", "DA", 8, 4)],
)
def test_legacy_header_geometry_respects_crop_limit_and_mean_without_rebinding_pixels(
    tmp_path, mode, crop, channel, width, expected_width
):
    import tifffile

    path = tmp_path / "original.tif"
    tifffile.imwrite(
        path, np.zeros((5, 6, width), dtype="u2"), photometric="minisblack"
    )
    project = {
        "dataset_dict": {
            "old": {
                "path": str(path),
                "multichannel_mode": mode,
                "import_crop_mode": crop,
                "import_limit": 5,
                "frame_averaging": True,
                "pixel_size": 100.0,
                "exposure_time": 20.0,
            }
        },
        "localisation_dict": {
            "old": {
                channel: {
                    "localisations": {"frame": [0], "x": [2.0], "y": [3.0]}
                }
            }
        },
    }
    s = model_io.from_legacy(project)
    assert s.image("old", channel).shape == (1, 6, expected_width)
    assert not s.image_available("old", channel)
    acq = s.model.acquisitions[s.partition("old", channel)[0]]
    assert acq.metadata["legacy_geometry_status"] == "source_header_only"
