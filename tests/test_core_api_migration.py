"""Integration contracts for MolCore 0.2 adapters."""

import numpy as np
from molcore import (
    DatasetSnapshot,
    ExecutionContext,
    PartitionSelection,
    SerialExecutor,
)
from molcore.localisation import detect
from test_model_processing import make_session, request

from moltrack import backend, model_processing


def test_prepared_processing_owns_mutable_staging_and_shares_immutable_pixels():
    session = make_session()
    prepared = model_processing.prepare(session, request())
    prepared.session.model.update_metadata({"worker": "private"})
    assert "worker" not in session.model.metadata
    assert (
        prepared.session.model.acquisitions["camera"].source
        is session.model.acquisitions["camera"].source
    )


def test_custom_detection_obeys_typed_worker_boundary():
    session = make_session()
    snapshot = session.model.snapshot()
    assert isinstance(snapshot, DatasetSnapshot)
    selection = PartitionSelection("camera", "image")
    result = detect(
        snapshot,
        selection,
        backend=backend.MoltrackDetection(1),
        context=ExecutionContext(cpu=SerialExecutor()),
    )
    assert result.columns["frame"].dtype.kind in "iu"
    assert not result.columns["x"].flags.writeable


def test_ad_hoc_geometry_source_does_not_allocate_synthetic_movie():
    rows = np.rec.fromarrays([[99999], [1.0], [1.0]], names="frame,x,y")
    model, _ = backend._model(rows=rows, shape=(1000, 1000))
    source = model.acquisitions[backend.ACQUISITION].source
    assert source.shape == (100000, 1000, 1000)
    assert source.array.strides == (0, 0, 0)


def test_gui_projection_preserves_integer_nulls_and_large_values():
    import pandas as pd

    session = make_session()
    session.model.localizations.update_measurements(
        [1], quality=np.array([2**63 + 1], dtype=np.uint64)
    )
    rows = session.query_localizations("a", "image", space="raw")
    assert int(rows.quality.iloc[0]) == 2**63 + 1
    assert pd.isna(rows.quality.iloc[1])
    from moltrack.model_views import gui_records

    records = gui_records(rows)
    assert int(records["quality"][0]) == 2**63 + 1
    assert pd.isna(records["quality"][1])


def test_image_free_project_preserves_scientific_measurements(tmp_path):
    from moltrack.model_io import load_project, save_project

    session = make_session()
    session.model.observations.add(
        acquisition_id="camera",
        channel="image",
        frame=0,
        molecule_id="m",
        intensity=12.0,
        localization_id=1,
    )
    path = tmp_path / "measurement.h5"
    save_project(session, path, include_images=False)
    with load_project(path) as restored:
        assert restored.model.observations.query()["intensity"].tolist() == [
            12.0
        ]


def test_ad_hoc_movie_source_owns_pixels_during_detached_work():
    movie = np.ones((2, 3, 3))
    model, _ = backend._model(movie)
    movie[:] = 99
    np.testing.assert_array_equal(
        model.acquisitions[backend.ACQUISITION].source[0], 1
    )
