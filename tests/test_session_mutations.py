import numpy as np
import pandas as pd
import pytest
from molcore import Acquisition, ArraySource, Dataset, Segmentation
from molcore.staging import state_token

from moltrack.session import ProjectSession


def make_session():
    model = Dataset()
    for name in ("a", "b"):
        model.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((5, 8, 8))),
                id=name,
                pixel_size=100,
                metadata={"exposure_s": 0.1},
            )
        )
    session = ProjectSession(model)
    session.bind("data", "donor", "a", "image")
    session.bind("data", "acceptor", "b", "image")
    return session


def rows():
    return pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "x": [2.0, 3.0, 4.0],
            "y": [2.0, 3.0, 4.0],
            "photons": [10.0, 20.0, 30.0],
        }
    )


def test_copy_detections_allocates_fresh_ids_and_preserves_measurements():
    s = make_session()
    old = s.replace_localizations("data", "donor", rows())
    s.copy_rows(("data", "donor"), ("data", "acceptor"))
    copied = s.query_localizations("data", "acceptor")
    assert not set(copied.id) & set(old)
    assert copied.photons.tolist() == [10, 20, 30]
    s.update_measurements(copied.id, {"photons": [1, 2, 3]})
    assert s.query_localizations("data", "donor").photons.tolist() == [
        10,
        20,
        30,
    ]


def test_copy_tracks_handles_colliding_mixed_labels_without_merging_identity():
    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    original = s.model.tracks.add(ids, metadata={"particle": "1"})
    s.import_tracks("data", "donor", rows().assign(particle=1))
    source = s.query_tracks("data", "donor")
    assert source.track_id.nunique() == 2
    s.copy_rows(("data", "donor"), ("data", "acceptor"), tracks=True)
    copied = s.query_tracks("data", "acceptor")
    assert copied.track_id.nunique() == 2
    assert not set(copied.track_id) & set(source.track_id)
    assert not set(copied.id) & set(source.id)
    assert original in source.track_id.values


def test_track_retain_preserves_detection_rows_and_track_id():
    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    track = s.model.tracks.add(ids)
    s.retain_rows("data", "donor", ids[:2], tracks=True)
    assert s.query_tracks("data", "donor").track_id.unique().tolist() == [
        track
    ]
    assert s.query_tracks("data", "donor").id.tolist() == ids[:2].tolist()
    assert s.query_localizations("data", "donor").id.tolist() == ids.tolist()
    s.delete_rows("data", "donor", tracks=True)
    assert s.query_tracks("data", "donor").empty
    assert len(s.query_localizations("data", "donor")) == 3


def test_detection_delete_cascades_memberships_and_empty_retain_clears():
    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    s.model.tracks.add(ids)
    s.delete_rows("data", "donor", ids=ids[:1])
    assert s.query_tracks("data", "donor").id.tolist() == ids[1:].tolist()
    s.retain_rows("data", "donor", [])
    assert s.query_tracks("data", "donor").empty
    assert s.query_localizations("data", "donor").empty


def test_annotations_detach_and_invalidate_workers_and_masks():
    s = make_session()
    s.model.add_segmentation(
        Segmentation(
            id="mask",
            acquisition_id="a",
            channel="image",
            frame=0,
            labels=np.ones((8, 8), dtype=np.int32),
        )
    )
    original = {
        "data": [np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]])],
        "properties": {"name": ["cell"]},
        "mask_ids": ["mask"],
    }
    s.set_annotations("segmentations", original)
    before = state_token(s.model)
    edited = dict(original)
    edited.pop("mask_ids")
    s.set_annotations("segmentations", edited)
    original["properties"]["name"][0] = "changed"
    assert s.annotations["segmentations"]["properties"]["name"] == ["cell"]
    assert state_token(s.model) != before
    assert "mask" not in s.model.segmentations


def test_invalid_copy_is_atomic():
    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    s.replace_localizations("data", "acceptor", rows())
    s.model.add_acquisition(
        Acquisition(ArraySource(np.zeros((1, 8, 8))), id="short")
    )
    s.bind("data", "acceptor", "short", "image")
    before = state_token(s.model)
    with pytest.raises(ValueError):
        s.copy_rows(("data", "donor"), ("data", "acceptor"))
    assert state_token(s.model) == before
    assert s.query_localizations("data", "donor").id.tolist() == ids.tolist()


def test_persistent_analysis_cache_recomputes_after_calibration_and_is_detached():
    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    s.model.tracks.add(ids)
    first = s.analyze_tracks(fit_lags=2, rolling_window=3)
    assert first.speed.iloc[1] == pytest.approx(np.sqrt(2))
    first.loc[:, "speed"] = 999
    assert s.analyze_tracks(fit_lags=2, rolling_window=3).speed.iloc[1] != 999
    s.update_metadata("data", pixel_size=200)
    assert s.analyze_tracks(fit_lags=2, rolling_window=3).speed.iloc[
        1
    ] == pytest.approx(2 * np.sqrt(2))


def test_annotation_edit_rejects_prepared_analysis_and_processing():
    from molcore import StaleResultError

    from moltrack.model_analysis import (
        compute_analysis,
        prepare_analysis,
        publish_analysis,
    )

    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    s.model.tracks.add(ids)
    prepared = prepare_analysis(s, fit_lags=2)
    pending = compute_analysis(prepared)
    s.set_annotations("cells", {"data": [], "properties": {"name": []}})
    with pytest.raises(StaleResultError):
        publish_analysis(s, pending)


def test_special_segmentation_image_excluded_from_ordinary_processing():
    from moltrack.model_processing import selected_partitions

    s = make_session()
    s.model.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((1, 8, 8))),
            id="seg",
            metadata={"role": "segmentation_image"},
        )
    )
    s.bind("Segmentation Image", "Segmentation Image", "seg", "image")
    assert s.datasets() == ("data", "Segmentation Image")
    assert selected_partitions(s) == (("data", "donor"), ("data", "acceptor"))


def test_copy_display_positions_into_transformed_target_without_double_correction():
    from molcore import Calibration

    s = make_session()
    s.replace_localizations("data", "donor", rows())
    for acquisition, shift in [("a", 2.0), ("b", 5.0)]:
        s.model.transforms.add_calibration(
            Calibration(
                acquisition,
                np.array(
                    [[1.0, 0.0, shift], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
            )
        )
        s.model.transforms.select(acquisition, "image", acquisition)
    s.copy_rows(("data", "donor"), ("data", "acceptor"))
    np.testing.assert_allclose(
        s.query_localizations("data", "acceptor").x,
        s.query_localizations("data", "donor").x,
    )
    np.testing.assert_allclose(
        s.query_localizations("data", "acceptor", space="raw").x,
        [-1.0, 0.0, 1.0],
    )


def test_calibration_and_shape_edits_clear_stale_published_statistics():
    from moltrack.model_analysis import (
        compute_analysis,
        prepare_analysis,
        publish_analysis,
    )

    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    s.model.tracks.add(ids)
    publish_analysis(s, compute_analysis(prepare_analysis(s, fit_lags=2)))
    assert s.query_tracks("data", "donor").speed.iloc[1] > 0
    s.update_metadata("data", pixel_size=200)
    assert s.query_tracks("data", "donor").speed.isna().all()
    publish_analysis(s, compute_analysis(prepare_analysis(s, fit_lags=2)))
    s.set_annotations("cells", {"data": [], "properties": {}})
    assert s.query_tracks("data", "donor").speed.isna().all()


def test_repeated_track_copy_cleans_old_import_rows_without_losing_detections():
    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    s.model.tracks.add(ids)
    s.replace_localizations("data", "acceptor", rows())
    s.copy_rows(("data", "donor"), ("data", "acceptor"), tracks=True)
    count = len(s.model.localizations)
    s.copy_rows(("data", "donor"), ("data", "acceptor"), tracks=True)
    assert len(s.model.localizations) == count
    assert len(s.query_localizations("data", "acceptor")) == 3


def test_application_provenance_roundtrips_and_cascades_without_native_run_collision(
    tmp_path,
):
    from molcore.storage import open_dataset, save_dataset

    from moltrack import model_processing as proc

    s = make_session()
    for ch in ("donor", "acceptor"):
        s.replace_localizations("data", ch, rows())
    native = {"acquisitions": ["a"], "settings": {"native": True}}
    s.model.processing.settings["photometry"] = {"native-run": native}
    request = proc.ProcessingRequest(
        "photometry",
        s.selected_partitions(),
        proc.PhotometrySettings(spot_size=3),
    )
    proc.publish(s, proc.compute(proc.prepare(s, request)))
    assert s.model.processing.settings["photometry"] == {"native-run": native}
    assert (
        s.model.processing.settings["moltrack_operations"]["a"]["image"][
            "photometry"
        ]["settings"]["spot_size"]
        == 3
    )
    path = tmp_path / "native.h5"
    save_dataset(s.model, path)
    restored = open_dataset(path)
    assert restored.processing.settings == s.model.processing.settings
    restored.remove_acquisition("a", cascade=True)
    assert "a" not in restored.processing.settings["moltrack_operations"]
    assert "b" in restored.processing.settings["moltrack_operations"]
    assert restored.processing.settings["photometry"] == {}


@pytest.mark.parametrize("operation", ["analysis", "merge", "cells"])
def test_cancel_after_computation_preserves_all_native_state(operation):
    from concurrent.futures import CancelledError
    from threading import Event

    s = make_session()
    ids = s.replace_localizations("data", "donor", rows())
    s.model.tracks.add(ids)
    s.model.localizations.update_measurements(ids, speed=[0.0, 7.0, 8.0])
    s.model.add_segmentation(
        Segmentation(
            id="keep-mask",
            acquisition_id="a",
            channel="image",
            frame=0,
            labels=np.ones((8, 8), dtype=np.int32),
        )
    )
    event = Event()
    if operation == "analysis":
        from moltrack.model_analysis import (
            compute_analysis,
            prepare_analysis,
            publish_analysis,
        )

        prepared = prepare_analysis(s, fit_lags=2, cancel_event=event)
        pending = compute_analysis(prepared)

        def publish():
            return publish_analysis(s, pending)
    elif operation == "merge":
        from moltrack.model_edits import (
            compute_merge,
            prepare_merge,
            publish_merge,
        )

        prepared = prepare_merge(s, "data", 1.0, cancel_event=event)
        pending = compute_merge(prepared)

        def publish():
            return publish_merge(s, pending)
    else:
        from moltrack.model_cells import (
            finish_cells,
            prepare_cells,
            validate_cells,
        )

        prepared = prepare_cells(s, {"data": []}, cancel_event=event)
        pending = finish_cells(prepared, {"fitted": True})

        def publish():
            return validate_cells(s, pending)

    before = state_token(s.model)
    event.set()
    with pytest.raises(CancelledError):
        publish()
    assert state_token(s.model) == before
    assert s.query_tracks("data", "donor").speed.tolist() == [0.0, 7.0, 8.0]
    assert "keep-mask" in s.model.segmentations


@pytest.mark.parametrize("operation", ["analysis", "merge", "cells"])
def test_cancel_before_computation_skips_work(operation):
    from concurrent.futures import CancelledError
    from threading import Event

    event = Event()
    event.set()
    s = make_session()
    if operation == "analysis":
        from moltrack.model_analysis import compute_analysis, prepare_analysis

        prepared = prepare_analysis(s, cancel_event=event)

        def compute():
            return compute_analysis(prepared)
    elif operation == "merge":
        from moltrack.model_edits import compute_merge, prepare_merge

        prepared = prepare_merge(s, "data", 1.0, cancel_event=event)

        def compute():
            return compute_merge(prepared)
    else:
        from moltrack.model_cells import finish_cells, prepare_cells

        prepared = prepare_cells(s, {"data": []}, cancel_event=event)

        def compute():
            return finish_cells(prepared, {"fitted": True})

    before = state_token(s.model)
    with pytest.raises(CancelledError):
        compute()
    assert state_token(s.model) == before
