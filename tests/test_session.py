import numpy as np
import pandas as pd
import pytest
from molcore import Acquisition, ArraySource, Dataset


def session():
    from moltrack.session import ProjectSession

    model = Dataset()
    model.add_acquisition(
        Acquisition(ArraySource(np.zeros((4, 8, 8))), id="camera")
    )
    result = ProjectSession(model)
    result.bind("experiment", "donor", "camera", "image")
    return result


def rows(**extra):
    return pd.DataFrame(
        dict(
            frame=[0, 1],
            x=[2.0, 3.0],
            y=[4.0, 5.0],
            photons=[10.0, 20.0],
            custom=["a", "b"],
            **extra,
        )
    )


def test_detached_queries_preserve_columns_and_ids():
    s = session()
    ids = s.replace_localizations("experiment", "donor", rows())
    data = s.query_localizations("experiment", "donor")
    assert data.id.tolist() == ids.tolist()
    assert data.custom.tolist() == ["a", "b"]
    data.loc[0, "photons"] = 999
    data.loc[0, "custom"] = "changed"
    assert s.query_localizations("experiment", "donor").photons.tolist() == [
        10,
        20,
    ]
    assert s.query_localizations("experiment", "donor").custom.tolist() == [
        "a",
        "b",
    ]
    assert s.datasets() == ("experiment",)
    assert s.channels("experiment") == ("donor",)
    assert s.partition("experiment", "donor") == ("camera", "image")


def test_preview_replacement_retains_other_frame_ids():
    s = session()
    ids = s.replace_localizations("experiment", "donor", rows())
    s.replace_localizations(
        "experiment", "donor", rows().iloc[:1], frame_index=0
    )
    data = s.query_localizations("experiment", "donor")
    assert data.loc[data.frame == 1, "id"].item() == ids[1]
    assert data.loc[data.frame == 0, "id"].item() != ids[0]
    s.replace_localizations(
        "experiment", "donor", rows().iloc[:0], frame_index=0
    )
    assert s.query_localizations("experiment", "donor").id.tolist() == [ids[1]]


def test_independent_tracks_do_not_merge_coincident_detections():
    s = session()
    detections = s.replace_localizations("experiment", "donor", rows())
    imported = rows(particle=[7, 7])
    imported["photons"] = [100, 200]
    track_ids = s.import_tracks("experiment", "donor", imported)
    data = s.query_tracks("experiment", "donor")
    assert data.particle.tolist() == [7, 7]
    assert data.track_id.unique().tolist() == track_ids.tolist()
    assert not set(data.id) & set(detections)
    assert data.photons.tolist() == [100, 200]
    assert (
        s.query_localizations("experiment", "donor").id.tolist()
        == detections.tolist()
    )
    data.loc[0, "photons"] = 900
    assert s.query_tracks("experiment", "donor").photons.tolist() == [100, 200]
    s.replace_localizations("experiment", "donor", rows().iloc[:0])
    assert s.query_tracks("experiment", "donor").photons.tolist() == [100, 200]


def test_candidate_ids_survive_fitting_and_failed_replacement_is_atomic():
    s = session()
    ids = s.replace_localizations("experiment", "donor", rows())
    fitted = rows()
    fitted["id"] = ids
    fitted["x"] += 0.25
    assert (
        s.replace_localizations("experiment", "donor", fitted).tolist()
        == ids.tolist()
    )
    before = s.query_localizations("experiment", "donor")
    fitted.loc[0, "x"] = np.nan
    with pytest.raises(ValueError):
        s.replace_localizations("experiment", "donor", fitted)
    pd.testing.assert_frame_equal(
        before, s.query_localizations("experiment", "donor")
    )


def test_failed_track_import_keeps_previous_results():
    s = session()
    s.import_tracks("experiment", "donor", rows(particle=[7, 7]))
    before = s.query_tracks("experiment", "donor")
    invalid = rows(particle=[8, 8])
    invalid["frame"] = 0
    with pytest.raises(ValueError):
        s.import_tracks("experiment", "donor", invalid)
    pd.testing.assert_frame_equal(
        before, s.query_tracks("experiment", "donor")
    )


def test_channel_frames_and_napari_projections():
    from molcore import ChannelSpec

    from moltrack.model_views import (
        gui_records,
        localization_points,
        track_points,
    )
    from moltrack.session import ProjectSession

    model = Dataset()
    model.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((5, 8, 8))),
            id="alex",
            channel_specs={"D": ChannelSpec(frame_start=1, frame_step=2)},
        )
    )
    s = ProjectSession(model)
    s.bind("ALEX", "donor", "alex", "D")
    ids = s.replace_localizations("ALEX", "donor", rows())
    assert model.localizations.query(space="raw")["frame"].tolist() == [1, 3]
    data = s.query_localizations("ALEX", "donor")
    assert data.frame.tolist() == [0, 1]
    np.testing.assert_array_equal(
        localization_points(data), [[0, 4, 2], [1, 5, 3]]
    )
    assert gui_records(data).id.tolist() == ids.tolist()
    s.import_tracks("ALEX", "donor", rows(particle=[7, 7]))
    np.testing.assert_array_equal(
        track_points(s.query_tracks("ALEX", "donor"))[:, 1:],
        [[0, 4, 2], [1, 5, 3]],
    )
    s.replace_localizations("ALEX", "donor", rows().iloc[:0], frame_index=0)
    assert s.query_localizations("ALEX", "donor").frame.tolist() == [1]


def test_candidate_numeric_updates_preserve_membership_and_text_changes_fail():
    s = session()
    ids = s.replace_localizations("experiment", "donor", rows())
    track = s.model.tracks.add(ids)
    fitted = rows()
    fitted["id"] = ids
    fitted["photons"] = [31.0, 32.0]
    fitted["precision"] = [0.1, 0.2]
    s.replace_localizations("experiment", "donor", fitted)
    assert s.model.tracks.query(track_id=track)["id"].tolist() == ids.tolist()
    assert s.query_localizations("experiment", "donor").photons.tolist() == [
        31,
        32,
    ]
    before = s.query_localizations("experiment", "donor")
    fitted.loc[0, "custom"] = "changed"
    with pytest.raises(ValueError, match="nonnumeric"):
        s.replace_localizations("experiment", "donor", fitted)
    pd.testing.assert_frame_equal(
        before, s.query_localizations("experiment", "donor")
    )


def test_bind_independent_sources_and_reject_cross_origin_ids():
    s = session()
    s.model.add_acquisition(
        Acquisition(ArraySource(np.zeros((4, 8, 8))), id="other")
    )
    s.bind("experiment", "acceptor", "other", "image")
    s.replace_localizations("experiment", "donor", rows())
    ids = s.replace_localizations("experiment", "acceptor", rows())
    before = s.query_localizations("experiment", "donor")
    invalid = rows()
    invalid["id"] = ids
    with pytest.raises(ValueError):
        s.replace_localizations("experiment", "donor", invalid)
    pd.testing.assert_frame_equal(
        before, s.query_localizations("experiment", "donor")
    )
    assert s.channels("experiment") == ("donor", "acceptor")


def test_string_import_labels_keep_detection_track_groups_distinct():
    s = session()
    detection_rows = pd.concat([rows(), rows()], ignore_index=True)
    ids = s.replace_localizations("experiment", "donor", detection_rows)
    # Multi-digit IDs catch accidental truncation in a short Unicode column.
    for _ in range(9):
        temporary = s.model.tracks.add(ids[:2])
        s.model.tracks.delete(temporary)
    first = s.model.tracks.add(ids[:2])
    second = s.model.tracks.add(ids[2:])
    imported = s.import_tracks(
        "experiment", "donor", rows(particle=["e", "e"])
    )
    before = s.model.tracks.query(space="raw")
    data = s.query_tracks("experiment", "donor")
    assert data.particle.tolist() == [first, first, second, second, "e", "e"]
    assert [
        len(group) for _, group in data.groupby("particle", sort=False)
    ] == [2, 2, 2]
    assert data.track_id.unique().tolist() == [first, second, imported.item()]
    after = s.model.tracks.query(space="raw")
    for column in before:
        np.testing.assert_array_equal(before[column], after[column])
