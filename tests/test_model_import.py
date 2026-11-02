import numpy as np

from moltrack.session import ProjectSession


def job(mode="FRET", **extra):
    return dict(
        dataset_name="test",
        path="camera.tif",
        source_images=np.arange(5 * 4 * 8).reshape(5, 4, 8),
        multichannel_mode=mode,
        channel_name="image",
        import_crop_mode="None",
        import_limit="None",
        frame_averaging=False,
        pixel_size=117,
        exposure_time=20,
        **extra,
    )


def test_split_channels_share_source_and_roi():
    from moltrack.model_io import import_images

    s = ProjectSession()
    j = job()
    import_images(s, [j])
    a, _ = s.partition("test", "Donor")
    b, _ = s.partition("test", "Acceptor")
    assert a == b
    np.testing.assert_array_equal(
        s.image("test", "Acceptor")[1], j["source_images"][1, :, 4:]
    )
    assert s.model.acquisitions[a].source.shape == (5, 4, 8)


def test_odd_alex_frame_maps_and_units():
    from moltrack.model_io import import_images, import_rows

    s = ProjectSession()
    j = job("ALEX")
    import_images(s, [j])
    assert s.image("test", "DD").shape == (3, 4, 4)
    assert s.image("test", "DA").shape == (2, 4, 4)
    np.testing.assert_array_equal(
        s.image("test", "AD")[2], j["source_images"][4, :, 4:]
    )
    a, _ = s.partition("test", "DA")
    acq = s.model.acquisitions[a]
    assert acq.pixel_size == 117
    np.testing.assert_allclose(acq.timestamps, np.arange(5) * 0.020)
    import_rows(s, "test", "DA", {"frame": [1], "x": [2.0], "y": [1.0]})
    assert s.query_localizations("test", "DA").frame.tolist() == [1]
    assert s.query_localizations("test", "DA").x.tolist() == [2.0]
    assert s.model.localizations.query(
        acquisition_id=a, channel="DA", space="raw"
    ).to_columns()["frame"].tolist() == [3]


def test_multifile_channels_have_independent_sources():
    from moltrack.model_io import import_images

    s = ProjectSession()
    a = job("Multi File")
    b = job("Multi File")
    b["path"] = "other.tif"
    import_images(s, [a, b])
    assert (
        s.partition("test", "camera.tif")[0]
        != s.partition("test", "other.tif")[0]
    )
    assert len(s.model.acquisitions) == 2


def test_average_import_records_derived_source():
    from moltrack.model_io import import_images

    s = ProjectSession()
    j = job()
    j["frame_averaging"] = True
    import_images(s, [j])
    a, _ = s.partition("test", "Donor")
    acq = s.model.acquisitions[a]
    assert acq.metadata["derived_operation"] == "temporal_mean"
    assert acq.metadata["source_acquisition_id"] in s.model.acquisitions
    np.testing.assert_array_equal(
        s.image("test", "Donor")[0], j["source_images"][:, :, :4].mean(axis=0)
    )
    assert acq.metadata["source_frames"]["Donor"] == [0, 1, 2, 3, 4]


def test_explicit_concatenation_retains_boundary_provenance():
    from moltrack.model_io import import_images

    s = ProjectSession()
    a = job()
    b = job()
    b["dataset_name"] = "second"
    b["path"] = "second.tif"
    import_images(s, [a, b], concatenate=True)
    assert s.image("test", "Donor").shape == (10, 4, 4)
    acq = s.model.acquisitions[s.partition("test", "Donor")[0]]
    assert acq.metadata["derived_operation"] == "concatenate"
    assert acq.metadata["source_segments"][1]["frame_offset"] == 5
    assert acq.metadata["source_segments"][1]["path"] == "second.tif"


def test_gui_queries_are_detached_and_keep_identity():
    import pytest

    from moltrack.model_io import import_images, import_rows
    from moltrack.model_views import gui_query, legacy_snapshot

    s = ProjectSession()
    import_images(s, [job()])
    import_rows(
        s,
        "test",
        "Donor",
        {"frame": [0], "x": [1.0], "y": [2.0], "custom": [42.0]},
    )
    rows = gui_query(s, "test", "Donor")
    assert rows["dataset"].tolist() == ["test"]
    assert rows["custom"].tolist() == [42.0]
    rows["x"] = 99
    assert s.query_localizations("test", "Donor").x.tolist() == [1.0]
    with pytest.raises(TypeError):
        legacy_snapshot(s, "images")["test"]["pixel_size"] = 999


def test_display_projection_does_not_read_full_movie(monkeypatch):
    from moltrack.model_io import import_images
    from moltrack.model_views import legacy_snapshot

    s = ProjectSession()
    import_images(s, [job()])
    acq = s.model.acquisitions[s.partition("test", "Donor")[0]]
    calls = []
    original = acq.source._frame
    monkeypatch.setattr(
        acq.source, "_frame", lambda i: (calls.append(i), original(i))[1]
    )
    data = legacy_snapshot(s, "images")["test"]["images"]["Donor"]
    assert calls == []
    data[2].compute()
    assert calls == [2]


def test_worker_import_returns_descriptors_without_publication():
    from moltrack.funcs.import_utils import _import_utils

    host = _import_utils()
    host.process_compute_jobs = lambda jobs, progress_callback=None: jobs
    host.populate_import_dataset_dict = lambda jobs: (_ for _ in ()).throw(
        AssertionError("worker published")
    )
    result = host.import_data(import_jobs=[job()])
    assert result["jobs"][0]["dataset_name"] == "test"


def test_failed_clear_import_preserves_existing_session():
    from types import SimpleNamespace

    import pytest

    from moltrack.funcs.import_utils import _import_utils
    from moltrack.model_io import import_images

    host = _import_utils()
    host.session = ProjectSession()
    import_images(host.session, [job()])
    previous = host.session
    host.gui = SimpleNamespace(
        import_clear=SimpleNamespace(isChecked=lambda: True),
        import_concatenate=SimpleNamespace(isChecked=lambda: False),
    )
    broken = job()
    broken["source_images"] = np.zeros((5, 4, 7))
    with pytest.raises(ValueError):
        host.populate_import_dataset_dict([broken])
    assert host.session is previous
    assert host.session.channels("test") == ("Donor", "Acceptor")


def test_segmentation_import_preserves_crop_and_average():
    from moltrack.funcs.import_utils import _import_utils

    host = _import_utils()
    j = job("None")
    j.update(
        import_crop_mode="Right",
        frame_averaging=True,
        import_mode="Segmentation Image",
    )
    host.publish_image_import(
        {"jobs": [j], "import_mode": "Segmentation Image"}
    )
    np.testing.assert_array_equal(
        host.segmentation_image,
        j["source_images"][:, :, 4:].mean(axis=0)[None],
    )


def test_failed_nonclear_import_is_atomic():
    import pytest
    from molcore.staging import state_token

    from moltrack.model_io import import_images

    s = ProjectSession()
    import_images(s, [job()])
    model = s.model
    before = state_token(model)
    binding = s.partition("test", "Donor")
    first = job()
    first["dataset_name"] = "new"
    second = job()
    second["dataset_name"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        import_images(s, [first, second])
    assert s.model is model
    assert state_token(model) == before
    assert s.datasets() == ("test",)
    assert s.partition("test", "Donor") == binding


def test_failure_while_adding_acquisitions_is_atomic(monkeypatch):
    import pytest
    from molcore import Dataset
    from molcore.staging import state_token

    from moltrack.model_io import import_images

    s = ProjectSession()
    import_images(s, [job()])
    before = state_token(s.model)
    add = Dataset.add_acquisition

    def fail_second(model, acquisition):
        if acquisition.metadata.get("dataset_name") == "second":
            raise RuntimeError("rejected acquisition")
        return add(model, acquisition)

    monkeypatch.setattr(Dataset, "add_acquisition", fail_second)
    first = job()
    first["dataset_name"] = "new"
    second = job()
    second["dataset_name"] = "second"
    with pytest.raises(RuntimeError, match="rejected"):
        import_images(s, [first, second])
    assert state_token(s.model) == before
    assert s.datasets() == ("test",)


def test_odd_alex_concatenation_preserves_source_timing():
    from moltrack.model_io import import_images

    s = ProjectSession()
    a = job("ALEX")
    b = job("ALEX")
    b["path"] = "second.tif"
    import_images(s, [a, b], concatenate=True)
    for name, expected in [
        ("DD", [0, 0.04, 0.08, 0.10, 0.14, 0.18]),
        ("AD", [0, 0.04, 0.08, 0.10, 0.14, 0.18]),
        ("DA", [0.02, 0.06, 0.12, 0.16]),
        ("AA", [0.02, 0.06, 0.12, 0.16]),
    ]:
        acq = s.model.acquisitions[s.partition("test", name)[0]]
        np.testing.assert_allclose(acq.timestamps, expected)
        segments = acq.metadata["source_segments"]
        assert segments[0]["timestamp_offset_s"] == 0
        assert segments[1]["timestamp_offset_s"] == 0.1
        for segment in segments:
            original = s.model.acquisitions[segment["acquisition_id"]]
            source_times = original.timestamps[segment["source_frames"]]
            np.testing.assert_allclose(
                segment["source_timestamps_s"], source_times
            )
