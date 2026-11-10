"""One persistent model carries an odd-frame ALEX workflow through a project."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd

from moltrack import backend, config
from moltrack import model_processing as proc
from moltrack.model_io import import_images, load_project, save_project
from moltrack.session import ProjectSession


def test_alex_import_fit_track_measure_edit_correct_project_chain(
    monkeypatch, tmp_path
):
    def forbidden_facade(*args, **kwargs):
        raise AssertionError(
            "normal model workflows cannot construct array facades"
        )

    monkeypatch.setattr(backend, "_model", forbidden_facade)
    monkeypatch.setattr(config, "NUM_CPUS", 6)
    movie = np.full((5, 17, 34), 100.0, dtype=np.float32)
    movie[:, 8, 8] = 1100.0
    session = ProjectSession()
    import_images(
        session,
        [
            {
                "dataset_name": "alex",
                "path": "camera.tif",
                "source_images": movie,
                "pixel_size": 117.0,
                "exposure_time": 20.0,
                "multichannel_mode": "ALEX",
                "channel_name": "image",
                "import_crop_mode": "None",
                "frame_averaging": False,
            }
        ],
    )
    model = session.model
    binding = session.partition("alex", "DD")
    acquisition = model.acquisitions[binding[0]]
    np.testing.assert_allclose(acquisition.timestamps, np.arange(5) * 0.020)
    assert session.image("alex", "DD").shape == (3, 17, 17)
    assert session.image("alex", "DA").shape == (2, 17, 17)
    cameras = []

    def detect(model, selection, settings, frame_range=None, **kwargs):
        acquisition_id, channel = selection.acquisition_id, selection.channel
        assert kwargs["context"].num_cpus == 6
        view = model.acquisitions[acquisition_id].channels[channel]
        frames = view.source_frames if frame_range != (0, 0) else []
        return proc.localisation.DetectionResult(
            acquisition_id=acquisition_id,
            channel=channel,
            settings=settings,
            source_token=0,
            frame_range=frame_range,
            columns={
                "frame": np.asarray(frames, dtype=np.int64),
                "x": np.full(len(frames), 8.0),
                "y": np.full(len(frames), 8.0),
                "net_gradient": np.full(len(frames), 1000.0),
                "n_id": np.arange(len(frames), dtype=np.uint32),
            },
        )

    def fit(model, detected, camera, **kwargs):
        assert kwargs["context"].num_cpus == 6
        cameras.append(asdict(camera))
        return SimpleNamespace(
            failed_fits=0,
            columns=dict(
                detected.columns,
                photons=np.full(len(detected.columns["frame"]), 1000.0),
                bg=np.zeros(len(detected.columns["frame"])),
                lpx=np.full(len(detected.columns["frame"]), 0.2),
                lpy=np.full(len(detected.columns["frame"]), 0.2),
            ),
        )

    monkeypatch.setattr(proc.localisation, "detect", detect)
    monkeypatch.setattr(proc.localisation, "fit", fit)

    def check_worker_count(operation):
        def checked(*args, **kwargs):
            assert kwargs["context"].num_cpus == 6
            return operation(*args, **kwargs)

        return checked

    monkeypatch.setattr(
        proc.tracking, "track", check_worker_count(proc.tracking.track)
    )
    monkeypatch.setattr(
        proc.photometry,
        "measure_localizations",
        check_worker_count(proc.photometry.measure_localizations),
    )
    partitions = (("alex", "DD"), ("alex", "DA"))

    def run(operation, settings):
        pending = proc.compute(
            proc.prepare(
                session,
                proc.ProcessingRequest(operation, partitions, settings),
            )
        )
        proc.publish(session, pending)
        assert session.model is model

    run("detect", proc.DetectionSettings())
    ids = {
        p: session.query_localizations(*p).id.to_numpy(copy=True)
        for p in partitions
    }
    run("detect", proc.DetectionSettings(detect=False, fit=True))
    assert len(cameras) == 2
    assert cameras[0] == asdict(proc.CameraSettings(100.0, 1.0, 1.0))
    run("tracking", proc.TrackingSettings(search_range=1.0, min_length=2))
    track_ids = {}
    for p in partitions:
        rows = session.query_tracks(*p)
        np.testing.assert_array_equal(rows.id, ids[p])
        track_ids[p] = rows.track_id.unique().tolist()
    assert set(track_ids[partitions[0]]).isdisjoint(track_ids[partitions[1]])
    run("photometry", proc.PhotometrySettings(tracks=True))
    for p in partitions:
        rows = session.query_localizations(*p)
        np.testing.assert_array_equal(rows.id, ids[p])
        np.testing.assert_allclose(rows.pixel_sum, 1900.0)
    # Filtering retains original members; editing writes measurements explicitly.
    session.retain_rows("alex", "DD", ids[("alex", "DD")][:2])
    kept_ids = ids[("alex", "DD")][:2]
    session.update_measurements(
        kept_ids, {"manual_score": np.array([4.0, 5.0])}
    )
    assert (
        session.query_tracks("alex", "DD").track_id.unique().tolist()
        == track_ids[("alex", "DD")]
    )
    raw = {
        p: session.query_localizations(*p, space="raw").copy()
        for p in partitions
    }
    run(
        "registration",
        proc.RegistrationSettings(
            ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
        ),
    )
    run(
        "registration",
        proc.RegistrationSettings(
            ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
        ),
    )
    for p in partitions:
        corrected = session.query_tracks(*p)
        np.testing.assert_allclose(corrected.x, raw[p].x + 2.0)
        np.testing.assert_allclose(corrected.y, raw[p].y + 1.0)
        pixels = session.image(*p)
        for row in corrected.itertuples():
            assert pixels[int(row.frame)][int(row.y), int(row.x)] == 1100.0
        pd.testing.assert_frame_equal(
            session.query_localizations(*p, space="raw"), raw[p]
        )
    session.set_annotations(
        "segmentations",
        {
            "data": [
                np.array([[1.0, 1.0], [1.0, 4.0], [4.0, 4.0], [4.0, 1.0]])
            ],
            "shape_types": ["polygon"],
            "properties": {"label": np.array(["ROI"])},
            "scale": [117.0, 117.0],
            "mask_ids": [],
        },
    )
    session.analyze_tracks()
    assert session._analysis_cache
    destination = tmp_path / "chain.moltrack"
    save_project(session, destination)
    with load_project(destination) as restored:
        assert restored.partition("alex", "DD") == binding
        assert restored._analysis_cache == {}
        assert restored.annotations["segmentations"]["properties"][
            "label"
        ].tolist() == ["ROI"]
        for p in partitions:
            pd.testing.assert_frame_equal(
                restored.query_tracks(*p),
                session.query_tracks(*p),
                check_like=True,
            )
            pd.testing.assert_frame_equal(
                restored.query_localizations(*p, space="raw"),
                raw[p],
                check_like=True,
            )
            np.testing.assert_array_equal(
                restored.image(*p)[:], session.image(*p)[:]
            )
            np.testing.assert_array_equal(
                restored.image(*p, space="raw")[:],
                session.image(*p, space="raw")[:],
            )
        np.testing.assert_array_equal(
            restored.image("alex", "AD", space="raw")[:], movie[::2, :, 17:]
        )
        np.testing.assert_allclose(
            restored.model.acquisitions[binding[0]].timestamps,
            np.arange(5) * 0.020,
        )
        assert (
            "registration"
            in restored.model.processing.settings["moltrack_operations"][
                binding[0]
            ][binding[1]]
        )
