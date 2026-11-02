"""Regression tests for the GUI-independent MolCore adapters."""

import threading
from concurrent.futures import CancelledError

import numpy as np
import pandas as pd
import pytest

from moltrack import backend


def test_transform_preserves_measurements_and_input():
    locs = np.rec.fromarrays(
        [[0, 2], [3.0, 4.0], [5.0, 6.0], [100.0, 200.0]],
        names="frame,x,y,photons",
    )
    original = locs.copy()
    result = backend.transform_locs(locs, [[1, 0, 2], [0, 1, -1]])
    np.testing.assert_array_equal(locs, original)
    np.testing.assert_array_equal(result.photons, original.photons)
    np.testing.assert_allclose(result.x, [5, 6])
    np.testing.assert_allclose(result.y, [4, 5])


def test_transform_movie_keeps_shape_and_dtype():
    movie = np.zeros((3, 7, 9), dtype=np.uint16)
    movie[:, 3, 2] = 100
    progress = []
    result = backend.transform_image(
        movie, [[1, 0, 1], [0, 1, 0]], progress.append
    )
    assert result.shape == movie.shape
    assert result.dtype == movie.dtype
    np.testing.assert_array_equal(result[:, 3, 3], [100] * 3)
    assert progress[-1] == 100


def test_tracking_honours_empty_frame_memory_and_keeps_columns():
    rows = pd.DataFrame(
        {
            "frame": [0, 2, 3],
            "x": [4.0, 4.1, 4.2],
            "y": [5.0] * 3,
            "photons": [100, 200, 300],
            "dataset": ["a"] * 3,
        }
    )
    linked = backend.link_locs(rows, 1, memory=1, min_length=3)
    assert len(linked) == 3
    assert linked.particle.nunique() == 1
    assert linked.photons.tolist() == rows.photons.tolist()
    assert backend.link_locs(rows, 1, memory=0, min_length=3).empty


def test_render_uses_requested_geometry_and_counts():
    locs = np.rec.fromarrays([[0], [3.0], [4.0]], names="frame,x,y")
    count, image = backend.render_locs(locs, (8, 10), oversampling=2)
    assert count == 1
    assert image.shape == (16, 20)
    assert image.sum() == pytest.approx(1)


def test_cancel_does_not_change_input():
    event = threading.Event()
    event.set()
    rows = pd.DataFrame({"frame": [0], "x": [1.0], "y": [1.0]})
    with pytest.raises(CancelledError):
        backend.link_locs(rows, 1, cancel_event=event)
    assert list(rows.columns) == ["frame", "x", "y"]


def test_localisation_roi_and_frame_preview():
    y, x = np.indices((23, 23))
    image = 100 + 1000 * np.exp(-((x - 11) ** 2 + (y - 12) ** 2) / 2)
    movie = np.stack([image, image]).astype(np.float32)
    result = backend.localize_movie(
        movie,
        threshold=50,
        box_size=7,
        frame_index=1,
        fit=False,
        roi=((5, 5), (18, 18)),
    )
    assert len(result)
    assert set(result.frame) == {1}
    np.testing.assert_allclose(result.x, 11)
    np.testing.assert_allclose(result.y, 12)
    excluded = backend.localize_movie(
        movie, threshold=50, box_size=7, fit=False, roi=((0, 0), (6, 6))
    )
    assert len(excluded) == 0


def test_fitting_existing_candidates_preserves_frame_and_metadata():
    y, x = np.indices((23, 23))
    image = 100 + 1000 * np.exp(-((x - 11.2) ** 2 + (y - 12.1) ** 2) / 2)
    movie = np.stack([image, image]).astype(np.float32)
    candidates = np.rec.fromarrays([[1], [11.0], [12.0], [7], ['molecule A']],
                                  names="frame,x,y,cell_index,label")
    result = backend.localize_movie(
        movie,
        threshold=50,
        box_size=7,
        candidates=candidates,
        frame_index=1,
        fit=True,
    )
    assert len(result) == 1
    assert result.frame[0] == 1
    assert result.photons[0] > 0
    assert result.cell_index[0] == 7
    assert result.label[0] == 'molecule A'
    np.testing.assert_allclose(
        [result.x[0], result.y[0]], [11.2, 12.1], atol=0.1
    )


def test_photometry_keeps_row_order_and_maps_gui_metrics():
    movie = np.full((2, 15, 15), 2.0, dtype=np.float32)
    movie[0, 6:9, 6:9] = 10
    movie[1, 6:9, 6:9] = 20
    rows = pd.DataFrame(
        {"frame": [1, 0], "x": [7.0, 7.0], "y": [7.0, 7.0], "particle": [8, 9]}
    )
    result = backend.measure_locs(
        movie,
        rows,
        spot_size=3,
        spot_shape="square",
        background_buffer=1,
        background_width=1,
    )
    assert result.particle.tolist() == [8, 9]
    np.testing.assert_allclose(result.pixel_sum, [180, 90])
    np.testing.assert_allclose(result.pixel_mean_bg, [2, 2])
    np.testing.assert_allclose(result.pixel_sum_bg, [18, 18])


def test_drift_applies_same_displacement_to_pixels_and_coordinates():
    movie = np.zeros((2, 9, 9), dtype=np.uint16)
    movie[:, 4, 4] = 100
    rows = np.rec.fromarrays(
        [[0, 1], [4.0, 4.0], [4.0, 4.0], [123.0, 456.0]],
        names="frame,x,y,photons",
    )
    result = type(
        "Drift",
        (),
        {
            "frames": np.array([0, 1]),
            "displacement": np.array([[1.0, 0.0], [1.0, 0.0]]),
        },
    )()
    corrected_movie, corrected_rows = backend.correct_drift(
        movie, rows, result
    )
    np.testing.assert_array_equal(corrected_movie[:, 4, 3], [100, 100])
    np.testing.assert_allclose(corrected_rows.x, [3.0, 3.0])
    np.testing.assert_array_equal(corrected_rows.photons, rows.photons)


def test_drift_accepts_shorter_paired_channel_and_checks_cancellation():
    movie = np.zeros((2, 9, 9), dtype=np.uint16)
    result = type('Drift', (), {'frames': np.array([0, 1, 2]),
                               'displacement': np.zeros((3, 2))})()
    corrected, _ = backend.correct_drift(movie, None, result)
    assert corrected.shape == movie.shape
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        backend.correct_drift(movie, None, result, cancel_event=event)


def test_drift_interpolates_between_alternating_camera_frames():
    movie = np.zeros((2, 9, 9), dtype=np.uint16)
    rows = np.rec.fromarrays([[0, 1], [4., 4.], [4., 4.]], names='frame,x,y')
    result = type('Drift', (), {'frames': np.array([0, 1, 2]),
        'displacement': np.array([[0., 0.], [2., 0.], [4., 0.]])})()
    _, corrected = backend.correct_drift(movie, rows, result, images=False,
        source_frames=np.array([0, 2, 4]), target_frames=np.array([1, 3]))
    np.testing.assert_allclose(corrected.x, [3., 1.])


def test_track_statistics_use_actual_frames_for_speed_and_time():
    rows = pd.DataFrame(
        {"frame": [0, 2, 3, 5], "x": [0.0, 2.0, 3.0, 5.0], "y": [0.0] * 4}
    )
    stats = backend.track_statistics(
        rows, pixel_size=0.1, time_step=0.5, fit_lags=2
    )
    np.testing.assert_allclose(stats["time"], [0, 1.0, 1.5, 2.5])
    np.testing.assert_allclose(stats["speed"], [0, 0.2, 0.2, 0.2])
    assert np.isfinite(stats["D"])


class ControlledSegmentation:
    version = "controlled-1"

    def compute(self, payload, reporter):
        from molcore import Segmentation

        request, frame, identifier, image, source_path, settings, _, _ = (
            payload
        )
        reporter.check_cancelled()
        labels = (image > 0).astype(np.uint16)
        return Segmentation(
            identifier,
            request.acquisition_id,
            labels,
            image,
            channel=request.channel,
            frame=frame,
        )


def test_cellpose_adapter_preserves_geometry_and_order():
    from molcore import CellposeSettings

    first = np.zeros((7, 9), dtype=np.float32)
    first[2:5, 3:6] = 1
    second = np.ones((5, 6), dtype=np.float32)
    output = backend.segment_images(
        [first, second],
        settings=CellposeSettings(),
        backend=ControlledSegmentation(),
    )
    assert [mask.shape for mask in output] == [first.shape, second.shape]
    np.testing.assert_array_equal(output[0], first)
    np.testing.assert_array_equal(output[1], second)


def test_picasso_gpu_extension_uses_unified_fitter(monkeypatch):
    from molcore import CameraSettings, LocalisationSettings
    from picasso import localize

    calls = []

    def fit(spots, **kwargs):
        calls.append(kwargs)
        return (
            np.array([[100, 3.0, 3.0, 1.0, 1.0, 1.0]]),
            None,
            np.array([3]),
            np.array([1.0]),
        )

    monkeypatch.setattr(localize, "fit_spots_gauss", fit)
    candidate = {
        "frame": np.array([5]),
        "x": np.array([7]),
        "y": np.array([7]),
        "net_gradient": np.array([100.0]),
    }
    reporter = type(
        "Reporter",
        (),
        {"report": lambda *a: None, "check_cancelled": lambda *a: None},
    )()
    columns, failures = backend.PicassoGpu().compute(
        (
            np.ones((1, 15, 15)) * 101,
            np.array([5]),
            LocalisationSettings(),
            CameraSettings(100.0, 1.0, 1.0),
            candidate,
            True,
        ),
        reporter,
    )
    assert calls[0]["use_gpu"] is True
    assert columns["frame"].tolist() == [5]
    assert failures == 0


def test_merge_only_clusters_within_each_frame():
    rows = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "x": [3.0, 3.1, 3.0],
            "y": [4.0, 4.0, 4.0],
            "channel": ["donor", "acceptor", "donor"],
        }
    )
    result = backend.merge_locs(rows, 0.5)
    assert result.frame.tolist() == [0, 1]
    assert result.x.tolist() == [3.0, 3.0]


def test_filter_retains_strict_gui_bounds_and_background_subtraction():
    rows = np.rec.fromarrays(
        [[0] * 4, [2.0] * 4, [3.0] * 4, [2.0, 3.0, 4.0, np.nan], [1.0] * 4],
        names="frame,x,y,pixel_sum,pixel_sum_bg",
    )
    result = backend.filter_locs(
        rows, "pixel_sum", 1.0, 3.0, subtract_background=True
    )
    assert result.pixel_sum.tolist() == [3.0]


def test_read_movie_uses_tiff_frame_limit(tmp_path):
    import tifffile

    movie = np.arange(3 * 7 * 9, dtype=np.uint16).reshape(3, 7, 9)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, movie, photometric="minisblack")
    frames = backend.read_movie(path, limit=2)
    np.testing.assert_array_equal(frames, movie[:2])
