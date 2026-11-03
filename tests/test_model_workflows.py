"""Persistent model workflows preserve raw state and scientific identities."""

import threading

import numpy as np
import pytest
from molcore import CellposeSettings, DriftSettings, StaleResultError
from test_model_processing import make_session


def run(session, operation, settings, partitions=(("a", "image"),)):
    from moltrack import model_processing as proc

    pending = proc.compute(
        proc.prepare(
            session, proc.ProcessingRequest(operation, partitions, settings)
        )
    )
    proc.publish(session, pending)
    return pending


def test_photometry_preserves_identity_and_metric_correspondence():
    from moltrack.model_processing import PhotometrySettings

    session = make_session()
    ids = session.query_localizations("a", "image").id.to_numpy()
    run(session, "photometry", PhotometrySettings(spot_size=3))
    rows = session.query_localizations("a", "image", space="raw")
    np.testing.assert_array_equal(rows.id, ids)
    assert rows.pixel_sum.tolist() == [0.0, 0.0]
    assert rows.pixel_sum_bg.tolist() == [0.0, 0.0]
    assert len(rows) == 2


def test_registration_points_and_pixels_use_same_transform():
    from moltrack.model_processing import RegistrationSettings

    session = make_session()
    raw_image = session.image("a", "image", space="raw")[:].copy()
    raw_rows = session.query_localizations("a", "image", space="raw").copy()
    settings = RegistrationSettings(
        ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    )
    run(session, "registration", settings)
    run(session, "registration", settings)
    rows = session.query_localizations("a", "image")
    np.testing.assert_allclose(rows.x, raw_rows.x + 2)
    np.testing.assert_allclose(rows.y, raw_rows.y + 1)
    np.testing.assert_array_equal(
        session.image("a", "image", space="raw")[:], raw_image
    )
    np.testing.assert_array_equal(
        session.query_localizations("a", "image", space="raw")[["x", "y"]],
        raw_rows[["x", "y"]],
    )


def test_repeat_drift_does_not_accumulate(monkeypatch):
    from types import SimpleNamespace

    from moltrack import model_processing as proc

    session = make_session()
    raw = session.query_localizations("a", "image", space="raw").copy()
    monkeypatch.setattr(
        proc.undrifting,
        "estimate_drift",
        lambda model, acquisition, channel, **kw: SimpleNamespace(
            frames=np.array([0, 1]),
            displacement=np.array([[1.0, 2.0], [2.0, 3.0]]),
        ),
    )
    settings = proc.DriftSettings(DriftSettings(), (("a", "image"),))
    run(session, "drift", settings)
    first = session.query_localizations("a", "image").copy()
    run(session, "drift", settings)
    np.testing.assert_allclose(
        session.query_localizations("a", "image")[["x", "y"]],
        first[["x", "y"]],
    )
    np.testing.assert_allclose(first.x, raw.x - [1.0, 2.0])
    np.testing.assert_allclose(
        session.query_localizations("a", "image", space="raw")[["x", "y"]],
        raw[["x", "y"]],
    )


def test_render_result_rejects_stale_rows():
    from moltrack import model_processing as proc

    session = make_session()
    pending = proc.compute(
        proc.prepare(
            session,
            proc.ProcessingRequest(
                "render",
                (("a", "image"),),
                proc.RenderingSettings(oversampling=2),
            ),
        )
    )
    assert pending.results[0][2].image.shape == (30, 30)
    session.update_metadata("a", pixel_size=120.0)
    with pytest.raises(StaleResultError):
        proc.publish(session, pending)


def test_segmentation_publishes_model_bound_masks(monkeypatch):
    from molcore import Segmentation

    from moltrack import model_processing as proc

    session = make_session()

    def segment(model, requests, **kwargs):
        return {
            r.segmentation_id: Segmentation(
                id=r.segmentation_id,
                acquisition_id=r.acquisition_id,
                channel=r.channel,
                frame=r.frame,
                labels=np.zeros((15, 15), dtype=np.int32),
            )
            for r in requests
        }

    monkeypatch.setattr(proc.cellpose, "segment_many", segment)
    run(
        session,
        "segmentation",
        proc.SegmentationSettings(CellposeSettings(), frame_index=1),
    )
    assert len(session.model.segmentations) == 1
    segmentation = next(iter(session.model.segmentations.values()))
    assert segmentation.acquisition_id == "camera"
    assert segmentation.frame == 1


def test_cancelled_photometry_does_not_publish():
    from concurrent.futures import CancelledError

    from moltrack import model_processing as proc

    session = make_session()
    pending = proc.compute(
        proc.prepare(
            session,
            proc.ProcessingRequest(
                "photometry", (("a", "image"),), proc.PhotometrySettings()
            ),
        )
    )
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        proc.publish(session, pending, cancel_event=event)
    assert "pixel_sum" not in session.query_localizations("a", "image")


def test_tracking_uses_detection_ids_and_preserves_independent_imports():
    import pandas as pd

    from moltrack import model_processing as proc

    session = make_session()
    session.replace_localizations(
        "a",
        "image",
        pd.DataFrame(
            {"frame": [0, 1, 0], "x": [5.0, 6.0, 12.0], "y": [5.0, 6.0, 12.0]}
        ),
    )
    ids = session.query_localizations("a", "image", space="raw").id.to_numpy()
    session.import_tracks(
        "a",
        "image",
        pd.DataFrame(
            {
                "frame": [0, 1],
                "x": [5.0, 6.0],
                "y": [5.0, 6.0],
                "particle": [99, 99],
            }
        ),
    )
    imported = session.query_tracks("a", "image", space="raw").copy()
    run(
        session,
        "tracking",
        proc.TrackingSettings(
            search_range=2.0, min_length=2, remove_unlinked=True
        ),
    )
    rows = session.query_localizations("a", "image", space="raw")
    assert rows.id.tolist() == ids[:2].tolist()
    tracks = session.query_tracks("a", "image", space="raw")
    assert set(imported.id) <= set(tracks.id)
    assert set(tracks.id) == set(ids[:2]) | set(imported.id)
    assert len(tracks) == 4
    run(
        session,
        "tracking",
        proc.TrackingSettings(search_range=2.0, min_length=2),
    )
    assert len(session.query_tracks("a", "image")) == 4


def test_segmentation_image_import_retains_persistent_source():
    from moltrack.funcs.import_utils import _import_utils

    host = _import_utils()
    host.session = make_session()
    image = np.zeros((2, 15, 15))
    host.publish_image_import(
        {
            "import_mode": "Segmentation Image",
            "jobs": [
                {
                    "dataset_name": "seg",
                    "path": "",
                    "source_images": image,
                    "pixel_size": 100.0,
                    "exposure_time": 10.0,
                }
            ],
        }
    )
    binding = host.session.annotations["segmentation_image_binding"]
    np.testing.assert_array_equal(
        host.session.image(*binding, space="raw")[:], image
    )


def test_generated_track_labels_do_not_merge_with_imported_tracks():
    import pandas as pd

    from moltrack import model_processing as proc

    session = make_session()
    session.import_tracks(
        "a",
        "image",
        pd.DataFrame(
            {
                "frame": [0, 1],
                "x": [5.0, 6.0],
                "y": [5.0, 6.0],
                "particle": [1, 1],
            }
        ),
    )
    run(
        session,
        "tracking",
        proc.TrackingSettings(search_range=2.0, min_length=2),
    )
    tracks = session.query_tracks("a", "image")
    assert tracks.groupby("particle").track_id.nunique().max() == 1
    assert tracks.particle.nunique() == 2


@pytest.mark.parametrize(
    "operation",
    [
        "tracking",
        "photometry",
        "render",
        "segmentation",
        "registration",
        "drift",
    ],
)
def test_gui_publication_runs_on_qt_thread(operation, qtbot, monkeypatch):
    from types import SimpleNamespace

    import napari
    from molcore import Segmentation
    from qtpy.QtCore import QThread

    from moltrack import model_processing as proc
    from moltrack._widget import MolTrack

    monkeypatch.setattr(MolTrack, "import_dev_data", lambda self: None)
    viewer = napari.Viewer(show=False)
    host = MolTrack(viewer)
    qtbot.addWidget(host)
    host.session = make_session()
    host.session.update_metadata("a", pixel_size=100.0)
    host.session.annotations["partitions"] = {"a": {"image": {"fitted": True}}}
    for name in (
        "update_ui",
        "draw_localisations",
        "draw_tracks",
        "update_active_image",
        "tracking_finished",
        "compute_pixstats_finished",
        "picasso_render_finished",
        "run_cellpose_finished",
        "_apply_fret_transform_matrix_finished",
    ):
        monkeypatch.setattr(host, name, lambda *a, **kw: None)
    for name in (
        "tracking_dataset",
        "picasso_render_dataset",
        "cellpose_dataset",
        "undrift_dataset",
    ):
        combo = getattr(host.gui, name)
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("a")
        combo.blockSignals(False)
    for name in (
        "tracking_channel",
        "picasso_render_channel",
        "cellpose_channel",
        "undrift_channel",
    ):
        combo = getattr(host.gui, name)
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("image")
        combo.blockSignals(False)
    host.session.bind("a", "Donor", "camera", "image")
    host.gui.tform_apply_target.blockSignals(True)
    host.gui.tform_apply_target.clear()
    host.gui.tform_apply_target.addItem("Donor")
    host.gui.tform_apply_target.blockSignals(False)
    host.transform_matrix = np.eye(3)
    published, computed = [], []

    def compute(prepared, **kwargs):
        assert QThread.currentThread() != host.thread()
        computed.append(prepared)
        results = []
        for dataset, channel in prepared.request.partitions:
            acquisition, name = prepared.session.partition(dataset, channel)
            if operation == "tracking":
                value = (
                    (
                        prepared.session.query_localizations(
                            dataset, channel
                        ).id.to_numpy(),
                    ),
                    {},
                )
            elif operation == "photometry":
                value = prepared.session.query_localizations(
                    dataset, channel
                ).assign(pixel_sum=42.0)
            elif operation == "render":
                value = SimpleNamespace(image=np.zeros((15, 15)))
            elif operation == "registration":
                value = np.eye(3)
            elif operation == "drift":
                value = (np.array([0, 1]), np.zeros((2, 2)))
            else:
                segment = Segmentation(
                    id="test",
                    acquisition_id=acquisition,
                    channel=name,
                    frame=0,
                    labels=np.zeros((15, 15), dtype=np.int32),
                )
                value = {"test": segment}
            results.append((dataset, channel, value))
        return proc.PendingOperation(prepared, tuple(results))

    real_publish = proc.publish

    def publish(*args, **kwargs):
        assert QThread.currentThread() == host.thread()
        result = real_publish(*args, **kwargs)
        published.append(result)
        return result

    monkeypatch.setattr(proc, "compute", compute)
    monkeypatch.setattr(proc, "publish", publish)
    try:
        if operation == "tracking":
            host.initialise_tracking()
        elif operation == "photometry":
            host.initialise_pixstats("locs")
        elif operation == "render":
            host.initialise_picasso_render()
        elif operation == "segmentation":
            host.initialise_cellpose()
        elif operation == "registration":
            host.apply_fret_transform_matrix()
        else:
            host.undrifting_tools.rcc_undrift()
        qtbot.waitUntil(lambda: bool(published), timeout=10000)
        assert computed
        host.threadpool.waitForDone(10000)
    finally:
        viewer.close()


@pytest.mark.parametrize(
    "operation",
    [
        "tracking",
        "photometry",
        "render",
        "segmentation",
        "registration",
        "drift",
    ],
)
def test_gui_workflows_capture_detached_requests(operation):
    from types import SimpleNamespace as NS

    from moltrack.funcs.picasso_render_utils import _picasso_render_utils
    from moltrack.funcs.pixstats_utils import _pixstats_utils
    from moltrack.funcs.segmentation_utils import _segmentation_utils
    from moltrack.funcs.tracking_utils import _tracking_utils
    from moltrack.funcs.transform_utils import _transform_utils
    from moltrack.funcs.undrift_utils import MoltrackUndrifting

    class Control:
        def __init__(self, text="a", number=1, checked=False):
            self._text, self._number, self._checked = text, number, checked

        def currentText(self):
            return self._text

        def text(self):
            return str(self._number)

        def value(self):
            return self._number

        def isChecked(self):
            return self._checked

        def blockSignals(self, value):
            pass

        def setChecked(self, value):
            pass

    class Gui:
        def __getattr__(self, name):
            if name.endswith("channel"):
                return Control("image")
            if name == "cellpose_model":
                return Control("cyto3")
            if "spot_shape" in name:
                return Control("square")
            if "spot_size" in name:
                return Control("3")
            if "bg_buffer" in name or "bg_width" in name:
                return Control("1")
            if name == "tform_apply_target":
                return Control("donor")
            return Control()

    types = {
        "tracking": _tracking_utils,
        "photometry": _pixstats_utils,
        "render": _picasso_render_utils,
        "segmentation": _segmentation_utils,
        "registration": _transform_utils,
    }
    host = types.get(operation, _tracking_utils)()
    host.session = make_session()
    host.session.update_metadata("a", pixel_size=100.)
    host.session.annotations["partitions"] = {"a": {"image": {"fitted": True}}}
    if operation == "registration":
        host.session.bind("a", "Donor", "camera", "image")
    host.gui = Gui()
    host.stop_event = threading.Event()
    host.dataset_dict = {
        "a": {"images": {"image": np.zeros((2, 15, 15))}, "pixel_size": 100.0}
    }
    host.localisation_dict = {
        "a": {"image": {"fitted": True, "localisations": [1]}}
    }
    host.tracking_dict = {}
    host.update_ui = lambda **kw: None
    host.moltrack_progress = lambda *a, **kw: None
    host.get_shapes = lambda *a, **kw: []
    host.get_locs = lambda *a, **kw: (
        [
            {
                "dataset": "a",
                "channel": "image",
                "image_shape": (2, 15, 15),
                "localisations": host.session.query_localizations(
                    "a", "image"
                ).to_records(index=False),
            }
        ]
        if kw.get("return_dict")
        else host.session.query_localizations("a", "image")
        .assign(dataset="a", channel="image")
        .to_records(index=False)
    )
    host.viewer = NS(dims=NS(current_step=(1,)))
    host.transform_matrix = np.eye(3)
    workers = []
    host.threadpool = NS(start=workers.append)
    host.tracking_finished = host.compute_pixstats_finished = (
        host.picasso_render_finished
    ) = host.run_cellpose_finished = (
        host._apply_fret_transform_matrix_finished
    ) = lambda: None
    if operation == "tracking":
        host.initialise_tracking()
    elif operation == "photometry":
        host.initialise_pixstats("locs")
    elif operation == "render":
        host.initialise_picasso_render()
    elif operation == "segmentation":
        host.initialise_cellpose()
    elif operation == "registration":
        host.apply_fret_transform_matrix()
    else:
        MoltrackUndrifting(None, host).rcc_undrift()
    assert len(workers) == 1
    worker = workers[0]
    assert getattr(worker.fn, "__self__", None) is None
    prepared = worker.args[0]
    assert prepared.request.operation == operation
    assert prepared.session.model is not host.session.model
    assert prepared.owner() is host.session.model


def test_photometry_alex_fret_aligns_rows_and_clears_previous_metrics():
    import pandas as pd
    from molcore import Acquisition, ArraySource

    from moltrack import model_processing as proc

    session = make_session()
    session.model.add_acquisition(
        Acquisition(ArraySource(np.full((2, 15, 15), 20.0)), id="acceptor")
    )
    session.bind("a", "DD", "camera", "image")
    session.bind("a", "DA", "acceptor", "image")
    rows = pd.DataFrame(
        {
            "frame": [1, 0],
            "x": [6.0, 5.0],
            "y": [6.0, 5.0],
            "photons": [20.0, 10.0],
            "bg": [2.0, 1.0],
        }
    )
    session.replace_localizations("a", "DD", rows)
    rows.photons = [30.0, 40.0]
    session.replace_localizations("a", "DA", rows)
    partitions = (("a", "DD"), ("a", "DA"))
    run(
        session,
        "photometry",
        proc.PhotometrySettings(calculate_fret=True),
        partitions,
    )
    donor = session.query_localizations("a", "DD")
    acceptor = session.query_localizations("a", "DA")
    # frame 0: (40-1)/((40-1)+(10-1)); frame 1: (30-2)/((30-2)+(20-2)).
    expected = {0: 39 / 48, 1: 28 / 46}
    for rows in (donor, acceptor):
        np.testing.assert_allclose(
            rows.photons_fret, [expected[f] for f in rows.frame]
        )
    run(
        session,
        "photometry",
        proc.PhotometrySettings(calculate_fret=False),
        partitions,
    )
    assert session.query_localizations("a", "DD").photons_fret.isna().all()


def test_failed_photometry_second_publication_is_atomic():
    from dataclasses import replace

    import pandas as pd
    from molcore import Acquisition, ArraySource

    from moltrack import model_processing as proc

    session = make_session()
    session.model.add_acquisition(
        Acquisition(ArraySource(np.zeros((2, 15, 15))), id="second")
    )
    session.bind("a", "second", "second", "image")
    session.replace_localizations(
        "a", "second", pd.DataFrame({"frame": [0], "x": [5.0], "y": [5.0]})
    )
    request = proc.ProcessingRequest(
        "photometry",
        (("a", "image"), ("a", "second")),
        proc.PhotometrySettings(),
    )
    pending = proc.compute(proc.prepare(session, request))
    bad = pending.results[1][2].copy()
    bad["id"] = [999999]
    pending = replace(
        pending, results=(pending.results[0], ("a", "second", bad))
    )
    with pytest.raises(ValueError, match="existing localization IDs"):
        proc.publish(session, pending)
    assert "pixel_sum" not in session.query_localizations("a", "image")


def test_registration_pixels_and_points_remain_aligned():
    import pandas as pd
    from molcore import Acquisition, ArraySource, Dataset

    from moltrack import model_processing as proc
    from moltrack.session import ProjectSession

    image = np.zeros((2, 15, 15), dtype=np.float32)
    image[0, 5, 5] = image[1, 6, 6] = 100.0
    model = Dataset()
    model.add_acquisition(Acquisition(ArraySource(image), id="signal"))
    session = ProjectSession(model)
    session.bind("a", "image", "signal", "image")
    session.replace_localizations(
        "a",
        "image",
        pd.DataFrame({"frame": [0, 1], "x": [5.0, 6.0], "y": [5.0, 6.0]}),
    )
    run(
        session,
        "registration",
        proc.RegistrationSettings(
            ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
        ),
    )
    pixels = session.image("a", "image")[:]
    points = session.query_localizations("a", "image")
    for _, row in points.iterrows():
        assert pixels[int(row.frame), int(row.y), int(row.x)] == 100.0
    np.testing.assert_array_equal(
        session.image("a", "image", space="raw")[:], image
    )


def test_drift_applies_same_correction_to_pixels_and_points(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd
    from molcore import Acquisition, ArraySource, Dataset

    from moltrack import model_processing as proc
    from moltrack.session import ProjectSession

    movie = np.zeros((2, 15, 15), dtype=np.float32)
    movie[0, 5, 5] = movie[1, 6, 6] = 100.0
    model = Dataset()
    model.add_acquisition(Acquisition(ArraySource(movie), id="camera"))
    session = ProjectSession(model)
    session.bind("a", "image", "camera", "image")
    session.replace_localizations(
        "a",
        "image",
        pd.DataFrame({"frame": [0, 1], "x": [5.0, 6.0], "y": [5.0, 6.0]}),
    )
    original_ids = session.query_localizations(
        "a", "image", space="raw"
    ).id.to_numpy()

    def estimate(model, acquisition, channel, **kwargs):
        np.testing.assert_array_equal(kwargs["ids"], original_ids)
        return SimpleNamespace(
            frames=np.array([0, 1]),
            displacement=np.array([[1.0, 2.0], [1.0, 2.0]]),
        )

    monkeypatch.setattr(proc.undrifting, "estimate_drift", estimate)
    run(
        session,
        "drift",
        proc.DriftSettings(DriftSettings(), (("a", "image"),)),
    )
    for _, row in session.query_localizations("a", "image").iterrows():
        assert (
            session.image("a", "image")[int(row.frame)][int(row.y), int(row.x)]
            == 100.0
        )
    np.testing.assert_array_equal(
        session.image("a", "image", space="raw")[:], movie
    )


def test_deleted_acquisition_rejects_pending_registration():
    from moltrack import model_processing as proc

    session = make_session()
    pending = proc.compute(
        proc.prepare(
            session,
            proc.ProcessingRequest(
                "registration",
                (("a", "image"),),
                proc.RegistrationSettings(
                    ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
                ),
            ),
        )
    )
    session.model.remove_acquisition("camera", cascade=True)
    with pytest.raises(StaleResultError):
        proc.publish(session, pending)
    assert not session.model.acquisitions


def test_tracking_segmentation_groups_keep_detection_ids():
    import pandas as pd

    from moltrack import model_processing as proc

    session = make_session()
    ids = session.replace_localizations(
        "a",
        "image",
        pd.DataFrame(
            {
                "frame": [0, 1, 0, 1],
                "x": [5.0, 6.0, 12.0, 13.0],
                "y": [5.0, 6.0, 12.0, 13.0],
            }
        ),
    )
    polygon = ((3.0, 3.0), (8.0, 3.0), (8.0, 8.0), (3.0, 8.0))
    run(
        session,
        "tracking",
        proc.TrackingSettings(
            search_range=2.0,
            min_length=2,
            polygons=(polygon,),
            segmentation_column="cell_index",
        ),
    )
    tracks = session.query_tracks("a", "image")
    assert set(tracks.id) == set(ids[:2])
    assert tracks.cell_index.tolist() == [0.0, 0.0]
    assert len(session.query_localizations("a", "image")) == 4


def test_processing_settings_publish_with_results():
    from moltrack import model_processing as proc

    session = make_session()
    request = proc.ProcessingRequest(
        "photometry", (("a", "image"),), proc.PhotometrySettings(spot_size=5)
    )
    pending = proc.compute(proc.prepare(session, request))
    assert "photometry" not in session.model.processing.settings
    proc.publish(session, pending)
    assert (
        session.model.processing.settings["moltrack_operations"]["camera"]["image"]["photometry"][
            "settings"
        ]["spot_size"]
        == 5
    )


@pytest.mark.parametrize(
    "source_frames,logical_frames,memory,minimum,expected_members",
    [
        ([0, 1, 2], [0, 1, 2], 0, 3, 3),
        ([0, 2, 4], [0, 1, 2], 0, 3, 3),
        ([0, 2, 4], [0, 2], 0, 2, 0),
        ([0, 2, 4], [0, 2], 1, 2, 2),
        ([0, 1, 3, 4, 6], [1, 2, 3], 0, 3, 3),
        ([0, 1, 3, 4, 6], [0, 2], 0, 2, 0),
        ([0, 1, 3, 4, 6], [0, 2], 1, 2, 2),
    ],
)
def test_gui_tracking_memory_uses_logical_channel_frames(
    source_frames, logical_frames, memory, minimum, expected_members
):
    import pandas as pd
    from molcore import Acquisition, ArraySource, ChannelSpec, Dataset

    from moltrack import model_processing as proc
    from moltrack.session import ProjectSession

    model = Dataset()
    irregular = source_frames == [0, 1, 3, 4, 6]
    spec = (
        ChannelSpec(excitation="D")
        if irregular
        else ChannelSpec(frame_step=source_frames[1] - source_frames[0])
    )
    model.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((source_frames[-1] + 1, 15, 15))),
            id="camera",
            channel_specs={"DD": spec},
            excitation_sequence=("D", "D", "A") if irregular else (),
        )
    )
    session = ProjectSession(model)
    session.bind("a", "DD", "camera", "DD")
    np.testing.assert_array_equal(
        session._view("a", "DD").source_frames, source_frames
    )
    rows = pd.DataFrame(
        {
            "frame": logical_frames,
            "x": np.arange(len(logical_frames)) * 0.1 + 5.0,
            "y": np.full(len(logical_frames), 5.0),
        }
    )
    ids = session.replace_localizations("a", "DD", rows)
    before = model.localizations.query(space="raw").to_columns()
    run(
        session,
        "tracking",
        proc.TrackingSettings(
            search_range=2.0,
            memory=memory,
            min_length=minimum,
            remove_unlinked=bool(expected_members),
        ),
        (("a", "DD"),),
    )
    tracks = session.query_tracks("a", "DD")
    assert len(tracks) == expected_members
    np.testing.assert_array_equal(
        session.query_localizations("a", "DD", space="raw").id, ids
    )
    after = model.localizations.query(space="raw").to_columns()
    for key in ("id", "x", "y", "frame"):
        np.testing.assert_array_equal(after[key], before[key])
