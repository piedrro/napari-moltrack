from types import SimpleNamespace as NS

import numpy as np
import pytest
from shapely.geometry import Polygon
from test_session_mutations import make_session, rows

from moltrack.funcs.loc_filter_utils import _loc_filter_utils
from moltrack.funcs.management_utils import _management_utils
from moltrack.funcs.track_filter_utils import _track_filter_utils
from moltrack.funcs.tracking_utils import _tracking_utils
from moltrack.funcs.trackstats_utils import _trackstats_utils
from moltrack.model_views import gui_query, legacy_snapshot


def control(value):
    return NS(
        currentText=lambda: value,
        value=lambda: value,
        currentIndex=lambda: value,
        isChecked=lambda: value,
        setEnabled=lambda *args: None,
    )


class Host(
    _management_utils,
    _loc_filter_utils,
    _track_filter_utils,
    _tracking_utils,
    _trackstats_utils,
):
    def __init__(self):
        self.session = make_session()
        self.calls = []
        self.gui = NS()
        self.moltrack_metrics = {"Photons": "photons"}

    @property
    def localisation_dict(self):
        return legacy_snapshot(self.session, "localisations")

    @property
    def tracking_dict(self):
        return legacy_snapshot(self.session, "tracks")

    @property
    def dataset_dict(self):
        return legacy_snapshot(self.session, "images")

    def get_locs(self, dataset, channel, **kw):
        return gui_query(self.session, dataset, channel, **kw)

    def get_tracks(self, dataset, channel, **kw):
        return gui_query(self.session, dataset, channel, tracks=True, **kw)

    def update_ui(self, *args, **kw):
        pass

    def draw_localisations(self, *args, **kw):
        self.calls.append("locs")

    def draw_tracks(self, *args, **kw):
        self.calls.append("tracks")

    def update_filter_criterion(self):
        pass

    def update_criterion_ranges(self):
        pass

    def update_track_criterion_ranges(self):
        pass

    def plot_diffusion_graph(self):
        pass


def populated():
    host = Host()
    for ch in ("donor", "acceptor"):
        ids = host.session.replace_localizations("data", ch, rows())
        host.session.model.tracks.add(ids, metadata={"particle": 1})
    return host


def test_gui_copy_and_clear_all_use_session():
    h = populated()
    h.gui.copy_locs_dataset = control("data")
    h.gui.copy_locs_channel = control("donor")
    h.gui.copy_locs_mode = control(0)
    before = h.session.query_localizations("data", "acceptor").id.tolist()
    h.copy_data()
    assert (
        h.session.query_localizations("data", "acceptor").id.tolist() != before
    )
    h.gui.delete_locs_dataset = control("All Datasets")
    h.gui.delete_locs_channel = control("All Channels")
    h.delete_data()
    assert all(
        h.session.query_localizations(*p).empty
        for p in h.session.selected_partitions()
    )


def test_gui_localization_filter_clears_every_selected_row():
    h = populated()
    for name, value in {
        "picasso_filter_dataset": "data",
        "picasso_filter_channel": "All Channels",
        "filter_criterion": "Photons",
        "filter_min": 90,
        "filter_max": 100,
        "filter_subtract_bg": False,
    }.items():
        setattr(h.gui, name, control(value))
    h.gui.filter_localisations = control(None)
    h.pixseq_filter_localisations()
    assert all(
        h.session.query_localizations(*p).empty
        for p in h.session.selected_partitions()
    )


def test_gui_polygon_filter_strict_boundaries_and_overlap_preserve_unique_ids():
    h = populated()
    h.get_shapes = lambda *args, **kw: [
        Polygon([(2, 2), (2, 5), (5, 5), (5, 2)]),
        Polygon([(2, 2), (2, 6), (6, 6), (6, 2)]),
    ]
    for name, value in {
        "remove_seglocs_dataset": "data",
        "remove_seglocs_channel": "All Channels",
        "remove_seglocs_segmentation": "Segmentations",
    }.items():
        setattr(h.gui, name, control(value))
    h.remove_seglocs()
    result = h.session.query_localizations("data", "donor")
    assert result.x.tolist() == [3.0, 4.0]
    assert result.id.is_unique
    assert result.segmentation_index.tolist() == [0.0, 0.0]


def test_gui_track_filter_empty_result_clears_all_memberships_retains_detections():
    h = populated()
    for name, value in {
        "track_filter_dataset": "data",
        "track_filter_channel": "All Channels",
        "track_filter_criterion": "Photons",
        "track_filter_metric": "Mean",
        "track_filter_subtract_bg": False,
        "track_filter_min": 90,
        "track_filter_max": 100,
    }.items():
        setattr(h.gui, name, control(value))
    h.filter_tracks()
    assert all(
        h.session.query_tracks(*p).empty
        for p in h.session.selected_partitions()
    )
    assert len(h.session.query_localizations("data", "donor")) == 3


def test_gui_segtrack_removal_empty_result_retains_detection_rows():
    h = populated()
    for name, value in {
        "segtracks_dataset": "data",
        "segtracks_channel": "All Channels",
        "segtracks_seg": "Segmentations",
        "segtracks_detect": "Entire Track",
        "segtracks_remove": "Track",
        "min_track_length": 2,
    }.items():
        setattr(h.gui, name, control(value))
    h.get_shapes = lambda *args, **kw: [
        Polygon([(6, 6), (6, 7), (7, 7), (7, 6)])
    ]
    h.remove_segtracks()
    assert all(
        h.session.query_tracks(*p).empty
        for p in h.session.selected_partitions()
    )
    assert len(h.session.query_localizations("data", "donor")) == 3


def test_gui_track_stats_updates_measurements_by_stable_id():
    h = populated()
    from moltrack.model_analysis import compute_analysis, prepare_analysis

    ids = h.session.query_tracks("data", "donor").id.tolist()
    pending = compute_analysis(prepare_analysis(h.session, fit_lags=2))
    h.process_track_stats_result(pending)
    data = h.session.query_tracks("data", "donor")
    assert data.id.tolist() == ids
    np.testing.assert_allclose(data.speed, [0.0, np.sqrt(2), np.sqrt(2)])


def test_gui_merge_returns_detached_pending_then_publishes_on_result():
    from moltrack.funcs.loc_filter_utils import (
        compute_merge,
        prepare_merge,
        publish_merge,
    )

    h = populated()
    before = h.session.query_localizations("data", "donor").id.tolist()
    pending = compute_merge(prepare_merge(h.session, "data", 1.0))
    assert h.session.query_localizations("data", "donor").id.tolist() == before
    publish_merge(h.session, pending)
    merged = h.session.query_localizations("data", "donor")
    assert len(merged) == 3
    assert not set(merged.id) & set(before)


def test_track_filter_keeps_sample_std_skips_first_and_groups_stable_ids():
    h = populated()
    h.session.import_tracks(
        "data", "donor", rows().assign(particle=1, photons=[999, 40, 60])
    )
    for name, value in {
        "track_filter_dataset": "data",
        "track_filter_channel": "donor",
        "track_filter_criterion": "Photons",
        "track_filter_metric": "Standard Deviation",
        "track_filter_subtract_bg": False,
        "track_filter_min": 0,
        "track_filter_max": 100,
    }.items():
        setattr(h.gui, name, control(value))
    stats = h.track_statistics_filtering()
    assert len(stats) == 2
    assert sorted(stats.stat.tolist()) == sorted(
        [np.std([20, 30], ddof=1), np.std([40, 60], ddof=1)]
    )


def test_trackplot_revision_cache_refreshes_and_selects_colliding_label_tracks():
    from moltrack.funcs.trackplot_utils import _trackplot_utils

    class PlotHost(Host, _trackplot_utils):
        pass

    h = PlotHost()
    ids = h.session.replace_localizations("data", "donor", rows())
    first = h.session.model.tracks.add(ids, metadata={"particle": 1})
    second = h.session.import_tracks(
        "data", "donor", rows().assign(particle=1)
    ).item()
    for name, value in {
        "trackplot_dataset": "data",
        "trackplot_channel": "donor",
        "trackplot_subtrack_background": False,
        "trackplot_xaxis": "Frame Index",
        "trackplot_slider": 1,
    }.items():
        setattr(h.gui, name, control(value))
    for i in range(1, 5):
        setattr(
            h.gui, f"trackplot_metric{i}", control("Photons" if i == 1 else "")
        )
    h.init_trackplot_tracks()
    result = h.get_trackplot_data()
    assert result["track_id"] == second
    assert len(result["data"][0]["values"]) == 3
    h.session.delete_rows("data", "donor", tracks=True, ids=ids)
    refreshed = h.init_trackplot_tracks(reset=False)
    assert first not in refreshed.track_id.values


def test_trace_export_keeps_colliding_tracks_separate_and_original_labels(
    tmp_path,
):
    import json

    from moltrack.funcs.traces_utils import _traces_utils

    class TraceHost(Host, _traces_utils):
        pass

    h = TraceHost()
    ids = h.session.replace_localizations("data", "donor", rows())
    h.session.model.tracks.add(ids, metadata={"particle": 1})
    h.session.import_tracks("data", "donor", rows().assign(particle=1))
    for name, value in {
        "traces_export_dataset": "All Datasets",
        "traces_export_channel": "All Channels",
        "traces_export_metric": "Photons",
        "traces_export_subtract_background": False,
    }.items():
        setattr(h.gui, name, control(value))
    path = tmp_path / "traces.json"
    h.get_traces_path = lambda dataset: str(path)
    h.export_traces()
    content = json.loads(path.read_text())
    traces = content["data"]["data"]
    assert len(traces) == 2
    assert len({trace["track_id"] for trace in traces}) == 2
    assert [trace["particle"] for trace in traces] == [1, 1]
    assert all(
        trace["trace_dict"]["donor"] == [10.0, 20.0, 30.0] for trace in traces
    )


def test_segmentation_image_translation_updates_persistent_corrected_view():
    from moltrack.funcs.events_utils import _events_utils

    h = _events_utils()
    h.session = make_session()
    binding = ("data", "donor")
    h.session.set_annotations("segmentation_image_binding", binding)
    image = h.session.model.acquisitions["a"].source[:].copy()
    h.segmentation_image = image.copy()
    h._segmentation_source = h.segmentation_image
    h.segmentation_layer = NS(data=h.segmentation_image, refresh=lambda: None)
    h.gui = NS(
        translation_target=control("Segmentation Image"),
        translation_size=control(2.0),
    )
    before = h.session.model.transforms.matrix("a", "image", 0)
    h.moltract_translation(direction="right")
    after = h.session.model.transforms.matrix("a", "image", 0)
    np.testing.assert_allclose(after[:2, 2], before[:2, 2] + [2.0, 0.0])
    np.testing.assert_array_equal(
        h.session.model.acquisitions["a"].source[:], image
    )
    assert h._segmentation_source is h.segmentation_image


@pytest.mark.parametrize(
    "operation", ["delete", "copy", "filter", "partial", "polygon", "merge"]
)
def test_localization_cascades_refresh_visible_tracks_and_consumers(operation):
    from molcore import Acquisition, ArraySource
    from napari.components import ViewerModel

    from moltrack.model_edits import compute_merge, prepare_merge

    class ProjectionHost(Host):
        def draw_tracks(self, *args, **kwargs):
            self.calls.append("tracks")
            _tracking_utils.draw_tracks(self, *args, **kwargs)

    h = ProjectionHost()
    for channel in ("donor", "acceptor"):
        ids = h.session.replace_localizations("data", channel, rows())
        h.session.model.tracks.add(ids, metadata={"particle": 1})
    h.session.model.add_acquisition(
        Acquisition(
            ArraySource(np.zeros((5, 8, 8))), id="unaffected", pixel_size=100
        )
    )
    h.session.bind("other", "donor", "unaffected", "image")
    unaffected_ids = h.session.replace_localizations("other", "donor", rows())
    unaffected_track = h.session.model.tracks.add(unaffected_ids)
    acceptor_ids = h.session.query_tracks("data", "acceptor").id.tolist()
    h.viewer = ViewerModel()
    for name, value in {
        "moltrack_dataset_selector": "data",
        "moltrack_channel_selector": "donor",
        "track_colour_metric": "",
        "track_colour_colourmap": "viridis",
        "show_tracks": True,
    }.items():
        setattr(h.gui, name, control(value))
    h.select_track = lambda *args: None
    if operation == "partial":
        h.moltrack_metrics["Speed"] = "speed"
        h.session.update_measurements(
            h.session.query_tracks("data", "donor").id,
            {"speed": [0.0, 1.0, 2.0]},
        )
        h.gui.track_colour_metric = control("Speed")
        for name in ("track_colour_min", "track_colour_max"):
            setattr(
                h.gui,
                name,
                NS(
                    blockSignals=lambda *args: None,
                    setRange=lambda *args: None,
                    setValue=lambda *args: None,
                    setSingleStep=lambda *args: None,
                ),
            )
    h.draw_tracks()
    assert "Tracks" in h.viewer.layers
    h.calls.clear()
    h.trackplot_tracks = h.get_tracks("All Datasets", "All Channels")

    def update_plots():
        h.calls.append("plot_options")
        h.trackplot_tracks = h.get_tracks("All Datasets", "All Channels")

    h.update_trackplot_options = update_plots
    h.update_track_filter_criterion = lambda: h.calls.append("filter_options")
    h.plot_tracks = lambda: h.calls.append("plot")
    h.update_diffusion_range = lambda: h.calls.append("range")
    h.plot_diffusion_graph = lambda: h.calls.append("diffusion")
    h.update_traces_export_options = lambda: h.calls.append("traces")
    h._heatmap_is_current = lambda: h.calls.append("heatmap")
    if operation == "delete":
        h.gui.delete_locs_dataset = control("data")
        h.gui.delete_locs_channel = control("donor")
        h.delete_data()
    elif operation == "copy":
        h.gui.copy_locs_dataset = control("data")
        h.gui.copy_locs_channel = control("acceptor")
        h.gui.copy_locs_mode = control(0)
        h.copy_data()
    elif operation in ("filter", "partial"):
        for name, value in {
            "picasso_filter_dataset": "data",
            "picasso_filter_channel": "donor",
            "filter_criterion": "Photons",
            "filter_min": 15 if operation == "partial" else 90,
            "filter_max": 35 if operation == "partial" else 100,
            "filter_subtract_bg": False,
        }.items():
            setattr(h.gui, name, control(value))
        h.gui.filter_localisations = control(None)
        h.pixseq_filter_localisations()
    elif operation == "polygon":
        for name, value in {
            "remove_seglocs_dataset": "data",
            "remove_seglocs_channel": "donor",
            "remove_seglocs_segmentation": "Segmentations",
        }.items():
            setattr(h.gui, name, control(value))
        h.get_shapes = lambda *args, **kw: [
            Polygon([(6, 6), (6, 7), (7, 7), (7, 6)])
        ]
        h.remove_seglocs()
    else:
        h.process_merge_localisations(
            compute_merge(prepare_merge(h.session, "data", 1.0))
        )
        h.merge_localisations_finished()
    if operation == "partial":
        assert len(h.session.query_tracks("data", "donor")) == 2
        assert len(h.viewer.layers["Tracks"].data) == 2
    else:
        assert h.session.query_tracks("data", "donor").empty
        assert "Tracks" not in h.viewer.layers
    assert {
        "tracks",
        "plot_options",
        "plot",
        "range",
        "diffusion",
        "traces",
        "heatmap",
    } <= set(h.calls)
    assert set(h.trackplot_tracks["track_id"]) == set(
        h.session.model.tracks.query()["track_id"]
    )
    assert h.session.query_tracks(
        "other", "donor"
    ).track_id.unique().tolist() == [unaffected_track]
    assert (
        h.session.query_localizations("other", "donor").id.tolist()
        == unaffected_ids.tolist()
    )
    if operation != "merge":
        assert (
            h.session.query_tracks("data", "acceptor").id.tolist()
            == acceptor_ids
        )
