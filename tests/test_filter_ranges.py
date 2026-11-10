import weakref
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QPushButton

from moltrack.funcs.loc_filter_utils import _loc_filter_utils
from moltrack.funcs.track_filter_utils import _track_filter_utils


@pytest.fixture(params=["localisations", "tracks"])
def histogram(request, qtbot):
    canvas = pg.GraphicsLayoutWidget()
    low, high = QDoubleSpinBox(), QDoubleSpinBox()
    for widget in (canvas, low, high):
        qtbot.addWidget(widget)
    for box in (low, high):
        box.setRange(-1e9, 1e9)
    if request.param == "localisations":
        host = _loc_filter_utils()
        host.filter_graph_canvas = canvas
        host.gui = SimpleNamespace(filter_min=low, filter_max=high)

        def draw(values):
            host.plot_filter_graph("photons", values)
    else:
        host = _track_filter_utils()
        host.track_graph_canvas = canvas
        host.gui = SimpleNamespace(
            track_filter_min=low,
            track_filter_max=high,
            track_filter_criterion=SimpleNamespace(
                currentText=lambda: "Photons"
            ),
            track_filter_metric=SimpleNamespace(currentText=lambda: "Mean"),
        )
        draw = host.plot_track_filter_graph
    return SimpleNamespace(
        canvas=canvas,
        low=low,
        high=high,
        draw=draw,
        host=host,
        kind=request.param,
    )


def regions(histogram):
    return [
        item
        for plot in histogram.canvas.ci.items
        if isinstance(plot, pg.PlotItem)
        for item in plot.items
        if isinstance(item, pg.LinearRegionItem)
    ]


def test_histogram_range_synchronises_controls_and_rounds_drag(histogram):
    histogram.draw(np.array([0.0, 2.0, 5.0, 10.0]))
    assert len(regions(histogram)) == 1
    region = regions(histogram)[0]
    assert region.getRegion() == (0.0, 10.0)
    region.setRegion((2.1234, 7.5678))
    assert (histogram.low.value(), histogram.high.value()) == (2.12, 7.57)
    assert region.getRegion() == (2.12, 7.57)
    histogram.low.setValue(3)
    histogram.high.setValue(6)
    assert region.getRegion() == (3.0, 6.0)
    # Crossing a limit collapses the other boundary to preserve ordering.
    histogram.low.setValue(8)
    assert region.getRegion() == (8.0, 8.0)
    assert histogram.high.value() == 8
    histogram.high.setValue(1)
    assert region.getRegion() == (1.0, 1.0)
    assert histogram.low.value() == 1


def test_range_clamps_and_rebuild_disconnects_old_region(histogram):
    histogram.draw(np.array([-5.0, 0.0, 10.0]))
    old = regions(histogram)[0]
    old_selector = weakref.ref(histogram.canvas.filter_range)
    old.setRegion((-100, 100))
    assert old.getRegion() == (-5.0, 10.0)
    histogram.draw(np.array([20.0, 25.0, 30.0]))
    current = regions(histogram)[0]
    assert current.getRegion() == (20.0, 30.0)
    assert old_selector() is None
    assert (histogram.low.value(), histogram.high.value()) == (20.0, 30.0)
    histogram.low.setValue(23)
    assert current.getRegion() == (23.0, 30.0)
    histogram.high.setValue(100)
    assert current.getRegion() == (23.0, 30.0)


def test_final_drag_position_updates_fields(histogram):
    histogram.draw(np.array([0.0, 10.0]))
    region = regions(histogram)[0]
    # PyQtGraph moves both lines with intermediate signals suppressed when
    # dragging the band, and emits only the finished signal on release.
    region.blockLineSignal = True
    for line, value in zip(region.lines, (2.123, 8.456), strict=True):
        line.setValue(value)
    region.blockLineSignal = False
    region.sigRegionChangeFinished.emit(region)
    assert (histogram.low.value(), histogram.high.value()) == (2.12, 8.46)
    assert region.getRegion() == (2.12, 8.46)


def test_context_refresh_resets_selection_and_clears_empty_data(
    histogram, qtbot
):
    import pandas as pd

    host = histogram.host
    host.moltrack_metrics = {"Photons": "photons"}
    background = QCheckBox()
    qtbot.addWidget(background)
    selector = SimpleNamespace(currentText=lambda: "Photons")
    values = [0.0, 5.0, 10.0]
    if histogram.kind == "localisations":
        host.gui.picasso_filter_dataset = selector
        host.gui.picasso_filter_channel = selector
        host.gui.filter_criterion = selector
        host.gui.filter_subtract_bg = background
        host.get_locs = lambda *args: np.array(
            list(zip(values)), dtype=[("photons", float)]
        )
        refresh = host.update_criterion_ranges
    else:
        host.gui.track_filter_subtract_bg = background
        host.track_statistics_filtering = lambda *args: pd.DataFrame(
            {"stat": values}
        )
        refresh = host.update_track_criterion_ranges
    refresh()
    regions(histogram)[0].setRegion((2.0, 8.0))
    values[:] = [20.0, 25.0, 30.0]
    refresh()
    assert regions(histogram)[0].getRegion() == (20.0, 30.0)
    values[:] = [np.nan, np.inf]
    refresh()
    assert regions(histogram) == []
    assert (histogram.low.value(), histogram.high.value()) == (0.0, 0.0)


@pytest.mark.parametrize("values", [[], [np.nan, np.inf, -np.inf], None])
def test_empty_or_nonfinite_data_clears_range(histogram, values, capsys):
    histogram.draw(np.array([0.0, 10.0]))
    old_selector = weakref.ref(histogram.canvas.filter_range)
    histogram.draw(None if values is None else np.array(values))
    assert regions(histogram) == []
    assert old_selector() is None
    assert (histogram.low.value(), histogram.high.value()) == (0.0, 0.0)
    assert not capsys.readouterr().out


def test_range_only_filters_when_button_is_clicked(histogram, qtbot):
    import pandas as pd
    from test_model_mutation_gui import Host, control
    from test_session_mutations import rows

    host = Host()
    is_tracks = histogram.kind == "tracks"
    incoming = pd.concat(
        [
            rows().assign(photons=photons, particle=int(photons))
            for photons in (10.0, 20.0, 30.0)
        ],
        ignore_index=True,
    )
    if is_tracks:
        host.session.import_tracks("data", "donor", incoming)
    else:
        host.session.replace_localizations("data", "donor", incoming)
    prefix = "track_filter_" if is_tracks else "filter_"
    controls = {
        "track_filter_dataset"
        if is_tracks
        else "picasso_filter_dataset": "data",
        "track_filter_channel"
        if is_tracks
        else "picasso_filter_channel": "donor",
        prefix + "criterion": "Photons",
        prefix + "subtract_bg": False,
        "track_filter_metric": "Mean",
    }
    for name, value in controls.items():
        setattr(host.gui, name, control(value))
    setattr(host.gui, prefix + "min", histogram.low)
    setattr(host.gui, prefix + "max", histogram.high)
    button = QPushButton()
    qtbot.addWidget(button)
    host.gui.filter_localisations = button
    button.clicked.connect(
        host.filter_tracks if is_tracks else host.pixseq_filter_localisations
    )
    query = (
        host.session.query_tracks
        if is_tracks
        else host.session.query_localizations
    )
    before = query("data", "donor").copy()
    histogram.draw(np.array([10.0, 20.0, 30.0]))
    regions(histogram)[0].setRegion((15.0, 25.0))
    pd.testing.assert_frame_equal(query("data", "donor"), before)
    button.click()
    result = query("data", "donor")
    assert result.photons.tolist() == [20.0, 20.0, 20.0]


def test_constant_and_partially_nonfinite_data(histogram, capsys):
    histogram.draw(np.array([np.nan, np.inf, -np.inf, 4.0, 4.0]))
    assert len(regions(histogram)) == 1
    region = regions(histogram)[0]
    assert region.getRegion() == (4.0, 4.0)
    region.setRegion((0, 10))
    assert region.getRegion() == (4.0, 4.0)
    assert (histogram.low.value(), histogram.high.value()) == (4.0, 4.0)
    assert not capsys.readouterr().out


def test_small_values_keep_a_usable_full_range(histogram):
    histogram.draw(np.array([0.001, 0.004]))
    assert (histogram.low.value(), histogram.high.value()) == (0.0, 0.01)
    assert regions(histogram)[0].getRegion() == (0.0, 0.01)
    histogram.draw(np.array([-0.004, -0.001]))
    assert (histogram.low.value(), histogram.high.value()) == (-0.01, 0.0)
    assert regions(histogram)[0].getRegion() == (-0.01, 0.0)
