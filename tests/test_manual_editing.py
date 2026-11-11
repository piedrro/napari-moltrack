"""Regression coverage for complete manual cells and drag-to-extend editing."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Shapes
from napari.utils.key_bindings import coerce_keybinding
from shapely.geometry import Polygon

from moltrack.funcs.cell_events import _cell_events
from moltrack.funcs.segmentation_events import _segmentation_events
from moltrack.session import ProjectSession


@pytest.mark.parametrize("points", [
    [[1., 3.], [4., 3.]], [[4., 3.], [1., 3.]],
    [[3., 1.], [3., 4.]], [[3., 4.], [3., 1.]],
])
def test_manual_refit_preserves_pole_order(points):
    from shapely.geometry import LineString
    from moltrack.funcs.cell_events import _validated_cell_fit

    midline = np.asarray(points)
    outline = np.asarray(LineString(midline).buffer(1).exterior.coords)
    _, fitted, _, _ = _validated_cell_fit(outline, midline, 1.)
    np.testing.assert_allclose(fitted[[0, -1]], midline, atol=1e-6)


class EditingHost(_cell_events, _segmentation_events):
    def update_segmentation_combos(self):
        pass


def host():
    h = EditingHost()
    h.session = ProjectSession()
    h.viewer = ViewerModel()
    h.active_dataset = None
    h.segmentation_image_pixel_size = 100.
    h.segmentation_mode = "panzoom"
    h.gui = NS(show_shapes=NS(isChecked=lambda: True),
               default_cell_radius=NS(value=lambda: 2.))
    return h


def assert_complete_cells(h, count):
    layer = h.cellLayer
    saved = h.session.annotations["cells"]
    assert len(layer.data) == count * 2
    assert saved["shape_types"] == list(layer.shape_type)
    names = layer.properties["name"]
    assert len(set(names)) == count
    for name in set(names):
        cell = h.get_cell(name)
        assert cell is not None
        assert Polygon(cell["polygon_coords"]).is_valid
        assert cell["radius"] == 2.
        for i in (cell["midline_index"], cell["polygon_index"]):
            props = layer.properties["cell"][i]
            assert props["name"] == name
            np.testing.assert_allclose(props["cell_poles"], cell["midline_coords"][[0, -1]])
            np.testing.assert_array_equal(saved["data"][i], layer.data[i])
            np.testing.assert_allclose(saved["properties"]["cell"][i]["cell_poles"], props["cell_poles"])


@pytest.mark.parametrize("framed", [False, True])
def test_manual_cells_are_complete_repeatable_editable_and_persisted(framed, tmp_path):
    from moltrack import model_io

    h = host()
    ndim = 3 if framed else 2
    h.initialise_cellLayer(prepared_layer=Shapes(name="Cells", ndim=ndim))
    for path in ([[10., 10.], [10., 30.]], [[40., 10.], [45., 20.], [40., 30.]]):
        path = np.asarray(path)
        if framed:
            path = np.column_stack((np.full(len(path), 3.), path))
        h.cellLayer.add_paths(path)
    assert_complete_cells(h, 2)
    if framed:
        assert all(np.all(s[:, 0] == 3) for s in h.cellLayer.data)
    np.testing.assert_array_equal(h.cellLayer.scale[-2:], [100., 100.])
    shapes = deepcopy(h.cellLayer.data)
    shapes[0][:, -2] += 1
    h.cellLayer.data = shapes
    assert_complete_cells(h, 2)
    destination = tmp_path / "cells.moltrack.h5"
    model_io.save_project(h.session, destination)
    with model_io.load_project(destination) as loaded:
        assert loaded.annotations["cells"]["shape_types"] == list(h.cellLayer.shape_type)
        for actual, expected in zip(loaded.annotations["cells"]["data"], h.cellLayer.data, strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("failure", ["fit", "duplicate", "radius", "bad_result"])
def test_failed_first_creation_leaves_empty_layer_and_can_retry(monkeypatch, failure):
    import moltrack.funcs.cell_events as module

    h = host()
    h.initialise_cellLayer()
    original_fit = module.manual_fit
    notices = []
    monkeypatch.setattr(module, "show_info", notices.append)
    if failure == "fit":
        def broken(*args):
            raise RuntimeError("fit failed")
        monkeypatch.setattr(module, "manual_fit", broken)
    elif failure == "bad_result":
        monkeypatch.setattr(module, "manual_fit", lambda *args: (None, None, None, None))
    elif failure == "radius":
        h.gui.default_cell_radius = NS(value=lambda: 0)
    path = [[10., 10.], [10., 30.]] if failure != "duplicate" else [[10., 10.], [10., 10.]]
    h.cellLayer.add_paths(path)
    assert len(h.cellLayer.data) == 0
    assert notices
    monkeypatch.setattr(module, "manual_fit", original_fit)
    h.gui.default_cell_radius = NS(value=lambda: 2.)
    h.cellLayer.add_paths([[10., 10.], [10., 30.]])
    assert_complete_cells(h, 1)


@pytest.mark.parametrize("failure", ["fit", "duplicate", "radius", "bad_result"])
def test_failed_creation_preserves_existing_cells_and_can_retry(monkeypatch, failure):
    import moltrack.funcs.cell_events as module

    h = host()
    h.initialise_cellLayer()
    h.cellLayer.add_paths([[10., 10.], [10., 30.]])
    before = deepcopy(h.cellLayer.data)
    before_names = list(h.cellLayer.properties["name"])
    notices = []
    monkeypatch.setattr(module, "show_info", notices.append)
    original_fit = module.manual_fit
    if failure == "fit":
        def broken(*args):
            raise RuntimeError("fit failed")
        monkeypatch.setattr(module, "manual_fit", broken)
    elif failure == "bad_result":
        monkeypatch.setattr(module, "manual_fit", lambda *args: (None, None, None, None))
    elif failure == "radius":
        h.gui.default_cell_radius = NS(value=lambda: 0)
    path = [[40., 10.], [40., 30.]] if failure != "duplicate" else [[40., 10.], [40., 10.]]
    h.cellLayer.add_paths(path)
    assert len(h.cellLayer.data) == 2
    assert list(h.cellLayer.properties["name"]) == before_names
    for a, b in zip(h.cellLayer.data, before, strict=True):
        np.testing.assert_array_equal(a, b)
    assert notices
    assert_complete_cells(h, 1)
    monkeypatch.setattr(module, "manual_fit", original_fit)
    h.gui.default_cell_radius = NS(value=lambda: 2.)
    h.cellLayer.add_paths([[40., 10.], [40., 30.]])
    assert_complete_cells(h, 2)


def segmentation_host(framed=False):
    h = host()
    target = np.array([[0., 0.], [0., 10.], [10., 10.], [10., 0.]])
    neighbour = target + [0., 12.]
    if framed:
        target, neighbour = [np.column_stack((np.full(4, 3.), p)) for p in (target, neighbour)]
    h.initialise_segLayer([target, neighbour], shape_properties={"name": ["target", "neighbour"]})
    if framed:
        h.viewer.dims.set_current_step(0, 3)
    return h


def drag(h, coords, *, cancel=False):
    layer = h.segLayer
    h.modify_mode(mode="extend")
    positions = [tuple(layer.data_to_world(p)) for p in coords]
    event = NS(position=positions[0], type="mouse_press", button=1, handled=False)
    gesture = h.seg_drag_event(layer, event)
    next(gesture)
    for point in positions[1:-1]:
        event.type = "mouse_move"
        event.position = point
        next(gesture)
    if cancel:
        h.modify_mode(mode="pan_zoom")
    event.position = positions[-1]
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gesture)


@pytest.mark.parametrize("framed", [False, True])
def test_extend_exact_union_preserves_identity_neighbours_and_saved_geometry(framed, tmp_path):
    from moltrack import model_io

    h = segmentation_host(framed)
    neighbour = h.segLayer.data[1].copy()
    coords = np.array([[2., 8.], [2., 16.], [8., 16.], [8., 8.]])
    if framed:
        coords = np.column_stack((np.full(len(coords), 3.), coords))
    drag(h, coords)
    assert len(h.segLayer.data) == 2
    assert Polygon(h.segLayer.data[0][:, -2:]).area == pytest.approx(136.)
    np.testing.assert_array_equal(h.segLayer.data[1], neighbour)
    assert list(h.segLayer.properties["name"]) == ["target", "neighbour"]
    assert h.segmentation_mode == "panzoom"
    assert h.segLayer.mode == "pan_zoom"
    if framed:
        assert np.all(h.segLayer.data[0][:, 0] == 3)
    np.testing.assert_array_equal(h.session.annotations["segmentations"]["data"][0], h.segLayer.data[0])
    destination = tmp_path / "extended.moltrack.h5"
    model_io.save_project(h.session, destination)
    with model_io.load_project(destination) as loaded:
        np.testing.assert_array_equal(loaded.annotations["segmentations"]["data"][0], h.segLayer.data[0])
        assert list(loaded.annotations["segmentations"]["properties"]["name"]) == ["target", "neighbour"]
    drag(h, coords)
    assert len(h.segLayer.data) == 2


@pytest.mark.parametrize("coords,cancel", [
    ([[2., 11.], [2., 16.], [8., 16.], [8., 8.]], False),
    ([[2., 8.], [2., 16.], [8., 16.]], False),
    ([[2., 8.], [2., 16.], [8., 16.], [8., 14.]], False),
    ([[2., 8.], [2., 16.], [2., 8.]], False),
    ([[2., 8.], [8., 16.], [2., 16.], [8., 8.]], False),
    ([[2., 8.], [2., 16.], [8., 16.], [8., 8.]], True),
])
def test_invalid_or_cancelled_extension_leaves_no_temporary_shapes(coords, cancel):
    h = segmentation_host()
    before = deepcopy(h.segLayer.data)
    drag(h, coords, cancel=cancel)
    for actual, expected in zip(h.segLayer.data, before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert h.segmentation_mode == "panzoom"
    assert h.segLayer.mode == "pan_zoom"


def test_extension_cancels_if_frame_changes():
    h = segmentation_host(True)
    before = deepcopy(h.segLayer.data)
    drag(h, [[3., 2., 8.], [3., 2., 16.], [4., 8., 16.], [4., 8., 8.]])
    for actual, expected in zip(h.segLayer.data, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_e_shortcut_from_cells_selects_segmentations_and_escape_cancels():
    h = segmentation_host()
    h.initialise_cellLayer()
    h.viewer.layers.selection.select_only(h.cellLayer)
    h.cellLayer.keymap[coerce_keybinding("e")](h.cellLayer)
    assert h.viewer.layers.selection.active is h.segLayer
    assert h.segmentation_mode == "extend"
    h.segLayer.keymap[coerce_keybinding("Escape")](h.segLayer)
    assert h.segmentation_mode == "panzoom"


@pytest.mark.parametrize("cancel", [False, True])
def test_extension_shows_transient_stroke_without_recording_preview(cancel):
    h = segmentation_host()
    layer = h.segLayer
    h.modify_mode(mode="extend")
    event = NS(position=layer.data_to_world([2., 8.]), type="mouse_press", button=1, handled=False)
    gesture = h.seg_drag_event(layer, event)
    next(gesture)
    event.type = "mouse_move"
    event.position = layer.data_to_world([2., 16.])
    next(gesture)
    preview = h._extension_preview
    assert preview in h.viewer.layers
    assert h.viewer.layers.selection.active is layer
    np.testing.assert_array_equal(preview.data[0], [[2., 8.], [2., 16.]])
    assert len(h.session.annotations["segmentations"]["data"]) == 2
    assert not layer.mouse_pan
    if cancel:
        h.modify_mode(mode="pan_zoom")
        assert preview not in h.viewer.layers
    event.type = "mouse_release"
    event.position = layer.data_to_world([8., 8.])
    with pytest.raises(StopIteration):
        next(gesture)
    assert preview not in h.viewer.layers
    assert layer.mouse_pan


def test_native_midline_drawing_completion_creates_one_complete_pair():
    from napari.layers.shapes._shapes_mouse_bindings import (
        add_path_polygon,
        polygon_creating,
    )

    h = host()
    h.modify_mode(mode="midline")
    layer = h.cellLayer
    for point in ([10., 10.], [10., 30.]):
        event = NS(position=layer.data_to_world(point), pos=np.asarray(point),
                   modifiers=(), button=1, type="mouse_press")
        if layer._is_creating:
            polygon_creating(layer, event)
        add_path_polygon(layer, event)
    layer._finish_drawing()
    assert_complete_cells(h, 1)


def test_abandoned_native_midline_does_not_recreate_an_existing_last_path():
    from napari.layers.shapes._shapes_mouse_bindings import add_path_polygon

    h = host()
    layer = h.initialise_cellLayer()
    layer.add_paths([[10., 10.], [10., 30.]])
    # Imported cells can have their polygon before their path.
    h.update_cellLayer_shapes(
        list(reversed(layer.data)), list(reversed(layer.shape_type)),
        {key: values[::-1].copy() for key, values in layer.properties.items()},
    )
    before = deepcopy(layer.data)
    names = list(layer.properties["name"])
    edge_colors, face_colors = layer.edge_color.copy(), layer.face_color.copy()
    h.modify_mode(mode="midline")
    event = NS(position=layer.data_to_world([40., 10.]), pos=np.array([40., 10.]),
               modifiers=(), button=1, type="mouse_press")
    add_path_polygon(layer, event)
    layer._finish_drawing()
    assert_complete_cells(h, 1)
    assert list(layer.properties["name"]) == names
    np.testing.assert_array_equal(layer.edge_color, edge_colors)
    np.testing.assert_array_equal(layer.face_color, face_colors)
    for actual, expected in zip(layer.data, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_boundary_endpoints_can_extend_a_segmentation():
    h = segmentation_host()
    drag(h, [[2., 10.], [2., 16.], [8., 16.], [8., 10.]])
    assert Polygon(h.segLayer.data[0]).area == pytest.approx(136.)


def test_native_toolbar_mode_change_cancels_extension_and_preserves_chosen_mode():
    h = segmentation_host()
    layer = h.segLayer
    before = deepcopy(layer.data)
    h.modify_mode(mode="extend")
    event = NS(position=layer.data_to_world([2., 8.]), type="mouse_press", button=1, handled=False)
    gesture = h.seg_drag_event(layer, event)
    next(gesture)
    event.type = "mouse_move"
    event.position = layer.data_to_world([2., 16.])
    next(gesture)
    preview = h._extension_preview
    layer.mode = "select"
    assert preview not in h.viewer.layers
    event.type = "mouse_release"
    event.position = layer.data_to_world([8., 8.])
    with pytest.raises(StopIteration):
        next(gesture)
    assert layer.mode == "select"
    for actual, expected in zip(layer.data, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_editing_shortcuts_through_qt_key_events(qtbot):
    import napari
    from qtpy.QtCore import Qt

    h = host()
    viewer = napari.Viewer(show=False)
    h.viewer = viewer
    try:
        h.initialise_segLayer([np.array([[0., 0.], [0., 10.], [10., 10.], [10., 0.]])])
        canvas = viewer.window._qt_viewer
        qtbot.keyClick(canvas, Qt.Key_M)
        assert viewer.layers.selection.active is h.cellLayer
        assert h.cellLayer.mode == "add_path"
        qtbot.keyClick(canvas, Qt.Key_S)
        assert h.cellLayer.mode == "direct"
        qtbot.keyClick(canvas, Qt.Key_E)
        assert viewer.layers.selection.active is h.segLayer
        assert h.segmentation_mode == "extend"
        qtbot.keyClick(canvas, Qt.Key_Escape)
        assert h.segmentation_mode == "panzoom"
        assert h.segLayer.mouse_pan
    finally:
        viewer.close()
