"""Display adapters work without creating an OpenGL canvas."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel

from moltrack.funcs.events_utils import _events_utils


@pytest.mark.parametrize("target", ["Segmentations", "Cells"])
@pytest.mark.parametrize("framed", [False, True])
def test_shape_translation_reuses_meshes_and_records_vertices(target, framed):
    from napari.layers import Shapes

    from moltrack.model_views import connect_layer_annotations
    from moltrack.session import ProjectSession

    host = _events_utils()
    host.session = ProjectSession()
    polygon = np.array([[1., 1.], [1., 4.], [4., 4.], [4., 1.]])
    shapes = [polygon, polygon[:3] + 10]
    if framed:
        shapes = [np.column_stack((np.full(len(s), i), s)) for i, s in enumerate(shapes)]
    shape_types = ["polygon", "path"] if target == "Cells" else "polygon"
    layer = Shapes(shapes, shape_type=shape_types, properties={"name": ["a", "b"]})
    name = "segmentations" if target == "Segmentations" else "cells"
    setattr(host, "segLayer" if target == "Segmentations" else "cellLayer", layer)
    for callback in (
        "seg_drag_event", "delete_clicked", "update_shapes", "dilate_segmentation",
        "celllayer_clicked", "dilate_cell", "update_cells", "store_cell_shapes",
    ):
        setattr(host, callback, lambda *args: None)
    layer.mouse_drag_callbacks.append(host.seg_drag_event if target == "Segmentations" else host.celllayer_clicked)
    layer.mouse_double_click_callbacks.append(host.delete_clicked)
    layer.mouse_wheel_callbacks.append(host.dilate_segmentation if target == "Segmentations" else host.dilate_cell)
    handler = host.update_shapes if target == "Segmentations" else host.update_cells
    layer.events.data.connect(handler)
    connect_layer_annotations(host.session, layer, name)
    host.gui = SimpleNamespace(
        translation_target=SimpleNamespace(currentText=lambda: target),
        translation_size=SimpleNamespace(value=lambda: 2),
    )
    before = [s.copy() for s in layer.data]
    models = list(layer._data_view.shapes)
    triangles = layer._data_view._mesh.triangles.copy()
    mesh_vertices = layer._data_view._mesh.vertices.copy()

    host.moltract_translation(direction="right")

    for old, translated, saved in zip(before, layer.data, host.session.annotations[name]["data"], strict=True):
        expected = old.copy()
        expected[:, -1] += 2
        np.testing.assert_array_equal(translated, expected)
        np.testing.assert_array_equal(saved, expected)
    assert all(old is current for old, current in zip(models, layer._data_view.shapes, strict=True))
    np.testing.assert_array_equal(layer._data_view._mesh.triangles, triangles)
    np.testing.assert_allclose(layer._data_view._mesh.vertices, mesh_vertices + [0, 2])
    assert list(layer.properties["name"]) == ["a", "b"]


def test_auto_contrast_computes_and_applies_limits():
    host = _events_utils()
    host.contrast_dict = {}
    image = np.arange(1000, dtype=float).reshape(10, 10, 10)
    host.image_layer = ViewerModel().add_image(image)

    limits = host.image_layer_auto_contrast(image, "a", "image")

    np.testing.assert_allclose(limits, [0.999, 998.9001])
    np.testing.assert_allclose(host.image_layer.contrast_limits, limits)
    assert np.isfinite(host.image_layer.gamma)


def test_auto_contrast_restores_saved_settings():
    host = _events_utils()
    image = np.arange(100, dtype=float).reshape(10, 10)
    host.image_layer = ViewerModel().add_image(image)
    host.contrast_dict = {"a": {"image": {"contrast_limits": [10, 80], "gamma": 2}}}

    assert host.image_layer_auto_contrast(image, "a", "image") == [10, 80]
    assert host.image_layer.gamma == 2
    assert host.image_layer.contrast_limits == [10, 80]


def test_segmentation_layer_has_physical_units_without_deprecated_scale_bar():
    host = _events_utils()
    host.viewer = ViewerModel()
    host.segmentation_image = np.arange(100).reshape(10, 10)
    host.segmentation_image_pixel_size = 100
    host.gui = SimpleNamespace(show_data=SimpleNamespace(isChecked=lambda: True))
    host.viewer.add_image(np.zeros((2, 10, 10)), units=("pixel", "nm", "nm"))
    inserted_units = []
    host.viewer.layers.events.inserted.connect(
        lambda event: inserted_units.append(host.viewer.layers.extent.units)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        host.draw_segmentation_image()

    assert [str(unit) for unit in host.segmentation_layer.units] == ["nanometer"] * 2
    np.testing.assert_allclose(host.segmentation_layer.scale, [100, 100])
    assert inserted_units and all(units is not None for units in inserted_units)


@pytest.mark.parametrize("kind", ["segmentations", "cells"])
def test_shape_layers_use_units_compatible_with_images(kind):
    from moltrack.funcs.cell_events import _cell_events
    from moltrack.funcs.segmentation_events import _segmentation_events

    host = _segmentation_events() if kind == "segmentations" else _cell_events()
    from moltrack.session import ProjectSession
    host.session = ProjectSession()
    host.viewer = ViewerModel()
    host.viewer.add_image(np.zeros((2, 10, 10)), units=("pixel", "nm", "nm"))
    host.segmentation_image_pixel_size = 100
    host.gui = SimpleNamespace(show_shapes=SimpleNamespace(isChecked=lambda: True))
    for name in (
        "seg_drag_event", "delete_clicked", "update_shapes",
        "update_segmentation_combos", "move_polygons_to_front",
        "update_cells", "celllayer_clicked", "dilate_cell",
        "register_shape_layer_keybinds", "store_cell_shapes",
    ):
        setattr(host, name, lambda *args, **kwargs: None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        if kind == "segmentations":
            layer = host.initialise_segLayer()
        else:
            layer = host.initialise_cellLayer()

    assert [str(unit) for unit in layer.units] == ["nanometer"] * 2
    assert host.viewer.layers.extent.units is not None


def test_smlm_render_uses_units_compatible_with_images():
    import threading
    from test_model_processing import make_session

    from moltrack import model_processing as proc
    from moltrack.funcs.picasso_render_utils import _picasso_render_utils

    host = _picasso_render_utils()
    host.session = make_session()
    host.session.update_metadata("a", pixel_size=100.)
    host.stop_event = threading.Event()
    host.viewer = ViewerModel()
    host.viewer.add_image(np.zeros((2, 10, 10)), units=("pixel", "nm", "nm"))
    host.gui = SimpleNamespace(show_render=SimpleNamespace(isChecked=lambda: True))
    pending = proc.compute(proc.prepare(host.session, proc.ProcessingRequest(
        "render", (("a", "image"),), proc.RenderingSettings(oversampling=2)
    )))
    host.draw_picasso_render(pending)

    assert host.viewer.layers.extent.units is not None
    np.testing.assert_allclose(host.render_layer.scale, [50, 50])


@pytest.mark.parametrize("shape", [(10, 10), (2, 10, 10)])
def test_active_image_sets_spatial_units_and_preserves_frame_units(shape):
    host = _events_utils()
    host.viewer = ViewerModel()
    image = np.arange(np.prod(shape), dtype=float).reshape(shape)
    from moltrack.session import ProjectSession
    from moltrack.model_io import import_images
    host.session = ProjectSession()
    import_images(host.session, [dict(dataset_name="a", path="camera", source_images=image[None] if image.ndim == 2 else image,
        pixel_size=100, exposure_time=20, multichannel_mode="None", channel_name="image")])
    host.contrast_dict = {}
    selector = SimpleNamespace(
        blockSignals=lambda value: None,
        clear=lambda: None,
        addItems=lambda values: None,
        setCurrentIndex=lambda index: None,
    )
    host.gui = SimpleNamespace(
        show_data=SimpleNamespace(isChecked=lambda: True),
        moltrack_dataset_selector=selector,
        moltrack_channel_selector=selector,
    )
    host.update_contrast_dict = lambda: None
    host.draw_localisations = lambda: None
    host.draw_tracks = lambda: None
    host.update_overlay_text = lambda: None

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        host.update_active_image(dataset="a", channel="image")
        host.session.update_metadata("a", pixel_size=120)
        host.update_active_image(dataset="a", channel="image")

    assert [str(unit) for unit in host.image_layer.units[-2:]] == ["nanometer"] * 2
    if len(shape) == 3:
        assert str(host.image_layer.units[0]) == "pixel"
    np.testing.assert_allclose(host.image_layer.scale[-2:], [120, 120])


def test_model_backed_active_image_and_pixel_size():
    from moltrack.model_io import import_images
    from moltrack.session import ProjectSession

    host = _events_utils()
    host.session = ProjectSession()
    import_images(
        host.session,
        [
            dict(
                dataset_name="a",
                path="camera",
                source_images=np.arange(200).reshape(2, 10, 10),
                pixel_size=100,
                exposure_time=20,
                multichannel_mode="None",
                channel_name="image",
            )
        ],
    )
    host.viewer = ViewerModel()
    host.contrast_dict = {}
    selector = SimpleNamespace(
        blockSignals=lambda value: None,
        clear=lambda: None,
        addItems=lambda values: None,
        setCurrentIndex=lambda index: None,
    )
    host.gui = SimpleNamespace(
        show_data=SimpleNamespace(isChecked=lambda: True),
        moltrack_dataset_selector=selector,
        moltrack_channel_selector=selector,
    )
    for name in (
        "update_contrast_dict",
        "draw_localisations",
        "draw_tracks",
        "update_overlay_text",
    ):
        setattr(host, name, lambda: None)
    host.update_active_image(dataset="a", channel="image")
    assert host.image_layer.data.shape == (2, 10, 10)
    host.session.update_metadata("a", pixel_size=120)
    host.update_active_image(dataset="a", channel="image")
    np.testing.assert_allclose(host.image_layer.scale[-2:], [120, 120])
