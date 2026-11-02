"""Display adapters work without creating an OpenGL canvas."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel

from moltrack.funcs.events_utils import _events_utils


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
    from moltrack.funcs.picasso_render_utils import _picasso_render_utils

    host = _picasso_render_utils()
    host.viewer = ViewerModel()
    host.viewer.add_image(np.zeros((2, 10, 10)), units=("pixel", "nm", "nm"))
    host.gui = SimpleNamespace(show_render=SimpleNamespace(isChecked=lambda: True))
    host.draw_picasso_render((np.zeros((10, 10, 3)), 100, 2))

    assert host.viewer.layers.extent.units is not None
    np.testing.assert_allclose(host.render_layer.scale, [50, 50])


@pytest.mark.parametrize("shape", [(10, 10), (2, 10, 10)])
def test_active_image_sets_spatial_units_and_preserves_frame_units(shape):
    host = _events_utils()
    host.viewer = ViewerModel()
    image = np.arange(np.prod(shape), dtype=float).reshape(shape)
    host.dataset_dict = {"a": {"images": {"image": image}, "pixel_size": 100}}
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
        host.dataset_dict["a"]["pixel_size"] = 120
        host.update_active_image(dataset="a", channel="image")

    assert [str(unit) for unit in host.image_layer.units[-2:]] == ["nanometer"] * 2
    if len(shape) == 3:
        assert str(host.image_layer.units[0]) == "pixel"
    np.testing.assert_allclose(host.image_layer.scale[-2:], [120, 120])
