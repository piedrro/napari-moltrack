import numpy as np
from molcore import Segmentation
from molcore.staging import state_token
from test_session_mutations import make_session


def test_shape_layers_roundtrip_edits_properties_and_model_mask_invalidation(
    qtbot, monkeypatch
):
    import napari

    from moltrack._widget import MolTrack

    monkeypatch.setattr(MolTrack, "import_dev_data", lambda self: None)
    viewer = napari.Viewer(show=False)
    try:
        widget = MolTrack(viewer)
        qtbot.addWidget(widget)
        widget.session = make_session()
        polygon = np.array([[1.0, 1.0], [1.0, 5.0], [5.0, 5.0], [5.0, 1.0]])
        mask = Segmentation(
            id="mask",
            acquisition_id="a",
            channel="image",
            frame=0,
            labels=np.ones((8, 8), dtype=np.int32),
        )
        widget.session.model.add_segmentation(mask)
        widget.initialise_segLayer([polygon], 100, model_mask_ids=["mask"])
        assert "mask" in widget.session.model.segmentations
        before = state_token(widget.session.model)
        widget.segLayer.data = [polygon + 1]
        assert "mask" not in widget.session.model.segmentations
        assert state_token(widget.session.model) != before
        np.testing.assert_array_equal(
            widget.session.annotations["segmentations"]["data"][0], polygon + 1
        )
        widget.initialise_cellLayer(
            [polygon], ["polygon"], {"name": ["cell"], "radius": [2.0]}
        )
        widget.cellLayer.properties = {
            "name": np.array(["renamed"]),
            "radius": np.array([3.0]),
        }
        assert widget.session.annotations["cells"]["properties"][
            "name"
        ].tolist() == ["renamed"]
        widget.segLayer.data = []
        assert widget.session.annotations["segmentations"]["data"] == []
    finally:
        viewer.dims.events.current_step.disconnect()
        viewer.close()


def test_regular_selectors_and_track_layer_use_stable_ids_and_clear_empty(
    qtbot, monkeypatch
):
    import napari
    from molcore import Acquisition, ArraySource
    from test_session_mutations import rows

    from moltrack._widget import MolTrack

    monkeypatch.setattr(MolTrack, "import_dev_data", lambda self: None)
    viewer = napari.Viewer(show=False)
    try:
        w = MolTrack(viewer)
        qtbot.addWidget(w)
        w.session = make_session()
        ids = w.session.replace_localizations("data", "donor", rows())
        w.session.model.tracks.add(ids, metadata={"particle": 1})
        w.session.import_tracks("data", "donor", rows().assign(particle=1))
        w.session.model.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((1, 8, 8))),
                id="seg",
                metadata={"role": "segmentation_image"},
            )
        )
        w.session.bind(
            "Segmentation Image", "Segmentation Image", "seg", "image"
        )
        w.segmentation_image = np.zeros((8, 8))
        w.populate_dataset_selectors()
        assert [
            w.gui.tracking_dataset.itemText(i)
            for i in range(w.gui.tracking_dataset.count())
        ] == ["data"]
        assert w.gui.cellpose_dataset.itemText(0) == "Segmentation Image"
        w.gui.moltrack_dataset_selector.setCurrentText("data")
        w.gui.moltrack_channel_selector.setCurrentText("donor")
        w.draw_tracks()
        assert len(np.unique(w.track_layer.data[:, 0])) == 2
        w.session.delete_rows("data", "donor", tracks=True)
        w.draw_tracks()
        assert "Tracks" not in viewer.layers
    finally:
        viewer.dims.events.current_step.disconnect()
        viewer.close()


def test_real_cell_midline_edit_fits_current_vertices_and_retains_final_properties(
    monkeypatch,
):
    from copy import deepcopy
    from types import SimpleNamespace as NS

    from napari.components import ViewerModel

    import moltrack.funcs.cell_events as module

    host = module._cell_events()
    host.session = make_session()
    host.viewer = ViewerModel()
    host.active_dataset = None
    host.gui = NS(show_shapes=NS(isChecked=lambda: True))
    host.move_polygons_to_front = lambda: None
    host.update_segmentation_combos = lambda: None
    host.register_shape_layer_keybinds = lambda layer: None
    polygon = np.array([[0.0, 0.0], [0.0, 6.0], [6.0, 6.0], [6.0, 0.0]])
    original = np.array([[2.0, 1.0], [2.0, 4.0]])
    parameters = {
        "name": "cell",
        "radius": 2.0,
        "poly_params": np.array([0.0, 1.0]),
        "cell_poles": original.copy(),
    }
    layer = host.initialise_cellLayer(
        [original, polygon],
        ["path", "polygon"],
        {
            "name": ["cell", "cell"],
            "cell": [deepcopy(parameters), deepcopy(parameters)],
        },
    )
    received = []

    def fit(polygon_coords, midline_coords, radius):
        received.append(midline_coords.copy())
        return (
            polygon_coords.copy(),
            midline_coords.copy(),
            np.array([4.0, 5.0]),
            3.5,
        )

    monkeypatch.setattr(module, "manual_fit", fit)
    for edited in (
        np.array([[3.0, 2.0], [3.0, 5.0]]),
        np.array([[4.0, 2.0], [4.0, 5.0]]),
    ):
        # The second edit also exercises the reconnect after a compound fit.
        host.store_cell_shapes(init=True)
        shapes = deepcopy(layer.data)
        shapes[0] = edited
        layer.data = shapes
        np.testing.assert_array_equal(received[-1], edited)
        assert (
            len(received) <= 2
        )  # no recursive fit on companion shape updates
        annotations = host.session.annotations["cells"]
        np.testing.assert_array_equal(annotations["data"][0], edited)
        for properties in annotations["properties"]["cell"]:
            assert properties["radius"] == 3.5
            np.testing.assert_array_equal(
                properties["poly_params"], [4.0, 5.0]
            )
        np.testing.assert_array_equal(
            host.get_cell("cell")["midline_coords"], edited
        )
