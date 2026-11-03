import traceback
from functools import partial

from bactfit.preprocess import data_to_cells
from napari.utils.notifications import show_info

from moltrack.funcs.compute_utils import Worker


class _bactfit_utils:

    def run_bactfit_finished(self):

        self.update_ui()
        self.update_segmentation_combos()

    def run_bactfit_results(self, celllist):

        from moltrack.model_cells import validate_cells
        try:
            celllist = validate_cells(self.session, celllist)
        except Exception as exc:
            show_info(str(exc))
            return
        if celllist is None:
            return

        self.celllist = celllist

        data = celllist.get_cell_polygons()

        cell_names = data["names"]
        cell_polygons = data["polygons"]
        cell_radii = data["cell_radii"]
        cell_params = data["poly_params"]
        cell_poles = data["cell_poles"]
        cell_midlines = data["midlines"]

        layer_names = [layer.name for layer in self.viewer.layers]

        if "fitted_cells" in layer_names:
            self.viewer.layers.remove("fitted_cells")

        shapes = []
        shape_types = []
        properties = {"name": [], "cell": []}

        for name, polygon, radius, midline, params, poles in zip(cell_names, cell_polygons,
                cell_radii, cell_midlines, cell_params, cell_poles):

            try:

                fit_params = {"name": name, "radius": radius,
                              "poly_params": params, "cell_poles": poles}

                shapes.append(polygon)
                shape_types.append("polygon")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

                shapes.append(midline)
                shape_types.append("path")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

            except:
                pass

        self.cellLayer = self.initialise_cellLayer(shapes=shapes,
            shape_types=shape_types, properties=properties)

        self.store_cell_shapes()


    def initialise_bactfit(self):

        if hasattr(self, "segLayer"):

            segmentations = self.session.annotations.get("segmentations", {}).get("data", [])

            if len(segmentations) == 0:
                return

            self.update_ui(init=True)

            from moltrack.model_cells import prepare_cells
            prepared = prepare_cells(self.session, {"segmentations": segmentations, "settings": {
                "min_radius": float(self.gui.fit_min_radius.value()),
                "max_radius": float(self.gui.fit_max_radius.value()),
                "max_error": float(self.gui.fit_max_error.value())}}, cancel_event=self.stop_event)
            worker = Worker(fit_cells_worker, prepared)
            worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.bactfit_progressbar))
            worker.signals.result.connect(self.run_bactfit_results)
            worker.signals.finished.connect(self.run_bactfit_finished)
            worker.signals.error.connect(self.update_ui)
            self.threadpool.start(worker)


def fit_cells_worker(prepared, progress_callback=None):
    from moltrack.model_cells import finish_cells
    from moltrack.model_processing import _cancel
    _cancel(prepared.cancel_event)
    cells = data_to_cells(prepared.payload["segmentations"])
    if not len(cells.data):
        return finish_cells(prepared, None)
    cells.optimise(**prepared.payload["settings"], progress_callback=progress_callback)
    return finish_cells(prepared, cells)
