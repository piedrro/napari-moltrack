import numpy as np
import traceback
from moltrack.funcs.compute_utils import Worker
from bactfit.preprocess import data_to_cells
from functools import partial
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from bactfit.cell import CellList
from napari.utils.notifications import show_info

class _bactfit_utils:

    def run_bactfit_finished(self):

        self.update_ui()
        self.update_segmentation_combos()

    def run_bactfit_results(self, celllist):

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


    def run_bactfit(self, segmentations, progress_callback=None):

        try:

            min_radius = float(self.gui.fit_min_radius.value())
            max_radius = float(self.gui.fit_max_radius.value())
            max_error = float(self.gui.fit_max_error.value())

            show_info(f"Building CellList")

            celllist = data_to_cells(segmentations)

            n_cells = len(celllist.data)

            if n_cells == 0:
                return None

            show_info(f"BactFit Fitting {n_cells} cells")

            celllist.optimise(max_radius=max_radius, min_radius=min_radius,
                max_error=max_error, progress_callback=progress_callback)

            error_list = [cell.fit_error for cell in celllist.data]
            error_list = [error for error in error_list if error is not None]
            print(f"Max error: {max(error_list)}")

        except:
            print(traceback.format_exc())
            return None

        return celllist

    def initialise_bactfit(self):

        if hasattr(self, "segLayer"):

            segmentations = self.segLayer.data

            if len(segmentations) == 0:
                return

            self.update_ui(init=True)

            worker = Worker(self.run_bactfit, segmentations)
            worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.bactfit_progressbar))
            worker.signals.result.connect(self.run_bactfit_results)
            worker.signals.finished.connect(self.run_bactfit_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.threadpool.start(worker)






