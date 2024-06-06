import numpy as np
import traceback
from moltrack.funcs.compute_utils import Worker
from moltrack.bactfit.fit import BactFit
from moltrack.bactfit.preprocess import data_to_cells
from moltrack.bactfit.fit import BactFit
from functools import partial
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt

class _bactfit_utils:

    def run_bactfit_finished(self):
        self.update_ui()

    def run_bactfit_results(self, cell_list):

        if cell_list is None:
            return

        data = cell_list.get_cell_fits()

        cell_names = data["names"]
        cell_fits = data["fits"]
        cell_widths = data["widths"]
        cell_midlines = data["midlines"]

        layer_names = [layer.name for layer in self.viewer.layers]

        if "fitted_cells" in layer_names:
            self.viewer.layers.remove("fitted_cells")

        shapes = []
        shape_types = []
        properties = {"name": [], "width": []}

        for name, fit, width, midline in zip(cell_names, cell_fits,
                cell_widths, cell_midlines):

            try:

                shapes.append(fit)
                shape_types.append("polygon")
                properties["name"].append(name)
                properties["width"].append(width)

                shapes.append(midline)
                shape_types.append("path")
                properties["name"].append(name)
                properties["width"].append(width)

            except:
                pass

        self.cellLayer = self.initialise_cellLayer(shapes=shapes,
            shape_types=shape_types, properties=properties)

        self.store_cell_shapes()


    def run_bactfit(self, segmentations, progress_callback=None):

        try:

            min_radius = float(self.gui.fit_min_radius.value())
            max_radius = float(self.gui.fit_max_radius.value())

            cell_list = data_to_cells(segmentations)

            cell_list.optimise(refine_fit=True, parallel=True,
                min_radius=min_radius, max_radius=max_radius,
                progress_callback=progress_callback)

        except:
            print(traceback.format_exc())
            return None

        return cell_list

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






