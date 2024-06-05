import numpy as np
import traceback
from moltrack.funcs.compute_utils import Worker
from moltrack.bactfit.fit import BactFit
from moltrack.bactfit.preprocess import data_to_cells
from functools import partial


class _bactfit_utils:

    def run_bactfit_finished(self):
        self.update_ui()

    def run_bactfit_results(self, cell_list):

        if cell_list is None:
            return

        fitted_cells, cell_names = cell_list.get_segmentations()

        layer_names = [layer.name for layer in self.viewer.layers]

        if "fitted_cells" in layer_names:
            self.viewer.layers.remove("fitted_cells")

        properties = {"name": cell_names,}

        self.viewer.add_shapes(fitted_cells,
            shape_type="polygon", name="fitted_cells",
            properties=properties)


    def run_bactfit(self, segmentations, progress_callback=None):

        try:

            cell_list = data_to_cells(segmentations)

            cell_list.optimise(refine_fit=True, parallel=True,
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






