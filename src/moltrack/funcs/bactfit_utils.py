import numpy as np

from moltrack.funcs.compute_utils import Worker
from moltrack.bactfit.fit import BactFit
from functools import partial


class _bactfit_utils:

    def run_bactfit_finished(self):
        self.update_ui()

    def run_bactfit_results(self, results):

        fitted_cells = []

        for result in results:
            seg = result["cell_fit"]
            seg = seg[1:]
            fitted_cells.append(seg)

        self.viewer.add_shapes(fitted_cells,
            shape_type="polygon", name="fitted_cells")


    def run_bactfit(self, segmentations, progress_callback=None):

        fit_data = None

        if segmentations[0].shape[1] == 2:

            bf = BactFit()

            fit_data = bf.fit_cell_contours(segmentations,
                fit=True, parallel=False, progress_callback=progress_callback)

        else:
            print("3D")

        return fit_data

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






