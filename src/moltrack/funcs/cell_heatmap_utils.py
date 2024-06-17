import pandas as pd
import traceback
import numpy as np
from functools import partial
from moltrack.bactfit.preprocess import data_to_cells
from moltrack.bactfit.cell import CellList, ModelCell
import matplotlib.pyplot as plt
import pyqtgraph as pg

from moltrack.funcs.compute_utils import Worker

class _cell_heatmap_utils:


    def cell_heatmap_compute_finished(self):

        try:

            self.plot_heatmap()
            self.update_ui()
            print("Cell heatmap computed.")

        except:
            print(traceback.format_exc())
            pass

    def cell_heatmap_compute(self, celllist, model,
            progress_callback=None):

        try:

            print("Computing cell heatmap...")

            celllist.transform_locs(model,
                progress_callback=progress_callback)

            self.celllist = celllist

        except:
            print(traceback.format_exc())
            pass

    def populate_celllist(self):

        try:

            name_list = self.cellLayer.properties["name"].copy()
            cell_list = []

            for name in name_list:
                cell = self.get_cell(name, bactfit=True)
                cell_list.append(cell)

            if len(cell_list) == 0:
                return

            cells = CellList(cell_list)

        except:
            print(traceback.format_exc())
            pass

        return cells





    def compute_cell_heatmap(self):

        try:

            dataset = self.gui.heatmap_dataset.currentText()
            channel = self.gui.heatmap_channel.currentText()

            locs = self.get_locs(dataset, channel)

            if len(locs) == 0:
                return
            if hasattr(self, "cellLayer") == False:
                return

            celllist = self.populate_celllist()

            celllist.add_localisations(locs)
            model = ModelCell(length=10, width=5)

            self.update_ui(init=True)

            worker = Worker(self.cell_heatmap_compute, celllist, model)
            worker.signals.finished.connect(self.cell_heatmap_compute_finished)
            worker.signals.progress.connect(partial(self.moltrack_progress, progress_bar=self.gui.heatmap_progressbar, ))
            self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()


    def plot_heatmap(self):

        try:

            if hasattr(self, "celllist") == False:
                return
            if self.celllist is None:
                return

            celllocs = self.celllist.get_locs()

            if len(celllocs) == 0:
                return

            self.heatmap_graph_canvas.clear()

            # create heatmap
            heatmap, xedges, yedges = np.histogram2d(celllocs["x"], celllocs["y"], bins=30, density=False)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            image_item = pg.ImageItem(heatmap.T)
            image_item.setRect(pg.QtCore.QRectF(extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2]))
            self.heatmap_graph_canvas.addItem(image_item)
            self.heatmap_graph_canvas.setAspectLocked(True)
            self.heatmap_graph_canvas.setTitle('Heatmap of Cell Locations')







        except:
            print(traceback.format_exc())
            pass


    def export_cell_heatmap(self):

        pass