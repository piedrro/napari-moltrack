import pandas as pd
import traceback
import numpy as np
from functools import partial
from moltrack.bactfit.preprocess import data_to_cells
from moltrack.bactfit.cell import CellList, ModelCell
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyqtgraph as pg
from io import BytesIO
from picasso.render import render
from PyQt5.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QVBoxLayout, QWidget, QMainWindow, QSpinBox

from moltrack.funcs.compute_utils import Worker


class CustomPyQTGraphWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None


class _cell_heatmap_utils:

    def initialise_heatmap_controls(self):

        try:

            self.heatmap_binning = QSpinBox()
            self.heatmap_blur_method = QComboBox()
            self.heatmap_min_blur_width = QDoubleSpinBox()
            self.heatmap_oversampling = QSpinBox()

            self.heatmap_binning.setRange(1, 100)
            self.heatmap_binning.setSingleStep(1)
            self.heatmap_binning.setValue(30)

            self.heatmap_min_blur_width.setRange(0.1, 10)
            self.heatmap_min_blur_width.setSingleStep(0.1)
            self.heatmap_min_blur_width.setValue(0.2)

            self.heatmap_oversampling.setRange(1, 100)
            self.heatmap_oversampling.setSingleStep(1)
            self.heatmap_oversampling.setValue(20)

            blur_methods = ["One-Pixel-Blur", "Global Localisation Precision",
                            "Individual Localisation Precision, iso",
                            "Individual Localisation Precision"]

            self.heatmap_blur_method.clear()
            self.heatmap_blur_method.addItems(blur_methods)

            self.heatmap_binning.blockSignals(True)
            self.heatmap_blur_method.blockSignals(True)
            self.heatmap_min_blur_width.blockSignals(True)
            self.heatmap_oversampling.blockSignals(True)

            self.heatmap_binning.valueChanged.connect(self.plot_heatmap)
            self.heatmap_blur_method.currentIndexChanged.connect(self.plot_heatmap)
            self.heatmap_min_blur_width.valueChanged.connect(self.plot_heatmap)
            self.heatmap_oversampling.valueChanged.connect(self.plot_heatmap)

            self.heatmap_binning.blockSignals(False)
            self.heatmap_blur_method.blockSignals(False)
            self.heatmap_min_blur_width.blockSignals(False)
            self.heatmap_oversampling.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass

    def update_heatmap_options(self):

        mode = self.gui.heatmap_mode.currentText()
        layout = self.gui.heatmap_settings_layout

        core_settings = ["heatmap_dataset", "heatmap_channel", "heatmap_mode",
                         "heatmap_dataset_label", "heatmap_channel_label", "heatmap_mode_label"]

        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            widget_name = widget.objectName()
            if widget_name.lower() not in core_settings:
                if widget is not None:
                    widget.deleteLater()

        self.initialise_heatmap_controls()

        if mode == "Heatmap":
            layout.addRow("Binning", self.heatmap_binning)
        elif mode == "Render":
            layout.addRow("Blur Method", self.heatmap_blur_method)
            layout.addRow("Min Blur Width", self.heatmap_min_blur_width)
            layout.addRow("Oversampling", self.heatmap_oversampling)


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

            dataset = "All Datasets"
            channel = "All Channels"

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

            heatmap_datset = self.gui.heatmap_dataset.currentText()
            heatmap_channel = self.gui.heatmap_channel.currentText()
            heatmap_mode = self.gui.heatmap_mode.currentText()

            self.heatmap_canvas.clear()

            if hasattr(self, "celllist") == False:
                return
            if self.celllist is None:
                return

            celllocs = self.celllist.get_locs()

            celllocs = pd.DataFrame(celllocs)

            if "dataset" in celllocs.columns:
                if heatmap_datset != "All Datasets":
                    celllocs = celllocs[celllocs["dataset"] == heatmap_datset]
            if "channel" in celllocs.columns:
                if heatmap_channel != "All Channels":
                    celllocs = celllocs[celllocs["channel"] == heatmap_channel]

            celllocs = celllocs.to_records(index=False)

            if len(celllocs) == 0:
                return

            print(f"Plotting {len(celllocs)} localisations...")

            if heatmap_mode == "Heatmap":
                self.plot_cell_heatmap(celllocs)
            elif heatmap_mode == "Render":
                self.plot_cell_render(celllocs)
            else:
                pass

        except:
            print(traceback.format_exc())
            pass



    def plot_cell_heatmap(self, celllocs):

        try:

            print("Generating heatmap...")

            bins = self.heatmap_binning.value()

            heatmap, xedges, yedges = np.histogram2d(celllocs["x"], celllocs["y"], bins=bins)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='inferno')
            ax.axis('off')

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)
            cax.set_facecolor('black')

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight',
                pad_inches=0, facecolor='black', dpi=300)
            buf.seek(0)
            heatmap = plt.imread(buf)

            # Close the figure
            plt.close(fig)

            #rotate heatmap -90
            heatmap = np.rot90(heatmap, k=3)
            #flip heatmap
            heatmap = np.fliplr(heatmap)

            self.heatmap_canvas.clear()
            self.heatmap_canvas.setImage(heatmap)

            self.heatmap_canvas.ui.histogram.hide()
            self.heatmap_canvas.ui.roiBtn.hide()
            self.heatmap_canvas.ui.menuBtn.hide()

        except:
            print(traceback.format_exc())
            pass

    def plot_cell_render(self, celllocs):

        try:

            print("Generating render...")

            blur_method = self.heatmap_blur_method.currentText()
            min_blur_width = self.heatmap_min_blur_width.value()
            oversampling = self.heatmap_oversampling.value()

            celllocs = pd.DataFrame(celllocs)

            picasso_columns = ["frame",
                               "y", "x",
                               "photons", "sx", "sy", "bg",
                               "lpx", "lpy",
                               "ellipticity", "net_gradient",
                               "group", "iterations", ]

            column_filter = [col for col in picasso_columns if col in celllocs.columns]
            celllocs = celllocs[column_filter]
            celllocs = celllocs.to_records(index=False)

            xmin, xmax = celllocs["x"].min(), celllocs["x"].max()
            ymin, ymax = celllocs["y"].min(), celllocs["y"].max()

            h,w = int(ymax-ymin)+3, int(xmax-xmin)+3

            viewport = [(float(0), float(0)), (float(h), float(w))]

            if blur_method == "One-Pixel-Blur":
                blur_method = "smooth"
            elif blur_method == "Global Localisation Precision":
                blur_method = "convolve"
            elif blur_method == "Individual Localisation Precision, iso":
                blur_method = "gaussian_iso"
            elif blur_method == "Individual Localisation Precision":
                blur_method = "gaussian"
            else:
                blur_method = None

            n_rendered_locs, image = render(celllocs,
                viewport=viewport,
                blur_method=blur_method,
                min_blur_width=min_blur_width,
                oversampling=oversampling, ang=0, )

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='inferno')
            ax.axis('off')

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight',
                pad_inches=0, facecolor='black', dpi=300)
            buf.seek(0)
            image = plt.imread(buf)
            plt.close(fig)

            #rotate and flip
            image = np.rot90(image, k=3)
            image = np.fliplr(image)

            self.heatmap_canvas.clear()
            self.heatmap_canvas.setImage(image)

            self.heatmap_canvas.ui.histogram.hide()
            self.heatmap_canvas.ui.roiBtn.hide()
            self.heatmap_canvas.ui.menuBtn.hide()

        except:
            print(traceback.format_exc())
            pass



    def export_cell_heatmap(self):

        pass