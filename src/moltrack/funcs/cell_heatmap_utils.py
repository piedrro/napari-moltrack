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
from PyQt5.QtWidgets import QFileDialog
import os

from moltrack.funcs.compute_utils import Worker


class CustomPyQTGraphWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None


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

    def compute_cell_heatmap(self, viewer=None, model_length_um=5, model_width_um=2):

        try:

            data_type = self.gui.heatmap_data.currentText()

            datasets = self.dataset_dict.keys()

            pixel_size_nm = list(set([self.dataset_dict[dataset]["pixel_size"] for dataset in datasets]))
            pixel_size_um = pixel_size_nm[0] / 1000

            model_length = model_length_um / pixel_size_um
            model_width = (model_width_um / pixel_size_um)/2

            if data_type.lower() == "localisations":
                locs = self.get_locs("All Datasets", "All Channels")
            elif data_type.lower() == "tracks":
                locs = self.get_tracks("All Datasets", "All Channels")

            if len(locs) == 0:
                return
            if hasattr(self, "cellLayer") == False:
                return

            celllist = self.populate_celllist()

            celllist.add_localisations(locs)
            model = ModelCell(length=model_length, width=model_width)

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
                pad_inches=0, facecolor='black', dpi=500)
            buf.seek(0)
            heatmap = plt.imread(buf)

            # Close the figure
            plt.close(fig)

            self.heatmap_image = heatmap

            heatmap = np.rot90(heatmap, k=3)
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
                pad_inches=0, facecolor='black', dpi=500)
            buf.seek(0)
            image = plt.imread(buf)
            plt.close(fig)

            self.heatmap_image = image

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

        try:

            if hasattr(self, "heatmap_image") == False:
                return

            if self.heatmap_image is None:
                return

            image = self.heatmap_image.copy()

            mode = self.gui.heatmap_mode.currentText()

            dataset_list = list(self.dataset_dict.keys())
            path = self.dataset_dict[dataset_list[0]]["path"]

            directory = os.path.dirname(path)
            file_name = os.path.basename(path)
            base, ext = os.path.splitext(file_name)
            path = os.path.join(directory, base + f"_cell_{mode.lower()}" + ".png")
            path = QFileDialog.getSaveFileName(self, "Save Image", path, "PNG (*.png,*.tif)")[0]

            if path == "":
                return

            plt.imsave(path, image, cmap='inferno')

            print(f"Exported cell {mode.lower()} to {path}")

        except:
            print(traceback.format_exc())