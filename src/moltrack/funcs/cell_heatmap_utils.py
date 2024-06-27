import pandas as pd
import traceback
import numpy as np
from functools import partial
from moltrack.bactfit.preprocess import data_to_cells
from moltrack.bactfit.cell import CellList, ModelCell
from moltrack.bactfit.postprocess import remove_locs_outside_cell
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyqtgraph as pg
from io import BytesIO
from picasso.render import render
from PyQt5.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QVBoxLayout, QWidget, QMainWindow, QSpinBox
from PyQt5.QtWidgets import QFileDialog
import os
import cv2
from shapely.geometry import Polygon
from matplotlib.colors import ListedColormap
from napari.utils.notifications import show_info
from moltrack.funcs.compute_utils import Worker
from napari.utils.notifications import show_info

class CustomPyQTGraphWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None


class _cell_heatmap_utils:

    def update_render_length_range(self):

        try:

            if hasattr(self, "celllist") == False:
                return

            if self.celllist is None:
                return

            cell_lengths = self.celllist.get_cell_lengths()

            if len(cell_lengths) > 0:
                min_length = min(cell_lengths)
                max_length = max(cell_lengths)
                self.gui.heatmap_min_length.setRange(min_length, max_length)
                self.gui.heatmap_max_length.setRange(min_length, max_length)
                self.gui.heatmap_min_length.setValue(min_length)
                self.gui.heatmap_max_length.setValue(max_length)

        except:
            print(traceback.format_exc())
            pass

    def update_render_msd_range(self):

        try:

            self.gui.heatmap_min_msd.setRange(0, 0)
            self.gui.heatmap_max_msd.setRange(0, 0)
            self.gui.heatmap_min_msd.setValue(0)
            self.gui.heatmap_max_msd.setValue(0)
            self.gui.heatmap_min_msd.setEnabled(False)
            self.gui.heatmap_max_msd.setEnabled(False)

            dataset = self.gui.heatmap_dataset.currentText()
            channel = self.gui.heatmap_channel.currentText()

            if hasattr(self, "tracking_dict") == False:
                return

            if self.tracking_dict is None:
                return

            tracks = self.get_tracks(dataset, channel)
            tracks = pd.DataFrame(tracks)

            if "msd" not in tracks.columns:
                return

            if len(tracks)== 0:
                return

            self.gui.heatmap_min_msd.setEnabled(True)
            self.gui.heatmap_max_msd.setEnabled(True)

            msd = tracks["msd"].tolist()
            msd = [x for x in msd if x > 0]

            min_msd = min(msd)
            max_msd = max(msd)

            self.gui.heatmap_min_msd.setRange(min_msd, max_msd)
            self.gui.heatmap_max_msd.setRange(min_msd, max_msd)
            self.gui.heatmap_min_msd.setValue(min_msd)
            self.gui.heatmap_max_msd.setValue(max_msd)

        except:
            print(traceback.format_exc())
            pass


    def cell_heatmap_compute_finished(self):

        try:

            self.update_render_length_range()
            self.update_render_msd_range()
            self.heatmap_canvas.clear()
            self.plot_heatmap()
            self.update_ui()
            show_info("Cell heatmap computed.")

        except:
            print(traceback.format_exc())
            pass

    def cell_heatmap_compute(self, celllist, model,
            progress_callback=None):

        try:

            celllist.transform_locs(model,
                progress_callback=progress_callback)

            self.celllist = celllist

        except:
            print(traceback.format_exc())
            pass

    def populate_celllist(self):

        try:

            name_list = self.cellLayer.properties["name"].copy()
            name_list = list(set(name_list))
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

            # self.update_ui(init=True)

            celllist = self.populate_celllist()
            n_cells = len(celllist.data)

            celllist.add_localisations(locs)
            model = ModelCell(length=model_length, width=model_width)

            show_info(f"Computing cell heatmap for {n_cells} cells")

            worker = Worker(self.cell_heatmap_compute, celllist, model)
            worker.signals.finished.connect(self.cell_heatmap_compute_finished)
            worker.signals.progress.connect(partial(self.moltrack_progress,
                progress_bar=self.gui.heatmap_progressbar, ))
            self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()


    def plot_heatmap(self):

        try:

            self.update_ui(init=True)

            heatmap_datset = self.gui.heatmap_dataset.currentText()
            heatmap_channel = self.gui.heatmap_channel.currentText()
            heatmap_mode = self.gui.heatmap_mode.currentText()
            colourmap_name = self.gui.heatmap_colourmap.currentText()
            draw_outline = self.gui.render_draw_outline.isChecked()
            min_length = self.gui.heatmap_min_length.value()
            max_length = self.gui.heatmap_max_length.value()
            min_msd = self.gui.heatmap_min_msd.value()
            max_msd = self.gui.heatmap_max_msd.value()
            symmetry = self.gui.render_symmetry.isChecked()

            self.heatmap_canvas.clear()

            if hasattr(self, "celllist") == False:
                return
            if self.celllist is None:
                return

            celllist = self.celllist
            celllist = celllist.filter_by_length(min_length, max_length)

            if len(celllist.data) == 0:
                return

            polygon = celllist.data[0].cell_polygon
            polygon_coords = np.array(polygon.exterior.coords)

            celllocs = celllist.get_locs(symmetry=symmetry)
            celllocs = remove_locs_outside_cell(celllocs, polygon)
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

            if "msd" in celllocs.dtype.names:
                celllocs = celllocs[celllocs["msd"] > min_msd]
                celllocs = celllocs[celllocs["msd"] < max_msd]

            n_cells = len(celllist.data)
            n_locs = len(celllocs)
            if symmetry:
                n_locs = int(n_locs/4)

            show_info(f"Generating Cell {heatmap_mode.lower()} with {n_locs} localisations from {n_cells} cells")

            if heatmap_mode == "Heatmap":
                self.plot_cell_heatmap(celllocs, polygon_coords,
                    colourmap_name, draw_outline)
            elif heatmap_mode == "Render":
                self.plot_cell_render(celllocs, polygon_coords,
                    colourmap_name, draw_outline)
            else:
                pass

            self.update_ui()

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass


    def plot_cell_heatmap(self, celllocs, polygon_coords,
            colourmap_name="inferno", draw_outline=True):

        try:

            bins = self.heatmap_binning.value()

            heatmap, xedges, yedges = np.histogram2d(celllocs["x"], celllocs["y"], bins=bins)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            cmap = self.get_custom_cmap(colour=colourmap_name)

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap)
            if draw_outline:
                ax.plot(*polygon_coords.T, color='white', linewidth=1)
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


    def get_custom_cmap(self, colour = "jet"):

        cmap = plt.get_cmap(colour.lower())
        new_cmap = cmap(np.arange(cmap.N))

        new_cmap[0] = [0, 0, 0, 1]

        new_cmap = ListedColormap(new_cmap)

        return new_cmap


    def plot_cell_render(self, celllocs, polygon_coords,
            colourmap_name="inferno", draw_outline=True):

        try:

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

            xmin, xmax = polygon_coords[:, 0].min(), polygon_coords[:, 0].max()
            ymin, ymax = polygon_coords[:, 1].min(), polygon_coords[:, 1].max()

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

            #stretch polygon to image size
            polygon_coords = np.array(polygon_coords)
            polygon_coords = polygon_coords * oversampling

            cmap = self.get_custom_cmap(colour=colourmap_name)

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            ax.imshow(image, cmap=cmap)
            if draw_outline:
                ax.plot(*polygon_coords.T, color='white')
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

            if type(path) == list:
                path = path[0]

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

