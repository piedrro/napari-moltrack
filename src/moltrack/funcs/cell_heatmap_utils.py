import os
import traceback
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import yaml
from bactfit.cell import CellList, ModelCell
from napari.utils.notifications import show_info
from PyQt5.QtWidgets import QFileDialog

from moltrack.funcs.compute_utils import Worker


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

    def cell_heatmap_compute(self, celllist, model,method,
            progress_callback=None):

        try:

            celllist.transform_cells(model,method = method,
                progress_callback=progress_callback)

            self.celllist = celllist

        except:
            print(traceback.format_exc())

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

        return cells


    def compute_cell_heatmap(self, viewer=None, model_length_um=5, model_radius_um=2):

        try:

            data_type = self.gui.heatmap_data.currentText()
            compute_method = self.gui.heatmap_compute_method.currentText()
            datasets = self.dataset_dict.keys()

            if "angular" in compute_method.lower():
                method = "angular"
            else:
                method = "perpendicular"

            pixel_size_nm = list(set([self.dataset_dict[dataset]["pixel_size"] for dataset in datasets]))
            pixel_size_um = pixel_size_nm[0] / 1000

            model_length = model_length_um / pixel_size_um
            model_radius = (model_radius_um / pixel_size_um) / 2

            if data_type.lower() == "localisations":
                locs = self.get_locs("All Datasets", "All Channels")
            elif data_type.lower() == "tracks":
                locs = self.get_tracks("All Datasets", "All Channels")

            if len(locs) == 0:
                return
            if hasattr(self, "cellLayer") == False:
                show_info("Cells must be fitted with BactFit before computing cell heatmap")
                return

            self.update_ui(init=True)

            show_info("Polpulating BactFit CellList")

            celllist = self.populate_celllist()
            n_cells = len(celllist.data)

            show_info("Adding localisations to CellList")

            celllist.add_localisations(locs)
            model = ModelCell(length=model_length, radius=model_radius)

            show_info(f"Computing cell heatmap for {n_cells} cells")

            worker = Worker(self.cell_heatmap_compute, celllist, model, method)
            worker.signals.finished.connect(self.cell_heatmap_compute_finished)
            worker.signals.progress.connect(partial(self.moltrack_progress,
                progress_bar=self.gui.heatmap_progressbar, ))
            worker.signals.error.connect(self.update_ui)
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
            bins = self.heatmap_binning.value()
            blur_method = self.heatmap_blur_method.currentText()
            min_blur_width = self.heatmap_min_blur_width.value()
            oversampling = self.heatmap_oversampling.value()

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

            self.heatmap_locs = celllocs
            self.heatmap_polygon = polygon_coords

            show_info(f"Generating Cell {heatmap_mode.lower()} with {n_locs} localisations from {n_cells} cells")

            if heatmap_mode == "Heatmap":

                heatmap = celllist.plot_heatmap(symmetry=symmetry, bins=bins,
                    cmap=colourmap_name, draw_outline=draw_outline,
                    show=False, save=False, path=None, dpi=500)

                self.heatmap_image = heatmap
                self.show_heatmap(heatmap)

            elif heatmap_mode == "Render":

                render = celllist.plot_render(symmetry=symmetry, oversampling=oversampling,
                    blur_method=blur_method, min_blur_width=min_blur_width,
                    cmap=colourmap_name, draw_outline=draw_outline,
                    show=False, save=False, path=None, dpi=500)

                self.heatmap_image = render
                self.show_heatmap(render)

            else:
                pass

            self.update_ui()

        except:
            self.update_ui()
            print(traceback.format_exc())


    def show_heatmap(self, image):

        try:

            image = np.rot90(image, k=3)
            image = np.fliplr(image)

            self.heatmap_canvas.clear()
            self.heatmap_canvas.setImage(image)

            self.heatmap_canvas.ui.histogram.hide()
            self.heatmap_canvas.ui.roiBtn.hide()
            self.heatmap_canvas.ui.menuBtn.hide()

        except:
            print(traceback.format_exc())


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


    def export_heatmap_locs(self):

        try:

            self.update_ui(init=True)

            if hasattr(self, "heatmap_locs") == False:
                show_info("No heatmap localisations to export")
                return

            locs = self.heatmap_locs
            polygon_coords = self.heatmap_polygon

            if len(locs) == 0:
                show_info("No heatmap localisations to export")
                return

            dataset_list = list(self.dataset_dict.keys())
            path = self.dataset_dict[dataset_list[0]]["path"]

            if type(path) == list:
                path = path[0]

            directory = os.path.dirname(path)
            file_name = os.path.basename(path)
            base, ext = os.path.splitext(file_name)
            path = os.path.join(directory, base + "_heatmap_locs.csv")
            options = QFileDialog.Options()
            file_filter = "CSV (*.csv);;Picasso HDF5 (*.hdf5);; POS.OUT (*.pos.out)"
            path, filter = QFileDialog.getSaveFileName(self, "Save Image", path, file_filter, options=options)

            if path == "":
                return None

            if filter == "CSV (*.csv)":

                locs = pd.DataFrame(locs)
                locs.to_csv(path, index=False)

                show_info("Exported heatmap CSV localisations")

            elif filter == "Picasso HDF5 (*.hdf5)":

                xmin, xmax = polygon_coords[:, 0].min(), polygon_coords[:, 0].max()
                ymin, ymax = polygon_coords[:, 1].min(), polygon_coords[:, 1].max()

                h,w = int(ymax-ymin)+3, int(xmax-xmin)+3

                image_shape = (0,h,w)

                locs = pd.DataFrame(locs)

                dataset_name = locs["dataset"].unique()[0]
                channel_name = locs["channel"].unique()[0]

                picasso_columns = ["frame", "y", "x", "photons",
                                   "sx", "sy", "bg", "lpx", "lpy",
                                   "ellipticity", "net_gradient", "group", "iterations", ]

                for column in locs.columns:
                    if column not in picasso_columns:
                        locs.drop(column, axis=1, inplace=True)

                locs = locs.to_records(index=False)

                import_path = self.dataset_dict[dataset_name]["path"]

                box_size = int(self.gui.picasso_box_size.value())
                picasso_info = self.get_picasso_info(import_path, image_shape, box_size)

                info_path = path.replace(".hdf5", ".yaml")

                with h5py.File(path, "w") as hdf_file:
                    hdf_file.create_dataset("locs", data=locs)

                # Save to temporary YAML file
                with open(info_path, "w") as file:
                    yaml.dump_all(picasso_info, file, default_flow_style=False)

                show_info("Exported heatmap HDF5 localisations")

            elif filter == "POS.OUT (*.pos.out)":

                localisation_data = pd.DataFrame(locs)

                pos_locs = localisation_data[["frame", "x", "y", "photons", "bg", "sx", "sy", ]].copy()

                pos_locs.dropna(axis=0, inplace=True)

                pos_locs.columns = ["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "S_X", "S_Y", ]

                pos_locs.loc[:, "I0"] = 0
                pos_locs.loc[:, "THETA"] = 0
                pos_locs.loc[:, "ECC"] = pos_locs["S_X"] / pos_locs["S_Y"]
                pos_locs.loc[:, "FRAME"] = pos_locs["FRAME"] + 1

                pos_locs = pos_locs[["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "I0", "S_X", "S_Y", "THETA", "ECC", ]]

                pos_locs.to_csv(path, sep="\t", index=False)

                show_info("Exported heatmap POS.OUT localisations")

            else:
                print("File format not supported")

        except:
            print(traceback.format_exc())
            self.update_ui()

        self.update_ui()
