import matplotlib.pyplot as plt
import tifffile

from moltrack.funcs.compute_utils import Worker
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import concurrent
import os
import h5py
import yaml
import tempfile
import shutil
from pathlib import Path
import traceback
import numpy as np
from qtpy.QtWidgets import QFileDialog
import pandas as pd
import cv2
import json
from moltrack.funcs.compute_utils import Worker
from napari.utils.notifications import show_info
import pickle

def format_picasso_path(path):
    if "%" in str(path):
        path = path.replace("%", "%%")

    path = os.path.normpath(path)

    if os.name == "nt":
        if path.startswith("\\\\"):
            path = "\\\\?\\UNC\\" + path[2:]

    return Path(path)


def initialise_data_export(loc_data):
    try:
        export_mode = loc_data["export_mode"]

        if export_mode == "Picasso HDF5":
            export_picasso_localisation(loc_data)
        else:
            export_localisation_data(loc_data)

    except:
        print(traceback.format_exc())
        pass


def export_localisation_data(loc_data):
    try:
        export_mode = loc_data["export_mode"]
        export_path = loc_data["export_path"]
        locs = loc_data["data"]

        if export_mode == "CSV":
            df = pd.DataFrame(locs)

            df.to_csv(export_path, index=False)

        elif export_mode == "POS.OUT":
            localisation_data = pd.DataFrame(locs)

            pos_locs = localisation_data[["frame", "x", "y", "photons", "bg", "sx", "sy", ]].copy()

            pos_locs.dropna(axis=0, inplace=True)

            pos_locs.columns = ["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "S_X", "S_Y", ]

            pos_locs.loc[:, "I0"] = 0
            pos_locs.loc[:, "THETA"] = 0
            pos_locs.loc[:, "ECC"] = pos_locs["S_X"] / pos_locs["S_Y"]
            pos_locs.loc[:, "FRAME"] = pos_locs["FRAME"] + 1

            pos_locs = pos_locs[["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "I0", "S_X", "S_Y", "THETA", "ECC", ]]

            pos_locs.to_csv(export_path, sep="\t", index=False)

    except:
        print(traceback.format_exc())


def export_picasso_localisation(loc_data):
    try:
        locs = loc_data["data"]

        locs = pd.DataFrame(locs)

        picasso_columns = ["frame", "y", "x", "photons", "sx", "sy", "bg", "lpx", "lpy", "ellipticity", "net_gradient", "group", "iterations", ]

        for column in locs.columns:
            if column not in picasso_columns:
                locs.drop(column, axis=1, inplace=True)

        locs = locs.to_records(index=False)

        h5py_path = loc_data["hdf5_path"]
        yaml_path = loc_data["info_path"]
        info = loc_data["picasso_info"]

        h5py_path = format_picasso_path(h5py_path)
        yaml_path = format_picasso_path(yaml_path)

        # Create temporary files
        temp_h5py_path = tempfile.NamedTemporaryFile(delete=False).name
        temp_yaml_path = tempfile.NamedTemporaryFile(delete=False).name

        h5py_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to temporary HDF5 file
        with h5py.File(temp_h5py_path, "w") as hdf_file:
            hdf_file.create_dataset("locs", data=locs)

        # Save to temporary YAML file
        with open(temp_yaml_path, "w") as file:
            yaml.dump_all(info, file, default_flow_style=False)

        try:
            shutil.move(temp_h5py_path, h5py_path)
            shutil.move(temp_yaml_path, yaml_path)
        except:
            show_info("Could not move files to import directory. Saving to desktop instead.")

            desktop_dir = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")

            desktop_h5py_path = os.path.join(desktop_dir, h5py_path.name)
            desktop_yaml_path = os.path.join(desktop_dir, yaml_path.name)

            shutil.move(temp_h5py_path, desktop_h5py_path)
            shutil.move(temp_yaml_path, desktop_yaml_path)

    except Exception as e:
        print(traceback.format_exc())


class _export_utils:

    def get_export_shapes_path(self, mode="Binary Mask"):
        export_path = None
        export_dir = None
        file_name = None

        for dataset in self.dataset_dict.keys():
            if "path" in self.dataset_dict[dataset]:
                path = self.dataset_dict[dataset]["path"]

                if type(path) == list:
                    path = path[0]

                export_dir = os.path.dirname(path)
                file_name, ext = os.path.splitext(path)
                file_name = dataset

        if export_dir is None:
            export_dir = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")

        if file_name is None:
            file_name = "moltrack"

        if mode == "Binary Mask":
            file_name = file_name + "_mask.tif"
            ext = ".tif"
        elif mode == "JSON":
            file_name = file_name + "_shapes.json"
            ext = ".json"
        elif mode == "Oufti/MicrobTracker Mesh":
            file_name = file_name + "_mesh.mat"
            ext = ".mat"
        else:
            return None

        export_path = os.path.join(export_dir, file_name)

        if os.path.exists(export_dir):
            export_path = QFileDialog.getSaveFileName(self, f"Export {mode}", export_path, f"(*{ext})")[0]

        if export_path == "":
            return None

        export_dir = os.path.dirname(export_path)

        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        return export_path

    def get_export_mask_shape(self):
        frame_shape = None

        try:
            shape_layers = ["segmentations", "cells", "localisations"]
            image_layers = [layer for layer in self.viewer.layers if layer.name.lower() not in shape_layers]
            frame_shapes = list(set([layer.data.shape[-2:] for layer in image_layers]))

            if len(frame_shapes) > 0:
                frame_shape = frame_shapes[0]

        except:
            pass

        return frame_shape

    def export_shapes_mask(self, path):
        try:
            export_polygons = self.get_export_polygons()

            frame_shape = self.get_export_mask_shape()

            if frame_shape is None:
                show_info("Could not determine mask shape for export.")
                return

            mask = np.zeros(frame_shape, dtype=np.uint8)

            contours = [np.array(polygon).reshape(-1, 1, 2) for polygon in export_polygons]
            contours = [contour[:, :, ::-1] for contour in contours]
            contours = [np.round(contour).astype(np.int32) for contour in contours]

            for contour_index, contour in enumerate(contours):
                colour = contour_index + 1
                cv2.drawContours(mask, [contour], -1, colour, -1)

            if path is None:
                show_info("No path provided for export.")
                return

            tifffile.imwrite(path, mask)

        except:
            print(traceback.format_exc())
            pass

    def export_shapes_json(self, path):
        try:
            export_polygons = self.get_export_polygons()

            json_keys = export_polygons[0].keys()

            json_dict = {}

            for key in json_keys:
                json_dict[key] = []

            for polygon in export_polygons:
                for key in json_keys:
                    json_dict[key].append(polygon[key])

            if path is None:
                show_info("No path provided for export.")
                return

            with open(path, "w") as f:
                json.dump(json_dict, f)

        except:
            print(traceback.format_exc())
            pass

    def export_segmentations(self, export_mode, path=None):
        if export_mode == "Binary Mask":
            self.export_shapes_mask(path)

        elif export_mode == "JSON":
            self.export_shapes_json(path)

    def export_cells(self, export_mode, path=None):
        if export_mode == "Binary Mask":
            self.export_shapes_mask(path)

        elif export_mode == "JSON":
            self.export_shapes_json(path)

        elif export_mode == "Oufti/MicrobTracker Mesh":
            self.export_mesh(path)

    def get_export_polygons(self):

        export_data = self.gui.shapes_export_data.currentText()
        export_mode = self.gui.shapes_export_mode.currentText()

        shape_layer = self.viewer.layers[export_data]

        shapes = shape_layer.data.copy()
        shape_types = shape_layer.shape_type.copy()
        properties = shape_layer.properties.copy()

        if export_data == "Segmentations":
            if export_mode == "Binary Mask":
                export_polygons = [shapes[i] for i, shape_type in enumerate(shape_types) if shape_type == "polygon"]

                return export_polygons

            else:
                export_polygons = [shapes[i] for i, shape_type in enumerate(shape_types) if shape_type == "polygon"]
                export_polygons = [{"polygon_coords": polygon.tolist()} for polygon in export_polygons]

                return export_polygons

        else:
            export_polygons = []

            names = set(properties["name"])

            for name in names:
                cell = self.get_cell(name, json=True)

                if cell is not None:
                    export_polygons.append(cell)

            return export_polygons

    def export_shapes_data_finished(self):
        self.update_ui()

    def export_shapes_data(self, path, progress_callback=None):
        export_data = self.gui.shapes_export_data.currentText()
        export_mode = self.gui.shapes_export_mode.currentText()

        if export_data == "Segmentations":
            self.export_segmentations(export_mode, path=path)

        if export_data == "Cells":
            self.export_cells(export_mode, path=path)

    def init_export_shapes_data(self):
        export_data = self.gui.shapes_export_data.currentText()
        export_mode = self.gui.shapes_export_mode.currentText()

        layer_names = [layer.name for layer in self.viewer.layers]

        if export_data in layer_names:
            path = self.get_export_shapes_path(export_mode)

            if path is not None:
                self.update_ui(init=True)

                worker = Worker(self.export_shapes_data, path)
                worker.signals.finished.connect(self.export_shapes_data_finished)
                self.threadpool.start(worker)

    def update_shape_export_options(self):
        export_data = self.gui.shapes_export_data.currentText()

        if export_data == "Segmentations":
            self.gui.shapes_export_mode.clear()
            self.gui.shapes_export_mode.addItems(["Binary Mask", "JSON"])

        if export_data == "Cells":
            self.gui.shapes_export_mode.clear()
            self.gui.shapes_export_mode.addItems(["Binary Mask", "JSON", "Oufti/MicrobTracker Mesh"])

    def get_export_locs(self, dataset, channel):

        locs = []
        fitted = False
        box_size = int(self.gui.picasso_box_size.value())
        net_gradient = 1000

        locs_export_data = self.gui.locs_export_data.currentText()

        if dataset in self.localisation_dict.keys():
            if channel in self.localisation_dict[dataset].keys():
                loc_dict = self.localisation_dict[dataset][channel]

                if "localisations" in loc_dict.keys():
                    fitted = loc_dict["fitted"]
                    box_size = loc_dict["box_size"]

                    if "net_gradient" in loc_dict.keys():
                        net_gradient = loc_dict["net_gradient"]

                    locs = loc_dict["localisations"]

                if locs_export_data == "Tracks":
                    if dataset in self.tracking_dict.keys():
                        locs = self.tracking_dict[dataset]
                    else:
                        locs = []

        return locs, fitted, box_size, net_gradient

    def get_picasso_info(self, import_path,
            image_shape, box_size):

        picasso_info = []

        try:
            picasso_info = [{"Byte Order": "<",
                             "Data Type": "uint16",
                             "File": import_path,
                             "Frames": image_shape[0],
                             "Height": image_shape[1],
                             "Micro-Manager Acquisiton Comments": "",
                             "Width": image_shape[2], },
                {"Box Size": box_size,
                 "Fit method": "LQ, Gaussian",
                 "Generated by": "Picasso Localize",
                 "Pixelsize": 130, "ROI": None, }, ]

        except:
            print(traceback.format_exc())
            pass

        return picasso_info

    def export_locs(self, export_list, export_data,
            export_loc_mode, progress_callback=None):

        try:
            picasso_info = []
            fitted = False
            export_loc_jobs = []

            for dat in export_list:
                dataset_name = dat["dataset"]
                channel_name = dat["channel"]
                data = dat[export_data.lower()]

                n_data = len(data)

                if n_data > 0:
                    import_path = self.dataset_dict[dataset_name]["path"]
                    image_dict = self.dataset_dict[dataset_name]["images"]
                    image_shape = image_dict[channel_name].shape

                    if type(import_path) == list:
                        import_path = import_path[0]

                    export_dir = os.path.dirname(import_path)

                    hdf5_file_name = (dataset_name + f"_{channel_name}_moltrack_{export_data.lower()}.hdf5")
                    info_file_name = (dataset_name + f"_{channel_name}_moltrack_{export_data.lower()}.yaml")

                    hdf5_path = os.path.join(export_dir, hdf5_file_name)
                    info_path = os.path.join(export_dir, info_file_name)

                    if export_loc_mode == "CSV":
                        export_file_name = (dataset_name + f"_{channel_name}_moltrack_{export_data.lower()}.csv")
                        export_path = os.path.join(export_dir, export_file_name)
                    elif export_loc_mode == "POS.OUT":
                        export_file_name = (dataset_name + f"_{channel_name}_moltrack_{export_data.lower()}.pos.out")
                        export_path = os.path.join(export_dir, export_file_name)
                    else:
                        export_path = ""

                        box_size = dat["box_size"]
                        fitted = dat["fitted"]

                        picasso_info = self.get_picasso_info(import_path, image_shape, box_size)

                    export_loc_job = {"dataset_name": dataset_name,
                                      "channel_name": channel_name,
                                      "export_data": export_data,
                                      "data": data,
                                      "fitted": fitted,
                                      "export_mode": export_loc_mode,
                                      "hdf5_path": hdf5_path,
                                      "info_path": info_path,
                                      "export_path": export_path,
                                      "picasso_info": picasso_info, }

                    export_loc_jobs.append(export_loc_job)

            if len(export_loc_jobs) > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    futures = [executor.submit(initialise_data_export, job) for job in export_loc_jobs]

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except:
                            print(traceback.format_exc())
                            pass

                        progress = int(100 * (len(export_loc_jobs) - len(futures)) / len(export_loc_jobs))

                        if progress_callback is not None:
                            progress_callback.emit(progress)

        except:
            print(traceback.format_exc())
            pass

    def export_locs_finished(self):
        try:
            print("Exporting locs finished")
            self.update_ui()

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

    def sort_export_cols(self, data):
        order = ["dataset", "channel", "group", "particle", "frame", "cell_index",
                 "segmentation_index", "x", "y", "photons", "bg", "sx", "sy", "lpx", "lpy",
                 "ellipticity", "net_gradient", "iterations", ]

        mask = []

        for col in data.columns:
            if col in order:
                mask.append(col)

        data = data[mask]

        return data

    def initialise_export_locs(self, event=None, export_dataset="", export_channel=""):
        try:
            if (export_dataset == "" or export_dataset not in self.dataset_dict.keys()):
                export_dataset = self.gui.locs_export_dataset.currentText()
            if (export_channel == "" or export_channel not in self.dataset_dict[export_dataset]["images"].keys()):
                export_channel = self.gui.locs_export_channel.currentText()

            export_data = self.gui.locs_export_data.currentText()
            export_loc_mode = self.gui.locs_export_mode.currentText()
            locs_export_concat = self.gui.locs_export_concat.isChecked()

            if export_data == "Localisations":
                export_list = self.get_locs(export_dataset, export_channel, return_dict=True, include_metadata=True, )
            else:
                export_list = self.get_tracks(export_dataset, export_channel, return_dict=True, include_metadata=True, )

            if len(export_list) == 0:
                return

            if locs_export_concat:
                concat_dict = {}

                for key, value in export_list[0].items():
                    if key != export_data.lower():
                        concat_dict[key] = value
                locs = []
                for dat in export_list:
                    locs.extend(dat[export_data.lower()])
                locs = np.hstack(locs).view(np.recarray).copy()
                concat_dict[export_data.lower()] = locs
                export_list = [concat_dict]

            self.update_ui(init=True)

            self.worker = Worker(self.export_locs, export_list=export_list, export_data=export_data, export_loc_mode=export_loc_mode, )
            self.worker.signals.progress.connect(partial(self.moltrack_progress, progress_bar=self.gui.export_progressbar, ))
            self.worker.signals.finished.connect(self.export_locs_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.threadpool.start(self.worker)

        except:
            self.update_ui()
            pass

    def export_moltrack_project(self, viewer = None, path=None):

        try:

            self.update_ui(init=True)

            if path is None:

                dataset_list = list(self.dataset_dict.keys())

                if len(dataset_list) > 0:
                    path = self.dataset_dict[dataset_list[0]]["path"]

                    if type(path) == list:
                        path = path[0]

                    directory = os.path.dirname(path)
                    file_name = os.path.basename(path)
                    base, ext = os.path.splitext(file_name)
                    file_name = base + ".moltrack"
                    export_path = os.path.join(directory, file_name)

                else:
                    desktop_dir = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
                    file_name = "moltrack_project.moltrack"
                    export_path = os.path.join(desktop_dir, file_name)

                path = QFileDialog.getSaveFileName(self, "Export Moltrack Project", export_path, "(*.moltrack)")[0]

            if path == "":
                return

            moltrack_project = {}

            if hasattr(self, "dataset_dict"):

                export_project_images = self.gui.export_project_images.isChecked()

                moltrack_project["dataset_dict"] = {}

                for dataset_name in self.dataset_dict.keys():

                    dataset_dict = {}

                    for key, value in self.dataset_dict[dataset_name].items():
                        if key != "images" or export_project_images:
                            if key == "images":
                                print("Exporting images")

                            dataset_dict[key] = value

                    moltrack_project["dataset_dict"][dataset_name] = dataset_dict

            if hasattr(self, "segmentation_layer"):

                segmentation_image = self.segmentation_layer.data.copy()
                scale = self.segmentation_layer.scale
                pixel_size = scale[0]

                moltrack_project["segmentation_image"] = {"data": segmentation_image,
                                                          "pixel_size": pixel_size}

            if hasattr(self, "localisation_dict"):

                moltrack_project["localisation_dict"] = self.localisation_dict

                print("localisation data added to moltrack project")

            if hasattr(self,"tracking_dict"):

                moltrack_project["tracking_dict"] = self.tracking_dict

            if hasattr(self, "segLayer"):

                shapes = self.segLayer.data.copy()
                shape_types = self.segLayer.shape_type.copy()
                scale = self.segLayer.scale
                pixel_size = scale[0]

                moltrack_project["segmentations"] = {"shapes": shapes,
                                                    "shape_types": shape_types,
                                                    "scale": scale,
                                                    "pixel_size": pixel_size}

            if hasattr(self, "cellLayer"):

                shapes = self.cellLayer.data.copy()
                shape_types = self.cellLayer.shape_type.copy()
                properties = self.cellLayer.properties.copy()
                scale = self.cellLayer.scale

                moltrack_project["cells"] = {"shapes": shapes,
                                             "shape_types": shape_types,
                                             "properties": properties,
                                             "scale": scale,
                                             "pixel_size": scale[0],}

            with open(path, "wb") as file:
                pickle.dump(moltrack_project, file)

            show_info("Moltrack project exported")

            self.update_ui()

        except:
            print(traceback.format_exc())
            self.update_ui()
            pass

