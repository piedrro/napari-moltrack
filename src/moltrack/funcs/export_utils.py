import json
import os
import traceback
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from bactfit import fileIO
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog

from moltrack.funcs.compute_utils import Worker


def format_picasso_path(path):
    if "%" in str(path):
        path = path.replace("%", "%%")

    path = os.path.normpath(path)

    if os.name == "nt":
        if path.startswith("\\\\"):
            path = "\\\\?\\UNC\\" + path[2:]

    return Path(path)


def initialise_data_export(loc_data):
    export_mode = loc_data["export_mode"]
    if export_mode == "Picasso HDF5":
        export_picasso_localisation(loc_data)
    else:
        export_localisation_data(loc_data)


def export_localisation_data(loc_data):
    from molcore.files import write_table
    from moltrack.model_io import table_columns

    data = pd.DataFrame(loc_data["data"])
    if loc_data["export_mode"] == "POS.OUT":
        data = data[["frame", "x", "y", "photons", "bg", "sx", "sy"]].dropna().copy()
        data["I0"] = 0
        data["THETA"] = 0
        data["ECC"] = data.sx / data.sy
    write_table(loc_data["export_path"], table_columns(data))


def export_picasso_localisation(loc_data):
    from molcore.files import write_table
    from moltrack.model_io import table_columns

    # The export dialog already selected these columns; retain arbitrary
    # measurements instead of applying a second hard-coded field whitelist.
    write_table(format_picasso_path(loc_data["hdf5_path"]), table_columns(loc_data["data"]), metadata=loc_data["picasso_info"])


def export_locs(jobs, progress_callback=None):
    """Write detached table jobs captured on the GUI thread."""
    for index, job in enumerate(jobs):
        initialise_data_export(job)
        if progress_callback is not None:
            progress_callback.emit(int(100 * (index + 1) / len(jobs)))


def export_shapes(path, mode, polygons, frame_shape=None, progress_callback=None):
    """Write detached shape payloads without consulting a widget or layer."""
    if mode == "Binary Mask":
        if frame_shape is None:
            raise ValueError("Could not determine mask shape for export")
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for index, polygon in enumerate(polygons):
            contour = np.round(np.asarray(polygon)[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask, [contour], -1, index + 1, -1)
        tifffile.imwrite(path, mask)
    elif mode == "JSON":
        payload = {key: [polygon[key] for polygon in polygons] for key in polygons[0]} if polygons else {}
        with open(path, "w") as stream:
            json.dump(payload, stream)
    elif mode == "Oufti/MicrobTracker Mesh":
        from moltrack.funcs.oufti_utils import oufti
        adapter = oufti()
        meshes = [adapter.get_cell_mesh(polygon) for polygon in polygons]
        if len(meshes) > 1:
            adapter.export_oufti(meshes, path)
    else:
        raise ValueError(f"Unsupported shape export mode: {mode}")


class _export_utils:

    def get_export_shapes_path(self, mode="Binary Mask"):

        export_path = None
        export_dir = None
        file_name = None

        for dataset in self.session.ordinary_datasets():
            if "path" in self.session.metadata(dataset):
                path = self.session.metadata(dataset)["path"]

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

            show_info(f"Mask exported to {path}")

        except:
            print(traceback.format_exc())

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

        else:
            print("Export mode not recognised")

    def get_export_polygons(self):

        export_data = self.gui.shapes_export_data.currentText()
        export_mode = self.gui.shapes_export_mode.currentText()

        shape_layer = self.viewer.layers[export_data]

        shapes = shape_layer.data.copy()
        shape_types = shape_layer.shape_type.copy()
        properties = shape_layer.properties.copy()

        if export_data == "Segmentations":
            if export_mode == "Binary Mask":

                export_polygons = []

                for i, shape_type in enumerate(shape_types):
                    if shape_type == "polygon":
                        polygon = shapes[i]
                        if len(polygon) > 3:
                            export_polygons.append(polygon)

                return export_polygons

            else:

                export_polygons = []

                for i, shape_type in enumerate(shape_types):
                    if shape_type == "polygon":
                        polygon = shapes[i]
                        polygon_dict = {"polygon_coords": polygon.tolist()}
                        export_polygons.append(polygon_dict)

                return export_polygons

        else:
            if export_mode == "Binary Mask":

                export_polygons = []
                names = set(properties["name"])

                for name in names:
                    cell = self.get_cell(name, json=True)

                    if cell is not None:
                        polygon = cell["polygon_coords"]

                        if polygon is not None:

                            if len(polygon) > 3:
                                polygon = np.array(polygon)
                                export_polygons.append(polygon)

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

    def init_export_shapes_data(self):

        export_data = self.gui.shapes_export_data.currentText()
        export_mode = self.gui.shapes_export_mode.currentText()

        layer_names = [layer.name for layer in self.viewer.layers]

        if export_data in layer_names:
            path = self.get_export_shapes_path(export_mode)

            if path is not None:
                self.update_ui(init=True)

                from copy import deepcopy
                polygons = deepcopy(self.get_export_polygons())
                frame_shape = self.get_export_mask_shape() if export_mode == "Binary Mask" else None
                worker = Worker(export_shapes, path, export_mode, polygons, frame_shape)
                worker.signals.finished.connect(self.export_shapes_data_finished)
                worker.signals.error.connect(self.update_ui)
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
        from moltrack.model_views import gui_records
        annotations = self.session.annotations.get("partitions", {}).get(dataset, {}).get(channel, {})
        query = self.session.query_tracks if self.gui.locs_export_data.currentText() == "Tracks" else self.session.query_localizations
        locs = gui_records(query(dataset, channel))
        return locs, annotations.get("fitted", False), annotations.get("box_size", self.gui.picasso_box_size.value()), annotations.get("net_gradient", 1000)

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

        return picasso_info

    def prepare_export_locs(self, export_list, export_data, export_loc_mode):

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
                    import_path = self.session.metadata(dataset_name)["path"]
                    image_dict = {c: self.session.image(dataset_name, c).as_dask() for c in self.session.ordinary_channels(dataset_name)}
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
                        saved_info = self.session.annotations.get("partitions", {}).get(dataset_name, {}).get(channel_name, {}).get("picasso_info")
                        if saved_info is not None:
                            from copy import deepcopy
                            picasso_info = deepcopy(saved_info)

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

            from copy import deepcopy
            return deepcopy(export_loc_jobs)
        except Exception:
            print(traceback.format_exc())
            raise

    def export_locs_finished(self):
        try:
            print("Exporting locs finished")
            self.update_ui()

        except:
            self.update_ui()
            print(traceback.format_exc())

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
            if (export_dataset == "" or export_dataset not in self.session.ordinary_datasets()):
                export_dataset = self.gui.locs_export_dataset.currentText()
            if (export_channel == "" or export_channel not in {c: self.session.image(export_dataset, c).as_dask() for c in self.session.ordinary_channels(export_dataset)}.keys()):
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

            jobs = self.prepare_export_locs(export_list, export_data, export_loc_mode)
            self.worker = Worker(export_locs, jobs)
            self.worker.signals.progress.connect(partial(self.moltrack_progress, progress_bar=self.gui.export_progressbar, ))
            self.worker.signals.finished.connect(self.export_locs_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.threadpool.start(self.worker)

        except:
            self.update_ui()

    def export_moltrack_project(self, viewer=None, path=None):
        from copy import deepcopy
        from moltrack.model_io import save_project
        from moltrack.model_views import layer_annotations
        from moltrack.project_geometry import RECORD_ID, ensure_record_ids

        try:
            if path is None:
                datasets = self.session.ordinary_datasets()
                source = self.session.metadata(datasets[0]).get("path", "") if datasets else ""
                if isinstance(source, list):
                    source = source[0] if source else ""
                proposed = Path(source) if source else Path.home() / "Desktop" / "moltrack_project"
                if not proposed.name.lower().endswith(".moltrack.h5"):
                    proposed = proposed.with_suffix(".moltrack.h5")
                path = QFileDialog.getSaveFileName(
                    self, "Export MolTrack Project", str(proposed),
                    # Native Windows dialogs use this wildcard as the default
                    # extension; a compound suffix can be appended twice.
                    "MolTrack project (*.h5)",
                )[0]
                if path and Path(path).suffix.lower() not in {".h5", ".hdf5"}:
                    suffix = ".h5" if path.lower().endswith(".moltrack") else ".moltrack.h5"
                    path += suffix
            if not path:
                return
            self.update_ui(init=True)
            self.update_contrast_dict()
            self.session.annotations["contrast"] = deepcopy(self.contrast_dict)
            for attribute, name in (("segLayer", "segmentations"), ("cellLayer", "cells")):
                layer = getattr(self, attribute, None)
                if layer is not None:
                    # Capturing a display is not a geometry edit. Preserve native
                    # masks already bound to the current exact layer vertices.
                    mask_ids = self.session.annotations.get(name, {}).get("mask_ids", ())
                    annotation = layer_annotations(layer, mask_ids=mask_ids)
                    ids = ensure_record_ids(annotation, name)
                    if list(layer.properties.get(RECORD_ID, ())) != ids:
                        with layer.events.properties.blocker():
                            layer.properties = {**layer.properties, RECORD_ID: ids}
                    self.session.annotations[name] = annotation
            save_project(self.session, path, include_images=self.gui.export_project_images.isChecked())
            show_info("MolTrack project exported")
        except Exception as error:
            show_info(f"Could not save MolTrack project: {error}")
            print(traceback.format_exc())
        finally:
            self.update_ui()

    def export_bactfit(self):

        try:
            celllist = None

            dataset = self.gui.bactfit_export_dataset.currentText()
            channel = self.gui.bactfit_export_channel.currentText()
            export_data = self.gui.bactfit_export_data.currentText()

            if export_data == "Localisations":
                locs = self.get_locs(dataset, channel)
            else:
                locs = self.get_tracks(dataset, channel)

            if len(locs) == 0:
                locs = None

            if hasattr(self, "cellLayer") == False:
                show_info("BactFit export requires fitted cells")
                return

            show_info("Polpulating BactFit CellList")

            celllist = self.populate_celllist()

            if locs is not None:

                show_info("Adding localisations to CellList")
                celllist.add_localisations(locs)

            if celllist is None:
                show_info("No cells found")
                return

            path = self.session.metadata(dataset)["path"]

            if type(path) == list:
                path = path[0]

            export_dir = os.path.dirname(path)
            file_name = os.path.basename(path)
            base, ext = os.path.splitext(file_name)
            file_name = base + ".h5"

            export_path = os.path.join(export_dir, file_name)

            export_path = QFileDialog.getSaveFileName(self, "Export BactFit",
                export_path, "(*.h5)")[0]

            if export_path == "":
                return

            fileIO.save(export_path, celllist)
            show_info("BactFit exported")

        except:
            print(traceback.format_exc())
