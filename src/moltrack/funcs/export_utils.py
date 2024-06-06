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


def format_picasso_path(path):

    if "%" in str(path):
        path = path.replace("%", "%%")

    path = os.path.normpath(path)

    if os.name == "nt":
        if path.startswith("\\\\"):
            path = '\\\\?\\UNC\\' + path[2:]

    return Path(path)


def initialise_localisation_export(loc_data):

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
        locs = loc_data["locs"]

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
            pos_locs.loc[:, "ECC"] = (pos_locs["S_X"] / pos_locs["S_Y"])
            pos_locs.loc[:, "FRAME"] = pos_locs["FRAME"] + 1

            pos_locs = pos_locs[["FRAME", "XCENTER", "YCENTER", "BRIGHTNESS", "BG", "I0", "S_X", "S_Y", "THETA", "ECC", ]]

            pos_locs.to_csv(export_path, sep="\t", index=False)

    except:
        print(traceback.format_exc())



def export_picasso_localisation(loc_data):

    try:
        locs = loc_data["locs"]
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

            print("Could not move files to import directory. Saving to desktop instead.")

            desktop_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

            desktop_h5py_path = os.path.join(desktop_dir, h5py_path.name)
            desktop_yaml_path = os.path.join(desktop_dir, yaml_path.name)

            shutil.move(temp_h5py_path, desktop_h5py_path)
            shutil.move(temp_yaml_path, desktop_yaml_path)

    except Exception as e:
        print(traceback.format_exc())


class _export_utils:

    def export_segmentations(self, export_mode, mode = "Binary Mask"):

        pass

    def export_cells(self, export_mode, mode = "Binary Mask"):

        pass

    def export_shapes_data(self):

        export_data = self.gui.shapes_export_data.currentText()
        export_mode = self.gui.shapes_export_mode.currentText()

        layer_names = [layer.name for layer in self.viewer.layers]

        if export_data in layer_names:

            if export_data == "Segmentations":
                self.export_segmentations(export_mode, mode = export_mode)

            if export_data == "Cells":
                self.export_cells(export_mode, mode = export_mode)


    def update_shape_export_options(self):

        export_data = self.gui.shapes_export_data.currentText()

        if export_data == "Segmentations":

            self.gui.shapes_export_mode.clear()
            self.gui.shapes_export_mode.addItems(["Binary Mask", "JSON"])

        if export_data == "Cells":

            self.gui.shapes_export_mode.clear()
            self.gui.shapes_export_mode.addItems(["Binary Mask", "JSON", "Oufti/MicrobTracker Mesh"])



    def get_export_locs(self, dataset):

        locs = []
        fitted = False
        box_size = int(self.gui.picasso_box_size.currentText())
        min_net_gradient = int(self.gui.picasso_min_net_gradient.text())

        locs_export_data = self.gui.locs_export_data.currentText()

        if dataset in self.localisation_dict.keys():
            if "localisations" in self.localisation_dict[dataset].keys():

                loc_dict = self.localisation_dict[dataset]

                fitted = loc_dict["fitted"]
                box_size = loc_dict["box_size"]

                if "min_net_gradient" in loc_dict.keys():
                    min_net_gradient = loc_dict["min_net_gradient"]

                locs = loc_dict["localisations"]

            if locs_export_data == "Tracks":

                if dataset in self.tracking_dict.keys():
                    locs = self.tracking_dict[dataset]
                else:
                    locs = []

        return locs, fitted, box_size, min_net_gradient


    def export_locs(self, progress_callback = None, export_dataset = ""):

        try:

            export_loc_mode = self.gui.locs_export_mode.currentText()

            export_loc_jobs = []

            if export_dataset == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [export_dataset]

            for dataset_name in dataset_list:

                locs, fitted, box_size, min_net_gradient = self.get_export_locs(dataset_name)

                n_locs = len(locs)

                if n_locs > 0:

                    import_path = self.dataset_dict[dataset_name]["path"]
                    image_shape = self.dataset_dict[dataset_name]["data"].shape

                    base, ext = os.path.splitext(import_path)

                    hdf5_path = base + f"_moltrack_localisations.hdf5"
                    info_path = base + f"_moltrack_localisations.yaml"

                    if export_loc_mode == "CSV":
                        export_path = base + f"_moltrack_localisations.csv"
                    elif export_loc_mode == "POS.OUT":
                        export_path = base + f"_moltrack_localisations.pos.out"

                    else:
                        export_path = ""

                    picasso_info = [{"Byte Order": "<", "Data Type": "uint16", "File": import_path,
                                     "Frames": image_shape[0], "Height": image_shape[1],
                                     "Micro-Manager Acquisiton Comments": "", "Width":image_shape[2],},
                                    {"Box Size": box_size, "Fit method": "LQ, Gaussian", "Generated by": "Picasso Localize",
                                     "Min. Net Gradient": min_net_gradient, "Pixelsize": 130, "ROI": None, }]

                    export_loc_job = { "dataset_name": dataset_name,
                                       "locs": locs,
                                       "fitted": fitted,
                                       "export_mode": export_loc_mode,
                                       "hdf5_path": hdf5_path,
                                       "info_path": info_path,
                                       "export_path": export_path,
                                       "picasso_info": picasso_info,
                                      }

                    export_loc_jobs.append(export_loc_job)

            if len(export_loc_jobs) > 0:

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    futures = [executor.submit(initialise_localisation_export, job) for job in export_loc_jobs]

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
            self.update_ui()
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

    def initialise_export_locs(self, event=None, export_dataset = ""):

        try:

            if export_dataset == "" or export_dataset not in self.dataset_dict.keys():
                export_dataset = self.gui.locs_export_dataset.currentText()

            self.update_ui(init = True)

            self.worker = Worker(self.export_locs, export_dataset = export_dataset)
            self.worker.signals.progress.connect(partial(self.moltrack_progress,progress_bar=self.gui.export_progressbar))
            self.worker.signals.finished.connect(self.export_locs_finished)
            self.threadpool.start(self.worker)

        except:
            self.update_ui()
            pass


