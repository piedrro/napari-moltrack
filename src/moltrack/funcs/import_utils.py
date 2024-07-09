import traceback
import numpy as np
import os

import pandas as pd
from PIL import Image
from qtpy.QtWidgets import QFileDialog
from moltrack.funcs.compute_utils import Worker
import time
import multiprocessing
from multiprocessing import shared_memory, Manager
from functools import partial
import tifffile
import concurrent.futures
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from napari.utils.notifications import show_info
import h5py
import json
import mat4py
from shapely.geometry import Polygon

def crop_frame(image, crop_mode):
    try:
        if "left" in crop_mode.lower():
            crop = image[:, : image.shape[1] // 2]
        elif "right" in crop_mode.lower():
            crop = image[:, image.shape[1] // 2:]
        elif "brightest" in crop_mode.lower():
            left = image[:, : image.shape[1] // 2]
            right = image[:, image.shape[1] // 2:]
            crop = left if np.mean(left) > np.mean(right) else right
        else:
            crop = image

    except:
        print(traceback.format_exc())
        crop = image

    return crop


def import_image_data(dat, progress_dict={}, index=0):
    try:
        path = dat["path"]
        path_index = dat["path_index"]
        crop_mode = dat["import_crop_mode"]
        import_limit = dat["import_limit"]
        frame_averaging = dat["frame_averaging"]
        multichannel_mode = dat["multichannel_mode"]
        channel_name = dat["channel_name"]
        import_mode = dat["import_mode"]

        base, ext = os.path.splitext(path)

        image_dict = {}
        images = []

        if ext.lower() == ".tif":
            with Image.open(path) as image:
                n_frames = image.n_frames

                if import_limit != "None":
                    if import_limit.isdigit():
                        import_limit = int(import_limit)
                        if n_frames > import_limit:
                            n_frames = import_limit

                for frame_index in range(n_frames):
                    image.seek(frame_index)
                    img_frame = np.array(image)

                    img_frame = crop_frame(img_frame, crop_mode)

                    images.append(img_frame)

                    progress = int(((frame_index + 1) / n_frames) * 100)
                    progress_dict[index] = progress

        elif ext.lower() == ".fits":
            with fits.open(path) as hdul:
                n_frames = hdul[0].data.shape[0]

                if import_limit != "None":
                    if import_limit.isdigit():
                        import_limit = int(import_limit)
                        if n_frames > import_limit:
                            n_frames = import_limit

                for frame_index in range(n_frames):
                    img_frame = hdul[0].data[frame_index]

                    img_frame = crop_frame(img_frame, crop_mode)

                    images.append(img_frame)

                    progress = int(((frame_index + 1) / n_frames) * 100)
                    progress_dict[index] = progress

        if len(images) > 0:
            images = np.stack(images, axis=0)

            if import_mode != "Segmentation Image":
                if multichannel_mode == "None":
                    image_dict = {channel_name: images}
                if multichannel_mode == "FRET":
                    donor, acceptor = np.split(images, 2, axis=-1)
                    image_dict = {"Donor": donor, "Acceptor": acceptor}
                if multichannel_mode == "ALEX":
                    donor, acceptor = np.split(images, 2, axis=-1)
                    DD, DA = donor[::2], donor[1::2]
                    AD, AA = acceptor[::2], acceptor[1::2]
                    image_dict = {"DD": DD, "DA": DA, "AD": AD, "AA": AA}
                if multichannel_mode == "Multi File":
                    chanel_name = os.path.basename(path)
                    image_dict = {chanel_name: images}
            else:
                image_dict = {"Segmentation Image": images}

            if frame_averaging == True:
                for channel in image_dict.keys():
                    image = image_dict[channel]
                    image = np.mean(image, axis=0)
                    image = np.expand_dims(image, axis=0)
                    image_dict[channel] = image

            dat["images"] = image_dict

    except:
        print(traceback.format_exc())
        pass

    return dat


class _import_utils:

    def get_image_info(self, path):
        if self.verbose:
            print(f"Getting image info for {path}")

        base, ext = os.path.splitext(path)

        if ext.lower() == ".tif":
            image_size = os.path.getsize(path)  # Get file size directly

            with tifffile.TiffFile(path) as tif:
                n_frames = len(tif.pages)  # Number of pages (frames)
                page_shape = tif.pages[0].shape  # Dimensions of the first page
                dtype = tif.pages[0].dtype  # Data type of the first page

            image_shape = (n_frames, page_shape[0], page_shape[1])

        elif ext.lower() == ".fits":
            image_size = os.path.getsize(path)

            with fits.open(path, mode="readonly", ignore_missing_end=True) as hdul:
                header = hdul[0].header

                # Extract shape information from the header
                if header["NAXIS"] == 3:
                    image_shape = (header["NAXIS3"], header["NAXIS2"], header["NAXIS1"],)
                else:
                    image_shape = (header["NAXIS2"], header["NAXIS1"])

                n_frames = image_shape[0] if len(image_shape) == 3 else 1
                page_shape = (image_shape[1:] if len(image_shape) == 3 else image_shape)

                # Determine the data type from BITPIX
                bitpix_to_dtype = {8: np.dtype("uint8"), 16: np.dtype("uint16"), 32: np.dtype("uint32"), -32: np.dtype("float32"), -64: np.dtype("float64"), }

                dtype = bitpix_to_dtype[header["BITPIX"]]

        return n_frames, image_shape, dtype, image_size

    def format_import_path(self, path):
        try:
            path = os.path.normpath(path)

            if os.name == "nt":
                if path.startswith("\\\\"):
                    path = "\\\\?\\UNC\\" + path[2:]

                    if "%" in str(path):
                        path = path.replace("%", "%%")

                if path.startswith("UNC"):
                    path = "\\\\?\\" + path

                    if "%" in str(path):
                        path = path.replace("%", "%%")

        except:
            print(traceback.format_exc())
            pass

        return path

    def populate_import_jobs(self, progress_callback=None, paths=[]):
        import_jobs = []

        try:
            import_crop_mode = self.gui.import_crop_mode.currentText()
            import_limit = self.gui.import_limit.currentText()
            frame_averaging = self.gui.frame_averaging.isChecked()
            multichannel_mode = self.gui.import_multichannel_mode.currentText()
            channel_name = self.gui.import_channel_name.text()
            import_mode = self.gui.import_mode.currentText()
            pixel_size = float(self.gui.import_pixel_size.value())
            exposure_time = float(self.gui.import_exposure_time.value())

            for path_index, path in enumerate(paths):

                if multichannel_mode.lower() == "multi file":
                    dataset_name = self.gui.import_dataset_name.text()
                else:
                    path = self.format_import_path(path)
                    dataset_name = os.path.basename(path)

                image_dict = {"path": path,
                              "path_index": path_index,
                              "dataset_name": dataset_name,
                              "import_limit": import_limit,
                              "import_crop_mode": import_crop_mode,
                              "frame_averaging": frame_averaging,
                              "multichannel_mode": multichannel_mode,
                              "channel_name": channel_name,
                              "import_mode": import_mode,
                              "pixel_size": pixel_size,
                              "exposure_time": exposure_time, }

                import_jobs.append(image_dict)

        except:
            print(traceback.format_exc())

        return import_jobs

    def process_compute_jobs(self, compute_jobs, progress_callback=None):

        results = []

        if self.verbose:
            print(f"Processing {len(compute_jobs)} compute jobs.")

        cpu_count = int(multiprocessing.cpu_count() * 0.75)

        if len(compute_jobs) == 1:
            executor = ThreadPoolExecutor(max_workers=cpu_count)
        else:
            if len(compute_jobs) < cpu_count:
                cpu_count = len(compute_jobs)
            executor = ProcessPoolExecutor(max_workers=cpu_count)

        show_info(f"Importing {len(compute_jobs)} images")

        with Manager() as manager:
            progress_dict = manager.dict()

            with executor:
                # Submit all jobs and store the future objects
                futures = [executor.submit(import_image_data, job,
                    progress_dict, i) for i, job in enumerate(compute_jobs)]

                while any(not future.done() for future in futures):
                    # Calculate and emit progress
                    total_progress = sum(progress_dict.values())
                    overall_progress = int((total_progress / len(compute_jobs)))
                    if progress_callback is not None:
                        progress_callback.emit(overall_progress)
                    time.sleep(0.1)  # Update frequency

                # Wait for all futures to complete
                concurrent.futures.wait(futures)

                # Retrieve and process results
                results = [future.result() for future in futures if future.done()]

        if self.verbose:
            print("Finished processing compute jobs.")

        return results

    def populate_import_dataset_dict(self, import_list):
        try:
            concat_images = self.gui.import_concatenate.isChecked()

            if self.verbose:
                print("Populating dataset dict")

            import_dict = {}

            for import_data in import_list:
                if type(import_data) == dict:
                    if "images" in import_data.keys():
                        dataset_name = import_data["dataset_name"]
                        image_dict = import_data["images"]

                        if image_dict != {}:
                            if dataset_name not in import_dict.keys():
                                import_dict[dataset_name] = import_data
                            else:
                                for channel in image_dict.keys():
                                    import_dict[dataset_name]["images"][channel] = image_dict[channel]


            if concat_images == True:
                dataset_list = list(import_dict.keys())

                image_list = []
                path_list = []
                pixel_size_list = []
                exposure_time_list = []

                for dataset_name in dataset_list:
                    image_dict = import_dict[dataset_name].pop("images")

                    dataset_channel = list(image_dict.keys())[0]
                    dataset_image = image_dict[dataset_channel]

                    dataset_path = import_dict[dataset_name].pop("path")
                    dataset_pixel_size = import_dict[dataset_name].pop("pixel_size")
                    dataset_exposure_time = import_dict[dataset_name].pop("exposure_time")

                    dataset_path = [dataset_path] * dataset_image.shape[0]

                    image_list.append(dataset_image)
                    path_list.extend(dataset_path)
                    pixel_size_list.append(dataset_pixel_size)
                    exposure_time_list.append(dataset_exposure_time)

                image_list = np.concatenate(image_list, axis=0)

                if dataset_list[0] not in self.dataset_dict.keys():
                    self.dataset_dict[dataset_list[0]] = {"path": path_list,
                                                          "pixel_size": pixel_size_list[0],
                                                          "exposure_time": exposure_time_list[0],
                                                          "images": {dataset_channel: image_list}, }

            else:
                dataset_list = list(import_dict.keys())

                for dataset_name in dataset_list:
                    dataset_dict = import_dict.pop(dataset_name)
                    self.dataset_dict[dataset_name] = dataset_dict

        except:
            print(traceback.format_exc())
            pass

    def import_data(self, progress_callback=None, paths=[], import_mode="data"):
        import_jobs = self.populate_import_jobs(paths=paths)

        results = self.process_compute_jobs(import_jobs, progress_callback=progress_callback)

        if import_mode.lower() != "segmentation image":
            self.populate_import_dataset_dict(results)
        else:
            if len(results) > 0:
                if type(results[0]) == dict:
                    if "images" in results[0].keys():
                        self.segmentation_image = results[0]["images"]["Segmentation Image"]
                        self.segmentation_image_pixel_size = float(results[0]["pixel_size"])

    def import_data_finished(self):
        self.populate_dataset_selectors()
        self.update_active_image()
        self.draw_segmentation_image()
        self.update_fret_transform_target_channel()
        self.reset_slider()
        self.update_ui()

    def reset_slider(self):
        try:
            curent_step = self.viewer.dims.current_step
            curent_step = list(curent_step)
            curent_step[0] = 0
            curent_step = tuple(curent_step)
            self.viewer.dims.current_step = curent_step
        except:
            print(traceback.format_exc())
            pass

    def init_import_data(self, viewer=None, import_mode=None, import_path=None):

        try:

            if import_mode is None:
                import_mode = self.gui.import_mode.currentText()

            if import_path is None:
                desktop = os.path.expanduser("~/Desktop")

                if import_mode.lower() != ["segmentation image"]:
                    paths = QFileDialog.getOpenFileNames(self, "Open file", desktop, "Image files (*.tif *.fits)")[0]

                    paths = [path for path in paths if path != ""]

                else:
                    path = QFileDialog.getOpenFileName(self, "Open file", desktop, "Image files (*.tif *.fits)")[0]
                    paths = [path]
            else:
                if type(import_path) == str:
                    paths = [import_path]

            if paths != []:
                self.update_ui(init=True)

                self.worker = Worker(self.import_data, paths=paths, import_mode=import_mode)
                self.worker.signals.progress.connect(partial(self.moltrack_progress, progress_bar=self.gui.import_progressbar, ))
                self.worker.signals.finished.connect(self.import_data_finished)
                self.worker.signals.error.connect(self.update_ui)
                self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass




    def import_pos_out_locs(self, path, loc_cols=None):

        try:

            col_dict = {"FRAME": "frame",
                        "XCENTER": "x", "YCENTER": "y",
                        "BRIGHTNESS": "photons", "BG": "bg",
                        "S_X": "sx", "S_Y": "sy",
                        }

            locs = pd.read_csv(path, sep="\t")
            locs = locs.rename(columns=col_dict)

            for col in locs.columns:
                if col not in loc_cols:
                    locs = locs.drop(col, axis=1)

            locs_frames = np.array(locs["frame"].values)
            locs_frames = locs_frames - 1
            locs["frame"] = locs_frames

        except:
            locs = None

        return locs

    def import_csv_locs(self, path, loc_cols):

        print("importing csv locs")

        try:

            locs = pd.read_csv(path)

            for col in locs.columns:
                if col not in loc_cols:
                    locs = locs.drop(col, axis=1)

        except:
            locs = None

        return locs

    def import_hdf5_locs(self, path, loc_cols):

        try:

            dtype = [('frame', '<u4'), ('x', '<f4'), ('y', '<f4'),
                     ('photons', '<f4'), ('sx', '<f4'), ('sy', '<f4'),
                     ('bg', '<f4'), ('lpx', '<f4'), ('lpy', '<f4'),
                     ('ellipticity', '<f4'),('net_gradient', '<f4')]

            with h5py.File(path, "r") as f:
                locs = np.array(f["locs"], dtype=dtype).view(np.recarray)

            locs = pd.DataFrame(locs)

        except:
            print(traceback.format_exc())
            locs = None

        return locs


    def import_localisations(self, import_dataset=None, import_channel=None,
            import_data=None, import_mode=None, path = None):

        try:

            if import_dataset is None:
                import_dataset = self.gui.locs_import_dataset.currentText()
            if import_channel is None:
                import_channel = self.gui.locs_import_channel.currentText()
            if import_data is None:
                import_data = self.gui.locs_import_data.currentText()
            if import_mode is None:
                import_mode = self.gui.locs_import_mode.currentText()

            if import_dataset == "" or import_channel == "":
                show_info("Localisation import requires image data to be loaded first")
                return None

            loc_cols = ["dataset", "channel", "group", "particle", "frame",
                        "cell_index", "segmentation_index",
                        "x", "y", "photons", "bg", "sx", "sy", "lpx", "lpy",
                        "ellipticity", "net_gradient", "iterations",
                        "pixel_mean", "pixel_median", "pixel_sum", "pixel_min",
                        "pixel_max", "pixel_std", "pixel_mean_bg", "pixel_median_bg",
                        "pixel_sum_bg", "pixel_min_bg", "pixel_max_bg", "pixel_std_bg"]

            if "picasso" in import_mode.lower():
                ext = "HDF5 files (*.hdf5)"
                import_func = self.import_hdf5_locs
            elif import_mode == "CSV":
                ext = "CSV files (*.csv)"
                import_func = self.import_csv_locs
            elif import_mode == "POS.OUT":
                ext = "POS.OUT files (*.pos.out)"
                import_func = self.import_pos_out_locs
            else:
                return None

            if path is None:
                path = os.path.expanduser("~/Desktop")

                if hasattr(self, "dataset_dict"):
                    if import_dataset in self.dataset_dict.keys():
                        path = self.dataset_dict[import_dataset]["path"]
                        if type(path) == list:
                            path = path[0]

                path = QFileDialog.getOpenFileName(self, "Open file", path, ext)[0]

            if path != "":

                locs = import_func(path, loc_cols)

                if locs is not None:
                    locs["dataset"] = import_dataset
                    locs["channel"] = import_channel
                    locs = locs.to_records(index=False)

                    if import_data == "Localisations":

                        locs_cols = locs.dtype.names

                        if set(["dataset","channel","frame","x","y"]).issubset(locs_cols):

                            if import_dataset not in self.localisation_dict.keys():
                                self.localisation_dict[import_dataset] = {}
                            if import_channel not in self.localisation_dict[import_dataset].keys():
                                self.localisation_dict[import_dataset][import_channel] = {}

                            locdict = self.localisation_dict[import_dataset][import_channel]
                            locdict["localisations"] = locs

                            imported_locs = self.get_locs(import_dataset, import_channel)
                            show_info(f"Imported {len(imported_locs)} localisations")

                            self.draw_localisations()

                            self.update_filter_criterion()
                            self.update_criterion_ranges()

                            self.update_pixmap_options()

                            if len(locs) > 0:
                                self.gui.locs_export_data.clear()
                                self.gui.locs_export_data.addItems(["Localisations"])
                                self.gui.heatmap_data.clear()
                                self.gui.heatmap_data.addItems(["Localisations"])

                        else:
                            show_info("Missing required columns for localisation data import")

                    if import_data == "Tracks":

                        locs_cols = locs.dtype.names
                        if set(["dataset","channel","particle", "frame","x","y"]).issubset(locs_cols):

                            if import_dataset not in self.tracking_dict.keys():
                                self.tracking_dict[import_dataset] = {}
                            if import_channel not in self.tracking_dict[import_dataset].keys():
                                self.tracking_dict[import_dataset][import_channel] = {}

                            self.tracking_dict[import_dataset][import_channel]["tracks"] = locs
                            self.draw_tracks()

                            imported_tracks = self.get_tracks(import_dataset, import_channel)
                            show_info(f"Imported {len(imported_tracks)} tracks")

                            self.update_filter_criterion()
                            self.update_criterion_ranges()

                            self.update_track_filter_criterion()
                            self.update_track_criterion_ranges()
                            self.update_traces_export_options()
                            self.update_pixmap_options()

                            if len(locs) > 0:
                                self.gui.locs_export_data.clear()
                                self.gui.locs_export_data.addItems(["Localisations", "Tracks"])
                                self.gui.heatmap_data.clear()
                                self.gui.heatmap_data.addItems(["Localisations", "Tracks"])

                        else:
                            show_info("Missing required columns for tracking data import")
        except:
            print(traceback.format_exc())
            pass



    def import_binary_mask(self, path):

        try:

            mask = tifffile.imread(path)
            shapes = self.mask_to_shape(mask)

        except:
            shapes = None

        return shapes

    def import_cells(self, path):

        #import json file
        with open(path, 'r') as f:
            data = json.load(f)

        cell_names = data["name"]
        polygon_coords = data["polygon_coords"]
        midline_coords = data["midline_coords"]

        shapes = []
        shape_types = []
        properties = {"name": [], "cell": []}

        for cell_index, (name, polygon, midline) in enumerate(zip(
                cell_names, polygon_coords, midline_coords)):

            try:

                fit_params = {}
                if "name" in data.keys():
                    fit_params["name"] = data["name"][cell_index]
                if "width" in data.keys():
                    fit_params["width"] = data["width"][cell_index]
                if "poly_params" in data.keys():
                    fit_params["poly_params"] = data["poly_params"][cell_index]
                if "cell_poles" in data.keys():
                    fit_params["cell_poles"] = data["cell_poles"][cell_index]

                polygon = Polygon(polygon)
                polygon = np.array(polygon.exterior.coords)
                polygon = polygon[:-1]

                shapes.append(polygon)
                shape_types.append("polygon")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

                shapes.append(midline)
                shape_types.append("path")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

            except:
                print(traceback.format_exc())

        shapes = {"shapes": shapes,
                  "shape_types": shape_types,
                  "properties": properties}

        return shapes

    def import_mesh(self, path):

        try:

            mat_data = mat4py.loadmat(path)

            mat_data = mat_data["cellList"]

            shapes = []

            for dat in mat_data:
                try:
                    if type(dat) == dict:
                        shape = np.array(dat["model"])
                        shape = np.fliplr(shape)
                        shape = shape[:-1]
                        shapes.append(shape)
                except:
                    pass

        except:
            shapes = None

        return shapes

    def import_shapes(self, import_data=None, import_mode=None, pixel_size=None, path=None):

        try:

            if import_data is None:
                import_data = self.gui.shapes_import_data.currentText()
            if import_mode is None:
                import_mode = self.gui.shapes_import_mode.currentText()
            if pixel_size is None:
                pixel_size = float(self.gui.shapes_import_pixel_size.value())

            self.segmentation_image_pixel_size = pixel_size

            if import_mode == "Binary Mask":
                import_func = self.import_binary_mask
                ext = "*.tif"
            elif import_mode == "JSON":
                import_func = self.import_cells
                ext = "*.json"
            elif import_mode == "Oufti/MicrobTracker Mesh":
                import_func = self.import_mesh
                ext = "*.mat"
            else:
                return None

            if path is None:
                path = os.path.expanduser("~/Desktop")
                path = QFileDialog.getOpenFileName(self, "Open file", path, ext)[0]

            if path != "":

                shape_data = import_func(path)

                if shape_data is not None:

                    if import_data == "Segmentations":

                        if import_mode in ["Binary Mask","Oufti/MicrobTracker Mesh"]:
                            self.initialise_segLayer(shape_data, pixel_size)

                        else:
                            shape_types = shape_data["shape_types"]
                            polygons = [shape_data["shapes"][i] for i, shape_type in enumerate(shape_types) if shape_type == "polygon"]

                            self.initialise_segLayer(polygons, pixel_size)

                    else:

                        shapes = shape_data["shapes"]
                        shape_types = shape_data["shape_types"]
                        properties = shape_data["properties"]

                        self.cellLayer = self.initialise_cellLayer(shapes=shapes,
                            shape_types=shape_types, properties=properties)

                        self.store_cell_shapes()

        except:
            print(traceback.format_exc())



