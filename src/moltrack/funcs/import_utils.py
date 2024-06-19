import traceback
import numpy as np
import os
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
        timeout_duration = 10  # Timeout in seconds

        with Manager() as manager:
            progress_dict = manager.dict()

            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                # Submit all jobs and store the future objects
                futures = [executor.submit(import_image_data, job, progress_dict, i) for i, job in enumerate(compute_jobs)]

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

    def init_import_data(self):
        try:
            import_mode = self.gui.import_mode.currentText()
            desktop = os.path.expanduser("~/Desktop")

            if import_mode.lower() != ["segmentation image"]:
                paths = QFileDialog.getOpenFileNames(self, "Open file", desktop, "Image files (*.tif *.fits)")[0]

                paths = [path for path in paths if path != ""]

            else:
                path = QFileDialog.getOpenFileName(self, "Open file", desktop, "Image files (*.tif *.fits)")[0]
                paths = [path]

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
