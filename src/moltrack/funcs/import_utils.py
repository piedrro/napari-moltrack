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


def import_image_data(dat, progress_dict={}, index=0):

    try:

        path = dat["path"]
        import_crop_mode = dat["import_crop_mode"]
        n_frames = dat["n_frames"]

        base, ext = os.path.splitext(path)

        if ext.lower() == ".tif":

            with Image.open(path) as image:

                for frame_index in range(n_frames):

                    image.seek(frame_index)
                    img_frame = np.array(image)

                    shared_mem = dat["shared_image"]
                    np_array = np.ndarray(dat["image_shape"], dtype=dat["dtype"], buffer=shared_mem.buf)
                    np_array[frame_index] = img_frame

                    progress = int(((frame_index + 1) / n_frames)*100)
                    progress_dict[index] = progress

        elif ext.lower() == ".fits":

            with fits.open(path) as hdul:

                for frame_index in range(n_frames):

                    img_frame = hdul[0].data[frame_index]

                    shared_mem = dat["shared_image"]
                    np_array = np.ndarray(dat["image_shape"], dtype=dat["dtype"], buffer=shared_mem.buf)
                    np_array[frame_index] = img_frame

                    progress = int(((frame_index + 1) / n_frames)*100)
                    progress_dict[index] = progress

    except:
        print(traceback.format_exc())
        pass






















class _import_utils:

    def create_import_shared_image(self, image_size):

        shared_mem = None

        try:

            if self.verbose:
                print("Creating shared image...")

            shared_mem = shared_memory.SharedMemory(create=True, size=image_size)

        except:
            print(traceback.format_exc())
            pass

        return shared_mem

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

            with fits.open(path, mode='readonly', ignore_missing_end=True) as hdul:

                header = hdul[0].header

                # Extract shape information from the header
                if header['NAXIS'] == 3:
                    image_shape = (header['NAXIS3'], header['NAXIS2'], header['NAXIS1'])
                else:
                    image_shape = (header['NAXIS2'], header['NAXIS1'])

                n_frames = image_shape[0] if len(image_shape) == 3 else 1
                page_shape = image_shape[1:] if len(image_shape) == 3 else image_shape

                # Determine the data type from BITPIX
                bitpix_to_dtype = {8: np.dtype('uint8'),
                                   16: np.dtype('uint16'),
                                   32: np.dtype('uint32'),
                                   -32: np.dtype('float32'),
                                   -64: np.dtype('float64'),
                                   }

                dtype = bitpix_to_dtype[header['BITPIX']]

        return n_frames, image_shape, dtype, image_size

    def format_import_path(self, path):

        try:

            path = os.path.normpath(path)

            if os.name == "nt":
                if path.startswith("\\\\"):
                    path = '\\\\?\\UNC\\' + path[2:]

                    if "%" in str(path):
                        path = path.replace("%", "%%")

                if path.startswith("UNC"):
                    path = '\\\\?\\' + path

                    if "%" in str(path):
                        path = path.replace("%", "%%")

        except:
            print(traceback.format_exc())
            pass

        return path


    def populate_import_lists(self, progress_callback=None, paths=[]):

        image_list = []
        import_dict = {}
        shared_images = {}

        try:

            if self.verbose:
                print("Populating import lists/metadata...")

            import_crop_mode = self.gui.import_crop_mode.currentText()
            import_limit_combo = self.gui.import_limit.currentText()

            for path_index, path in enumerate(paths):

                path = self.format_import_path(path)
                dataset_name = os.path.basename(path)

                if dataset_name not in shared_images.keys():
                    shared_images[dataset_name] = {}

                n_frames, image_shape, dtype, image_size = self.get_image_info(path)

                if import_limit_combo != "None":
                    import_limit = int(import_limit_combo)
                else:
                    import_limit = n_frames

                if import_crop_mode != "None":
                    image_shape[2] = image_shape[2]//2

                image_shape = (import_limit, image_shape[1], image_shape[2])

                shared_image = self.create_import_shared_image(image_size)
                shared_images[dataset_name] = shared_image

                image_dict = {"path": path,
                              "dataset_name": dataset_name,
                              "shared_image": shared_image,
                              "import_crop_mode": import_crop_mode,
                              "n_frames": n_frames,
                              "image_shape": image_shape,
                              "dtype": dtype,
                              }

                image_list.append(image_dict)

                if dataset_name not in import_dict.keys():
                    import_dict[dataset_name] = {"path":path,
                                                 "import_limit": import_limit,
                                                 "import_crop_mode": import_crop_mode,
                                                 "image_shape": image_shape,
                                                 "dtype": dtype,}

        except:
            print(traceback.format_exc())

        return image_list, shared_images, import_dict

    def populate_import_compute_jobs(self, image_list):

        if self.verbose:
            print(f"Populating import compute jobs.")

        compute_jobs = []

        for image_dict in image_list:

            compute_jobs.append({"stop_event": self.stop_event,
                                 **image_dict})

        return compute_jobs

    def process_compute_jobs(self, compute_jobs, progress_callback=None):

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
                results = [future.result() for future in futures]

        if self.verbose:
            print("Finished processing compute jobs.")

    def populate_import_dataset_dict(self, import_dict):

        try:

            concat_images = self.gui.import_concatenate.isChecked()

            if self.verbose:
                print("Populating dataset dict")

            image_dict = {}

            for dataset_name, dataset_dict in import_dict.items():

                path = dataset_dict["path"]
                image_shape = dataset_dict["image_shape"]
                dtype = dataset_dict["dtype"]

                shared_mem = self.shared_images[dataset_name]

                image = np.ndarray(image_shape, dtype=dtype, buffer=shared_mem.buf).copy()
                image = image.astype(np.uint16)

                shared_mem.close()
                shared_mem.unlink()

                if dataset_name not in import_dict.keys():
                    import_dict[dataset_name] = {"data": []}

                import_dict[dataset_name]["data"] = image
                import_dict[dataset_name]["path"] = path

            if concat_images == True:

                dataset_list = list(import_dict.keys())

                image_list = []
                path_list = []

                for dataset_name in dataset_list:
                    dataset_image = import_dict[dataset_name].pop("data")
                    dataset_path = import_dict[dataset_name].pop("path")

                    dataset_path = [dataset_path]*dataset_image.shape[0]

                    image_list.append(dataset_image)
                    path_list.extend(dataset_path)

                image_list = np.concatenate(image_list, axis=0)

                if dataset_list[0] not in self.dataset_dict.keys():
                    self.dataset_dict[dataset_list[0]] = {"data": image_list, "path": path_list}

            else:
                dataset_list = list(import_dict.keys())

                for dataset_name in dataset_list:
                    dataset_dict = import_dict.pop(dataset_name)
                    self.dataset_dict[dataset_name] = dataset_dict

            print(self.dataset_dict.keys())

        except:
            print(traceback.format_exc())
            pass

    def import_data(self, progress_callback=None, paths=[]):

        image_list, self.shared_images, import_dict = self.populate_import_lists(paths=paths)

        compute_jobs = self.populate_import_compute_jobs(image_list)

        self.process_compute_jobs(compute_jobs, progress_callback=progress_callback)

        self.populate_import_dataset_dict(import_dict)

    def import_data_finished(self):

        self.populate_dataset_selectors()
        self.update_ui()
        # self.update_active_image()

    def populate_dataset_selectors(self):

        dataset_selectors = ["import_picasso_dataset",
                             "cellpose_dataset",
                             "moltrack_dataset_selector",
                             "picasso_dataset,"
                             "picasso_filter_dataset",
                             "picasso_render_dataset",
                             "tracking_dataset",
                             "locs_export_dataset"
                             ]

        for selector_name in dataset_selectors:

            if hasattr(self.gui, selector_name):
                getattr(self.gui, selector_name).clear()
                getattr(self.gui, selector_name).addItems(self.dataset_dict.keys())


    def init_import_data(self):

        try:

            desktop = os.path.expanduser("~/Desktop")
            paths = QFileDialog.getOpenFileNames(self, 'Open file', desktop, "Image files (*.tif *.fits)")[0]

            paths = [path for path in paths if path != ""]

            if paths != []:

                self.update_ui(init=True)

                self.worker = Worker(self.import_data, paths=paths)
                self.worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.import_progressbar))
                self.worker.signals.finished.connect(self.import_data_finished)
                self.worker.signals.error.connect(self.update_ui)
                self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass