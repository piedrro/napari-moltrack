import concurrent.futures
import math
import multiprocessing
import time
import traceback
from functools import partial
from multiprocessing import Manager, shared_memory

import cv2
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info
from picasso import gausslq, postprocess
from picasso.localize import get_spots, identify_frame
from skimage.feature import peak_local_max

from moltrack.funcs.compute_utils import Worker


def precompute_kernels(lnoise=0, lobject=1):

    # Determine the kernel size based on the larger of the two scales
    size = 2 * max(math.ceil(5 * lnoise), round(lobject)) + 1

    # Handling Gaussian Kernel
    if lnoise == 0:
        gaussian_kernel = np.array([[1]])
    else:
        gaussian_kernel_1d = cv2.getGaussianKernel(size, math.sqrt(2) * lnoise)
        gaussian_kernel = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)

    gaussian_kernel_convolved = cv2.filter2D(gaussian_kernel, -1, gaussian_kernel)

    # Handling Boxcar Kernel
    if lobject != 0:
        boxcar_kernel = np.ones((size, size))
        boxcar_kernel = boxcar_kernel / np.sum(boxcar_kernel)
        boxcar_kernel_convolved = cv2.filter2D(boxcar_kernel, -1, boxcar_kernel)
    else:
        boxcar_kernel_convolved = None

    lzero = round(max(lobject, math.ceil(5 * lnoise)))

    return gaussian_kernel_convolved, boxcar_kernel_convolved, lzero



def bandpass(image_array, kernels):

    gaussian_kernel_convolved, boxcar_kernel_convolved, lzero = kernels

    if boxcar_kernel_convolved is not None:
        image_filtered = cv2.filter2D(image_array, -1, gaussian_kernel_convolved - boxcar_kernel_convolved)
    else:
        image_filtered = cv2.filter2D(image_array, -1, gaussian_kernel_convolved)

    image_filtered[:lzero, :] = 0
    image_filtered[-lzero:, :] = 0
    image_filtered[:, :lzero] = 0
    image_filtered[:, -lzero:] = 0

    image_filtered[image_filtered < 0] = 0

    return image_filtered



def detect_moltrack_locs(dat, progress_list, fit_list):

    result = None

    try:

        box_size = dat["box_size"]
        dataset = dat["dataset"]
        channel = dat["channel"]
        start_index = dat["start_index"]
        remove_overlapping = dat["remove_overlapping"]
        stop_event = dat["stop_event"]
        threshold = dat["threshold"]
        kernel_size = dat["kernel_size"]

        loc_list = []
        spot_list = []

        if not stop_event.is_set():

            kernel_size = max(round(kernel_size), 1)
            kernels = precompute_kernels(1, kernel_size)

            # Access the shared memory
            shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            image_chunk = np_array.copy()

            for array_index, frame in enumerate(image_chunk):

                try:

                    frame_index = start_index + array_index

                    filtered_frame = bandpass(frame.copy(),kernels)

                    locs = peak_local_max(filtered_frame, min_distance=1,
                        threshold_abs=threshold)

                    locs = pd.DataFrame(locs, columns=["y", "x"])
                    locs.insert(0, "frame", 0)
                    locs = locs.to_records(index=False)

                    if remove_overlapping:
                        locs = remove_overlapping_locs(locs, box_size)

                    if len(locs) > 0:

                        image = np.expand_dims(frame.copy(), axis=0)
                        camera_info = {"Baseline": 100.0, "Gain": 1, "Sensitivity": 1.0, "qe": 0.9, }
                        spot_data = get_spots(image, locs, box_size, camera_info)

                        locs.frame = frame_index

                        locs = pd.DataFrame(locs)
                        locs.insert(0, "dataset", dataset)
                        locs.insert(1, "channel", channel)

                        locs = locs.to_records(index=False)

                        for loc, spot in zip(locs, spot_data):
                            loc_list.append(loc)
                            spot_list.append(spot)

                except:
                    pass

                progress_list.append(1)

        if len(loc_list) > 0:
            result = loc_list, spot_list

    except:
        print(traceback.format_exc())
        result = None

    return result



def remove_overlapping_locs(locs, box_size):

    try:

        coordinates = np.vstack((locs.y, locs.x)).T

        # Calculate all pairwise differences
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]

        # Calculate squared distances
        dist_squared = np.sum(diff ** 2, axis=-1)

        # Check if the array is of integer type
        if coordinates.dtype.kind in 'iu':
            # Use the maximum integer value for the diagonal if the array is of integer type
            max_int_value = np.iinfo(coordinates.dtype).max
            np.fill_diagonal(dist_squared, max_int_value)
        else:
            # Use infinity for the diagonal if the array is of float type
            np.fill_diagonal(dist_squared, np.inf)

        # Identify overlapping coordinates (distance less than X)
        overlapping = np.any(dist_squared < box_size ** 2, axis=1)

        non_overlapping_locs = locs[~overlapping]
        non_overlapping_locs = np.array(non_overlapping_locs).view(np.recarray)

    except:
        pass

    return non_overlapping_locs

def cut_spots(movie, ids_frame, ids_x, ids_y, box):

    n_spots = len(ids_x)
    r = int(box / 2)
    spots = np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r : yc + r + 1, xc - r : xc + r + 1]

    return spots

def locs_from_fits(locs, theta, box, em=False, gpu_fit=False):

    try:

        if gpu_fit:
            box_offset = int(box / 2)
            x = theta[:, 1] + locs.x - box_offset
            y = theta[:, 2] + locs.y - box_offset
            lpx = postprocess.localization_precision(theta[:, 0], theta[:, 3], theta[:, 5], em=em)
            lpy = postprocess.localization_precision(theta[:, 0], theta[:, 4], theta[:, 5], em=em)
            a = np.maximum(theta[:, 3], theta[:, 4])
            b = np.minimum(theta[:, 3], theta[:, 4])
            ellipticity = (a - b) / a
            photons = theta[:, 0]
            sx = theta[:, 3]
            sy = theta[:, 4]
            bg = theta[:, 5]
        else:
            x = theta[:, 0] + locs.x  # - box_offset
            y = theta[:, 1] + locs.y  # - box_offset
            lpx = postprocess.localization_precision(theta[:, 2], theta[:, 4], theta[:, 3], em=em)
            lpy = postprocess.localization_precision(theta[:, 2], theta[:, 5], theta[:, 3], em=em)
            a = np.maximum(theta[:, 4], theta[:, 5])
            b = np.minimum(theta[:, 4], theta[:, 5])
            ellipticity = (a - b) / a
            photons = theta[:, 2]
            sx = theta[:, 4]
            sy = theta[:, 5]
            bg = theta[:, 3]

        if "net_gradient" in locs.dtype.names:
            net_gradient = locs.net_gradient
        else:
            net_gradient = None

        locs = pd.DataFrame(locs)

        locs["x"] = x
        locs["y"] = y
        locs["photons"] = photons
        locs["sx"] = sx
        locs["sy"] = sy
        locs["bg"] = bg
        locs["lpx"] = lpx
        locs["lpy"] = lpy
        locs["ellipticity"] = ellipticity

        if net_gradient is not None:
            locs["net_gradient"] = net_gradient

        locs = locs.to_records(index=False)

    except:
        print(traceback.format_exc())

    return locs

def fit_spots_lq(spots, locs, box, progress_list):

    theta = np.empty((len(spots), 6), dtype=np.float32)
    theta.fill(np.nan)
    for i, spot in enumerate(spots):

        theta[i] = gausslq.fit_spot(spot)
        progress_list.append(1)

    locs = locs_from_fits(locs, theta, box, gpu_fit=False)

    return locs

def detect_picaso_locs(dat, progress_list, fit_list):

    result = None

    try:
        min_net_gradient = dat["threshold"]
        box_size = dat["box_size"]
        roi = dat["roi"]
        dataset = dat["dataset"]
        channel = dat["channel"]
        start_index = dat["start_index"]
        remove_overlapping = dat["remove_overlapping"]
        stop_event = dat["stop_event"]

        loc_list = []
        spot_list = []

        if not stop_event.is_set():

            # Access the shared memory
            shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            image_chunk = np_array.copy()

            for array_index, frame in enumerate(image_chunk):

                frame_index = start_index + array_index

                locs = identify_frame(frame, min_net_gradient,
                    box_size, 0, roi=roi)

                if remove_overlapping:
                    locs = remove_overlapping_locs(locs, box_size)


                if len(locs) > 0:

                    try:

                        image = np.expand_dims(frame, axis=0)
                        camera_info = {"Baseline": 100.0, "Gain": 1,
                                       "Sensitivity": 1.0, "qe": 0.9, }

                        spot_data = get_spots(image, locs, box_size, camera_info)

                        locs.frame = frame_index

                        locs = pd.DataFrame(locs)

                        #instert dataset at column 0
                        locs.insert(0, "dataset", dataset)
                        locs.insert(1, "channel", channel)

                        locs = locs.to_records(index=False)

                        for loc, spot in zip(locs, spot_data):
                            loc_list.append(loc)
                            spot_list.append(spot)

                        progress_list.append(1)

                    except:
                        print(traceback.format_exc())

            if len(loc_list) > 0:
                result = loc_list, spot_list

    except:
        print(traceback.format_exc())
        result = None

    return result

class _picasso_detect_utils:

    def _picasso_wrapper_finished(self):

        try:

            dataset_name = self.gui.picasso_dataset.currentText()
            channel_name = self.gui.picasso_channel.currentText()

            if dataset_name == "All Datasets":
                dataset_name = self.active_dataset
            if channel_name == "All Channels":
                channel_name = self.active_channel

            self.update_active_image(dataset=dataset_name,
                channel=channel_name)

            self.draw_localisations()

            if channel_name != "All Channels":
                self.gui.picasso_filter_channel.blockSignals(True)
                self.gui.picasso_filter_channel.setCurrentText(channel_name)
                self.gui.picasso_filter_channel.blockSignals(False)

                self.gui.picasso_render_channel.blockSignals(True)
                self.gui.picasso_render_channel.setCurrentText(channel_name)
                self.gui.picasso_render_channel.blockSignals(False)

                self.gui.tracking_channel.blockSignals(True)
                self.gui.tracking_channel.setCurrentText(channel_name)
                self.gui.tracking_channel.blockSignals(False)

            self.update_filter_criterion()
            self.update_criterion_ranges()

            self.update_ui()

        except:
            print(traceback.format_exc())

    def get_frame_locs(self, dataset_name, image_channel, frame_index):

        try:

            loc_dict, n_locs, _ = self.get_loc_dict(dataset_name,
                image_channel.lower(), type = "localisations")

            if "localisations" not in loc_dict.keys() or len(loc_dict["localisations"]) == 0:
                return None
            else:
                locs = loc_dict["localisations"]
                locs = locs[locs.frame == frame_index]

                return locs.copy()

        except:
            print(traceback.format_exc())
            return None

    def get_chunk_locs(self, dataset_name, image_channel,
            start_index, end_index):

        try:

            loc_dict, n_locs, _ = self.get_loc_dict(dataset_name,
                image_channel.lower(), type = "localisations")

            if "localisations" not in loc_dict.keys() or len(loc_dict["localisations"]) == 0:
                return None
            else:
                locs = loc_dict["localisations"]
                locs = locs[(locs.frame >= start_index) & (locs.frame <= end_index)]

                return locs.copy()

        except:
            print(traceback.format_exc())
            return None

    def get_fit_data(self, dataset_list, channel_list, box_size, frame_index=None):

        try:

            loc_list = []
            spot_list = []

            for dataset in dataset_list:

                if dataset in self.localisation_dict.keys():

                    for channel in channel_list:

                        if channel in self.localisation_dict[dataset].keys():

                            loc_dict = self.localisation_dict[dataset][channel]

                            locs = loc_dict["localisations"].copy()

                            if frame_index is not None:
                                locs = locs[locs.frame == frame_index]

                            if len(locs) > 0:

                                image_dict = self.dataset_dict[dataset]["images"]

                                if "dataset" not in locs.dtype.names:
                                    locs = pd.DataFrame(locs)
                                    locs.insert(0, "dataset", dataset)
                                    locs.insert(1, "channel", channel)
                                    locs = locs.to_records(index=False)

                                image = image_dict.pop(channel)

                                camera_info = {"Baseline": 100.0, "Gain": 1, "Sensitivity": 1.0, "qe": 0.9, }
                                spot_data = get_spots(image, locs, box_size, camera_info)

                                loc_list.append(locs)
                                spot_list.append(spot_data)

                                image_dict[channel] = image

        except:
            print(traceback.format_exc())

        if len(loc_list) > 0:
            loc_list = np.hstack(loc_list).view(np.recarray).copy()
            spot_list = np.concatenate(spot_list, axis=0)

        return loc_list, spot_list



    def populate_picasso_detect_jobs(self, detect, fit, roi_dict):

        try:

            compute_jobs = []
            n_frames = 0

            box_size = int(self.gui.picasso_box_size.value())
            remove_overlapping = self.gui.picasso_remove_overlapping.isChecked()
            threshold = int(self.gui.smlm_threshold.value())
            kernel_size = int(self.gui.moltrack_kernel_size.text())

            compute_jobs = []

            if self.verbose:
                print("Creating Picasso compute jobs...")

            for image_chunk in self.shared_chunks:

                dataset = image_chunk["dataset"]
                channel = image_chunk["channel"]

                compute_job = {"dataset": image_chunk["dataset"],
                               "channel": image_chunk["channel"],
                               "start_index": image_chunk["start_index"],
                               "end_index": image_chunk["end_index"],
                               "shared_memory_name": image_chunk["shared_memory_name"],
                               "shape": image_chunk["shape"],
                               "dtype": image_chunk["dtype"],
                               "detect": detect,
                               "fit": fit,
                               "threshold": int(threshold),
                               "box_size": int(box_size),
                               "kernel_size": kernel_size,
                               "roi": roi_dict[dataset][channel],
                               "remove_overlapping": remove_overlapping,
                               "stop_event": self.stop_event, }

                compute_jobs.append(compute_job)
                n_frames += image_chunk["end_index"] - image_chunk["start_index"]

            if self.verbose:
                print(f"Created {len(compute_jobs)} compute jobs...")

        except:
            print(traceback.format_exc())

        return compute_jobs, n_frames




    def detect_spots_parallel(self, detect_mode, detect_jobs, executor, manager,
            n_workers, n_frames, fit, progress_callback=None,
            timeout_duration = 100):

        progress_list = manager.list()
        fit_jobs = manager.list()

        if detect_mode == "Picasso":
            detect_fn = detect_picaso_locs
        else:
            detect_fn = detect_moltrack_locs

        start_time = time.time()

        futures = {executor.submit(detect_fn,
            job, progress_list, fit_jobs,): job for job in detect_jobs}

        while any(not future.done() for future in futures):
            progress = (sum(progress_list) / n_frames)

            if fit == True:
                progress = progress*50
            else:
                progress = progress*100

            if progress_callback is not None:
                progress_callback.emit(progress)

        locs = []
        spots = []

        for future in concurrent.futures.as_completed(futures):

            job = futures[future]
            try:
                result = future.result()  # Process result here

                if result is not None:
                    result_locs, result_spots = result
                    locs.extend(result_locs)
                    spots.extend(result_spots)

            except concurrent.futures.TimeoutError:
                print(f"Task {job} timed out after {timeout_duration} seconds.")
            except Exception as e:
                print(f"Error occurred in task {job}: {e}")


        end_time = time.time()
        show_info(f"Finished detecting spots in {n_frames} frames in {end_time - start_time} seconds")

        if len(locs) > 0:
            locs = np.hstack(locs).view(np.recarray).copy()
            spots = np.stack(spots, axis=0)

        return locs, spots



    def fit_spots_gpu(self, locs, spots, box_size,
            tolerance=1e-2, max_number_iterations=20):

        try:
            from pygpufit import gpufit as gf

            size = spots.shape[1]
            initial_parameters = gausslq.initial_parameters_gpufit(spots, size)
            spots.shape = (len(spots), (size * size))
            model_id = gf.ModelID.GAUSS_2D_ELLIPTIC

            result = gf.fit(
                spots,
                None,
                model_id,
                initial_parameters, tolerance=tolerance,
                max_number_iterations=max_number_iterations,
            )

            parameters, states, chi_squares, number_iterations, exec_time = result

            parameters[:, 0] *= 2.0 * np.pi * parameters[:, 3] * parameters[:, 4]

            locs = locs_from_fits(locs, parameters, box_size, gpu_fit=True)


        except:
            print(traceback.format_exc())

        return locs


    def fit_spots_parallel(self, locs, spots, box_size, executor, manager, n_workers,
             detect=False, progress_callback=None):

        try:

            num_spots = len(spots)
            num_tasks = 100 * n_workers

            progress_list = manager.list()

            # Calculate spots per task using divmod for quotient and remainder
            quotient, remainder = divmod(num_spots, num_tasks)
            spots_per_task = [quotient + 1 if i < remainder else quotient for i in range(num_tasks)]

            # Calculate start indices using numpy
            start_indices = np.cumsum([0] + spots_per_task[:-1])

            futures = [executor.submit(fit_spots_lq,
                spots[start:start + count],
                locs[start:start + count], box_size,
                progress_list) for start, count in zip(start_indices, spots_per_task)]

            while any(not future.done() for future in futures):
                progress = (sum(progress_list) / num_spots)

                if detect:
                    progress = 50 + (progress*50)
                else:
                    progress = progress*100

                if progress_callback is not None:
                    progress_callback.emit(progress)

            locs = [f.result() for f in futures]
            locs = np.hstack(locs).view(np.recarray).copy()

        except:
            print(traceback.format_exc())

        return locs


    def _picasso_wrapper(self, progress_callback, detect, fit,
            dataset_list = [], channel_list = [],
            frame_index = None, detect_mode = "Picasso", fit_mode = "Picasso"):

        try:
            locs, fitted = [], False

            frame_mode = self.gui.picasso_frame_mode.currentText()
            detect_mode = self.gui.smlm_detect_mode.currentText()
            box_size = int(self.gui.picasso_box_size.value())
            roi_dict = self.generate_roi()

            if frame_mode.lower() == "active":
                executor_class = concurrent.futures.ThreadPoolExecutor
                n_workers = 1
            else:
                executor_class = concurrent.futures.ProcessPoolExecutor
                n_workers = int(multiprocessing.cpu_count() * 0.9)

            with Manager() as manager:

                with executor_class(max_workers=n_workers) as executor:

                    if detect is True:

                        self.create_shared_image_chunks(dataset_list=dataset_list,
                            channel_list=channel_list, frame_index=frame_index,
                            chunk_size = 1000)

                        detect_jobs, n_frames = self.populate_picasso_detect_jobs(
                            detect, fit, roi_dict)

                        show_info(f"Starting Picasso {len(detect_jobs)} compute jobs...")

                        if len(detect_jobs) > 0:
                            if self.verbose:
                                print(f"Starting Picasso {len(detect_jobs)} compute jobs...")

                            show_info(f"Detecting spots in {n_frames} frames...")

                            locs, spots = self.detect_spots_parallel(detect_mode, detect_jobs,
                                executor, manager, n_workers, n_frames, fit, progress_callback)

                            show_info(f"Detected {len(locs)} spots")

                    if detect is False and fit is True:

                        locs, spots = self.get_fit_data(dataset_list, channel_list,
                            box_size, frame_index)

                    if len(locs) > 0 and fit == True:

                        if fit_mode == "GPUFit":

                            show_info(f"Fitting {len(locs)} spots on GPU...")

                            locs = self.fit_spots_gpu(locs, spots, box_size)

                        else:

                            show_info(f"Fitting {len(locs)} spots on CPU...")

                            locs = self.fit_spots_parallel(locs, spots, box_size, executor, manager,
                                n_workers, detect, progress_callback)

                            time.sleep(1)

                        fitted = True
                        show_info(f"Fitted {len(locs)} spots")

                    else:
                        fitted = False

            self.process_locs(locs, detect_mode, box_size, fitted=fitted)

            if progress_callback is not None:
                progress_callback.emit(100)

            self.restore_shared_image_chunks()
            self.update_ui()

        except:
            print(traceback.format_exc())

            self.restore_shared_image_chunks()
            self.update_ui()

        return locs, fitted


    def process_locs(self, locs, detect_mode, box_size, fitted=False):

        try:

            if len(locs) > 0:

                loc_columns = list(locs.dtype.names)

                if "lpx" in loc_columns and "lpy" in loc_columns:
                    locs = pd.DataFrame(locs)

                    lpx = np.array(locs["lpx"].values)
                    lpy = np.array(locs["lpy"].values)

                    sigma = np.sqrt((lpx ** 2 + lpy ** 2) / 2)
                    locs["sigma"] = sigma

                    locs = locs.to_records(index=False)

                dataset_list = list(set(locs["dataset"]))
                channel_list = list(set(locs["channel"]))

                for dataset in dataset_list:

                    if dataset not in self.localisation_dict.keys():
                        self.localisation_dict[dataset] = {}

                    dataset_locs = locs[locs["dataset"] == dataset]

                    for channel in channel_list:

                        channel_locs = dataset_locs[dataset_locs["channel"] == channel]

                        if "lpx" in channel_locs.dtype.names:
                            channel_locs = channel_locs[~np.isnan(channel_locs["lpx"])]
                            channel_locs = channel_locs[~np.isnan(channel_locs["lpy"])]

                        if channel not in self.localisation_dict[dataset].keys():
                            self.localisation_dict[dataset][channel] = {}

                        result_dict = self.localisation_dict[dataset][channel]

                        if len(channel_locs) == 0:

                            result_dict["localisations"] = []
                            result_dict["fitted"] = False
                            result_dict["box_size"] = box_size

                        else:

                            loc_cols = list(channel_locs.dtype.names)

                            if "dataset" in loc_cols or "channel" in loc_cols:
                                channel_locs = pd.DataFrame(channel_locs)

                                if "dataset" in loc_cols:
                                    channel_locs = channel_locs.drop(columns=["dataset"])

                                if "channel" in loc_cols:
                                    channel_locs = channel_locs.drop(columns=["channel"])

                                channel_locs = channel_locs.to_records(index=False)

                            result_dict["localisations"] = channel_locs.copy()
                            result_dict["fitted"] = fitted
                            result_dict["box_size"] = box_size

        except:
            print(traceback.format_exc())
            return None

        show_info("Finished processing locs")

    def init_picasso(self, detect = False, fit = False):

        try:
            if self.dataset_dict != {}:

                detect_mode = self.gui.smlm_detect_mode.currentText()
                fit_mode = self.gui.smlm_fit_mode.currentText()
                dataset_name = self.gui.picasso_dataset.currentText()
                channel_name = self.gui.picasso_channel.currentText()
                frame_mode = self.gui.picasso_frame_mode.currentText()
                minimise_ram = self.gui.picasso_minimise_ram.isChecked()
                smlm_fit_mode = self.gui.smlm_fit_mode.currentText()

                if dataset_name != "":

                    self.gui.picasso_progressbar.setValue(0)
                    self.gui.picasso_detect.setEnabled(False)
                    self.gui.picasso_fit.setEnabled(False)
                    self.gui.picasso_detectfit.setEnabled(False)

                    self.update_ui(init=True)

                    if minimise_ram == True and frame_mode.lower() != "active":
                        self.clear_live_images()

                    if frame_mode.lower() == "active":
                        frame_index = self.viewer.dims.current_step[0]
                    else:
                        frame_index = None

                    if dataset_name == "All Datasets":
                        dataset_list = list(self.dataset_dict.keys())
                    else:
                        dataset_list = [dataset_name]

                    if channel_name == "All Channels":

                        channel_list = []
                        for dataset_name in self.dataset_dict.keys():
                            try:
                                image_dict = self.dataset_dict[dataset_name]["images"]
                                channel_list.append(set(image_dict.keys()))
                            except:
                                pass

                        channel_list = set.intersection(*channel_list)
                        channel_list = list(channel_list)

                    else:
                        channel_list = [channel_name]

                    self.worker = Worker(self._picasso_wrapper,
                        detect=detect, fit=fit,
                        dataset_list=dataset_list,
                        channel_list=channel_list,
                        detect_mode=detect_mode,
                        fit_mode=fit_mode,
                        frame_index=frame_index)

                    self.worker.signals.progress.connect(partial(self.moltrack_progress,
                        progress_bar=self.gui.picasso_progressbar))
                    self.worker.signals.finished.connect(self._picasso_wrapper_finished)
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

        except:
            print(traceback.format_exc())

            self.update_ui()


    def generate_roi(self):

        if self.verbose:
            print("Generating ROI")

        border_width = self.gui.picasso_roi_border_width.text()
        window_cropping = self.gui.picasso_window_cropping .isChecked()

        roi_dict = {}

        try:

            generate_roi = False

            if window_cropping:
                layers_names = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisations"]]

                crop = self.viewer.layers[layers_names[0]].corner_pixels[:, -2:]
                [[y1, x1], [y2, x2]] = crop

                generate_roi = True

            else:

                if type(border_width) == str:
                    border_width = int(border_width)
                    if border_width > 0:
                        generate_roi = True
                elif type(border_width) == int:
                    if border_width > 0:
                        generate_roi = True

            if generate_roi:

                dataset = self.gui.picasso_dataset.currentText()
                channel = self.gui.picasso_channel.currentText()

                if dataset == "All Datasets":
                    dataset_list = list(self.dataset_dict.keys())
                else:
                    dataset_list = [dataset]

                for dataset_name in dataset_list:

                    if channel == "All Channels":
                        channel_list = list(self.dataset_dict[dataset_name]["images"].keys())
                    else:
                        channel_list = [channel]

                    for channel_name in channel_list:

                        if dataset_name not in roi_dict:
                            roi_dict[dataset_name] = {}
                        if channel_name not in roi_dict[dataset_name].keys():
                            roi_dict[dataset_name][channel_name] = {}

                        image_dict = self.dataset_dict[dataset_name]["images"]
                        image_shape = image_dict[channel_name].shape

                        frame_shape = image_shape[1:]

                        if window_cropping:

                            border_width = int(border_width)

                            if x1 < border_width:
                                x1 = border_width
                            if y1 < border_width:
                                y1 = border_width
                            if x2 > frame_shape[1] - border_width:
                                x2 = frame_shape[1] - border_width
                            if y2 > frame_shape[0] - border_width:
                                y2 = frame_shape[0] - border_width

                            roi = [[y1, x1], [y2, x2]]

                        else:

                            roi = [[int(border_width), int(border_width)],
                                   [int(frame_shape[0] - border_width), int(frame_shape[1] - border_width)]]

                        roi_dict[dataset_name][channel_name] = roi

        except:
            print(traceback.format_exc())

        return roi_dict
