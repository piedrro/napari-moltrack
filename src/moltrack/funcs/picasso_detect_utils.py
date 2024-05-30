import numpy as np
import traceback

import pandas as pd

from moltrack.funcs.compute_utils import Worker
import time
import os
from multiprocessing import shared_memory
from picasso import localize
from picasso.localize import get_spots, identify_frame
from picasso.gaussmle import gaussmle
from picasso import gausslq
from picasso import postprocess
from functools import partial
import concurrent.futures
import multiprocessing
from picasso.render import render
from shapely.geometry import Point, Polygon
from multiprocessing import Manager, Event
import numba
from numba import jit, types,typed
from numba.typed import Dict
import numpy as np
import pandas as pd






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
            net_gradient = locs.net_gradient
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
            net_gradient = locs.net_gradient

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
        locs["net_gradient"] = net_gradient

        locs = locs.to_records(index=False)

    except:
        pass

    return locs


def fit_spots_lq(spots, locs, box, progress_list):

    theta = np.empty((len(spots), 6), dtype=np.float32)
    theta.fill(np.nan)
    for i, spot in enumerate(spots):

        theta[i] = gausslq.fit_spot(spot)
        progress_list.append(1)

    locs = locs_from_fits(locs, theta, box, gpu_fit=False)

    return locs

def remove_segmentation_locs(locs, polygons):

    if len(polygons) > 0 and len(locs) > 0:

        loclist = pd.DataFrame(locs).to_dict(orient="records")

        filtered_locs = []

        for loc in loclist:
            point = Point(loc["x"], loc["y"])

            for polygon_index, polygon in enumerate(polygons):
                if polygon.contains(point):
                    loc["segmentation"] = polygon_index
                    filtered_locs.append(loc)

        if len(filtered_locs):
            locs = pd.DataFrame(filtered_locs).to_records(index=False)
        else:
            locs = []

    else:
        locs = []

    return locs



def detect_picaso_locs(dat, progress_list, fit_list):

    result = None

    try:
        min_net_gradient = dat["min_net_gradient"]
        box_size = dat["box_size"]
        roi = dat["roi"]
        dataset = dat["dataset"]
        start_index = dat["start_index"]
        end_index = dat["end_index"]
        detect = dat["detect"]
        fit = dat["fit"]
        remove_overlapping = dat["remove_overlapping"]
        stop_event = dat["stop_event"]
        polygon_filter = dat["polygon_filter"]
        polygons = dat["polygons"]

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

                if polygon_filter:
                    locs = remove_segmentation_locs(locs, polygons)

                if len(locs) > 0:

                    try:

                        image = np.expand_dims(frame, axis=0)
                        camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }
                        spot_data = get_spots(image, locs, box_size, camera_info)

                        locs.frame = frame_index

                        locs = pd.DataFrame(locs)

                        #instert dataset at column 0
                        locs.insert(0, "dataset", dataset)

                        locs = locs.to_records(index=False)

                        for loc, spot in zip(locs, spot_data):
                            loc_list.append(loc)
                            spot_list.append(spot)

                        progress_list.append(1)

                    except:
                        pass

            if len(loc_list) > 0:
                result = loc_list, spot_list

    except:
        print(traceback.format_exc())
        result = None

    return result

def picasso_detect(dat, progress_list):

    result = None

    try:

        frame_index = dat["frame_index"]
        min_net_gradient = dat["min_net_gradient"]
        box_size = dat["box_size"]
        roi = dat["roi"]
        dataset = dat["dataset"]
        channel = dat["channel"]
        detect = dat["detect"]
        fit = dat["fit"]
        remove_overlapping = dat["remove_overlapping"]
        stop_event = dat["stop_event"]
        polygon_filter = dat["polygon_filter"]
        polygons = dat["polygons"]

        if not stop_event.is_set():

            # Access the shared memory
            shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            frame = np_array.copy()

            if detect:
                locs = identify_frame(frame, min_net_gradient, box_size, 0, roi=roi)

                if remove_overlapping:
                    # overlapping removed prior to fitting to increase speed
                    locs = remove_overlapping_locs(locs, box_size)

            else:
                locs = dat["frame_locs"]

            expected_loc_length = 4

            if fit:
                expected_loc_length = 12
                try:
                    image = np.expand_dims(frame, axis=0)
                    camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }
                    spot_data = get_spots(image, locs, box_size, camera_info)

                    theta, CRLBs, likelihoods, iterations = gaussmle(spot_data, eps=0.001, max_it=1000, method="sigma")
                    locs = localize.locs_from_fits(locs.copy(), theta, CRLBs, likelihoods, iterations, box_size)

                    locs.frame = frame_index

                    if remove_overlapping:
                        # sometimes locs can overlap after fitting
                        locs = remove_overlapping_locs(locs, box_size)

                except:
                    # print(traceback.format_exc())
                    pass

            for loc in locs:
                loc.frame = frame_index

            if polygon_filter:

                if len(polygons) > 0 and len(locs) > 0:

                    expected_loc_length += 1

                    loclist = pd.DataFrame(locs).to_dict(orient="records")

                    filtered_locs = []

                    for loc in loclist:
                        point = Point(loc["x"], loc["y"])

                        for polygon_index, polygon in enumerate(polygons):
                            if polygon.contains(point):
                                loc["segmentation"] = polygon_index
                                filtered_locs.append(loc)

                    if len(filtered_locs):
                        locs = pd.DataFrame(filtered_locs).to_records(index=False)
                    else:
                        locs = []

                else:
                    locs = []

            if len(locs) > 0:

                locs = [loc for loc in locs if len(loc) == expected_loc_length]
                locs = np.array(locs).view(np.recarray)

            result = {"dataset": dataset, "channel": channel,
                      "frame_index": frame_index,"locs": locs}

    except:
        print(traceback.format_exc())
        pass

    progress_list.append(dat["frame_index"])

    return result


class _picasso_detect_utils:

    def populate_localisation_dict(self, loc_dict, render_loc_dict, detect_mode,
            image_channel, box_size, fitted=False):

        if self.verbose:
            print("Populating localisation dictionary...")

        detect_mode = detect_mode.lower()

        try:

            for dataset_name, locs in loc_dict.items():

                if detect_mode == "localisations":

                    if dataset_name not in self.localisation_dict["localisations"].keys():
                        self.localisation_dict["localisations"][dataset_name] = {}
                    if image_channel not in self.localisation_dict["localisations"][dataset_name].keys():
                        self.localisation_dict["localisations"][dataset_name][image_channel.lower()] = {}


                    fiducial_dict = {"localisations": []}

                    fiducial_dict["localisations"] = locs.copy()
                    fiducial_dict["fitted"] = fitted
                    fiducial_dict["box_size"] = box_size

                    self.localisation_dict["localisations"][dataset_name][image_channel.lower()] = fiducial_dict.copy()

                else:

                    self.localisation_dict["bounding_boxes"]["localisations"] = locs.copy()
                    self.localisation_dict["bounding_boxes"]["fitted"] = fitted
                    self.localisation_dict["bounding_boxes"]["box_size"] = box_size


        except:
            print(traceback.format_exc())
            self.gui.picasso_progressbar.setValue(0)
            self.gui.picasso_detect.setEnabled(True)
            self.gui.picasso_fit.setEnabled(True)
            self.gui.picasso_detectfit.setEnabled(True)

    def _picasso_wrapper_finished(self):

        try:

            dataset_name = self.gui.picasso_dataset.currentText()

            if dataset_name == "All Datasets":
                dataset_name = self.active_dataset

            self.update_active_image(dataset=dataset_name)

            self.draw_localisations()

            self.update_filter_criterion()
            self.update_criterion_ranges()

            self.update_ui()

        except:
            print(traceback.format_exc())

    def get_frame_locs(self, dataset_name, image_channel, frame_index):

        try:

            loc_dict, n_locs, _ = self.get_loc_dict(dataset_name,
                image_channel.lower(), type = "localisations")

            if "localisations" not in loc_dict.keys():
                return None
            elif len(loc_dict["localisations"]) == 0:
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

            if "localisations" not in loc_dict.keys():
                return None
            elif len(loc_dict["localisations"]) == 0:
                return None
            else:
                locs = loc_dict["localisations"]
                locs = locs[(locs.frame >= start_index) & (locs.frame <= end_index)]

                return locs.copy()

        except:
            print(traceback.format_exc())
            return None

    def get_fit_data(self, dataset_list, box_size, frame_index=None):

        try:

            loc_list = []
            spot_list = []

            for dataset in dataset_list:

                if dataset in self.localisation_dict.keys():

                    loc_dict = self.localisation_dict[dataset]

                    locs = loc_dict["localisations"].copy()

                    if frame_index is not None:
                        locs = locs[locs.frame == frame_index]

                    if len(locs) > 0:

                        image_dict = self.dataset_dict[dataset]

                        if "dataset" not in locs.dtype.names:
                            locs = pd.DataFrame(locs)
                            locs.insert(0, "dataset", dataset)
                            locs = locs.to_records(index=False)

                        image = image_dict.pop("data")

                        camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }
                        spot_data = get_spots(image, locs, box_size, camera_info)

                        loc_list.append(locs)
                        spot_list.append(spot_data)

                        image_dict["data"] = image

        except:
            print(traceback.format_exc())
            pass

        if len(loc_list) > 0:
            loc_list = np.hstack(loc_list).view(np.recarray).copy()
            spot_list = np.concatenate(spot_list, axis=0)

        return loc_list, spot_list



    def populate_picasso_detect_jobs(self, detect, fit,
            min_net_gradient, roi):

        try:

            compute_jobs = []
            n_frames = 0

            box_size = int(self.gui.picasso_box_size.currentText())
            remove_overlapping = self.gui.picasso_remove_overlapping.isChecked()
            polygon_filter = self.gui.picasso_segmentation_filtering.isChecked()

            segmentation_polygons = self.get_segmentation_polygons()

            compute_jobs = []

            if self.verbose:
                print("Creating Picasso compute jobs...")

            for image_chunk in self.shared_chunks:

                compute_job = {"dataset": image_chunk["dataset"],
                               "start_index": image_chunk["start_index"],
                               "end_index": image_chunk["end_index"],
                               "shared_memory_name": image_chunk["shared_memory_name"],
                               "shape": image_chunk["shape"],
                               "dtype": image_chunk["dtype"],
                               "detect": detect,
                               "fit": fit,
                               "min_net_gradient": int(min_net_gradient),
                               "box_size": int(box_size),
                               "roi": roi,
                               "remove_overlapping": remove_overlapping,
                               "polygon_filter":polygon_filter,
                               "polygons": segmentation_polygons,
                               "stop_event": self.stop_event, }

                compute_jobs.append(compute_job)
                n_frames += image_chunk["end_index"] - image_chunk["start_index"]

            if self.verbose:
                print(f"Created {len(compute_jobs)} compute jobs...")

        except:
            print(traceback.format_exc())

        return compute_jobs, n_frames


    def detect_spots_parallel(self, detect_jobs, executor, manager,
            n_workers, n_frames, fit, progress_callback=None,
            timeout_duration = 10):

        progress_list = manager.list()
        fit_jobs = manager.list()

        futures = {executor.submit(detect_picaso_locs,
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
                result = future.result(timeout=timeout_duration)  # Process result here

                if result is not None:
                    result_locs, result_spots = result
                    locs.extend(result_locs)
                    spots.extend(result_spots)

            except concurrent.futures.TimeoutError:
                print(f"Task {job} timed out after {timeout_duration} seconds.")
            except Exception as e:
                print(f"Error occurred in task {job}: {e}")

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
            pass

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
            min_net_gradient, dataset_list = [], frame_index = None, gpu_fit=True):

        try:
            locs, fitted = [], False

            frame_mode = self.gui.picasso_frame_mode.currentText()
            detect_mode = self.gui.picasso_detect_mode.currentText()
            box_size = int(self.gui.picasso_box_size.currentText())
            roi = self.generate_roi()

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
                            frame_index=frame_index, )

                        detect_jobs, n_frames = self.populate_picasso_detect_jobs(detect,
                            fit, min_net_gradient, roi)

                        print(f"Starting Picasso {len(detect_jobs)} compute jobs...")

                        if len(detect_jobs) > 0:
                            if self.verbose:
                                print(f"Starting Picasso {len(detect_jobs)} compute jobs...")

                            print(f"Detecting spots in {n_frames} frames...")

                            locs, spots = self.detect_spots_parallel(detect_jobs, executor, manager,
                                n_workers, n_frames, fit, progress_callback)

                            print(f"Detected {len(locs)} spots")

                    if detect is False and fit is True:

                        locs, spots = self.get_fit_data(dataset_list, box_size, frame_index)

                    if len(locs) > 0 and fit == True:

                        if gpu_fit:

                            print(f"Fitting {len(locs)} spots on GPU...")

                            locs = self.fit_spots_gpu(locs, spots, box_size)

                        else:

                            print(f"Fitting {len(locs)} spots on CPU...")

                            locs = self.fit_spots_parallel(locs, spots, box_size, executor, manager,
                                n_workers, detect, progress_callback)

                        fitted = True

                        print(f"Fitted {len(locs)} spots")

                    else:
                        fitted = False

            # time to process locs

            import time

            start = time.time()
            self.process_locs(locs, detect_mode, box_size, fitted=fitted)
            end = time.time()

            print(f"Processed {len(locs)} locs in {end-start} seconds")

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

                dataset_list = list(set(locs["dataset"]))

                for dataset in dataset_list:

                    if dataset not in self.localisation_dict.keys():
                        self.localisation_dict[dataset] = {}

                    result_dict = self.localisation_dict[dataset]

                    dataset_locs = locs[locs["dataset"] == dataset]

                    if len(dataset_locs) == 0:

                        result_dict["localisations"] = []
                        result_dict["fitted"] = False
                        result_dict["box_size"] = box_size

                    else:

                        loc_cols = list(dataset_locs.dtype.names)

                        if "dataset" in loc_cols or "channel" in loc_cols:
                            dataset_locs = pd.DataFrame(dataset_locs)

                            if "dataset" in loc_cols:
                                dataset_locs = dataset_locs.drop(columns=["dataset"])

                            dataset_locs = dataset_locs.to_records(index=False)

                        result_dict["localisations"] = dataset_locs.copy()
                        result_dict["fitted"] = fitted
                        result_dict["box_size"] = box_size

        except:
            print(traceback.format_exc())
            return None

        print("Finished processing locs")


    def init_picasso(self, detect = False, fit = False):

        try:
            if self.dataset_dict != {}:

                dataset_name = self.gui.picasso_dataset.currentText()
                min_net_gradient = self.gui.picasso_min_net_gradient.text()
                frame_mode = self.gui.picasso_frame_mode.currentText()
                minimise_ram = self.gui.picasso_minimise_ram.isChecked()
                smlm_fit_mode = self.gui.smlm_fit_mode.currentText()

                if self.gpufit_available and smlm_fit_mode == "GPUFit":
                    gpu_fit = True
                else:
                    gpu_fit = False

                if min_net_gradient.isdigit() and dataset_name != "":

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

                    self.worker = Worker(self._picasso_wrapper,
                        detect=detect, fit=fit,
                        min_net_gradient=min_net_gradient,
                        dataset_list=dataset_list,
                        gpu_fit=gpu_fit,
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

        roi = None

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

                if dataset == "All Datasets":
                    dataset = list(self.dataset_dict.keys())[0]

                image_shape = self.dataset_dict[dataset]["data"].shape

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

        except:
            print(traceback.format_exc())
            pass

        return roi

    def export_picasso_locs(self, locs):

        if self.verbos:
            print("Exporting Picasso locs")

        try:

            dataset_name = self.gui.picasso_dataset.currentText()
            image_channel = self.gui.picasso_channel.currentText()
            min_net_gradient = int(self.gui.picasso_min_net_gradient.text())
            box_size = int(self.gui.picasso_box_size.currentText())

            path = self.dataset_dict[dataset_name][image_channel.lower()]["path"]
            image_shape = self.dataset_dict[dataset_name][image_channel.lower()]["data"].shape

            base, ext = os.path.splitext(path)
            path = base + f"_{image_channel}_picasso_locs.hdf5"

            info = [{"Byte Order": "<", "Data Type": "uint16", "File": path,
                     "Frames": image_shape[0], "Height": image_shape[1],
                     "Micro-Manager Acquisiton Comments": "", "Width": image_shape[2], },
                    {"Box Size": box_size, "Fit method": "LQ, Gaussian",
                     "Generated by": "Picasso Localize",
                     "Min. Net Gradient": min_net_gradient, "Pixelsize": 130, "ROI": None, }]

            from picasso.io import save_locs
            # save_locs(path, locs, info)

        except:
            print(traceback.format_exc())
            pass
