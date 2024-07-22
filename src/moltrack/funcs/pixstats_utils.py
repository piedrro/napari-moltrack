from moltrack.funcs.compute_utils import Worker
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
from multiprocessing import Manager, shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pandas as pd
from napari.utils.notifications import show_info
import traceback
import numpy as np
import matplotlib.patches as mpatches
from io import BytesIO

class _pixstats_utils:

    def draw_pixstats_mask(self, mode = "tracks"):

        try:

            if mode == "tracks":

                spot_size = int(self.gui.tracks_pixstats_spot_size.currentText())
                spot_shape = self.gui.tracks_pixstats_spot_shape.currentText()
                background_buffer = int(self.gui.tracks_pixstats_bg_buffer.currentText())
                background_width = int(self.gui.tracks_pixstats_bg_width.currentText())

                canvas = self.tracks_pixstats_canvas

            else:

                spot_size = int(self.gui.locs_pixstats_spot_size.currentText())
                spot_shape = self.gui.locs_pixstats_spot_shape.currentText()
                background_buffer = int(self.gui.locs_pixstats_bg_buffer.currentText())
                background_width = int(self.gui.locs_pixstats_bg_width.currentText())

                canvas = self.locs_pixstats_canvas

            mask, buffer_mask, background_mask = _pixstats_utils.generate_localisation_mask(
                spot_size, spot_shape, background_buffer, background_width)


            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            rgb_mask[:,:,0] = mask * 255
            rgb_mask[:,:,2] = background_mask * 255

            ticks = np.arange(-0.5, mask.shape[1], 1)

            fig, ax = plt.subplots(figsize=(6, 6))

            plt.imshow(rgb_mask, interpolation='none')
            plt.xticks(ticks, [])
            plt.yticks(ticks, [])
            plt.grid(color='black', linestyle='-', linewidth=2)

            plt.tick_params(axis='x', which='both', bottom=False, top=False)
            plt.tick_params(axis='y', which='both', left=False, right=False)

            red_patch = mpatches.Patch(color='red', label='Mask')
            black_patch = mpatches.Patch(color='black', label='Buffer')
            blue_patch = mpatches.Patch(color='blue', label='Background')
            plt.legend(handles=[red_patch, black_patch, blue_patch],
                loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3,
                fontsize=18, columnspacing=0.5)

            buf = BytesIO()
            plt.savefig(buf, format='png',
                bbox_inches='tight', facecolor='black', dpi=300)
            buf.seek(0)
            image = plt.imread(buf)
            plt.close(fig)

            image = np.rot90(image, k=3)
            image = np.fliplr(image)

            canvas.clear()
            canvas.setImage(image)
            canvas.view.autoRange()
            canvas.view.setMouseEnabled(x=False, y=False)

        except:
            print(traceback.format_exc())
            pass

    def initialise_pixstats(self, mode="tracks"):

        pixstats_data = []

        if mode == "tracks":

            if hasattr(self, "tracking_dict") == False:
                show_info("No tracking data found")
                return None
            if self.tracking_dict == {}:
                show_info("No tracking data found")
                return None

            pixstats_data = self.get_tracks(dataset="All Datasets", channel= "All Channels")

            progressbar = partial(self.moltrack_progress,
                progress_bar=self.gui.tracks_pixstats_progressbar)

            spot_size = int(self.gui.tracks_pixstats_spot_size.currentText())
            spot_shape = self.gui.tracks_pixstats_spot_shape.currentText()
            background_buffer = int(self.gui.tracks_pixstats_bg_buffer.currentText())
            background_width = int(self.gui.tracks_pixstats_bg_width.currentText())

            calculate_fret = self.gui.tracks_pixstats_fret.isChecked()

        if mode == "locs":

            if hasattr(self, "localisation_dict") == False:
                show_info("No localisations data found")
                return None
            if self.localisation_dict == {}:
                show_info("No localisations data found")
                return None

            pixstats_data = self.get_locs(dataset="All Datasets", channel= "All Channels")

            progressbar = partial(self.moltrack_progress,
                progress_bar=self.gui.locs_pixstats_progressbar)

            spot_size = int(self.gui.locs_pixstats_spot_size.currentText())
            spot_shape = self.gui.locs_pixstats_spot_shape.currentText()
            background_buffer = int(self.gui.locs_pixstats_bg_buffer.currentText())
            background_width = int(self.gui.locs_pixstats_bg_width.currentText())

            calculate_fret = self.gui.locs_pixstats_fret.isChecked()

        if len(pixstats_data) == 0:
            show_info("No data found")
            return None

        self.update_ui(init=True)

        pixstats_data = pd.DataFrame(pixstats_data)

        channels = pixstats_data["channel"].unique()
        channels = [chan.lower() for chan in channels]

        if calculate_fret:
            if set(["donor", "acceptor"]).issubset(channels):
                calculate_fret = True
                show_info("PixStats FRET calculation enabled")
            elif set(["dd", "da"]).issubset(channels):
                calculate_fret = True
                show_info("PixStats ALEX FRET calculation enabled")
            else:
                show_info("FRET calculation requires Donor+Acceptor or DD+DA channels")
                calculate_fret = False
                self.gui.tracks_pixstats_fret.setChecked(False)

        worker = Worker(self.compute_pixstats, pixstats_data, mode, calculate_fret,
            spot_size, spot_shape,background_buffer, background_width)

        worker.signals.progress.connect(progressbar)
        worker.signals.result.connect(self.process_pixstats_result)
        worker.signals.finished.connect(self.compute_pixstats_finished)
        worker.signals.error.connect(self.update_ui)
        self.threadpool.start(worker)


    def compute_pixstats_finished(self):

        self.update_ui()

        self.draw_localisations()

        self.update_filter_criterion()
        self.update_criterion_ranges()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()

        self.update_trackplot_options()
        self.plot_tracks()


    def process_pixstats_result(self, result):

        pixstats_data, mode = result

        if pixstats_data is None:
            return

        pixstats_data = pd.DataFrame(pixstats_data)

        for (dataset, channel), data in pixstats_data.groupby(["dataset", "channel"]):

            if "index" in data.columns:
                data = data.drop("index", axis=1)

            data = data.to_records(index=False)

            if mode == "tracks":
                self.tracking_dict[dataset][channel]["tracks"] = data
            else:
                self.localisation_dict[dataset][channel]["localisations"] = data

    def pixstats_calculate_fret(self, data):

        try:
            np.seterr(all='ignore')

            data = pd.DataFrame(data)

            fret_dataset = []

            for dataset, dataset_data in data.groupby("dataset"):

                channel_list = dataset_data["channel"].unique()
                channel_list = [chan.lower() for chan in channel_list]

                if set(["donor", "acceptor"]).issubset(channel_list):
                    fret_data = self.pixstats_compute_fret(dataset_data)
                elif set(["dd", "da"]).issubset(channel_list):
                    fret_data = self.pixstats_compute_alex_fret(dataset_data)
                else:
                    fret_data = dataset_data

                fret_dataset.append(fret_data)

            if len(fret_dataset) > 0:

                fret_dataset = pd.concat(fret_dataset, ignore_index=True)
                fret_dataset.reset_index(drop=True, inplace=True)

        except:
            print(traceback.format_exc())
            pass

        return fret_dataset

    def pixstats_compute_fret(self, data):

        for metric in data.columns:
            try:
                if metric in self.moltrack_metrics.values():
                    if "pixel" in metric:
                        metric_name = metric
                        bg_name = metric + "_bg"
                    elif "photons" in metric:
                        metric_name = metric
                        bg_name = "bg"
                    else:
                        continue

                    donor_data = data[data["channel"].str.lower() == "donor"][metric_name]
                    acceptor_data = data[data["channel"].str.lower() == "acceptor"][metric_name]

                    donor_bg = data[data["channel"].str.lower() == "donor"][bg_name]
                    acceptor_bg = data[data["channel"].str.lower() == "acceptor"][bg_name]

                    fret = (acceptor_data - donor_data) / acceptor_data
                    fret_bg = (acceptor_bg - donor_bg) / acceptor_bg

                    fret_metric = metric + "_fret"

                    data[fret_metric] = fret
                    data[fret_metric + "_bg"] = fret_bg

            except:
                print(traceback.format_exc())
                pass

        return data

    def pixstats_compute_alex_fret(self, data):

        for metric in list(data.columns):
            try:
                if metric in self.moltrack_metrics.values():
                    if "fret" in metric:
                        continue
                    elif "pixel" in metric:
                        metric_name = metric
                        bg_name = metric + "_bg"
                    elif "photons" in metric:
                        metric_name = metric
                        bg_name = "bg"
                    else:
                        continue

                    dd_data = data[data["channel"].str.lower() == "dd"][metric_name].copy()
                    da_data = data[data["channel"].str.lower() == "da"][metric_name].copy()

                    dd_bg = data[data["channel"].str.lower() == "dd"][bg_name].copy()
                    da_bg = data[data["channel"].str.lower() == "da"][bg_name].copy()

                    dd_data = np.array(dd_data)
                    da_data = np.array(da_data)
                    dd_bg = np.array(dd_bg)
                    da_bg = np.array(da_bg)

                    dd_data = dd_data - dd_bg
                    da_data = da_data - da_bg

                    fret = da_data / (da_data + dd_data)
                    fret = np.clip(fret, 0, 1)

                    fret_metric = metric + "_fret"

                    for channel in data["channel"].unique():
                        data.loc[data["channel"] == channel, fret_metric] = fret

            except:
                print(traceback.format_exc())
                pass

        return data



    @staticmethod
    def generate_localisation_mask(spot_size, spot_shape="square", buffer_size=0, bg_width=1):

        box_size = spot_size + (bg_width * 2) + (buffer_size * 2)

        # Create a grid of coordinates
        y, x = np.ogrid[:box_size, :box_size]

        # Adjust center based on box size
        if box_size % 2 == 0:
            center = (box_size / 2 - 0.5, box_size / 2 - 0.5)
        else:
            center = (box_size // 2, box_size // 2)

        if spot_shape.lower() == "circle":
            # Calculate distance from the center for circular mask
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Central spot mask
            inner_radius = spot_size // 2
            mask = distance <= inner_radius

            # Buffer mask
            buffer_outer_radius = inner_radius + buffer_size
            buffer_mask = (distance > inner_radius) & (distance <= buffer_outer_radius)

            # Background mask (outside the buffer zone)
            background_outer_radius = buffer_outer_radius + bg_width
            background_mask = (distance > buffer_outer_radius) & (distance <= background_outer_radius)

        elif spot_shape.lower() == "square":
            # Create square mask
            half_size = spot_size // 2
            mask = (abs(x - center[0]) <= half_size) & (abs(y - center[1]) <= half_size)

            # Create square background mask (one pixel larger on each side)
            buffer_mask = (abs(x - center[0]) <= half_size + buffer_size) & (abs(y - center[1]) <= half_size + buffer_size)
            background_mask = (abs(x - center[0]) <= half_size + buffer_size + bg_width) & (abs(y - center[1]) <= half_size + buffer_size + bg_width)
            background_mask = background_mask & ~buffer_mask

        return mask, buffer_mask, background_mask

    @staticmethod
    def generate_spot_bounds(locs, box_size):

        spot_bounds = []

        for loc_index, loc in enumerate(locs):

            x,y = loc.x, loc.y

            if box_size % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)+1
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)+1

            spot_bounds.append([x1,x2,y1,y2])

        return spot_bounds

    @staticmethod
    def crop_spot_data(image_shape, spot_bounds,
            spot_mask, background_mask=None):
        try:
            x1, x2, y1, y2 = spot_bounds
            crop = [0, spot_mask.shape[1], 0, spot_mask.shape[0]]

            if x1 < 0:
                crop[0] = abs(x1)
                x1 = 0
            if x2 > image_shape[1]:
                crop[1] = spot_mask.shape[1] - (x2 - image_shape[1])
                x2 = image_shape[1]
            if y1 < 0:
                crop[2] = abs(y1)
                y1 = 0
            if y2 > image_shape[0]:
                crop[3] = spot_mask.shape[0] - (y2 - image_shape[0])
                y2 = image_shape[0]

            corrected_bounds = [x1, x2, y1, y2]

            if spot_mask is not None:
                loc_mask = spot_mask.copy()
                loc_mask = loc_mask[crop[2]:crop[3], crop[0]:crop[1]]
            else:
                loc_mask = None

            if background_mask is not None:
                loc_bg_mask = background_mask.copy()
                loc_bg_mask = loc_bg_mask[crop[2]:crop[3], crop[0]:crop[1]]
            else:
                loc_bg_mask = None

        except:
            loc_mask = spot_mask
            loc_bg_mask = background_mask
            corrected_bounds = spot_bounds
            print(traceback.format_exc())

        return corrected_bounds, loc_mask, loc_bg_mask

    @staticmethod
    def generate_background_overlap_mask(locs, spot_mask,
            spot_background_mask, image_mask_shape):

        global_spot_mask = np.zeros(image_mask_shape, dtype=np.uint8)
        global_background_mask = np.zeros(image_mask_shape, dtype=np.uint8)

        spot_mask = spot_mask.astype(np.uint16)
        spot_background_mask = spot_background_mask.astype(np.uint16)

        spot_bounds = _pixstats_utils.generate_spot_bounds(locs,  len(spot_mask[0]))

        for loc_index, bounds in enumerate(spot_bounds):

            [x1, x2, y1, y2], loc_mask, log_bg_mask = _pixstats_utils.crop_spot_data(
                image_mask_shape, bounds, spot_mask,spot_background_mask)

            global_spot_mask[y1:y2, x1:x2] += loc_mask
            global_background_mask[y1:y2, x1:x2] += log_bg_mask

        global_spot_mask[global_spot_mask > 0] = 1
        global_background_mask[global_background_mask > 0] = 1

        intersection_mask = global_spot_mask & global_background_mask

        global_background_mask = global_background_mask - intersection_mask

        return global_background_mask, global_spot_mask



    @staticmethod
    def pixstats_compute_func(dat, progress_list=None):

        try:

            pixstats_data = dat["pixstats_data"]
            spot_size = dat["spot_size"]
            spot_shape = dat["spot_shape"]
            background_buffer = dat["background_buffer"]
            background_width = dat["background_width"]
            start_index = dat["start_index"]
            frame_shape = dat["shape"][-2:]

            n_pixels = spot_size ** 2

            if type(pixstats_data) == pd.DataFrame:
                pixstats_data = pixstats_data.to_records(index=False)

            # Access the shared memory
            shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            image_chunk = np_array.copy()

            spot_mask, buffer_mask, spot_background_mask = _pixstats_utils.generate_localisation_mask(
                spot_size, spot_shape, background_buffer, background_width)

            pixstats_data = pd.DataFrame(pixstats_data)
            pixstats_data = pixstats_data.reset_index()
            pixstats_data["pixel_mean"] = np.nan
            pixstats_data["pixel_median"] = np.nan
            pixstats_data["pixel_sum"] = np.nan
            pixstats_data["pixel_max"] = np.nan
            pixstats_data["pixel_std"] = np.nan
            pixstats_data["pixel_mean_bg"] = np.nan
            pixstats_data["pixel_median_bg"] = np.nan
            pixstats_data["pixel_sum_bg"] = np.nan
            pixstats_data["pixel_max_bg"] = np.nan
            pixstats_data["pixel_std_bg"] = np.nan
            pixstats_data = pixstats_data.to_records(index=False)

            for array_index, frame in enumerate(image_chunk):

                frame_index = start_index + array_index

                frame_dat = pixstats_data[pixstats_data["frame"] == frame_index]

                if len(frame_dat) == 0:
                    continue

                for dat in frame_dat:

                    try:

                        bounds = _pixstats_utils.generate_spot_bounds([dat], len(spot_mask[0]))[0]

                        [x1, x2, y1, y2], cropped_mask, cropped_background_mask = _pixstats_utils.crop_spot_data(
                            frame_shape, bounds, spot_mask, spot_background_mask)

                        cropped_mask = np.logical_not(cropped_mask).astype(int)
                        cropped_background_mask = np.logical_not(cropped_background_mask).astype(int)

                        spot_values = frame[y1:y2, x1:x2].copy()
                        spot_background = frame[y1:y2, x1:x2].copy()

                        spot_values = np.ma.array(
                            spot_values,mask=cropped_mask)
                        spot_background = np.ma.array(
                            spot_background,mask=cropped_background_mask)

                        spot_mean = float(np.ma.mean(spot_values))
                        spot_median = float(np.ma.median(spot_values))
                        spot_sum = float(np.ma.sum(spot_values))
                        spot_max = float(np.ma.max(spot_values))
                        spot_std = float(np.ma.std(spot_values))

                        spot_mean_bg = float(np.ma.mean(spot_background))
                        spot_median_bg = float(np.ma.median(spot_background))
                        spot_sum_bg = spot_mean_bg * n_pixels
                        spot_max_bg = float(np.ma.max(spot_background))
                        spot_std_bg = float(np.ma.std(spot_background))

                        dat["pixel_mean"] = spot_mean
                        dat["pixel_median"] = spot_median
                        dat["pixel_sum"] = spot_sum
                        dat["pixel_max"] = spot_max
                        dat["pixel_std"] = spot_std
                        dat["pixel_mean_bg"] = spot_mean_bg
                        dat["pixel_median_bg"] = spot_median_bg
                        dat["pixel_sum_bg"] = spot_sum_bg
                        dat["pixel_max_bg"] = spot_max_bg
                        dat["pixel_std_bg"] = spot_std_bg

                        dat_index = dat["index"]
                        pixstats_data[dat_index] = dat

                    except:
                        pass

        except:
            print(traceback.format_exc())
            pass

        if progress_list is not None:
            progress_list.append(1)

        return pixstats_data



    def compute_pixstats(self, pixmap_data, mode, calculate_fret, spot_size, spot_shape,
            background_buffer, background_width, progress_callback=None):

        try:
            results = None

            dataset_list = pixmap_data["dataset"].unique()
            channel_list = pixmap_data["channel"].unique()

            self.create_shared_image_chunks(dataset_list=dataset_list,
                channel_list=channel_list, chunk_size=100)

            compute_jobs = self.get_pixstats_compute_jobs(pixmap_data, spot_size, spot_shape,
                background_buffer, background_width)

            if len(compute_jobs) == 0:
                return None

            with Manager() as manager:

                progress_list = manager.list()

                with ProcessPoolExecutor() as executor:

                    futures = [executor.submit(_pixstats_utils.pixstats_compute_func,
                        job, progress_list) for job in compute_jobs]

                    for future in as_completed(futures):
                        if progress_callback is not None:
                            progress = int(len(progress_list) / len(compute_jobs) * 100)
                            progress_callback.emit(progress)

                results = [future.result() for future in futures if future.result() is not None]

            if len(results) > 0:

                results = [pd.DataFrame(result) for result in results]
                results = pd.concat(results, ignore_index=True)

                if calculate_fret:
                    results = self.pixstats_calculate_fret(results)
                else:
                    columns = results.columns
                    fret_columns = [col for col in columns if "_fret" in col]
                    results = results.drop(fret_columns, axis=1)

                results = results.to_records(index=False)

            self.restore_shared_image_chunks()

        except:
            print(traceback.format_exc())
            self.restore_shared_image_chunks()
            self.update_ui()
            pass

        return results, mode

    def get_pixstats_compute_jobs(self, pixstats_data, spot_size, spot_shape,
            background_buffer, background_width):

        compute_jobs = []

        try:

            for image_chunk in self.shared_chunks:

                dataset = image_chunk["dataset"]
                channel = image_chunk["channel"]

                if dataset in pixstats_data["dataset"].unique() and channel in pixstats_data["channel"].unique():

                    frame_start = image_chunk["start_index"]
                    frame_end = image_chunk["end_index"]

                    pixstats_chunks = pixstats_data.copy()

                    pixstats_chunks = pixstats_chunks[(pixstats_chunks["dataset"] == dataset) &
                                                      (pixstats_chunks["channel"] == channel)]

                    pixstats_chunks = pixstats_chunks.to_records(index=False)

                    pixstats_chunks = pixstats_chunks[(pixstats_chunks["frame"] >= frame_start) &
                                                      (pixstats_chunks["frame"] <= frame_end)]

                    if len(pixstats_chunks) > 0:
                        job = {"dataset": dataset,
                               "channel": channel,
                               "start_index": image_chunk["start_index"],
                               "end_index": image_chunk["end_index"],
                               "pixstats_data": pixstats_chunks,
                               "spot_size": spot_size,
                               "spot_shape": spot_shape,
                               "background_buffer": background_buffer,
                               "background_width": background_width,
                               "shared_memory_name": image_chunk["shared_memory_name"],
                               "shape": image_chunk["shape"],
                               "dtype": image_chunk["dtype"],}

                        compute_jobs.append(job)
        except:
            pass

        return compute_jobs