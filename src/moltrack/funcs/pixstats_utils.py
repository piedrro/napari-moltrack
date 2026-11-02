import traceback
from functools import partial
from io import BytesIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info

from moltrack.funcs.compute_utils import Worker


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

        self.update_traces_export_options()

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









    def compute_pixstats(self, pixmap_data, mode, calculate_fret, spot_size, spot_shape,
                         background_buffer, background_width, progress_callback=None):
        from moltrack.backend import measure_locs

        measured = []
        for (dataset, channel), rows in pixmap_data.groupby(['dataset', 'channel'], sort=False):
            movie = self.dataset_dict[dataset]['images'][channel]
            measured.append(measure_locs(movie, rows, spot_size=spot_size, spot_shape=spot_shape,
                background_buffer=background_buffer, background_width=background_width,
                progress_callback=progress_callback, cancel_event=self.stop_event))
        results = pd.concat(measured, ignore_index=True) if measured else pd.DataFrame(pixmap_data).copy()
        if calculate_fret:
            results = self.pixstats_calculate_fret(results)
        else:
            results = results.drop(columns=[col for col in results if '_fret' in col])
        return results.to_records(index=False), mode
