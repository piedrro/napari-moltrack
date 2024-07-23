import traceback

import numpy as np
import pandas as pd
import trackpy as tp
from numba import njit
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from napari.utils.notifications import show_info
from moltrack.funcs.compute_utils import Worker
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
from multiprocessing import Manager, shared_memory
from functools import partial
import warnings
from napari.utils.notifications import show_info

warnings.filterwarnings("ignore", category=UserWarning)

@njit
def calculate_msd_numba(x_disp, y_disp, max_lag):
    msd_list = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        squared_displacements = np.zeros(len(x_disp) - lag)
        for i in range(len(x_disp) - lag):
            dx = np.sum(x_disp[i:i+lag])
            dy = np.sum(y_disp[i:i+lag])
            squared_displacements[i] = dx**2 + dy**2
        msd_list[lag] = np.mean(squared_displacements)
    return msd_list


@njit
def calculate_rolling_msd(x, y, pixel_size, window_size=3):
    n = len(x)
    half_window = window_size // 2
    rolling_msd = np.zeros(n)

    for i in range(n):

        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        msd_sum = 0
        count = 0

        for j in range(start, end):
            dx = (x[j] - x[i]) * pixel_size
            dy = (y[j] - y[i]) * pixel_size
            msd_sum += dx ** 2 + dy ** 2
            count += 1

        rolling_msd[i] = msd_sum / count if count > 0 else 0

    return rolling_msd



class _trackstats_utils:


    @staticmethod
    def get_track_stats(df):

        try:

            pixel_size = df["pixel_size (um)"].values[0]
            time_step = df["exposure_time (s)"].values[0]

            # Convert columns to numpy arrays for faster computation
            x = df['x'].values
            y = df['y'].values

            rolling_msd = calculate_rolling_msd(x, y, pixel_size, window_size=4)

            # Calculate the displacements using numpy diff
            x_disp = np.diff(x, prepend=x[0]) * pixel_size
            y_disp = np.diff(y, prepend=y[0]) * pixel_size

            step_size = np.sqrt(x_disp ** 2 + y_disp ** 2)

            # Calculate the MSD using numba
            max_lag = len(x) - 1
            msd_list = calculate_msd_numba(x_disp, y_disp, max_lag)

            # Calculate speed
            speed = np.sqrt(x_disp ** 2 + y_disp ** 2) / time_step

            # Create time array
            time = np.arange(0, max_lag + 1) * time_step

            if len(time) >= 4 and len(msd_list) >= 4:  # Ensure there are enough points to fit
                slope, intercept = np.polyfit(time[:4], msd_list[:4], 1)
                apparent_diffusion = abs(slope / 4)  # the slope of MSD vs time gives 4D in 2D
            else:
                apparent_diffusion = 0

        except:
            apparent_diffusion = None
            msd_list = None
            time = None
            speed = None
            step_size = None
            rolling_msd = None
            print(traceback.format_exc())

        # Append results to the dataframe
        df["time"] = time
        df["msd"] = msd_list
        df["D*"] = apparent_diffusion
        df["speed"] = speed
        df["step_size"] = step_size
        df["rolling_msd"] = rolling_msd

        return df


    def compute_track_stats(self, track_data, progress_callback=None):

        try:

            tracks_with_stats = []

            if type(track_data) == np.recarray:
                track_data = pd.DataFrame(track_data)

            with ProcessPoolExecutor() as executor:

                #split by dataset, channel and particle into list
                stats_jobs = [dat[1] for dat in track_data.groupby(["dataset", "channel", "particle"])]

                n_processed = 0

                futures = [executor.submit(_trackstats_utils.get_track_stats, df) for df in stats_jobs]

                for future in as_completed(futures):
                    n_processed += 1
                    if progress_callback is not None:
                        progress = int(n_processed / len(stats_jobs) * 100)
                        progress_callback.emit(progress)

            tracks_with_stats = [future.result() for future in futures if future.result() is not None]

        except:
            print(traceback.format_exc())
            pass

        if len(tracks_with_stats) > 0:
            track_data = pd.concat(tracks_with_stats, ignore_index=True)

        return track_data


    def compute_track_stats_finished(self):

        self.plot_diffusion_graph()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()

        self.update_traces_export_options()

        self.update_trackplot_options()
        self.plot_tracks()

        self.update_ui()

    def process_track_stats_result(self, track_data):

        if track_data is None:
            return

        remove_unlinked = self.gui.remove_unlinked.isChecked()
        layers_names = [layer.name for layer in self.viewer.layers]
        total_tracks = 0

        for (dataset, channel), tracks in track_data.groupby(["dataset", "channel"]):
            tracks = tracks.to_records(index=False)

            if dataset not in self.tracking_dict.keys():
                self.tracking_dict[dataset] = {}
            if channel not in self.tracking_dict[dataset].keys():
                self.tracking_dict[dataset][channel] = {}

            self.tracking_dict[dataset][channel] = {"tracks": tracks}

            n_tracks = len(np.unique(tracks["particle"]))
            total_tracks += n_tracks

        return tracks


    def initialise_track_stats(self):

        try:

            if hasattr(self, "tracking_dict"):

                tracks = self.get_tracks("All Datasets", "All Channels")

                if len(tracks) == 0:
                    show_info("No tracks found")
                    return

                self.update_ui(init=True)

                tracks = pd.DataFrame(tracks)
                tracks["pixel_size (um)"] = 0.0
                tracks["exposure_time (s)"] = 0.0

                for dataset, data in tracks.groupby("dataset"):
                    pixel_size_nm = float(self.dataset_dict[dataset]["pixel_size"])
                    exposure_time_ms = float(self.dataset_dict[dataset]["exposure_time"])
                    pixel_size_um = pixel_size_nm * 1e-3
                    exposure_time_s = exposure_time_ms * 1e-3
                    tracks.loc[data.index, "pixel_size (um)"] = pixel_size_um
                    tracks.loc[data.index, "exposure_time (s)"] = exposure_time_s

                tracks = tracks.to_records(index=False)

                worker = Worker(self.compute_track_stats, tracks)
                worker.signals.result.connect(self.process_track_stats_result)
                worker.signals.finished.connect(self.compute_track_stats_finished)
                worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.track_stats_progressbar))
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()


