import traceback
import numpy as np
import pandas as pd
import trackpy as tp
from moltrack.funcs.compute_utils import Worker
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager

class _diffusion_utils:

    @staticmethod
    def compute_adc(dat, diffusion_data):

        try:

            track = dat.pop("track")
            pixel_size_nm = dat["pixel_size"]
            exposure_time_ms = dat["exposure_time"]
            min = dat["min_track_length"]
            dataset = dat["dataset"]
            channel = dat["channel"]
            particle = dat["particle"]

            pixel_size_um = pixel_size_nm * 1e-3
            exposure_time_s = exposure_time_ms * 1e-3
            fps = 1 / exposure_time_s

            track = track[["particle", "frame", "x", "y"]]

            # Calculate MSD using trackpy
            msd_df = tp.motion.msd(track, mpp=0.1, fps=100, max_lagtime=100)

            # Linear fit to first 10 points of msd vs time
            msd_values = msd_df['msd'].values
            time = msd_df['lagt'].values

            if len(time) >= min and len(msd_values) >= min:  # Ensure there are enough points to fit
                slope, intercept = np.polyfit(time[:min], msd_values[:min], 1)
                apparent_diffusion = slope / 4  # the slope of MSD vs time gives 4D in 2D

                dat = {"dataset":dataset,
                       "channel": channel,
                       "particle": particle,
                       "apparent_diffusion": apparent_diffusion}

                diffusion_data.append(dat)

        except:
            print(traceback.format_exc())
            pass


    def populate_diffusion_compute_jobs(self, tracks):

        compute_jobs = []

        try:

            dataset_list = list(self.tracking_dict.keys())

            for dataset in dataset_list:

                channel_list = list(self.tracking_dict[dataset].keys())

                for channel in channel_list:

                    channel_tracks = tracks[(tracks["dataset"] == dataset) &
                                            (tracks["channel"] == channel)]

                    pixel_size = self.dataset_dict[dataset]["pixel_size"]
                    exposure_time = self.dataset_dict[dataset]["exposure_time"]

                    channel_tracks = pd.DataFrame(channel_tracks)

                    for particle, track in channel_tracks.groupby("particle"):

                        track = track.sort_values("frame")

                        particle_data = {"dataset": dataset,
                                         "channel": channel,
                                         "particle": particle,
                                         "pixel_size": pixel_size,
                                         "exposure_time": exposure_time,
                                         "track": track,
                                         "min_track_length": 10,
                                         }

                        compute_jobs.append(particle_data)
        except:
            print(traceback.format_exc())

        return compute_jobs


    def compute_diffusion_coefficient_finished(self):

        try:
            self.update_ui()
            self.plot_diffusion_histogram()

        except:
            print(traceback.format_exc())

    def compute_diffusion_coefficients(self, tracks, progress_callback=None):

        compute_jobs = self.populate_diffusion_compute_jobs(tracks)

        print(f"Computing diffusion coefficients for {len(compute_jobs)} particles.")

        n_compute_jobs = len(compute_jobs)

        if len(compute_jobs) == 0:
            return

        with Manager() as manager:

            diffusion_data = manager.list()

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_diffusion_utils.compute_adc,
                    dat, diffusion_data) for dat in compute_jobs]

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if progress_callback is not None:
                        progress = (completed / n_compute_jobs) * 100
                        progress_callback.emit(progress)

            diffusion_data = pd.DataFrame(list(diffusion_data))

            for (dataset, channel), group in diffusion_data.groupby(["dataset", "channel"]):

                diffusion_coefficients = group["apparent_diffusion"].values
                particle = group["particle"].values

                if dataset not in self.diffusion_dict:
                    self.diffusion_dict[dataset] = {}
                if channel not in self.diffusion_dict[dataset]:
                    self.diffusion_dict[dataset][channel] = {}

                self.diffusion_dict[dataset][channel]["particle"] = particle
                self.diffusion_dict[dataset][channel]["diffusion_coefficient"] = diffusion_coefficients



    def init_compute_diffusion_coefficients(self):
        try:
            dataset = "All Datasets"
            channel = "All Channels"

            tracks = self.get_tracks(dataset, channel)

            if len(tracks) == 0:
                return

            self.update_ui(init=True)

            self.worker = Worker(self.compute_diffusion_coefficients, tracks)
            self.worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.adc_progressbar))
            self.worker.signals.finished.connect(self.compute_diffusion_coefficient_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())

    def plot_diffusion_histogram(self):

        try:

            dataset = self.gui.adc_dataset.currentText()
            channel = self.gui.adc_channel.currentText()
            min_range = self.gui.adc_range_min.value()
            max_range = self.gui.adc_range_min.value()

            coefs = self.diffusion_dict[dataset][channel]["diffusion_coefficient"]

            coefs = np.array(coefs)
            # coefs = coefs[(coefs >= min_range) & (coefs <= max_range)]

            self.adc_graph_canvas.clear()

            if len(coefs) > 0:

                ax = self.adc_graph_canvas.addPlot()

                # Create histogram
                y, x = np.histogram(coefs, bins=20)

                ax.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 75))
                ax.setLabel('left', 'Frequency')
                ax.setLabel('bottom', 'Diffusion Coefficient (um^2/s)')


        except:
            print(traceback.format_exc())
            self.update_ui()

        pass

    def export_diffusion_coefficients(self, diffusion_coefficients):
        pass