import traceback
import numpy as np
import pandas as pd
import trackpy as tp
from moltrack.funcs.compute_utils import Worker
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pyqtgraph import LegendItem
import os
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg

class _diffusion_utils:

    def update_diffusion_options(self):

        try:
            plot = self.gui.adc_plot.currentText()

            adc_contols = [self.gui.adc_range_min, self.gui.adc_range_max, self.gui.adc_range_label,
                           self.gui.adc_bins, self.gui.adc_bins_label,
                           self.gui.adc_density, self.gui.adc_hide_first]

            if plot.lower() == "msd":
                for control in adc_contols:
                    control.setEnabled(False)
                    control.hide()
            else:
                for control in adc_contols:
                    control.setEnabled(True)
                    control.show()


        except Exception as e:
            print(traceback.format_exc())

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
            fps = 1 / (exposure_time_ms * 1e-3)

            track = track[["particle", "frame", "x", "y"]]

            # Calculate MSD using trackpy
            msd_df = tp.motion.msd(track, mpp=pixel_size_um, fps=fps, max_lagtime=100)

            msd_df = msd_df.reset_index(drop=True)
            msd_df = msd_df[["lagt", "msd"]]

            # Linear fit to first 10 points of msd vs time
            msd_values = msd_df['msd'].values
            time = msd_df['lagt'].values

            if len(time) >= min and len(msd_values) >= min:  # Ensure there are enough points to fit
                slope, intercept = np.polyfit(time[:min], msd_values[:min], 1)
                diffusion_coefficient = slope / 4  # the slope of MSD vs time gives 4D in 2D

                if diffusion_coefficient > 0:

                    ddat = {"dataset":dataset,
                            "channel": channel,
                            "particle": particle,
                            "pixel_size (nm)": pixel_size_nm,
                            "exposure_time (ms)": exposure_time_ms,
                            "msd": msd_df,
                            "diffusion_coefficient (um^2/s)": diffusion_coefficient}

                    diffusion_data.append(ddat)

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
            self.plot_diffusion_graph()

        except:
            print(traceback.format_exc())

    def compute_diffusion_coefficients_result(self, result):

        try:

            if result is None:
                return

            diffusion_data, msd_data = result

            self.diffusion_dict = {}
            self.msd_dict = msd_data

            diffusion_min = 0
            diffusion_max = 10

            for (dataset, channel), group in diffusion_data.groupby(["dataset", "channel"]):

                diffusion_coefficients = group["diffusion_coefficient (um^2/s)"].values

                dmin = np.min(diffusion_coefficients)
                dmax = np.max(diffusion_coefficients)

                if dmin < diffusion_min and dmin > 0:
                    diffusion_min = dmin
                if dmax > diffusion_max:
                    diffusion_max = dmax

                if dataset not in self.diffusion_dict:
                    self.diffusion_dict[dataset] = {}
                if channel not in self.diffusion_dict[dataset]:
                    self.diffusion_dict[dataset][channel] = {}

                for col in group.columns:
                    self.diffusion_dict[dataset][channel][col] = group[col].values

            self.gui.adc_range_min.blockSignals(True)
            self.gui.adc_range_max.blockSignals(True)

            self.gui.adc_range_min.setValue(diffusion_min)
            self.gui.adc_range_max.setValue(diffusion_max)

            self.gui.adc_range_min.blockSignals(False)
            self.gui.adc_range_max.blockSignals(False)

        except:
            print(traceback.format_exc())
            self.update_ui()


    def compute_diffusion_coefficients(self, tracks, progress_callback=None):

        diffusion_data = None
        msd_dict = None

        try:

            compute_jobs = self.populate_diffusion_compute_jobs(tracks)

            print(f"Computing diffusion coefficients for {len(compute_jobs)} particles.")

            n_compute_jobs = len(compute_jobs)

            if len(compute_jobs) == 0:
                return

            with Manager() as manager:

                diffusion_data = manager.list()

                with ProcessPoolExecutor() as executor:
                    futures = [executor.submit(_diffusion_utils.compute_adc,
                        dat, diffusion_data) for dat in compute_jobs]

                    completed = 0
                    for future in as_completed(futures):
                        completed += 1
                        if progress_callback is not None:
                            progress = (completed / n_compute_jobs) * 100
                            progress_callback.emit(progress)

                msd_dict = {}
                for d in diffusion_data:
                    msd = d.pop("msd")
                    dataset = d["dataset"]
                    channel = d["channel"]

                    if dataset not in msd_dict:
                        msd_dict[dataset] = {}
                    if channel not in msd_dict[dataset]:
                        msd_dict[dataset][channel] = []

                    msd_dict[dataset][channel].append(msd)

                for dataset in msd_dict:
                    for channel in msd_dict[dataset]:
                        msd = pd.concat(msd_dict[dataset][channel])
                        msd = msd.groupby('lagt')['msd'].agg(['mean', 'std', "sem"]).reset_index()
                        msd_dict[dataset][channel] = msd

                diffusion_data = pd.DataFrame(list(diffusion_data))

        except:
            diffusion_data = None
            print(traceback.format_exc())

        return diffusion_data, msd_dict


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
            self.worker.signals.result.connect(self.compute_diffusion_coefficients_result)
            self.worker.signals.finished.connect(self.compute_diffusion_coefficient_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())


    def plot_diffusion_graph(self):

        try:

            self.adc_graph_canvas.clear()

            plot = self.gui.adc_plot.currentText()

            if plot.lower() == "msd":
                self.plot_msd_curve()
            else:
                self.plot_diffusion_histogram()

        except:
            print(traceback.format_exc())
            pass


    def get_msd_data(self, dataset, channel):

        try:
            if hasattr(self, "msd_dict") == False:
                return None

            if dataset not in self.msd_dict.keys():
                return None

            if channel not in self.msd_dict[dataset].keys():
                return None

            msd = self.msd_dict[dataset][channel]

            return msd

        except:
            print(traceback.format_exc())
            return None



    def plot_msd_curve(self):

        try:

            if hasattr(self, "msd_dict") == True:

                dataset = self.gui.adc_dataset.currentText()
                channel = self.gui.adc_channel.currentText()

                msd = self.get_msd_data(dataset, channel)

                if msd is None:
                    return

                # Retrieve MSD values and errors
                msd_values = msd["mean"].values
                msd_error = msd["sem"].values
                lagt = msd["lagt"].values

                # Create a new plot
                ax = self.adc_graph_canvas.addPlot()

                # Plot the data points
                ax.plot(lagt, msd_values, pen=None, symbol='o', symbolBrush='b')

                # Create the error bars
                error = pg.ErrorBarItem(x=lagt, y=msd_values, top=msd_error, bottom=msd_error, beam=0.5)
                ax.addItem(error)

                # Set labels
                ax.setLabel('left', 'MSD (μm²)')
                ax.setLabel('bottom', 'Time (seconds)')
                ax.showGrid(x=True, y=True)


        except:
            print(traceback.format_exc())
            pass




    def plot_diffusion_histogram(self):

        try:

            dataset = self.gui.adc_dataset.currentText()
            channel = self.gui.adc_channel.currentText()
            min_range = self.gui.adc_range_min.value()
            max_range = self.gui.adc_range_max.value()
            bins = self.gui.adc_bins.value()
            density = self.gui.adc_density.isChecked()
            hide_first = self.gui.adc_hide_first.isChecked()

            self.adc_graph_canvas.clear()

            diffusion_coefficients = self.get_diffusion_coefficents(dataset, channel, return_dict=True)

            ax = self.adc_graph_canvas.addPlot()
            legend = LegendItem(offset=(-10, 10))
            legend.setParentItem(ax.graphicsItem())

            for ddata in diffusion_coefficients:

                dataset_name = ddata["dataset"][0]
                channel_name = ddata["channel"][0]
                coefs = ddata["diffusion_coefficient (um^2/s)"]

                if dataset_name == "All Datasets" and channel_name == "All Channels":
                    label = f"{dataset_name} - {channel_name}"
                elif dataset_name == "All Datasets":
                    label = f"{dataset_name}"
                elif channel_name == "All Channels":
                    label = f"{dataset_name}"
                else:
                    label = f"{dataset_name}"

                coefs = np.array(coefs)
                coefs = coefs[(coefs >= min_range) & (coefs <= max_range)]

                if len(coefs) > 0:

                    if density == True:
                        y_label = "Density"
                    else:
                        y_label = "Counts"

                    y, x = np.histogram(coefs, bins=bins, density=density)

                    if hide_first:
                        y = y[1:]
                        x = x[1:]

                    plotItem = ax.plot(x, y, stepMode=True, fillLevel=0,
                        brush=(0, 0, 255, 90), name=label)
                    ax.setLabel('left', y_label)
                    ax.setLabel('bottom', 'Apparent Diffusion Coefficient (μm²/s)')
                    legend.addItem(plotItem, label)

            ax.showGrid(x=True, y=True)

        except:
            print(traceback.format_exc())
            self.update_ui()

        pass



    def export_diffusion_graph(self):

        try:

            plot = self.gui.adc_plot.currentText()

            if plot.lower() == "msd":
                self.export_msd_curve()
            else:
                self.export_diffusion_coefficients()

        except:
            print(traceback.format_exc())
            self.update_ui()


    def export_msd_curve(self):

        try:

            dataset = self.gui.adc_dataset.currentText()
            channel = self.gui.adc_channel.currentText()

            msd = self.get_msd_data(dataset, channel)

            if msd is None:
                return

            dataset_list = list(self.dataset_dict.keys())
            file_path = self.dataset_dict[dataset_list[0]]["path"]

            if type(file_path) == list:
                file_path = file_path[0]

            base, ext = os.path.splitext(file_path)
            file_path = f"{base}_msd_curve.csv"

            file_path = QFileDialog.getSaveFileName(self, f"Export MSD Curve", file_path, f"(*.csv)")[0]

            if file_path == "":
                return

            msd.to_csv(file_path, index=False)

            print(f"MSD curve exported to {file_path}")

        except:
            print(traceback.format_exc())
            self.update_ui()


    def export_diffusion_coefficients(self):

        try:

            dataset = self.gui.adc_dataset.currentText()
            channel = self.gui.adc_channel.currentText()
            min_range = self.gui.adc_range_min.value()
            max_range = self.gui.adc_range_max.value()

            coefs = self.get_diffusion_coefficents(dataset, channel, return_dict=False)

            if len(coefs) == 0:
                return

            coefs = coefs[(coefs["diffusion_coefficient (um^2/s)"] >= min_range) &
                          (coefs["diffusion_coefficient (um^2/s)"] <= max_range)]

            if len(coefs) == 0:
                return

            coefs = pd.DataFrame(coefs)

            dataset_list = coefs["dataset"].unique()

            file_path = self.dataset_dict[dataset_list[0]]["path"]

            if type(file_path) == list:
                file_path = file_path[0]

            base, ext = os.path.splitext(file_path)
            file_path = f"{base}_diffusion_coefficients.csv"

            file_path = QFileDialog.getSaveFileName(self, f"Export Diffusion Coefficients", file_path, f"(*.csv)")[0]

            if file_path == "":
                return

            coefs.to_csv(file_path, index=False)

            print(f"Diffusion coefficients exported to {file_path}")

        except:
            print(traceback.format_exc())
            self.update_ui()



    def get_diffusion_coefficents(self, dataset, channel,
            return_dict=False, include_metadata=True):

        coefficient_data = []

        try:
            if dataset == "All Datasets":
                dataset_list = list(self.diffusion_dict.keys())
            else:
                dataset_list = [dataset]

            for dataset_name in dataset_list:
                if dataset_name not in self.diffusion_dict.keys():
                    continue

                if channel == "All Channels":
                    channel_list = list(self.diffusion_dict[dataset_name].keys())
                else:
                    channel_list = [channel]

                for channel_name in channel_list:
                    if channel_name not in self.diffusion_dict[dataset_name].keys():
                        continue

                    data = self.diffusion_dict[dataset_name][channel_name]

                    data = pd.DataFrame(data)

                    if include_metadata:
                        if "dataset" not in data.columns:
                            data.insert(0, "dataset", dataset_name)
                        if "channel" not in data.columns:
                            data.insert(1, "channel", channel_name)

                    data = data.to_records(index=False)
                    coefficient_data.append(data)

            if return_dict == False:
                if len(coefficient_data) == 0:
                    pass
                elif len(coefficient_data) == 1:
                    coefficient_data = coefficient_data[0]
                else:
                    coefficient_data = np.hstack(coefficient_data).view(np.recarray).copy()

        except:
            print(traceback.format_exc())

        return coefficient_data