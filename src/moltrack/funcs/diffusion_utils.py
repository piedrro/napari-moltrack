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
from napari.utils.notifications import show_info

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


    def plot_msd_curve(self):

        try:

            if hasattr(self, "tracking_dict") == True:

                dataset = self.gui.adc_dataset.currentText()
                channel = self.gui.adc_channel.currentText()

                tracks = self.get_tracks(dataset, channel)

                if len(tracks) == 0:
                    return

                tracks = pd.DataFrame(tracks)

                if "msd" not in tracks.columns:
                    return

                msd_data = tracks[["time", "msd"]]

                msd_data = msd_data.groupby("time")["msd"].agg(['mean', 'std', "sem"]).reset_index()

                msd_values = msd_data["mean"].values
                msd_error = msd_data["sem"].values
                lagt = msd_data["time"].values

                # Create a new plot
                ax = self.adc_graph_canvas.addPlot()

                # Plot the data points
                ax.plot(lagt, msd_values, pen=None, symbol='o', symbolBrush='b')

                # Create the error bars
                error = pg.ErrorBarItem(x=lagt, y=msd_values, top=msd_error, bottom=msd_error, beam=0.5)
                ax.addItem(error)

                # Set labels
                ax.setLabel('left', "msd")
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

            track_data = self.get_tracks(dataset, channel)

            if len(track_data) == 0:
                return

            ax = self.adc_graph_canvas.addPlot()
            legend = LegendItem(offset=(-10, 10))
            legend.setParentItem(ax.graphicsItem())

            track_data = pd.DataFrame(track_data)

            if "D*" not in track_data.columns:
                return

            for (dataset_name,channel_name), tracks in track_data.groupby(["dataset", "channel"]):

                coefs = []

                for particle, track in tracks.groupby("particle"):

                    diffusion = track["D*"].values[0]
                    coefs.append(diffusion)

                if dataset_name == "All Datasets" and channel_name == "All Channels":
                    label = f"{dataset_name} - {channel_name}"
                elif dataset_name == "All Datasets":
                    label = f"{dataset_name}"
                elif channel_name == "All Channels":
                    label = f"{dataset_name}"
                else:
                    label = f"{dataset_name}"

                coefs = np.array(coefs)
                coefs = coefs[(~np.isnan(coefs)) & (~np.isinf(coefs))]
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

            if hasattr(self, "tracking_dict") == True:

                dataset = self.gui.adc_dataset.currentText()
                channel = self.gui.adc_channel.currentText()

                tracks = self.get_tracks(dataset, channel)

                if len(tracks) == 0:
                    return

                tracks = pd.DataFrame(tracks)

                if "msd" not in tracks.columns:
                    return

                msd_data = tracks[["time", "msd"]]

                msd = msd_data.groupby("time")["msd"].agg(['mean', 'std', "sem"]).reset_index()

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
                show_info(f"MSD curve exported to {file_path}")

        except:
            print(traceback.format_exc())
            self.update_ui()


    def export_diffusion_coefficients(self):

        try:

            dataset = self.gui.adc_dataset.currentText()
            channel = self.gui.adc_channel.currentText()
            min_range = self.gui.adc_range_min.value()
            max_range = self.gui.adc_range_max.value()

            track_data = self.get_tracks(dataset, channel)

            track_data = pd.DataFrame(track_data)

            export_data = []
            for (dataset_name, channel_name), tracks in track_data.groupby(["dataset", "channel"]):
                dat = []
                for particle, track in tracks.groupby("particle"):
                    diffusion = track["D*"].values[0]
                    if diffusion >= min_range and diffusion <= max_range:
                        dat.append([particle,diffusion])

                coef_data = pd.DataFrame(dat, columns=["particle","D* (um2/s)"])
                coef_data.insert(0, "dataset", dataset_name)
                coef_data.insert(1, "channel", channel_name)
                export_data.append(coef_data)

            if len(export_data) == 0:
                return

            coefs = pd.concat(export_data)

            dataset_list = coefs["dataset"].unique()

            file_path = self.dataset_dict[dataset_list[0]]["path"]

            if type(file_path) == list:
                file_path = file_path[0]

            base, ext = os.path.splitext(file_path)
            file_path = f"{base}_diffusion_coefficients.csv"

            file_path = QFileDialog.getSaveFileName(self, f"Export Diffusion Coefficients",
                file_path, f"(*.csv)")[0]

            if file_path == "":
                return

            coefs.to_csv(file_path, index=False)
            show_info(f"Diffusion coefficients exported to {file_path}")

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