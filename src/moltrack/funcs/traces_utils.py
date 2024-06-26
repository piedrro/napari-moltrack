import traceback

import numpy as np
import pandas as pd
import json
import os
from PyQt5.QtWidgets import QFileDialog
from napari.utils.notifications import show_info


class _traces_utils:

    def get_traces_path(self, dataset, metric_name, mode="json"):

        json_path = None

        if hasattr(self, "dataset_dict"):
            path = self.dataset_dict[dataset]["path"]

            base, ext = os.path.splitext(path)
            file_name = os.path.basename(base)
            dir_name = os.path.dirname(path)

            json_name = f"{file_name}_{metric_name.lower()}_traces.json"

            json_path = os.path.join(dir_name, json_name)

            json_path = QFileDialog.getSaveFileName(self, "Save Traces",
                json_path, "JSON (*.json)")[0]

            if json_path == "":
                return None

        return json_path



    def export_traces(self):

        try:

            if hasattr(self, "tracking_dict"):

                dataset = self.gui.traces_export_dataset.currentText()
                channel = self.gui.traces_export_channel.currentText()
                metric = self.gui.traces_export_metric.currentText()
                subtract_background = self.gui.traces_export_subtract_background.isChecked()

                bg_metric_name = ""
                if metric == "Mean Squared Displacement":
                    metric_name = "msd"
                elif metric == "Speed":
                    metric_name = "speed"
                elif metric == "Apparent Diffusion Coefficient":
                    metric_name = "D*"
                elif metric == "Photons":
                    metric_name = "photons"
                elif metric == "Pixel Mean":
                    metric_name = "pixel_mean"
                    bg_metric_name = "pixel_mean_bg"
                elif metric == "Pixel Standard Deviation":
                    metric_name = "pixel_std"
                    bg_metric_name = "pixel_std_bg"
                elif metric == "Pixel Median":
                    metric_name = "pixel_median"
                    bg_metric_name = "pixel_median_bg"
                elif metric == "Pixel Min":
                    metric_name = "pixel_min"
                    bg_metric_name = "pixel_min_bg"
                elif metric == "Pixel Max":
                    metric_name = "pixel_max"
                    bg_metric_name = "pixel_max_bg"
                elif metric == "Pixel Sum":
                    metric_name = "pixel_sum"
                    bg_metric_name = "pixel_sum_bg"
                else:
                    metric_name = ""

                if metric_name == "":
                    show_info("Invalid metric, please try again")
                    return

                traces_path = self.get_traces_path(dataset, metric_name)

                if traces_path is None:
                    show_info("Invalid path, please try again")
                    return

                tracks = self.get_tracks(dataset, channel)

                if len(tracks) == 0:
                    show_info("No tracks found")
                    return

                json_dict = {"metadata": {}, "data": {}}
                track_cols = ["dataset", "channel", "particle", metric_name]
                if bg_metric_name in tracks.dtype.names:
                    track_cols.append(bg_metric_name)

                tracks = pd.DataFrame(tracks)
                tracks = tracks[track_cols]

                n_traces = 0

                for (dataset_name,channel_name), track_data in tracks.groupby(["dataset", "channel"]):

                    if dataset_name not in json_dict["data"]:
                        json_dict["data"][dataset_name] = []

                    particle_list = track_data["particle"].unique()

                    n_traces += len(particle_list)

                    for particle in particle_list:

                        particle_data = track_data[track_data["particle"] == particle]

                        data = particle_data[metric_name].tolist()
                        data = data[1:]

                        if bg_metric_name in particle_data.columns:
                            bg_data = particle_data[bg_metric_name].tolist()
                            bg_data = bg_data[1:]

                            if subtract_background:
                                data = np.array(data) - np.array(bg_data)
                                data = data.tolist()

                        dat = {"Data": data}
                        json_dict["data"][dataset_name].append(dat)

                with open(traces_path, "w") as f:
                    json.dump(json_dict, f)

                show_info(f"Eported {n_traces} traces to JSON file")

        except:
            print(traceback.format_exc())

    def update_traces_export_options(self):

        try:
            if hasattr(self, "tracking_dict"):

                tracks = self.get_tracks("All Datasets", "All Channels")

                if len(tracks) == 0:
                    return

                tracks = pd.DataFrame(tracks)

                export_metric = []

                if "photons" in tracks.columns:
                    export_metric.append("Photons")
                if "bg" in tracks.columns:
                    export_metric.append("Background")
                if "msd" in tracks.columns:
                    export_metric.append("Mean Squared Displacement")
                if "speed" in tracks.columns:
                    export_metric.append("Speed")
                if "D*" in tracks.columns:
                    export_metric.append("Apparent Diffusion Coefficient")
                if "pixel_mean" in tracks.columns:
                    export_metric.append("Pixel Mean")
                if "pixel_std" in tracks.columns:
                    export_metric.append("Pixel Standard Deviation")
                if "pixel_median" in tracks.columns:
                    export_metric.append("Pixel Median")
                if "pixel_min" in tracks.columns:
                    export_metric.append("Pixel Min")
                if "pixel_max" in tracks.columns:
                    export_metric.append("Pixel Max")
                if "pixel_sum" in tracks.columns:
                    export_metric.append("Pixel Sum")

                self.gui.traces_export_metric.blockSignals(True)
                self.gui.traces_export_metric.clear()
                self.gui.traces_export_metric.addItems(export_metric)
                self.gui.traces_export_metric.blockSignals(False)

        except:
            print(traceback.format_exc())