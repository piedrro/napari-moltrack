import traceback
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

                if metric == "Mean Squared Displacement":
                    metric_name = "msd"
                elif metric == "Speed":
                    metric_name = "speed"
                elif metric == "Apparent Diffusion Coefficient":
                    metric_name = "D*"
                elif metric == "Photons":
                    metric_name = "photons"
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

                tracks = pd.DataFrame(tracks)[track_cols]

                n_traces = 0

                for (dataset_name,channel_name), track_data in tracks.groupby(["dataset", "channel"]):

                    if dataset_name not in json_dict["data"]:
                        json_dict["data"][dataset_name] = []

                    particle_list = track_data["particle"].unique()

                    n_traces += len(particle_list)

                    for particle in particle_list:

                        data = track_data[track_data["particle"] == particle]
                        data = data[track_cols[-1]].tolist()
                        data = data[1:]

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

                self.gui.traces_export_metric.blockSignals(True)
                self.gui.traces_export_metric.clear()
                self.gui.traces_export_metric.addItems(export_metric)
                self.gui.traces_export_metric.blockSignals(False)

        except:
            print(traceback.format_exc())