import traceback

import numpy as np
import pandas as pd
import json
import os
from PyQt5.QtWidgets import QFileDialog
from napari.utils.notifications import show_info


class _traces_utils:

    def get_traces_path(self, dataset, mode="json"):

        json_path = None

        if hasattr(self, "dataset_dict"):
            path = self.dataset_dict[dataset]["path"]

            base, ext = os.path.splitext(path)
            file_name = os.path.basename(base)
            dir_name = os.path.dirname(path)

            json_name = f"{file_name}_traces.json"

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
                subtract_background = self.gui.traces_export_subtract_background.isChecked()

                traces_path = self.get_traces_path(dataset)

                if traces_path is None:
                    show_info("Invalid path, please try again")
                    return

                tracks = self.get_tracks(dataset, channel)

                if len(tracks) == 0:
                    show_info("No tracks found")
                    return

                n_traces = len(np.unique(tracks["particle"]))
                json_dict = {"metadata": {}, "data": {dataset: []}}

                tracks = pd.DataFrame(tracks)

                for (dataset_name,particle), track_data in tracks.groupby(["dataset", "particle"]):

                    track_data = track_data.sort_values("frame")

                    channel_list = track_data["channel"].unique()

                    channel_dict = {"trace_dict": {}}

                    for channel_name in channel_list:

                        channel_data = track_data[track_data["channel"] == channel_name]

                        channel_data.drop(columns=["dataset", "particle",
                                                   "channel","frame"], inplace=True)

                        channel_data = channel_data.to_dict(orient="list")

                        for metric_name, column_name in self.moltrack_metrics.items():

                            if column_name not in channel_data:
                                continue

                            data = channel_data[column_name]

                            if metric_name not in channel_dict["trace_dict"]:
                                channel_dict["trace_dict"][metric_name] = []

                            if subtract_background:
                                bg_metric_name = f"{column_name}_bg"
                                if bg_metric_name in channel_data:
                                    data = np.array(data) - np.array(channel_data[bg_metric_name])
                                    data = data.tolist()

                            if len(data) == 0:
                                continue

                            channel_dict["trace_dict"][metric_name] = list(data)

                    json_dict["data"][dataset_name].append(channel_dict)

                with open(traces_path, "w") as f:
                    json.dump(json_dict, f)

                show_info(f"Eported {n_traces} traces to JSON file")

        except:
            print(traceback.format_exc())

