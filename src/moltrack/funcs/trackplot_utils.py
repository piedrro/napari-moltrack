import traceback
import pandas as pd
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from PyQt5.QtGui import QColor

class _trackplot_utils:


    def select_track(self, viewer, event):

        try:

            if hasattr(self, "trackplot_tracks") == False:
                return

            tracks = self.trackplot_tracks

            click_coords = self.track_layer.world_to_data(event.position)
            click_coords = [click_coords[-1], click_coords[-2]]
            track_coords = tracks[["x", "y"]].values

            track_distances = np.linalg.norm(track_coords - click_coords, axis=1)
            closest_distance = np.min(track_distances)
            loc_index = np.argmin(track_distances)
            track_id = tracks.loc[loc_index].particle

            particle_list = tracks.particle.unique().tolist()
            slider_value = particle_list.index(track_id)

            self.gui.trackplot_slider.setValue(slider_value)

        except:
            print(traceback.format_exc())
            pass

        pass


    def initialise_trackplot_slider(self):

        try:

            if hasattr(self, "trackplot_tracks"):

                tracks = self.trackplot_tracks

                if isinstance(tracks, pd.DataFrame) == False:
                    return

                if "particle" in tracks.columns:

                    particle_list = tracks.particle.unique().tolist()
                    n_tracks = len(particle_list) - 1

                    slider = self.gui.trackplot_slider

                    slider.blockSignals(True)

                    slider.setMinimum(0)
                    slider.setMaximum(n_tracks)

                    slider.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass


    def update_trackplot_slider(self):

        try:

            if hasattr(self, "trackplot_tracks") == False:
                return

            slider = self.gui.trackplot_slider
            slider_label = self.gui.trackplot_slider_label

            tracks = self.trackplot_tracks

            if isinstance(tracks, pd.DataFrame) == False:
                return

            particle_list = tracks.particle.unique().tolist()
            track_id = particle_list[slider.value()]

            slider_label.setText(str(track_id))

        except:
            print(traceback.format_exc())
            pass

    def highlight_track(self, track_id):

        try:

            if hasattr(self, "track_layer") == False:
                return

            self.draw_tracks(track_id=track_id)

        except:
            print(traceback.format_exc())
            pass

    def reset_tracks(self):

        try:
            if hasattr(self, "track_layer"):

                if self.gui.trackplot_highlight.isChecked() == False:
                    self.draw_tracks()

        except:
            print(traceback.format_exc())
            pass


    def focus_on_track(self, track_id):

        try:
            if hasattr(self, "track_layer") == False:
                return

            tracks = self.track_layer.data
            scale = self.track_layer.scale
            track = tracks[tracks[:,0] == track_id]

            track_y = track[:, -2]
            track_x = track[:, -1]

            center_x = track_x.mean() * scale[-2]
            center_y = track_y.mean() * scale[-1]

            center = (center_y, center_x)

            self.viewer.camera.center = center

        except:
            print(traceback.format_exc())
            pass



    def initialise_trackplot_layout(self, trackplot_data):

        try:

            self.trackplot_layout = {}
            self.trackplot_canvas.clear()

            if len(trackplot_data["data"]) > 0:

                # Create a vertical layout
                layout = pg.GraphicsLayout()
                self.trackplot_canvas.setCentralItem(layout)

                plot0 = None

                for i in range(len(trackplot_data["data"])):

                    values = trackplot_data["data"][i]["values"]
                    x_axis = trackplot_data["data"][i]["x_axis"]
                    x_axis_label = trackplot_data["data"][i]["x_axis_label"]
                    metric = trackplot_data["data"][i]["metric"]

                    if metric == "Track Length":
                        y_axis_label = "Track Length (frames)"
                    elif metric == "Track Duration":
                        y_axis_label = "Track Duration (s)"
                    elif metric == "Mean Squared Displacement":
                        y_axis_label = f"Mean Squared Displacement (µm²)"
                    elif metric == "Speed":
                        y_axis_label = f"Speed (µm/s)"
                    elif metric == "Apparent Diffusion Coefficient":
                        y_axis_label = f"Apparent Diffusion Coefficient (µm²/s)"
                    else:
                        y_axis_label = metric

                    # Add a new plot row to the layout
                    p = layout.addPlot(row=i, col=0)

                    plot_line = p.plot(x_axis, values,
                        pen=(i, len(trackplot_data)), name=y_axis_label)

                    if plot0 is not None:
                        p.setXLink(plot0)

                    p.setLabel('left', y_axis_label)
                    p.setLabel('bottom', x_axis_label)
                    p.getAxis('left').setWidth(60)

                    # Add legend
                    legend = pg.LegendItem(offset=(-10, 10))
                    legend.setParentItem(p.graphicsItem())
                    legend.addItem(plot_line, y_axis_label)

                    for sample, label in legend.items:
                        label.setAttr('color', QColor(255, 255, 255))
                        label.setAttr('size', '8pt')
                        label.setAttr('weight', 'bold')

                    self.trackplot_layout[i] = plot_line

        except:
            print(traceback.format_exc())
            pass






    def plot_tracks(self, viewer=None, reset = False):

        trackplot_data = self.get_trackplot_data()

        if len(trackplot_data["data"]) == 0:
            self.trackplot_canvas.clear()
            return

        if hasattr(self, "trackplot_layout") == False:
            self.initialise_trackplot_layout(trackplot_data)
        if reset == True:
            self.initialise_trackplot_layout(trackplot_data)

        track_highlight = self.gui.trackplot_highlight.isChecked()
        track_focus = self.gui.trackplot_focus.isChecked()

        if len(trackplot_data["data"]) > 0:

            user_label = trackplot_data["data"][0]["user_label"]
            track_center = trackplot_data["track_center"]
            track_id = trackplot_data["track_id"]
            dataset = trackplot_data["data"][0]["dataset"]
            channel = trackplot_data["data"][0]["channel"]

            if track_highlight:
                self.highlight_track(track_id)

            if track_focus:
                self.focus_on_track(track_id)

            for i in range(len(trackplot_data["data"])):

                plot_line = self.trackplot_layout[i]

                values = trackplot_data["data"][i]["values"]
                x_axis = trackplot_data["data"][i]["x_axis"]

                plot_line.setData(x_axis, values)






    def get_trackplot_data(self):

        subtract_bg = self.gui.trackplot_subtrack_background.isChecked()
        trackplot_xaxis = self.gui.trackplot_xaxis.currentText()

        trackplot_data = {"track_id": None, "track_center": [], "data": []}

        try:

            if hasattr(self, "trackplot_tracks"):

                tracks = self.trackplot_tracks

                if isinstance(tracks, pd.DataFrame) == False:
                    return trackplot_data

                particle_list = tracks.particle.unique().tolist()
                slider = self.gui.trackplot_slider
                track_id = particle_list[slider.value()]

                if "particle" in tracks.columns:

                    track_data = tracks[tracks.particle == track_id]

                    track_frames = track_data["frame"].values
                    track_center = track_data[["x", "y"]].mean().values

                    trackplot_data["track_id"] = track_id
                    trackplot_data["track_center"] = track_center

                    n_metric_combos = 4

                    for i in range(1, n_metric_combos+1):

                        metric_combo = getattr(self.gui, f"trackplot_metric{i}")
                        metric_label = metric_combo.currentText()

                        if metric_label in self.moltrack_metrics.keys():
                            metric_name = self.moltrack_metrics[metric_label]

                            if metric_name in track_data.columns:

                                metric_values = track_data[metric_name].values.tolist()
                                dataset = track_data["dataset"].values[0]
                                channel = track_data["channel"].values[0]

                                exposure_time_ms = self.dataset_dict[dataset]["exposure_time"]
                                pixel_size = self.dataset_dict[dataset]["pixel_size"]
                                exposure_time_s = exposure_time_ms / 1000

                                if "user_label" in track_data.columns:
                                    user_label = track_data["user_label"].values[0]
                                else:
                                    user_label = None

                                if subtract_bg:
                                    bg_name = f"{metric_name}_bg"
                                    if bg_name in track_data.columns:
                                        bg_values = track_data[bg_name].values.tolist()
                                        metric_values = np.array(metric_values) - np.array(bg_values)
                                        metric_values = metric_values.tolist()

                                if trackplot_xaxis == "Frame Index":
                                    x_axis = list(track_frames)
                                    x_axis_label = "Frame Index"
                                elif trackplot_xaxis == "Time (s)":
                                    x_axis = [i * exposure_time_s for i in range(len(metric_values))]
                                    x_axis_label = "Time (s)"
                                elif trackplot_xaxis == "Time (ms)":
                                    x_axis = [i * exposure_time_ms for i in range(len(metric_values))]
                                    x_axis_label = "Time (ms)"
                                else:
                                    x_axis = [i for i in range(len(metric_values))]
                                    x_axis_label = "Segment Index"

                                trackplot_data["data"].append({"metric": metric_label,
                                                               "dataset": dataset,
                                                               "channel": channel,
                                                               "values": metric_values,
                                                               "x_axis": x_axis,
                                                               "x_axis_label": x_axis_label,
                                                               "user_label": user_label})
        except:
            print(traceback.format_exc())
            pass

        return trackplot_data




    def init_trackplot_tracks(self, reset = True):

        try:
            if hasattr(self, "trackplot_tracks") == False or reset == True:

                dataset = self.gui.trackplot_dataset.currentText()
                channel = self.gui.trackplot_channel.currentText()

                trackplot_tracks = self.get_tracks(dataset, channel)

                if len(trackplot_tracks) > 0:
                    self.trackplot_tracks = pd.DataFrame(trackplot_tracks)
                else:
                    self.trackplot_tracks = None

        except:
            print(traceback.format_exc())
            pass

        return self.trackplot_tracks



    def get_trackplot_metrics(self):

        trackplot_metrics = None

        if hasattr(self, "tracking_dict"):

            if isinstance(self.trackplot_tracks, pd.DataFrame) == False:
                return trackplot_metrics

            track_cols = self.trackplot_tracks.columns

            trackplot_metrics = {}

            for metric_name in list(self.moltrack_metrics.keys()):
                metric = self.moltrack_metrics[metric_name]

                if metric in track_cols:
                    trackplot_metrics[metric_name] = metric
                if metric + "_fret" in track_cols:
                    trackplot_metrics[metric_name + " FRET"] = metric + "_fret"
                    self.moltrack_metrics[metric_name + " FRET"] = metric + "_fret"

        return trackplot_metrics



    def update_trackplot_options(self):

        try:

            self.init_trackplot_tracks(reset=True)
            self.initialise_trackplot_slider()
            n_metric_combos = 4
            metric_list = [""]

            metric_dict = self.get_trackplot_metrics()

            if isinstance(metric_dict, dict):
                metric_list = list(metric_dict.keys())
                metric_list.insert(0, "")

            for i in range(1, n_metric_combos+1):

                metric_combo = getattr(self.gui, f"trackplot_metric{i}")

                metric_combo.blockSignals(True)

                metric_combo.clear()
                if i == 1:
                    metric_combo.addItems(metric_list[1:])
                else:
                    metric_combo.addItems(metric_list)

                metric_combo.blockSignals(False)

            self.plot_tracks()

        except:
            print(traceback.format_exc())
            pass
