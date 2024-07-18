import traceback
import pandas as pd
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

class _trackplot_utils:


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

            slider = self.gui.trackplot_slider
            slider_label = self.gui.trackplot_slider_label

            slider_label.setText(str(slider.value()))

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



    def plot_tracks(self):

        trackplot_data = self.get_trackplot_data()
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

            track_center = f"[{track_center[0]:.2f}, {track_center[1]:.2f}]"

            self.trackplot_canvas.clear()

            # Create a vertical layout
            layout = pg.GraphicsLayout()
            self.trackplot_canvas.setCentralItem(layout)

            title = f"Track ID: {trackplot_data['track_id']} | Track Center: {track_center}"
            if user_label is not None:
                title += f" | User Label: {user_label}"

            plot0 = None

            for i in range(len(trackplot_data["data"])):
                metric = trackplot_data["data"][i]["metric"]
                values = trackplot_data["data"][i]["values"]

                # Add a new plot row to the layout
                p = layout.addPlot(row=i, col=0)

                if plot0 is not None:
                    p.setXLink(plot0)

                curve = p.plot(values, pen=(i, len(trackplot_data)), name=metric)
                p.getAxis('left').setWidth(50)

                # Add legend
                legend = pg.LegendItem(offset=(-10, 10))  # Adjust the offset to position legend at the top right
                legend.setParentItem(p.graphicsItem())
                legend.addItem(curve, metric)
                legend.setBrush('w')  # White background for the legend

                if i == 0:
                    p.setTitle(title)
                    plot0 = p


    def get_trackplot_data(self):

        subtract_bg = self.gui.trackplot_subtrack_background.isChecked()

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

                                trackplot_data["data"].append({"metric": metric_label,
                                                               "dataset": dataset,
                                                               "channel": channel,
                                                               "values": metric_values,
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
