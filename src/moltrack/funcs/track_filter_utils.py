import traceback

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info


class _track_filter_utils:



    def update_track_filter_metric(self, viewer=None):

        try:

            criterion = self.gui.track_filter_criterion.currentText()

            if criterion in ["Track Length", "Track Duration"]:
                metrics = [""]
            else:
                metrics = ["Mean", "Median", "Standard Deviation", "Min", "Max"]

            self.gui.track_filter_metric.blockSignals(True)
            self.gui.track_filter_metric.clear()
            self.gui.track_filter_metric.addItems(metrics)
            self.gui.track_filter_metric.blockSignals(False)

            self.update_track_criterion_ranges()

        except:
            traceback.print_exc()

    def update_track_filter_criterion(self, viewer=None):

        try:

            if hasattr(self, "tracking_dict"):

                dataset = self.gui.track_filter_dataset.currentText()
                channel = self.gui.track_filter_channel.currentText()

                tracks = self.get_tracks(dataset, channel)

                critertion_options = []

                if len(tracks) > 0:
                    critertion_options.append("Track Length")

                    tracks = pd.DataFrame(tracks)

                    for metrix_name, metric in self.moltrack_metrics.items():
                        if metric in tracks.columns:
                            critertion_options.append(metrix_name)

                self.gui.track_filter_criterion.clear()
                self.gui.track_filter_criterion.addItems(critertion_options)

        except:
            traceback.print_exc()

    def update_track_criterion_ranges(self, viewer=None, plot=True):

        try:
            criterion = self.gui.track_filter_criterion.currentText()

            if "pixel" not in criterion.lower():
                self.gui.track_filter_subtract_bg.blockSignals(True)
                self.gui.track_filter_subtract_bg.setChecked(False)
                self.gui.track_filter_subtract_bg.setEnabled(False)
                self.gui.track_filter_subtract_bg.blockSignals(False)
            else:
                self.gui.track_filter_subtract_bg.setEnabled(True)

            stats = self.track_statistics_filtering(viewer)

            values = []

            if len(stats) > 0:

                values = np.array(stats['stat'])

                values = values.tolist()
                values = [v for v in values if v not in [None, np.nan]]
                values = np.array(values)

                stat_min = min(values)
                stat_max = max(values)

                self.gui.track_filter_min.blockSignals(True)
                self.gui.track_filter_max.blockSignals(True)

                self.gui.track_filter_min.setValue(stat_min)
                self.gui.track_filter_max.setValue(stat_max)

                self.gui.track_filter_min.blockSignals(False)
                self.gui.track_filter_max.blockSignals(False)

            if len(values) > 0 and plot:
                self.plot_track_filter_graph(values)
            else:
                self.track_graph_canvas.clear()

        except:
            traceback.print_exc()


    def track_statistics_filtering(self, viewer=None, mode="stats"):

        dataset = self.gui.track_filter_dataset.currentText()
        channel = self.gui.track_filter_channel.currentText()
        criterion = self.gui.track_filter_criterion.currentText()
        metric = self.gui.track_filter_metric.currentText()
        subtract_background = self.gui.track_filter_subtract_bg.isChecked()

        if "pixel" not in criterion.lower():
            subtract_background = False

        min_filter = self.gui.track_filter_min.value()
        max_filter = self.gui.track_filter_max.value()

        tracks = self.get_tracks(dataset, channel)

        filtered_tracks = []
        stats = []

        if len(tracks) > 0:

            tracks = pd.DataFrame(tracks)

            for (dataset, channel, particle), track in tracks.groupby(['dataset', 'channel', 'particle']):

                try:

                    if len(track) == 0:
                        continue

                    if criterion == "Track Length":
                        stat = len(track)

                    elif criterion == "Track Duration":
                        time_values = track['time'].values
                        duration = time_values[-1] - time_values[0]
                        stat = duration

                    else:
                        if criterion in self.moltrack_metrics:
                            data = track[self.moltrack_metrics[criterion]]
                        else:
                            data = []

                        if len(data) == 0:
                            continue

                        if subtract_background:
                            data_name = data.name
                            bg_data_name = f"{data_name}_bg"
                            if bg_data_name in track.columns:
                                bg_data = track[bg_data_name]
                                data = data - bg_data

                        data = data.iloc[1:]
                        data = data.dropna()

                        if metric.lower() == "mean":
                            stat = data.mean()
                        elif metric.lower() == "median":
                            stat = data.median()
                        elif metric.lower() == "standard deviation":
                            stat = data.std()
                        elif metric.lower() == "min":
                            stat = data.min()
                        elif metric.lower() == "max":
                            stat = data.max()
                        else:
                            stat = None

                    if stat is not None:
                        stats.append([dataset, channel, particle, stat])

                    if mode == "tracks":
                        if stat >= min_filter and stat <= max_filter:
                            filtered_tracks.append(track)

                except:
                    pass


        if mode == "stats":
            if len(stats) > 0:
                stats = pd.DataFrame(stats, columns=['dataset', 'channel', 'particle', 'stat'])

            return stats

        else:
            if len(filtered_tracks) > 0:
                filtered_tracks = pd.concat(filtered_tracks)
                n_removed = len(tracks) - len(filtered_tracks)

                return filtered_tracks, n_removed


    def plot_track_filter_graph(self, values):

        try:

            criterion = self.gui.track_filter_criterion.currentText()
            metric = self.gui.track_filter_metric.currentText()

            self.track_graph_canvas.clear()

            if values is not None:

                if len(values) > 0:

                    values = values[~np.isnan(values)]

                    if criterion == "Track Length":
                        xlabel = "Track Length (frames)"
                    elif criterion == "Track Duration":
                        xlabel = "Track Duration (s)"
                    elif criterion == "Mean Squared Displacement" or criterion == "MSD":
                        xlabel = f"{metric} MSD (µm²)"
                    elif criterion == "Speed":
                        xlabel = f"{metric} Speed (µm/s)"
                    elif criterion in ["D","Rolling D"]:
                        xlabel = f"{metric} D (µm²/s)"
                    elif criterion in ["D*","Rolling D*"]:
                        xlabel = f"{metric} D* (µm²/s)"
                    elif criterion == "Step Size":
                        xlabel = f"{metric} Step Size (µm)"
                    elif criterion == "Rolling MSD":
                        xlabel = f"{metric} Rolling MSD (µm²)"
                    elif criterion ==  "Membrane Distance":
                        xlabel = f"{metric} Membrane Distance (µm)"
                    elif criterion ==  "Midline Distance":
                        xlabel = f"{metric} Midline Distance (µm)"
                    elif criterion ==  "Centroid Distance":
                        xlabel = f"{metric} Centroid Distance (µm)"
                    elif criterion ==  "Cell Pole Distance":
                        xlabel = f"{metric} Cell Pole Distance (µm)"
                    elif criterion ==  "Angle":
                        xlabel = f"{metric} Angle (degrees)"
                    else:
                        xlabel = criterion

                    ax = self.track_graph_canvas.addPlot()

                    # Create histogram
                    y, x = np.histogram(values, bins=50)

                    ax.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 75))
                    ax.setLabel('bottom', xlabel)
                    ax.setLabel('left', 'Frequency')


        except:
            traceback.print_exc()



    def filter_tracks(self, viewer=None):

        try:

            filtered_tracks, n_removed = self.track_statistics_filtering(mode="tracks")

            for (dataset, channel), tracks in filtered_tracks.groupby(['dataset', 'channel']):

                tracks = tracks.to_dict('records')
                self.tracking_dict[dataset][channel]['tracks'] = tracks

            self.draw_tracks()
            self.update_track_criterion_ranges()
            self.plot_diffusion_graph()


            show_info(f"Removed {n_removed} tracks")

        except:
            print(traceback.print_exc())
