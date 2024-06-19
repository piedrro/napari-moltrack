import traceback
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint
from shapely.strtree import STRtree
import matplotlib.pyplot as plt


class _track_filter_utils:

    def update_track_filter_metric(self, viewer=None):

        try:

            criterion = self.gui.track_filter_criterion.currentText()

            if criterion == "Track Length" or criterion == "Track Duration":
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

                if len(tracks) > 0:

                    critertion_options = ["Track Length", "Track Duration"]

                    tracks = pd.DataFrame(tracks)

                    if "msd" in tracks.columns:
                        critertion_options.append("Mean Squared Displacement")
                    if "speed" in tracks.columns:
                        critertion_options.append("Speed")
                    if "D*" in tracks.columns:
                        critertion_options.append("Apparent Diffusion Coefficient")

                    self.gui.track_filter_criterion.blockSignals(True)
                    self.gui.track_filter_criterion.clear()
                    self.gui.track_filter_criterion.addItems(critertion_options)
                    self.gui.track_filter_criterion.blockSignals(False)

        except:
            traceback.print_exc()

    def update_track_criterion_ranges(self, viewer=None, plot=True):

        try:

            stats = self.track_statistics_filtering(viewer)

            if len(stats) > 0:

                values = np.array(stats['stat'])

                stat_min = values.min()
                stat_max = values.max()

                self.gui.track_filter_min.blockSignals(True)
                self.gui.track_filter_max.blockSignals(True)

                self.gui.track_filter_min.setValue(stat_min)
                self.gui.track_filter_max.setValue(stat_max)

                self.gui.track_filter_min.blockSignals(False)
                self.gui.track_filter_max.blockSignals(False)

                if plot:
                    self.plot_track_filter_graph(values)

        except:
            traceback.print_exc()







    def track_statistics_filtering(self, viewer=None, mode="stats"):

        dataset = self.gui.track_filter_dataset.currentText()
        channel = self.gui.track_filter_channel.currentText()
        criterion = self.gui.track_filter_criterion.currentText()
        metric = self.gui.track_filter_metric.currentText()

        min_filter = self.gui.track_filter_min.value()
        max_filter = self.gui.track_filter_max.value()

        tracks = self.get_tracks(dataset, channel)

        filtered_tracks = []
        stats = []

        if len(tracks) > 0:

            tracks = pd.DataFrame(tracks)

            for (dataset, channel, particle), track in tracks.groupby(['dataset', 'channel', 'particle']):

                try:

                    if criterion == "Track Length":
                        stat = len(track)

                    elif criterion == "Track Duration":
                        time_values = track['time'].values
                        duration = time_values[-1] - time_values[0]
                        stat = duration

                    else:
                        if criterion == "Mean Squared Displacement":
                            data = track['msd']
                        elif criterion == "Speed":
                            data = track['speed']
                        elif criterion == "Apparent Diffusion Coefficient":
                            data = track['D*']

                        data = data.iloc[1:]

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
                    print(traceback.print_exc())


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
                values = values[~np.isnan(values)]

                if len(values) > 0:

                    if criterion == "Track Length":
                        xlabel = "Track Length (frames)"
                    elif criterion == "Track Duration":
                        xlabel = "Track Duration (s)"
                    else:
                        if criterion == "Mean Squared Displacement":
                            xlabel = f"{metric} MSD (µm²)"
                        if criterion == "Speed":
                            xlabel = f"{metric} Speed (µm/s)"
                        if criterion == "Apparent Diffusion Coefficient":
                            xlabel = f"{metric} Apparent Diffusion Coefficient (µm²/s)"


                    ax = self.track_graph_canvas.addPlot()

                    # Create histogram
                    y, x = np.histogram(values, bins=100)

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

            print(f"Removed {n_removed} tracks")

        except:
            print(traceback.print_exc())
            pass