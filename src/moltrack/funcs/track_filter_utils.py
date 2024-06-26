import traceback
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
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

                    if "time" in tracks.columns:
                        critertion_options.append("Track Duration")
                    if "msd" in tracks.columns:
                        critertion_options.append("Mean Squared Displacement")
                    if "speed" in tracks.columns:
                        critertion_options.append("Speed")
                    if "D*" in tracks.columns:
                        critertion_options.append("Apparent Diffusion Coefficient")
                    if "pixel_mean" in tracks.columns:
                        critertion_options.append("Pixel Mean")
                    if "pixel_std" in tracks.columns:
                        critertion_options.append("Pixel Standard Deviation")
                    if "pixel_median" in tracks.columns:
                        critertion_options.append("Pixel Median")
                    if "pixel_min" in tracks.columns:
                        critertion_options.append("Pixel Min")
                    if "pixel_max" in tracks.columns:
                        critertion_options.append("Pixel Max")
                    if "pixel_sum" in tracks.columns:
                        critertion_options.append("Pixel Sum")

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
                        if criterion == "Mean Squared Displacement":
                            data = track['msd']
                        elif criterion == "Speed":
                            data = track['speed']
                        elif criterion == "Apparent Diffusion Coefficient":
                            data = track['D*']
                        elif criterion == "Pixel Mean":
                            data = track['pixel_mean']
                        elif criterion == "Pixel Standard Deviation":
                            data = track['pixel_std']
                        elif criterion == "Pixel Median":
                            data = track['pixel_median']
                        elif criterion == "Pixel Min":
                            data = track['pixel_min']
                        elif criterion == "Pixel Max":
                            data = track['pixel_max']
                        elif criterion == "Pixel Sum":
                            data = track['pixel_sum']

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
                    elif criterion in ["Pixel Mean", "Pixel Standard Deviation",
                                       "Pixel Median", "Pixel Min", "Pixel Max", "Pixel Sum"]:
                        xlabel = criterion
                    else:
                        if criterion == "Mean Squared Displacement":
                            xlabel = f"{metric} MSD (µm²)"
                        if criterion == "Speed":
                            xlabel = f"{metric} Speed (µm/s)"
                        if criterion == "Apparent Diffusion Coefficient":
                            xlabel = f"{metric} Apparent Diffusion Coefficient (µm²/s)"

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
            pass