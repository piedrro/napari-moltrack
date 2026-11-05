import math
import traceback
import warnings
from functools import partial

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info
from shapely.geometry import Point
from shapely.strtree import STRtree

from moltrack.funcs.compute_utils import Worker

warnings.filterwarnings("ignore", category=UserWarning)

class _tracking_utils:

    def update_segtrack_options(self):

        detect_mode = self.gui.segtracks_detect.currentText()

        if detect_mode.lower() == "entire track":
            self.gui.segtracks_remove.clear()
            self.gui.segtracks_remove.addItems(["Track"])
        else:
            self.gui.segtracks_remove.clear()
            self.gui.segtracks_remove.addItems(["Track", "Track Segment"])

    def remove_segtracks(self):

        try:
            dataset = self.gui.segtracks_dataset.currentText()
            channel = self.gui.segtracks_channel.currentText()
            segchannel = self.gui.segtracks_seg.currentText()
            detect_mode = self.gui.segtracks_detect.currentText()
            remove_mode = self.gui.segtracks_remove.currentText()

            min_track_length = int(self.gui.min_track_length.value())

            self.update_ui(init=True)

            polygons = self.get_shapes(segchannel, flipxy=True, polygon=True)

            if len(polygons) == 0:
                return

            tracks = self.get_tracks(dataset, channel, return_dict=False, include_metadata=True)

            tracks = pd.DataFrame(tracks)

            filtered_tracks = []

            track_groups = tracks.groupby(["dataset", "channel", "track_id"])

            n_tracks_removed = 0
            n_segments_removed = 0

            for (dataset, channel, particle), track in track_groups:

                track = track.reset_index(drop=True)

                coords = np.stack([track["x"], track["y"]], axis=1)
                points = [Point(coord) for coord in coords]

                spatial_index = STRtree(points)

                inside_indices = []

                for polygon_index, polygon in enumerate(polygons):
                    possible_points = spatial_index.query(polygon)

                    for point_index in possible_points:
                        point = points[point_index]

                        if polygon.contains(point):
                            inside_indices.append(point_index)

                outside_indices = list(set(range(len(points))) - set(inside_indices))

                if detect_mode.lower() == "entire track":
                    if len(outside_indices) == len(points):
                        n_tracks_removed += 1
                    else:
                        filtered_tracks.append(track)
                else:
                    if remove_mode.lower() == "track":
                        if len(outside_indices) == 0:
                            filtered_tracks.append(track)
                        else:
                            n_tracks_removed += 1
                    else:
                        if len(outside_indices) == 0:
                            filtered_tracks.append(track)
                        else:
                            track_length = len(track)

                            if track_length - len(outside_indices) >= min_track_length:
                                track = track.drop(outside_indices, axis=0)
                                filtered_tracks.append(track)
                                n_segments_removed += len(outside_indices)
                            else:
                                n_tracks_removed += 1

            retained = pd.concat(filtered_tracks, ignore_index=True) if filtered_tracks else tracks.iloc[:0]
            for partition in self.session.selected_partitions(
                self.gui.segtracks_dataset.currentText(), self.gui.segtracks_channel.currentText()
            ):
                selected = retained[(retained.dataset == partition[0]) & (retained.channel == partition[1])]
                self.session.retain_rows(*partition, selected.id.to_numpy(), tracks=True)

            if remove_mode == "Track":
                show_info(f"Removed {n_tracks_removed} complete tracks")
            else:
                if n_segments_removed == 0:
                    show_info(f"Removed {n_tracks_removed} complete tracks")
                else:
                    show_info(f"Removed {n_tracks_removed} complete tracks and {n_segments_removed} track segments")

            self.update_ui()
            self.draw_tracks()

        except:
            self.update_ui()
            print(traceback.format_exc())

    def get_tracks(self, dataset, channel, return_dict=False, include_metadata=True):
        from moltrack.model_views import gui_query
        return gui_query(self.session, dataset, channel, tracks=True,
                         return_dict=return_dict, include_metadata=include_metadata)


    def process_tracking_results(self, track_data):
        from moltrack.funcs.compute_utils import publish_model_result
        if not publish_model_result(self, track_data):
            return
        total = sum(len(value[0]) for _, _, value in track_data.results)
        if total:
            self.gui.locs_export_data.clear()
            self.gui.locs_export_data.addItems(['Localisations', 'Tracks'])
            self.gui.heatmap_data.clear()
            self.gui.heatmap_data.addItems(['Localisations', 'Tracks'])
        show_info(f'Tracking complete, {total} tracks found')
        return
    def tracking_finished(self):

        self.draw_localisations()
        self.draw_tracks()

        channel_name = self.gui.tracking_channel.currentText()

        self.gui.adc_channel.blockSignals(True)
        self.gui.adc_channel.setCurrentText(channel_name)
        self.gui.adc_channel.blockSignals(False)

        channel = self.gui.tracking_channel.currentText()

        if channel != "All Channels":
            self.gui.track_filter_channel.blockSignals(True)
            self.gui.trackplot_channel.blockSignals(True)
            self.gui.track_filter_channel.setCurrentText(channel)
            self.gui.trackplot_channel.setCurrentText(channel)
            self.gui.track_filter_channel.blockSignals(False)
            self.gui.trackplot_channel.blockSignals(False)

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()

        self.update_traces_export_options()

        self.update_trackplot_options()
        self.plot_tracks(reset=True)

        self.update_ui()

    def initialise_tracking(self):
        try:
            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()

            loc_data = self.get_locs(dataset, channel, return_dict=True, include_metadata=True)

            if len(loc_data) > 0:
                self.update_ui(init=True)

                self.gui.trackplot_highlight.blockSignals(True)
                self.gui.trackplot_highlight.setChecked(False)
                self.gui.trackplot_highlight.blockSignals(False)

                self.gui.trackplot_focus.blockSignals(True)
                self.gui.trackplot_focus.setChecked(False)
                self.gui.trackplot_focus.blockSignals(False)

                from moltrack.model_processing import (
                    ProcessingRequest, TrackingSettings, compute_worker, prepare,
                )
                segchannel = self.gui.tracking_segmentations.currentText()
                segmented = segchannel in ('Segmentations', 'Cells')
                shapes = self.get_shapes(segchannel, flipxy=True, polygon=False) if segmented else []
                polygons = tuple(tuple(tuple(map(float, point)) for point in shape) for shape in (shapes or []))
                settings = TrackingSettings(
                    search_range=float(self.gui.trackpy_search_range.value()),
                    memory=int(self.gui.trackpy_memory.value()),
                    min_length=int(self.gui.min_track_length.value()),
                    remove_unlinked=self.gui.remove_unlinked.isChecked(),
                    polygons=polygons,
                    segmentation_column=segchannel[:-1].lower() + '_index' if segmented else None)
                request = ProcessingRequest('tracking', tuple((d['dataset'], d['channel']) for d in loc_data), settings)
                worker = Worker(compute_worker, prepare(self.session, request), self.stop_event)
                worker.signals.result.connect(self.process_tracking_results)
                worker.signals.finished.connect(self.tracking_finished)
                worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.tracking_progressbar))
                self.threadpool.start(worker)

            else:
                show_info(f"No localisations found for {dataset}`")

        except:
            print(traceback.format_exc())
            self.update_ui()

    def nice_step(self, min_val, max_val, n_steps=50):
        range_val = max_val - min_val
        raw_step = range_val / n_steps
        exponent = math.floor(math.log10(raw_step))
        base = 10 ** exponent
        fraction = raw_step / base

        # Choose 1, 2, or 5 times the base for a "nice" step
        if fraction < 1.5:
            nice = 1
        elif fraction < 3.5:
            nice = 2
        elif fraction < 7.5:
            nice = 5
        else:
            nice = 10

        return nice * base


    def draw_tracks(self, viewer=None, track_id=None, reset_cmap_range = True):
        try:

            remove_tracks = True

            if hasattr(self, "session"):
                if hasattr(self, "tracks_layer"):
                    show_tracks = self.loc_layer.visible
                else:
                    show_tracks = True

            if show_tracks:
                layer_names = [layer.name for layer in self.viewer.layers]

                dataset_name = self.gui.moltrack_dataset_selector.currentText()
                channel_name = self.gui.moltrack_channel_selector.currentText()

                tracks = self.get_tracks(dataset_name, channel_name,
                    return_dict=False, include_metadata=True, )

                if len(tracks) > 0:
                    n_frames = self.session.image(dataset_name, channel_name).shape[0]

                    pixel_size = self.session._view(dataset_name, channel_name).acquisition.pixel_size
                    scale = [pixel_size or 1.0] * 2
                    units = ("pixel", "nm", "nm") if pixel_size is not None else ("pixel",) * 3

                    remove_tracks = False

                    render_tracks = pd.DataFrame(tracks)

                    cmap_metric = self.gui.track_colour_metric.currentText()
                    cmap_metric = self.moltrack_metrics.get(cmap_metric)
                    cmap_name = self.gui.track_colour_colourmap.currentText()

                    if cmap_metric in render_tracks.columns and np.isfinite(render_tracks[cmap_metric].to_numpy(dtype=float)).any():
                        metric_data = render_tracks[cmap_metric].to_numpy(dtype=float)
                        finite_metrics = metric_data[np.isfinite(metric_data)]

                        if reset_cmap_range:
                            cmap_min = min(finite_metrics)
                            cmap_max = max(finite_metrics)

                            step = self.nice_step(cmap_min,cmap_max) if cmap_max > cmap_min else 1.0
                            self.gui.track_colour_min.blockSignals(True)
                            self.gui.track_colour_max.blockSignals(True)

                            self.gui.track_colour_min.setRange(cmap_min, cmap_max)
                            self.gui.track_colour_max.setRange(cmap_min, cmap_max)

                            self.gui.track_colour_min.setValue(cmap_min)
                            self.gui.track_colour_max.setValue(cmap_max)

                            self.gui.track_colour_min.setSingleStep(step)
                            self.gui.track_colour_max.setSingleStep(step)

                            self.gui.track_colour_min.blockSignals(False)
                            self.gui.track_colour_max.blockSignals(False)
                        else:
                            cmap_min = self.gui.track_colour_min.value()
                            cmap_max = self.gui.track_colour_max.value()

                        if cmap_min > cmap_max:
                            cmap_min, cmap_max = cmap_max, cmap_min

                        metric_norm_data = (
                            np.clip((metric_data - cmap_min) / (cmap_max - cmap_min), 0, 1)
                            if cmap_max > cmap_min else np.zeros(len(metric_data))
                        )
                    else:
                        metric_norm_data = None

                    if track_id in render_tracks["track_id"].values:
                        selected = render_tracks["track_id"] == track_id
                        render_tracks = render_tracks[selected]
                        if metric_norm_data is not None:
                            metric_norm_data = metric_norm_data[selected]
                    render_tracks = render_tracks[["track_id", "frame", "y", "x"]]

                    render_tracks = render_tracks.to_records(index=False)
                    render_tracks = [list(track) for track in render_tracks]
                    render_tracks = np.array(render_tracks).copy()
                    render_tracks[:, 1] = 0

                    properties = {cmap_metric: metric_norm_data}

                    if "Tracks" not in layer_names:
                        self.track_layer = self.viewer.add_tracks(
                            render_tracks, name="Tracks",scale=scale,
                            colormap=cmap_name, properties=properties,
                            color_by="track_id", units=units,)

                        self.viewer.reset_view()
                        self.track_layer.mouse_double_click_callbacks.append(self.select_track)

                    else:

                        self.track_layer.color_by = "track_id"
                        self.track_layer.data = render_tracks
                        self.track_layer.scale = scale
                        self.track_layer.units = units
                        self.track_layer.colormap = cmap_name

                    if self.gui.show_tracks.isChecked() == False:
                        if self.track_layer in self.viewer.layers:
                            self.viewer.layers.remove(self.track_layer)

                    self.track_layer.properties = properties

                    if metric_norm_data is not None:
                        self.track_layer.color_by = cmap_metric

                    self.track_layer.tail_length = n_frames * 2
                    self.track_layer.blending = "opaque"

                    self.track_layer.selected_data = []

                    self.viewer.scale_bar.visible = True
                    self.track_layer.units = ("pixel", "nm", "nm")

            if remove_tracks:
                try:
                    if "Tracks" in layer_names:
                        self.viewer.layers.remove("Tracks")
                        if hasattr(self, "track_layer"):
                            del self.track_layer
                except:
                    print(traceback.format_exc())

            if hasattr(self, "track_layer"):
                self.track_layer.refresh()

        except:
            print(traceback.format_exc())
