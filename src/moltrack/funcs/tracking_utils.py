import math
import traceback
import warnings
from functools import partial

import numpy as np
import pandas as pd
import trackpy as tp
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

            track_groups = tracks.groupby(["dataset", "channel", "particle"])

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

            if len(filtered_tracks) > 0:
                filtered_tracks = pd.concat(filtered_tracks, ignore_index=True)

                for (dataset, channel), filtered_track in filtered_tracks.groupby(["dataset", "channel"]):
                    filtered_track.reset_index(drop=True, inplace=True)
                    filtered_track = filtered_track.to_records(index=False)
                    self.tracking_dict[dataset][channel]["tracks"] = filtered_track

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


        order = ["dataset", "channel", "group", "particle",
                 "frame", "cell_index", "segmentation_index",
                 "x", "y", "photons", "bg",
                 "sx", "sy", "lpx", "lpy",
                 "ellipticity", "net_gradient", "iterations"]

        track_data = []
        group = 0

        try:
            if dataset == "All Datasets":
                dataset_list = list(self.tracking_dict.keys())
            else:
                dataset_list = [dataset]

            for dataset_name in dataset_list:
                if dataset_name not in self.tracking_dict.keys():
                    continue

                if channel == "All Channels":
                    channel_list = list(self.tracking_dict[dataset_name].keys())
                else:
                    channel_list = [channel]

                for channel_name in channel_list:
                    if (channel_name not in self.tracking_dict[dataset_name].keys()):
                        continue

                    track_dict = self.tracking_dict[dataset_name][channel_name]

                    if "tracks" in track_dict.keys():
                        tracks = track_dict["tracks"].copy()

                        tracks = pd.DataFrame(tracks)

                        if include_metadata:
                            if "dataset" not in tracks.columns:
                                tracks.insert(0, "dataset", dataset_name)
                            if "channel" not in tracks.columns:
                                tracks.insert(1, "channel", channel_name)
                            if len(dataset_list) > 1:
                                if "group" not in tracks.columns:
                                    tracks.insert(2, "group", group)

                        mask = []

                        for col in tracks.columns:
                            if col not in order:
                                order.append(col)

                        for col in order:
                            if col in tracks.columns:
                                mask.append(col)

                        tracks = tracks[mask]

                        tracks = tracks.to_records(index=False)

                        n_tracks = len(tracks)

                        if n_tracks > 0:
                            if return_dict == False:
                                track_data.append(tracks)
                            else:
                                track_dict = {"dataset": dataset_name, "channel": channel_name, "tracks": tracks, }
                                track_data.append(track_dict)

        except:
            print(traceback.format_exc())

        if return_dict == False:
            if len(track_data) == 0:
                pass
            elif len(track_data) == 1:
                track_data = track_data[0]
            else:

                name_list = [set(dat.dtype.names) for dat in track_data]
                common_names = list(set.intersection(*name_list))

                if len(common_names) > 0:
                    track_data = [dat[common_names] for dat in track_data]
                    track_data = np.hstack(track_data).view(np.recarray).copy()
                else:
                    track_data = []

        return track_data


    def link_locs(self, f, search_range, pos_columns=None,
            t_column='frame', progress_callback = None, **kwargs):

        from trackpy.linking.linking import (
            coords_from_df,
            guess_pos_columns,
            link_iter,
            pandas_sort,
        )

        if pos_columns is None:
            pos_columns = guess_pos_columns(f)

        # copy the dataframe
        f = f.copy()
        # coerce t_column to integer type (use np.int64 to avoid 32-bit on windows)
        if not np.issubdtype(f[t_column].dtype, np.integer):
            f[t_column] = f[t_column].astype(np.int64)
        # sort on the t_column
        pandas_sort(f, t_column, inplace=True)

        coords_iter = coords_from_df(f, pos_columns, t_column)
        ids = []

        n_iter = 0
        for t, coords in coords_iter:
            n_iter += 1

        coords_iter = coords_from_df(f, pos_columns, t_column)

        for i, _ids in link_iter(coords_iter, search_range, **kwargs):
            ids.extend(_ids)

            if progress_callback is not None:
                progress = int(i / n_iter * 100)
                progress_callback.emit(progress)

        f['particle'] = ids

        return f


    def detect_tracks(self, loc_data, progress_callback=None):

        segchannel = self.gui.tracking_segmentations.currentText()
        search_range = float(self.gui.trackpy_search_range.value())
        memory = int(self.gui.trackpy_memory.value())
        min_track_length = int(self.gui.min_track_length.value())

        track_data = []

        for dat in loc_data:
            try:
                dataset = dat["dataset"]
                channel = dat["channel"]
                pixel_size_nm = float(self.dataset_dict[dataset]["pixel_size"])
                exposure_time_ms = float(self.dataset_dict[dataset]["exposure_time"])
                locs = dat["localisations"]


                pixel_size_um = pixel_size_nm * 1e-3
                exposure_time_s = exposure_time_ms * 1e-3

                if segchannel in ["Segmentations", "Cells"]:
                    locs, segcol = self.detect_seglocs(dataset, channel, segchannel)
                    columns = list(locs.dtype.names)
                    locdf = pd.DataFrame(locs, columns=columns)
                    self.tracking_segchannel = segchannel
                    self.tracking_segcol = segcol
                else:
                    columns = list(locs.dtype.names)
                    locdf = pd.DataFrame(locs, columns=columns)
                    segcol ="segmentation_index"
                    locdf[segcol] = 0
                    self.tracking_segchannel = None
                    self.tracking_segcol = None

                n_segmentations = len(locdf[segcol].unique())
                tp.quiet()

                if n_segmentations == 1:

                    tracks_df = self.link_locs(locdf, search_range=search_range,
                        memory=memory, progress_callback=progress_callback)
                    tracks_df = tracks_df.sort_values(by=[segcol, "particle", "frame"])
                    tracks_df.drop(columns=[segcol], inplace=True)

                else:

                    seg_tracks = []
                    track_index = 1

                    for seg_index, group in locdf.groupby(segcol):

                        seg_track = self.link_locs(group.copy(), search_range=search_range,
                            memory=memory)

                        particle_array = seg_track["particle"].values
                        particle_array = particle_array + track_index
                        seg_track["particle"] = particle_array
                        track_index = np.max(particle_array) + 1

                        seg_tracks.append(seg_track.copy())

                        if progress_callback is not None:
                            progress = int(len(seg_tracks) / n_segmentations * 100)
                            progress_callback.emit(progress)

                    tracks_df = pd.concat(seg_tracks, ignore_index=True)
                    tracks_df = tracks_df.sort_values(by=[segcol,"particle", "frame"])

                tracks_df.reset_index(drop=True, inplace=True)
                tracks_df = tp.filter_stubs(tracks_df, min_track_length).reset_index(drop=True)

                self.tracks = tracks_df

                track_index = 1
                for particle, group in tracks_df.groupby("particle"):
                    tracks_df.loc[group.index, "particle"] = track_index
                    track_index += 1

                tracks_df = tracks_df.sort_values(by=["particle", "frame"])

                tracks_df["pixel_size (um)"] = pixel_size_um
                tracks_df["exposure_time (s)"] = exposure_time_s

                track_data.append(tracks_df)


            except:
                print(traceback.format_exc())

        if len(track_data) > 0:
            track_data = pd.concat(track_data, ignore_index=True)

        return track_data


    def track_particles(self, loc_data, stats = True, progress_callback=None):

        show_info("Detecting tracks...")

        track_data = self.detect_tracks(loc_data,
            progress_callback=progress_callback)

        return track_data




    def process_tracking_results(self, track_data):
        try:
            if track_data is None:
                return

            remove_unlinked = self.gui.remove_unlinked.isChecked()
            layers_names = [layer.name for layer in self.viewer.layers]
            total_tracks = 0

            for (dataset, channel), tracks in track_data.groupby(["dataset", "channel"]):

                tracks = tracks.to_records(index=False)

                if dataset not in self.tracking_dict.keys():
                    self.tracking_dict[dataset] = {}
                if channel not in self.tracking_dict[dataset].keys():
                    self.tracking_dict[dataset][channel] = {}

                self.tracking_dict[dataset][channel] = {"tracks": tracks}

                n_tracks = len(np.unique(tracks["particle"]))
                total_tracks += n_tracks

                if remove_unlinked:
                    loc_dict = self.localisation_dict[dataset][channel]

                    locs = loc_dict["localisations"]
                    n_locs = len(locs)
                    n_filtered = len(tracks)

                    n_removed = n_locs - n_filtered

                    if n_removed > 0:
                        show_info(f"Removed {n_removed} unlinked localisations")
                        loc_dict["localisations"] = tracks

            if len(track_data) > 0:
                self.gui.locs_export_data.clear()
                self.gui.locs_export_data.addItems(["Localisations", "Tracks"])
                self.gui.heatmap_data.clear()
                self.gui.heatmap_data.addItems(["Localisations", "Tracks"])

                show_info(f"Tracking complete, {total_tracks} tracks found")

        except:
            print(traceback.format_exc())

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

                worker = Worker(self.track_particles, loc_data)
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

            if hasattr(self, "tracking_dict"):
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
                    image_dict = self.dataset_dict[dataset_name]["images"]
                    n_frames = image_dict[channel_name].shape[0]

                    pixel_size = float(self.dataset_dict[dataset_name]["pixel_size"])
                    scale = [pixel_size, pixel_size]

                    remove_tracks = False

                    render_tracks = pd.DataFrame(tracks)

                    cmap_metric = self.gui.track_colour_metric.currentText()
                    cmap_metric = self.moltrack_metrics.get(cmap_metric)
                    cmap_name = self.gui.track_colour_colourmap.currentText()

                    if cmap_metric in render_tracks.columns:
                        metric_data = render_tracks[cmap_metric].values

                        if reset_cmap_range:
                            cmap_min = min(metric_data)
                            cmap_max = max(metric_data)

                            step = self.nice_step(cmap_min,cmap_max)
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

                        metric_norm_data = np.clip((metric_data - cmap_min) / (cmap_max - cmap_min), 0, 1)
                    else:
                        metric_norm_data = None

                    render_tracks = render_tracks[["particle", "frame", "y", "x"]]

                    if track_id in render_tracks["particle"].values:
                        render_tracks = render_tracks[render_tracks["particle"] == track_id]

                    render_tracks = render_tracks.to_records(index=False)
                    render_tracks = [list(track) for track in render_tracks]
                    render_tracks = np.array(render_tracks).copy()
                    render_tracks[:, 1] = 0

                    properties = {cmap_metric: metric_norm_data}

                    if "Tracks" not in layer_names:
                        self.track_layer = self.viewer.add_tracks(
                            render_tracks, name="Tracks",scale=scale,
                            colormap=cmap_name, properties=properties,
                            color_by="track_id",)

                        self.viewer.reset_view()
                        self.track_layer.mouse_double_click_callbacks.append(self.select_track)

                    else:

                        self.track_layer.color_by = "track_id"
                        self.track_layer.data = render_tracks
                        self.track_layer.scale = scale
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
                    self.viewer.scale_bar.unit = "nm"

            if remove_tracks:
                try:
                    if "Tracks" in layer_names:
                        self.track_layer.selected_data = []
                        self.viewer.layers.remove(self.track_layer)
                        self.track_layer = self.viewer.add_tracks(np.array([[0,0,0,0]]), name="Tracks")
                except:
                    print(traceback.format_exc())

            if hasattr(self, "track_layer"):
                self.track_layer.refresh()

        except:
            print(traceback.format_exc())
