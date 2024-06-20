import traceback

import numpy as np
import pandas as pd
import trackpy as tp
from numba import njit
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from napari.utils.notifications import show_info
from moltrack.funcs.compute_utils import Worker
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
import matplotlib.pyplot as plt

@njit
def calculate_msd_numba(x_disp, y_disp, max_lag):
    msd_list = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        squared_displacements = np.zeros(len(x_disp) - lag)
        for i in range(len(x_disp) - lag):
            dx = np.sum(x_disp[i:i+lag])
            dy = np.sum(y_disp[i:i+lag])
            squared_displacements[i] = dx**2 + dy**2
        msd_list[lag] = np.mean(squared_displacements)
    return msd_list


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
            pass

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
                track_data = np.hstack(track_data).view(np.recarray).copy()

        return track_data


    def link_locs(self, f, search_range, pos_columns=None,
            t_column='frame', progress_callback = None, **kwargs):

        from trackpy.linking.linking import (coords_from_df, pandas_sort,
            guess_pos_columns, link_iter)

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
                progress = int(i / n_iter * 50)
                progress_callback.emit(progress)

        f['particle'] = ids

        return f


    def detect_tracks(self, loc_data, progress_callback=None):

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

                columns = list(locs.dtype.names)

                locdf = pd.DataFrame(locs, columns=columns)

                tp.quiet()
                tracks_df = self.link_locs(locdf, search_range=search_range,
                    memory=memory, progress_callback=progress_callback)

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

    @staticmethod
    def get_track_stats(df):

        try:

            pixel_size = df["pixel_size (um)"].values[0]
            time_step = df["exposure_time (s)"].values[0]

            # Convert columns to numpy arrays for faster computation
            x = df['x'].values
            y = df['y'].values

            # Calculate the displacements using numpy diff
            x_disp = np.diff(x, prepend=x[0]) * pixel_size
            y_disp = np.diff(y, prepend=y[0]) * pixel_size

            # Calculate the squared displacements
            sq_disp = x_disp ** 2 + y_disp ** 2

            # Calculate the MSD using numba
            max_lag = len(x) - 1
            msd_list = calculate_msd_numba(x_disp, y_disp, max_lag)

            # Calculate speed
            speed = np.sqrt(x_disp ** 2 + y_disp ** 2) / time_step

            # Create time array
            time = np.arange(0, max_lag + 1) * time_step

            if len(time) >= 4 and len(msd_list) >= 4:  # Ensure there are enough points to fit
                slope, intercept = np.polyfit(time[:4], msd_list[:4], 1)
                apparent_diffusion = abs(slope / 4)  # the slope of MSD vs time gives 4D in 2D
            else:
                apparent_diffusion = 0

        except:
            apparent_diffusion = None
            msd_list = None
            time = None
            speed = None
            print(traceback.format_exc())

        # Append results to the dataframe
        df["time"] = time
        df["msd"] = msd_list
        df["D*"] = apparent_diffusion
        df["speed"] = speed

        return df




    def compute_tracking_statistics(self, executor, track_data, progress_callback=None):

        try:

            pass

            #split by dataset, channel and particle into list
            stats_jobs = [dat[1] for dat in track_data.groupby(["dataset", "channel", "particle"])]

            n_processed = 0

            futures = [executor.submit(_tracking_utils.get_track_stats, df) for df in stats_jobs]

            for future in as_completed(futures):
                n_processed += 1
                if progress_callback is not None:
                    progress = int(n_processed / len(stats_jobs) * 50) + 50
                    progress_callback.emit(progress)

            tracks_with_stats = [future.result() for future in futures if future.result() is not None]

        except:
            print(traceback.format_exc())
            pass

        if len(tracks_with_stats) > 0:
            track_data = pd.concat(tracks_with_stats, ignore_index=True)

        return track_data



    def track_particles(self, loc_data, stats = True, progress_callback=None):

        with ProcessPoolExecutor() as executor:

            show_info("Detecting tracks...")

            track_data = self.detect_tracks(loc_data,
                progress_callback=progress_callback)

            if stats:

                show_info("Calculating tracking statistics...")

                track_data = self.compute_tracking_statistics(executor,track_data,
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
                        print(f"Removed {n_removed} unlinked localisations")
                        loc_dict["localisations"] = tracks

            if len(track_data) > 0:
                self.gui.locs_export_data.clear()
                self.gui.locs_export_data.addItems(["Localisations", "Tracks"])
                self.gui.heatmap_data.clear()
                self.gui.heatmap_data.addItems(["Localisations", "Tracks"])

                print(f"Tracking complete, {total_tracks} tracks found")

        except:
            print(traceback.format_exc())

    def tracking_finished(self):
        self.draw_localisations()
        self.draw_tracks()
        self.plot_diffusion_graph()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()

        self.update_ui()

    def initialise_tracking(self):
        try:
            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()

            loc_data = self.get_locs(dataset, channel, return_dict=True, include_metadata=True)

            if len(loc_data) > 0:
                self.update_ui(init=True)

                worker = Worker(self.track_particles, loc_data)
                worker.signals.result.connect(self.process_tracking_results)
                worker.signals.finished.connect(self.tracking_finished)
                worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.tracking_progressbar))
                self.threadpool.start(worker)

            else:
                print(f"No localisations found for {dataset}`")

        except:
            print(traceback.format_exc())
            self.update_ui()

    def draw_tracks(self, dataset=None, channel=None):
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

                tracks = self.get_tracks(dataset_name, channel_name, return_dict=False, include_metadata=True, )

                if len(tracks) > 0:
                    image_dict = self.dataset_dict[dataset_name]["images"]
                    n_frames = image_dict[channel_name].shape[0]

                    pixel_size = float(self.dataset_dict[dataset_name]["pixel_size"])
                    scale = [pixel_size, pixel_size]

                    remove_tracks = False

                    render_tracks = pd.DataFrame(tracks)

                    if "speed" in render_tracks.columns:
                        speed = render_tracks["speed"].values.tolist()
                    else:
                        speed = None

                    render_tracks = render_tracks[["particle", "frame", "y", "x"]]
                    render_tracks = render_tracks.to_records(index=False)
                    render_tracks = [list(track) for track in render_tracks]
                    render_tracks = np.array(render_tracks).copy()
                    render_tracks[:, 1] = 0

                    properties = {"speed": speed}

                    if "Tracks" not in layer_names:
                        self.track_layer = self.viewer.add_tracks(render_tracks, name="Tracks",
                            scale=scale, colormap="plasma", properties=properties, color_by="track_id",)
                        self.viewer.reset_view()
                    else:

                        self.track_layer.color_by = "track_id"
                        self.track_layer.data = render_tracks
                        self.track_layer.scale = scale


                    if self.gui.show_tracks.isChecked() == False:
                        if self.track_layer in self.viewer.layers:
                            self.viewer.layers.remove(self.track_layer)

                    self.track_layer.properties = properties

                    if speed is not None:
                        self.track_layer.color_by = "speed"

                    self.track_layer.tail_length = n_frames * 2
                    self.track_layer.blending = "opaque"

                    self.track_layer.selected_data = []

                    self.viewer.scale_bar.visible = True
                    self.viewer.scale_bar.unit = "nm"

            if remove_tracks:
                if "Tracks" in layer_names:
                    self.viewer.layers["Tracks"].data = []

            for layer in layer_names:
                self.viewer.layers[layer].refresh()

        except:
            print(traceback.format_exc())
