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
from multiprocessing import Manager, shared_memory
from functools import partial
import warnings
from napari.utils.notifications import show_info

warnings.filterwarnings("ignore", category=UserWarning)

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
                progress = int(i / n_iter * 100)
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




    def compute_track_stats(self, track_data, progress_callback=None):

        try:

            tracks_with_stats = []

            if type(track_data) == np.recarray:
                track_data = pd.DataFrame(track_data)

            with ProcessPoolExecutor() as executor:

                #split by dataset, channel and particle into list
                stats_jobs = [dat[1] for dat in track_data.groupby(["dataset", "channel", "particle"])]

                n_processed = 0

                futures = [executor.submit(_tracking_utils.get_track_stats, df) for df in stats_jobs]

                for future in as_completed(futures):
                    n_processed += 1
                    if progress_callback is not None:
                        progress = int(n_processed / len(stats_jobs) * 100)
                        progress_callback.emit(progress)

            tracks_with_stats = [future.result() for future in futures if future.result() is not None]

        except:
            print(traceback.format_exc())
            pass

        if len(tracks_with_stats) > 0:
            track_data = pd.concat(tracks_with_stats, ignore_index=True)

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

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()
        self.update_traces_export_options()
        self.update_pixmap_options()

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
                show_info(f"No localisations found for {dataset}`")

        except:
            print(traceback.format_exc())
            self.update_ui()


    def compute_track_stats_finished(self):

        self.plot_diffusion_graph()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()
        self.update_traces_export_options()
        self.update_pixmap_options()

        self.update_ui()

    def process_track_stats_result(self, track_data):

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

        return tracks


    def initialise_track_stats(self):

        try:

            if hasattr(self, "tracking_dict"):

                tracks = self.get_tracks("All Datasets", "All Channels")

                if len(tracks) == 0:
                    show_info("No tracks found")
                    return

                self.update_ui(init=True)

                tracks = pd.DataFrame(tracks)
                tracks["pixel_size (um)"] = 0.0
                tracks["exposure_time (s)"] = 0.0

                for dataset, data in tracks.groupby("dataset"):
                    pixel_size_nm = float(self.dataset_dict[dataset]["pixel_size"])
                    exposure_time_ms = float(self.dataset_dict[dataset]["exposure_time"])
                    pixel_size_um = pixel_size_nm * 1e-3
                    exposure_time_s = exposure_time_ms * 1e-3
                    tracks.loc[data.index, "pixel_size (um)"] = pixel_size_um
                    tracks.loc[data.index, "exposure_time (s)"] = exposure_time_s

                tracks = tracks.to_records(index=False)

                worker = Worker(self.compute_track_stats, tracks)
                worker.signals.result.connect(self.process_track_stats_result)
                worker.signals.finished.connect(self.compute_track_stats_finished)
                worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.track_stats_progressbar))
                self.threadpool.start(worker)

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

                tracks = self.get_tracks(dataset_name, channel_name,
                    return_dict=False, include_metadata=True, )

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
                try:
                    if "Tracks" in layer_names:
                        self.track_layer.selected_data = []
                        self.viewer.layers.remove(self.track_layer)
                        self.track_layer = self.viewer.add_tracks(np.array([[0,0,0,0]]), name="Tracks")
                except:
                    print(traceback.format_exc())
                    pass

            if hasattr(self, "track_layer"):
                self.track_layer.refresh()

        except:
            print(traceback.format_exc())



    def process_pixmap_result(self, pixmap_data):

        if pixmap_data is None:
            return

        pixmap_data = pd.DataFrame(pixmap_data)

        for (dataset, channel), data in pixmap_data.groupby(["dataset", "channel"]):

            data = data.dropna(axis=1, how="all")

            if "index" in data.columns:
                data = data.drop("index", axis=1)

            data = data.to_records(index=False)

            if self.gui.pixmap_data.currentText() == "Localisations":
                self.localisation_dict[dataset][channel]["localisations"] = data
            else:
                self.tracking_dict[dataset][channel] = {"tracks": data}

    def compute_pixmap_finished(self):

        self.update_ui()

        self.draw_localisations()

        self.update_filter_criterion()
        self.update_criterion_ranges()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()
        self.update_traces_export_options()
        self.update_pixmap_options()


    @staticmethod
    def generate_localisation_mask(spot_size, spot_shape="square", buffer_size=0, bg_width=1):

        box_size = spot_size + (bg_width * 2) + (buffer_size * 2)

        # Create a grid of coordinates
        y, x = np.ogrid[:box_size, :box_size]

        # Adjust center based on box size
        if box_size % 2 == 0:
            center = (box_size / 2 - 0.5, box_size / 2 - 0.5)
        else:
            center = (box_size // 2, box_size // 2)

        if spot_shape.lower() == "circle":
            # Calculate distance from the center for circular mask
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Central spot mask
            inner_radius = spot_size // 2
            mask = distance <= inner_radius

            # Buffer mask
            buffer_outer_radius = inner_radius + buffer_size
            buffer_mask = (distance > inner_radius) & (distance <= buffer_outer_radius)

            # Background mask (outside the buffer zone)
            background_outer_radius = buffer_outer_radius + bg_width
            background_mask = (distance > buffer_outer_radius) & (distance <= background_outer_radius)

        elif spot_shape.lower() == "square":
            # Create square mask
            half_size = spot_size // 2
            mask = (abs(x - center[0]) <= half_size) & (abs(y - center[1]) <= half_size)

            # Create square background mask (one pixel larger on each side)
            buffer_mask = (abs(x - center[0]) <= half_size + buffer_size) & (abs(y - center[1]) <= half_size + buffer_size)
            background_mask = (abs(x - center[0]) <= half_size + buffer_size + bg_width) & (abs(y - center[1]) <= half_size + buffer_size + bg_width)
            background_mask = background_mask & ~buffer_mask

        return mask, buffer_mask, background_mask

    @staticmethod
    def generate_spot_bounds(locs, box_size):

        spot_bounds = []

        for loc_index, loc in enumerate(locs):

            x,y = loc.x, loc.y

            if box_size % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)+1
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)+1

            spot_bounds.append([x1,x2,y1,y2])

        return spot_bounds

    @staticmethod
    def crop_spot_data(image_shape, spot_bounds,
            spot_mask, background_mask=None):
        try:
            x1, x2, y1, y2 = spot_bounds
            crop = [0, spot_mask.shape[1], 0, spot_mask.shape[0]]

            if x1 < 0:
                crop[0] = abs(x1)
                x1 = 0
            if x2 > image_shape[1]:
                crop[1] = spot_mask.shape[1] - (x2 - image_shape[1])
                x2 = image_shape[1]
            if y1 < 0:
                crop[2] = abs(y1)
                y1 = 0
            if y2 > image_shape[0]:
                crop[3] = spot_mask.shape[0] - (y2 - image_shape[0])
                y2 = image_shape[0]

            corrected_bounds = [x1, x2, y1, y2]

            if spot_mask is not None:
                loc_mask = spot_mask.copy()
                loc_mask = loc_mask[crop[2]:crop[3], crop[0]:crop[1]]
            else:
                loc_mask = None

            if background_mask is not None:
                loc_bg_mask = background_mask.copy()
                loc_bg_mask = loc_bg_mask[crop[2]:crop[3], crop[0]:crop[1]]
            else:
                loc_bg_mask = None

        except:
            loc_mask = spot_mask
            loc_bg_mask = background_mask
            corrected_bounds = spot_bounds
            print(traceback.format_exc())

        return corrected_bounds, loc_mask, loc_bg_mask

    @staticmethod
    def generate_background_overlap_mask(locs, spot_mask,
            spot_background_mask, image_mask_shape):

        global_spot_mask = np.zeros(image_mask_shape, dtype=np.uint8)
        global_background_mask = np.zeros(image_mask_shape, dtype=np.uint8)

        spot_mask = spot_mask.astype(np.uint16)
        spot_background_mask = spot_background_mask.astype(np.uint16)

        spot_bounds = _tracking_utils.generate_spot_bounds(locs,  len(spot_mask[0]))

        for loc_index, bounds in enumerate(spot_bounds):

            [x1, x2, y1, y2], loc_mask, log_bg_mask = _tracking_utils.crop_spot_data(
                image_mask_shape, bounds, spot_mask,spot_background_mask)

            global_spot_mask[y1:y2, x1:x2] += loc_mask
            global_background_mask[y1:y2, x1:x2] += log_bg_mask

        global_spot_mask[global_spot_mask > 0] = 1
        global_background_mask[global_background_mask > 0] = 1

        intersection_mask = global_spot_mask & global_background_mask

        global_background_mask = global_background_mask - intersection_mask

        return global_background_mask, global_spot_mask


    @staticmethod
    def pixmap_compute_func(dat, progress_list=None):

        try:

            pixmap_data = dat["pixmap_data"]
            spot_size = dat["spot_size"]
            spot_shape = dat["spot_shape"]
            background_buffer = dat["background_buffer"]
            background_width = dat["background_width"]
            start_index = dat["start_index"]
            frame_shape = dat["shape"][-2:]

            n_pixels = spot_size ** 2

            if type(pixmap_data) == pd.DataFrame:
                pixmap_data = pixmap_data.to_records(index=False)

            # Access the shared memory
            shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            image_chunk = np_array.copy()

            spot_mask, buffer_mask, spot_background_mask = _tracking_utils.generate_localisation_mask(
                spot_size, spot_shape, background_buffer, background_width)

            pixmap_data = pd.DataFrame(pixmap_data)
            pixmap_data = pixmap_data.reset_index()
            pixmap_data["pixel_mean"] = np.nan
            pixmap_data["pixel_median"] = np.nan
            pixmap_data["pixel_sum"] = np.nan
            pixmap_data["pixel_max"] = np.nan
            pixmap_data["pixel_std"] = np.nan
            pixmap_data["pixel_mean_bg"] = np.nan
            pixmap_data["pixel_median_bg"] = np.nan
            pixmap_data["pixel_sum_bg"] = np.nan
            pixmap_data["pixel_max_bg"] = np.nan
            pixmap_data["pixel_std_bg"] = np.nan
            pixmap_data = pixmap_data.to_records(index=False)

            for array_index, frame in enumerate(image_chunk):

                frame_index = start_index + array_index

                frame_dat = pixmap_data[pixmap_data["frame"] == frame_index]

                if len(frame_dat) == 0:
                    continue

                for dat in frame_dat:

                    try:

                        bounds = _tracking_utils.generate_spot_bounds([dat], len(spot_mask[0]))[0]

                        [x1, x2, y1, y2], cropped_mask, cropped_background_mask = _tracking_utils.crop_spot_data(
                            frame_shape, bounds, spot_mask, spot_background_mask)

                        cropped_mask = np.logical_not(cropped_mask).astype(int)
                        cropped_background_mask = np.logical_not(cropped_background_mask).astype(int)

                        spot_values = frame[y1:y2, x1:x2].copy()
                        spot_background = frame[y1:y2, x1:x2].copy()

                        spot_values = np.ma.array(
                            spot_values,mask=cropped_mask)
                        spot_background = np.ma.array(
                            spot_background,mask=cropped_background_mask)

                        spot_mean = float(np.ma.mean(spot_values))
                        spot_median = float(np.ma.median(spot_values))
                        spot_sum = float(np.ma.sum(spot_values))
                        spot_max = float(np.ma.max(spot_values))
                        spot_std = float(np.ma.std(spot_values))

                        spot_mean_bg = float(np.ma.mean(spot_background))
                        spot_median_bg = float(np.ma.median(spot_background))
                        spot_sum_bg = spot_mean_bg * n_pixels
                        spot_max_bg = float(np.ma.max(spot_background))
                        spot_std_bg = float(np.ma.std(spot_background))

                        dat["pixel_mean"] = spot_mean
                        dat["pixel_median"] = spot_median
                        dat["pixel_sum"] = spot_sum
                        dat["pixel_max"] = spot_max
                        dat["pixel_std"] = spot_std
                        dat["pixel_mean_bg"] = spot_mean_bg
                        dat["pixel_median_bg"] = spot_median_bg
                        dat["pixel_sum_bg"] = spot_sum_bg
                        dat["pixel_max_bg"] = spot_max_bg
                        dat["pixel_std_bg"] = spot_std_bg

                        dat_index = dat["index"]
                        pixmap_data[dat_index] = dat

                    except:
                        pass

        except:
            print(traceback.format_exc())
            pass

        if progress_list is not None:
            progress_list.append(1)

        return pixmap_data


    def get_pixmap_compute_jobs(self, pixmap_data):

        pixmap_dataset = self.gui.pixmap_dataset.currentText()
        pixmap_channel = self.gui.pixmap_channel.currentText()
        spot_size = int(self.gui.pixmap_spot_size.currentText())
        spot_shape = self.gui.pixmap_spot_shape.currentText()
        background_buffer = int(self.gui.pixmap_background_buffer.currentText())
        background_width = int(self.gui.pixmap_background_width.currentText())

        pixmap_columns = ["dataset", "channel",
                          "group", "particle", "frame",
                          "cell_index", "segmentation_index",
                          "x", "y", ]

        compute_jobs = []

        try:

            for image_chunk in self.shared_chunks:

                dataset = image_chunk["dataset"]
                channel = image_chunk["channel"]

                frame_start = image_chunk["start_index"]
                frame_end = image_chunk["end_index"]

                pixmap_data_chunks = pixmap_data.copy()

                if dataset == pixmap_dataset and channel == pixmap_channel:
                    pass
                else:

                    pixmap_data_chunks["dataset"] = dataset
                    pixmap_data_chunks["channel"] = channel

                    for col in pixmap_data_chunks.columns:
                        if col not in pixmap_columns:
                            pixmap_data_chunks.drop(col, axis=1, inplace=True)

                pixmap_data_chunks = pixmap_data_chunks.to_records(index=False)

                pixmap_data_chunks = pixmap_data_chunks[(pixmap_data_chunks["frame"] >= frame_start) &
                                            (pixmap_data_chunks["frame"] <= frame_end)]

                if len(pixmap_data_chunks) > 0:
                    job = {"dataset": dataset,
                           "channel": channel,
                           "start_index": image_chunk["start_index"],
                           "end_index": image_chunk["end_index"],
                           "pixmap_data": pixmap_data_chunks,
                           "spot_size": spot_size,
                           "spot_shape": spot_shape,
                           "background_buffer": background_buffer,
                           "background_width": background_width,
                           "shared_memory_name": image_chunk["shared_memory_name"],
                           "shape": image_chunk["shape"],
                           "dtype": image_chunk["dtype"],}

                    compute_jobs.append(job)

        except:
            pass

        return compute_jobs

    def compute_pixmap(self, pixmap_data, progress_callback=None):

        try:
            results = None

            dataset_list = pixmap_data["dataset"].unique()
            channel_list = list(self.dataset_dict[dataset_list[0]]["images"].keys())

            self.create_shared_image_chunks(dataset_list=dataset_list,
                channel_list=channel_list, chunk_size=100)

            compute_jobs = self.get_pixmap_compute_jobs(pixmap_data)

            if len(compute_jobs) == 0:
                return None

            with Manager() as manager:

                progress_list = manager.list()

                with ProcessPoolExecutor() as executor:

                    futures = [executor.submit(_tracking_utils.pixmap_compute_func,
                        job, progress_list) for job in compute_jobs]

                    for future in as_completed(futures):
                        if progress_callback is not None:
                            progress = int(len(progress_list) / len(compute_jobs) * 100)
                            progress_callback.emit(progress)

                results = [future.result() for future in futures if future.result() is not None]

                if len(results) > 0:

                    results = [pd.DataFrame(result) for result in results]
                    results = pd.concat(results, ignore_index=True)
                    results = results.to_records(index=False)

            self.restore_shared_image_chunks()

        except:
            print(traceback.format_exc())
            self.restore_shared_image_chunks()
            self.update_ui()
            pass

        return results

    def initialise_pixmap(self):

        try:

            if hasattr(self, "tracking_dict"):

                data = self.gui.pixmap_data.currentText()
                dataset = self.gui.pixmap_dataset.currentText()
                channel = self.gui.pixmap_channel.currentText()

                pixmap_data = []

                if data == "Tracks":
                    pixmap_data = self.get_tracks(dataset, channel,
                        return_dict=False, include_metadata=True)
                if data == "Localisations":
                    pixmap_data = self.get_locs(dataset, channel,
                        return_dict=False, include_metadata=True)

                if len(pixmap_data) == 0:
                    return

                self.update_ui(init=True)

                pixmap_data = pd.DataFrame(pixmap_data)

                worker = Worker(self.compute_pixmap, pixmap_data)
                worker.signals.result.connect(self.process_pixmap_result)
                worker.signals.finished.connect(self.compute_pixmap_finished)
                worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.compute_pixmap_progressbar))
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()


    def update_pixmap_combos(self):

        try:

            pixmap_data = []

            if self.gui.pixmap_data.currentText() == "Localisations":
                if hasattr(self, "localisation_dict"):
                    pixmap_data = self.get_locs("All Datasets", "All Channels")
            if self.gui.pixmap_data.currentText() == "Tracks":
                if hasattr(self, "tracking_dict"):
                    pixmap_data = self.get_tracks("All Datasets", "All Channels")

            if len(pixmap_data) > 0:

                pixmap_data = pd.DataFrame(pixmap_data)

                datasets = pixmap_data["dataset"].unique().tolist()
                channels = pixmap_data["channel"].unique().tolist()

                self.gui.pixmap_dataset.blockSignals(True)
                self.gui.pixmap_dataset.clear()
                self.gui.pixmap_dataset.addItems(datasets)
                self.gui.pixmap_dataset.blockSignals(False)

                self.gui.pixmap_channel.blockSignals(True)
                self.gui.pixmap_channel.clear()
                self.gui.pixmap_channel.addItems(channels)
                self.gui.pixmap_channel.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass


    def update_pixmap_options(self):

        try:

            pixmap_data = []

            if hasattr(self, "tracking_dict"):
                tracks = self.get_tracks("All Datasets", "All Channels")
                if len(tracks) > 0:
                    pixmap_data.append("Tracks")
            if hasattr(self, "localisation_dict"):
                locs = self.get_locs("All Datasets", "All Channels")
                if len(locs) > 0:
                    pixmap_data.append("Localisations")

            self.gui.pixmap_data.blockSignals(True)
            self.gui.pixmap_data.clear()
            self.gui.pixmap_data.addItems(pixmap_data)
            self.gui.pixmap_data.blockSignals(False)

            self.update_pixmap_combos()

        except:
            print(traceback.format_exc())
            pass