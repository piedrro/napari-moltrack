import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import trackpy
from napari.utils.notifications import show_info
from shapely.geometry import LineString, Point, Polygon

from moltrack.funcs.compute_utils import Worker

warnings.filterwarnings("ignore", category=UserWarning)

class _trackstats_utils:

    @staticmethod
    def calculate_track_angles(coords):
        # Calculate the difference between each consecutive point
        vectors = np.diff(coords, axis=0)

        # Calculate the angles of these vectors relative to the x-axis
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Convert angles from radians to degrees
        angles = np.degrees(angles)

        # Initialize the relative angles array with zeros
        relative_angles = np.zeros(len(coords))

        # The first angle is zero
        relative_angles[0] = 0

        # Calculate the difference in angles between consecutive vectors
        for i in range(1, len(angles) + 1):
            relative_angles[i] = (angles[i - 1] - angles[i - 2] + 360) % 360 if i > 1 else angles[i - 1]

        return relative_angles

    @staticmethod
    def get_track_shape_stats(track_data, shape_data, pixel_size):

        shape_stats = {}

        try:
            if shape_data is None:
                return shape_stats

            polygon_index = None

            track_coords = np.array(track_data[["x", "y"]].values)
            track_line = LineString(track_coords)

            if "polygons" in shape_data.keys():
                polygons = shape_data["polygons"]

                polygon_index = [polygon_index for polygon_index, polygon in enumerate(polygons) if track_line.intersects(polygon)]

                if len(polygon_index) == 0:
                    return shape_stats

                polygon_index = polygon_index[0]
                polygon = polygons[polygon_index]
                polygon_centroid = polygon.centroid.coords[0]

                try:
                    centroid_distance = [Point(coord).distance(Point(polygon_centroid)) for coord in track_coords]

                    if len(centroid_distance) == len(track_coords):
                        centroid_distance = np.array(centroid_distance)
                        centroid_distance = centroid_distance * pixel_size
                        centroid_distance = centroid_distance.tolist()
                        shape_stats["centroid_distance"] = centroid_distance

                except:
                    pass

                try:
                    polygon_coords = np.array(polygon.exterior.coords)
                    polygon_outline = LineString(polygon_coords)

                    membrane_distance = [Point(coord).distance(polygon_outline) for coord in track_coords]

                    if len(membrane_distance) == len(track_coords):
                        membrane_distance = np.array(membrane_distance)
                        membrane_distance = membrane_distance * pixel_size
                        membrane_distance = membrane_distance.tolist()
                        shape_stats["membrane_distance"] = membrane_distance
                except:
                    pass

            if "midlines" in shape_data.keys():

                if polygon_index is not None:

                    midlines = shape_data["midlines"]
                    midline = midlines[polygon_index]

                    try:

                        midline_distance = [Point(coord).distance(midline) for coord in track_coords]

                        if len(midline_distance) == len(track_coords):
                            midline_distance = np.array(midline_distance)
                            midline_distance = midline_distance * pixel_size
                            midline_distance = midline_distance.tolist()
                            shape_stats["midline_distance"] = midline_distance

                    except:
                        pass

                    try:

                        midline_coords = np.array(midline.coords)

                        cell_poles = [Point(midline_coords[0]), Point(midline_coords[-1])]

                        cell_pole_distance = [min([Point(coord).distance(pole) for pole in cell_poles]) for coord in track_coords]

                        if len(cell_pole_distance) == len(track_coords):
                            cell_pole_distance = np.array(cell_pole_distance)
                            cell_pole_distance = cell_pole_distance * pixel_size
                            cell_pole_distance = cell_pole_distance.tolist()
                            shape_stats["cell_pole_distance"] = cell_pole_distance

                    except:
                        pass

        except:
            print(traceback.format_exc())

        return shape_stats


    @staticmethod
    def get_track_stats(df, min_track_length = 4,
                        rolling_window_size = 4,
                        shape_data=None):

        stats = {}

        try:

            pixel_size = df["pixel_size (um)"].values[0]
            time_step = df["exposure_time (s)"].values[0]

            # Convert columns to numpy arrays for faster computation
            x = df['x'].values
            y = df['y'].values

            if "sigma" in df.columns:
                sigma_data = df["sigma"].values[:].copy()
            else:
                sigma_data = None

            # Calculate the displacements using numpy diff
            x_disp = np.diff(x, prepend=x[0]) * pixel_size
            y_disp = np.diff(y, prepend=y[0]) * pixel_size

            stats["step_size"] = np.sqrt(x_disp ** 2 + y_disp ** 2)

            # Calculate the MSD using numba
            max_lag = len(x)
            fps = 1 / time_step
            msd_data = trackpy.motion.msd(df, pixel_size, fps, max_lag-1)
            msd_list = msd_data["msd"].tolist()[:max_lag]
            msd_list = [0] + msd_list
            stats["msd"] = msd_list

            # Calculate speed
            speed = np.sqrt(x_disp ** 2 + y_disp ** 2) / time_step
            stats["speed"] = speed

            # Create time array
            time = np.arange(0, max_lag) * time_step
            stats["time"] = time

            min_track_length += 1

            if len(time) >= min_track_length and len(msd_list) >= min_track_length:  # Ensure there are enough points to fit

                # skip the first point as it is zero
                Dtime = np.array(time[1:min_track_length])
                Dmsd = np.array(msd_list[1:min_track_length])

                slope, intercept = np.polyfit(Dtime,Dmsd, 1)
                D = slope / 4
                stats["D"] = D

                #Correct D* for localization error, if available

                if sigma_data is not None:
                    sigma_window = sigma_data[1:min_track_length]
                    mean_sigma = np.mean(sigma_window)

                    D_star = D - mean_sigma**2 /time_step

                    stats["D*"] = D_star
                else:
                    stats["D*"] = np.nan

                roll_D, roll_Dstar = _trackstats_utils.calculate_rolling_diffusion(time,msd_list,time_step,
                                                                                   sigma_data,
                                                                                   window_size=rolling_window_size)
                if len(roll_D) > 0:
                    stats["rD"] = roll_D
                else:
                    stats["rD"] = np.nan

                if len(roll_Dstar) > 0:
                    stats["rD*"] = roll_Dstar
                else:
                    stats["rD*"] = np.nan

            else:
                stats["D"] = np.nan
                stats["D*"] = np.nan

            try:
                track_angles = _trackstats_utils.calculate_track_angles(np.array([x, y]).T)

                if len(track_angles) == len(x):
                    stats["angle"] = track_angles
            except:
                pass

            shape_stats = _trackstats_utils.get_track_shape_stats(df, shape_data, pixel_size)

            if shape_stats != {}:
                stats = {**stats, **shape_stats}
            else:
                shape_stats_cols = ["membrane_distance", "centroid_distance",
                                    "midline_distance", "cell_pole_distance"]

                shape_stats_cols = [col for col in shape_stats_cols if col in df.columns]
                df.drop(columns=shape_stats_cols, inplace=True)

            # append shape stats to df
            for key, value in stats.items():
                df[key] = value

        except:
            print(traceback.format_exc())

        return df


    @staticmethod
    def calculate_rolling_diffusion(time_array, msd_array, time_step,
                                   sigma_array=None, window_size=4):

        # Ensure inputs are numpy arrays for easier slicing
        time_array = np.array(time_array)
        msd_array = np.array(msd_array)
        if sigma_array is not None:
            sigma_array = np.array(sigma_array)

        # Initialize lists to store results
        roll_D = []
        roll_Dstar = []

        # Calculate rolling D and D* over the specified window
        for i in range(len(time_array)):

            try:

                start = i - window_size//2
                end = i + window_size//2

                if start < 0:
                    start = 0
                if end > len(time_array):
                    end = len(time_array)

                window_time = time_array[start:end].copy()
                window_msd = msd_array[start:end].copy()

                slope, intercept = np.polyfit(window_time, window_msd, 1)
                D = slope / 4

                roll_D.append(D)

                if sigma_array is not None:

                    try:

                        sigma_window = sigma_array[start:end].copy()
                        mean_sigma = np.mean(sigma_window)

                        Dstar = D - mean_sigma ** 2 / time_step

                        roll_Dstar.append(Dstar)

                    except:
                        roll_Dstar.append(np.nan)

                else:
                    roll_Dstar.append(np.nan)

            except:
                roll_D.append(np.nan)
                roll_Dstar.append(np.nan)


        return roll_D, roll_Dstar

    def compute_track_stats(self, track_data, shape_data, progress_callback=None):

        try:

            min_track_length = self.gui.trackstats_adc_track_length.value()
            rolling_window_size = self.gui.trackstats_roll_window_size.value()

            tracks_with_stats = []

            if type(track_data) == np.recarray:
                track_data = pd.DataFrame(track_data)

            with ProcessPoolExecutor() as executor:

                #split by dataset, channel and particle into list
                stats_jobs = [dat[1] for dat in track_data.groupby(["dataset", "channel", "particle"])]

                n_processed = 0

                futures = [executor.submit(_trackstats_utils.get_track_stats, df,
                    min_track_length, rolling_window_size, shape_data) for df in stats_jobs]

                for future in as_completed(futures):
                    n_processed += 1
                    if progress_callback is not None:
                        progress = int(n_processed / len(stats_jobs) * 100)
                        progress_callback.emit(progress)

            tracks_with_stats = [future.result() for future in futures if future.result() is not None]

        except:
            print(traceback.format_exc())

        if len(tracks_with_stats) > 0:
            track_data = pd.concat(tracks_with_stats, ignore_index=True)

        return track_data


    def compute_track_stats_finished(self):

        self.update_diffusion_range()
        self.plot_diffusion_graph()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()

        self.update_traces_export_options()

        self.update_trackplot_options()
        self.plot_tracks()

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



    def get_trackstats_polygons(self, type, flipxy=False):

        shape_data = {}

        try:

            layer = None

            if type == "":
                return shape_data

            if type.lower() == "segmentations":
                if hasattr(self, "segLayer"):
                    layer = self.segLayer
            elif type.lower() == "cells":
                if hasattr(self, "cellLayer"):
                    layer = self.cellLayer

            if layer == None:
                return shape_data

            shapes = layer.data.copy()
            shape_type = layer.shape_type.copy()

            if len(shapes) == 0:
                return

            if type.lower() == "segmentations":
                shape_data["polygons"] = []

                for shape_index, shape in enumerate(shapes):
                    if shape_type[shape_index] == "polygon":

                        if flipxy:
                            shape = np.flip(shape, axis=1)

                        shape = Polygon(shape)
                        shape_data["polygons"].append(shape)

            else:
                name_list = self.cellLayer.properties["name"].copy()
                name_list = list(set(name_list))

                if len(name_list) == 0:
                    return

                shape_data["polygons"] = []
                shape_data["midlines"] = []

                for name in name_list:
                    try:
                        cell = self.get_cell(name)
                        if cell is not None:
                            midline = cell["midline_coords"]
                            polygon = cell["polygon_coords"]

                            if flipxy:
                                midline = np.flip(midline, axis=1)
                                polygon = np.flip(polygon, axis=1)

                            midline = LineString(midline)
                            polygon = Polygon(polygon)

                            shape_data["polygons"].append(polygon)
                            shape_data["midlines"].append(midline)

                    except:
                        pass
        except:
            print(traceback.format_exc())

        return shape_data



    def initialise_track_stats(self):

        try:

            if hasattr(self, "tracking_dict"):

                tracks = self.get_tracks("All Datasets", "All Channels")
                segchannel = self.tracking_segchannel
                segcol = self.tracking_segcol

                if self.tracking_segchannel in ["Segmentations", "Cells"]:
                    shape_data = self.get_trackstats_polygons(segchannel, flipxy=True)
                    shape_data["segcol"] = segcol
                else:
                    shape_data = None

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

                worker = Worker(self.compute_track_stats, tracks, shape_data)
                worker.signals.result.connect(self.process_track_stats_result)
                worker.signals.finished.connect(self.compute_track_stats_finished)
                worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.track_stats_progressbar))
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()


