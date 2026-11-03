import traceback
import warnings
from functools import partial

import numpy as np
import pandas as pd
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

        from moltrack.model_analysis import publish_analysis
        try:
            return publish_analysis(self.session, track_data)
        except Exception as exc:
            show_info(str(exc))
            return None


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
            from moltrack.model_analysis import prepare_analysis, compute_analysis
            if not len(self.get_tracks("All Datasets", "All Channels")):
                show_info("No tracks found")
                return
            shape_data = None
            if self.tracking_segchannel in ("Segmentations", "Cells"):
                shape_data = self.get_trackstats_polygons(self.tracking_segchannel, flipxy=True)
            prepared = prepare_analysis(self.session,
                fit_lags=self.gui.trackstats_adc_track_length.value(),
                rolling_window=self.gui.trackstats_roll_window_size.value(), shape_data=shape_data,
                cancel_event=self.stop_event)
            self.update_ui(init=True)
            worker = Worker(compute_analysis, prepared)
            worker.signals.result.connect(self.process_track_stats_result)
            worker.signals.finished.connect(self.compute_track_stats_finished)
            worker.signals.progress.connect(partial(self.moltrack_progress, progress_bar=self.gui.track_stats_progressbar))
            self.threadpool.start(worker)
        except Exception:
            print(traceback.format_exc())
            self.update_ui()
