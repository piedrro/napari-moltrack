import traceback
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint
from shapely.strtree import STRtree
import matplotlib.pyplot as plt

class _loc_filter_utils:

    def get_shapes(self, type="segmentations", flipxy=True, polygon=True):

        try:
            shape_data = []
            layer = None

            if type.lower() == "segmentations":
                if hasattr(self, "segLayer"):
                    layer = self.segLayer
            elif type.lower() == "cells":
                if hasattr(self, "cellLayer"):
                    layer = self.cellLayer

            if layer is None:
                return

            shapes = layer.data.copy()
            shape_type = layer.shape_type.copy()

            if len(shapes) == 0:
                return

            for shape_index, shape in enumerate(shapes):
                if shape_type[shape_index] == "polygon":
                    shape_data.append(shape)

            if flipxy:
                shape_data = [np.fliplr(poly) for poly in shape_data]

            if polygon:
                shape_data = [Polygon(poly) for poly in shape_data]

        except:
            print(traceback.format_exc())

        return shape_data





    def remove_seglocs(self, viewer=None):

        try:

            segchannel = self.gui.remove_seglocs_segmentation.currentText()
            dataset = self.gui.remove_seglocs_dataset.currentText()
            channel = self.gui.remove_seglocs_channel.currentText()

            polygons = self.get_shapes(segchannel, flipxy=True, polygon=True)

            if len(polygons) == 0:
                return

            total_locs = 0
            total_filtered = 0

            locs = self.get_locs(dataset, channel, return_dict=False)

            n_locs = len(locs)
            n_filtered = 0

            if n_locs == 0:
                return

            filtered_locs = []

            coords = np.stack([locs["x"], locs["y"]], axis=1)
            points = [Point(coord) for coord in coords]

            # for polygon in polygons:
            #     polygon_coords = np.array(polygon.exterior.coords).copy()
            #     plt.plot(polygon_coords[:,0], polygon_coords[:,1], color="red")
            # points_coords = coords.copy()
            # plt.scatter(points_coords[:,0], points_coords[:,1], color="blue", s=0.5)
            # plt.gca().invert_yaxis()
            # plt.show()

            # print("Done")

            spatial_index = STRtree(points)

            polygon_point_indices = []

            for polygon_index, polygon in enumerate(polygons):

                possible_points = spatial_index.query(polygon)

                for point_index in possible_points:

                    point = points[point_index]

                    if polygon.contains(point):

                        polygon_point_indices.append(point_index)

            if len(polygon_point_indices) > 0:

                polygon_locs = locs[polygon_point_indices].copy()

                seg_name = segchannel[:-1].lower() + "_index"

                polygon_locs = pd.DataFrame(polygon_locs)

                if "cell_index" in polygon_locs.columns:
                    polygon_locs = polygon_locs.drop(columns=["cell_index"])
                if "segmentation_index" in polygon_locs.columns:
                    polygon_locs = polygon_locs.drop(columns=["segmentation_index"])

                polygon_locs[seg_name] = polygon_index
                polygon_locs = polygon_locs.to_records(index=False)

                filtered_locs.append(polygon_locs)

            if len(filtered_locs) > 0:

                filtered_locs = np.hstack(filtered_locs).view(np.recarray).copy()
                filtered_locs = pd.DataFrame(filtered_locs)

                for (dataset_name, channel_name), flocs in filtered_locs.groupby(["dataset", "channel"]):

                    flocs = flocs.to_records(index=False)
                    loc_dict = self.localisation_dict[dataset_name][channel_name]
                    loc_dict["localisations"] = flocs
                    n_filtered += len(flocs)

            n_removed = n_locs - n_filtered
            print(f"Removed {n_removed} localisations.")

            self.draw_localisations()
            self.update_filter_criterion()
            self.update_criterion_ranges()

        except:
            print(traceback.format_exc())

        pass



    def get_locs(self, dataset, channel,
            return_dict = False, include_metadata=True):

        order = ["dataset", "channel", "group", "particle", "frame",
                 "cell_index", "segmentation_index",
                 "x", "y", "photons", "bg", "sx", "sy",
                 "lpx", "lpy", "ellipticity", "net_gradient",
                 "iterations",
                 "pixel_mean", "pixel_median", "pixel_sum",
                 "pixel_min", "pixel_max","pixel_std",
                 "pixel_mean_bg", "pixel_median_bg", "pixel_sum_bg",
                    "pixel_min_bg", "pixel_max_bg", "pixel_std_bg"]

        loc_data = []

        fitted = False
        box_size = int(self.gui.picasso_box_size.currentText())
        min_net_gradient = int(self.gui.picasso_min_net_gradient.text())

        try:

            if dataset == "All Datasets":
                dataset_list = list(self.localisation_dict.keys())
            else:
                dataset_list = [dataset]

            group = 0

            for dataset_name in dataset_list:

                if dataset_name not in self.localisation_dict.keys():
                    continue

                if channel == "All Channels":
                    channel_list = list(self.localisation_dict[dataset_name].keys())
                else:
                    channel_list = [channel]

                for channel_name in channel_list:

                    if channel_name not in self.localisation_dict[dataset_name].keys():
                        continue

                    loc_dict = self.localisation_dict[dataset_name][channel_name]

                    if "localisations" in loc_dict.keys():

                        locs = loc_dict["localisations"].copy()
                        n_locs = len(locs)

                        if n_locs > 0:

                            locs = pd.DataFrame(locs)

                            if include_metadata:
                                if "dataset" not in locs.columns:
                                    locs.insert(0, "dataset", dataset_name)
                                if "channel" not in locs.columns:
                                    locs.insert(1, "channel", channel_name)
                                if len(dataset_list) > 1:
                                    if "group" not in locs.columns:
                                        locs.insert(2, "group", group)
                                    else:
                                        locs["group"] = group
                            else:
                                if "dataset" in locs.columns:
                                    locs = locs.drop(columns=["dataset"])
                                if "channel" in locs.columns:
                                    locs = locs.drop(columns=["channel"])
                                if "cell_index" in locs.columns:
                                    locs = locs.drop(columns=["cell_index"])
                                if "segmentation_index" in locs.columns:
                                    locs = locs.drop(columns=["segmentation_index"])
                                    if len(dataset_list) > 1:
                                        if "group" not in locs.columns:
                                            locs.insert(0, "group", group)
                                        else:
                                            locs["group"] = group

                            mask = []

                            for col in order:
                                if col in locs.columns:
                                    mask.append(col)

                            locs = locs[mask]
                            locs = locs.to_records(index=False)

                            if return_dict == False:
                                loc_data.append(locs)
                            else:

                                image_dict = self.dataset_dict[dataset_name]["images"]
                                image_shape = list(image_dict[channel_name].shape)

                                if "fitted" in loc_dict.keys():
                                    fitted = loc_dict["fitted"]
                                if "box_size" in loc_dict.keys():
                                    box_size = loc_dict["box_size"]
                                if "min_net_gradient" in loc_dict.keys():
                                    min_net_gradient = loc_dict["min_net_gradient"]

                                loc_dict = {"dataset": dataset_name,
                                            "channel": channel_name,
                                            "localisations": locs,
                                            "image_shape": image_shape,
                                            "fitted": fitted,
                                            "box_size": box_size,
                                            "min_net_gradient": min_net_gradient,
                                            }
                                loc_data.append(loc_dict)

                            group += 1

        except:
            print(traceback.format_exc())

        if return_dict == False:
            if len(loc_data) == 0:
                loc_data = []
            elif len(loc_data) == 1:
                loc_data = loc_data[0]
            else:
                loc_data = np.hstack(loc_data).view(np.recarray).copy()

        return loc_data




    def pixseq_filter_localisations(self, viewer=None):

        try:

            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            criterion = self.gui.filter_criterion.currentText()
            min_value = self.gui.filter_min.value()
            max_value = self.gui.filter_max.value()
            subtract_background = self.gui.filter_subtract_bg.isChecked()

            n_removed = 0

            loc_data = self.get_locs(dataset, channel, return_dict=True)

            for dat in loc_data:

                dataset_name = dat["dataset"]
                channel_name = dat["channel"]
                locs = dat["localisations"]

                if len(locs) > 0:

                    columns = list(locs.dtype.names)

                    if criterion in columns:

                        self.gui.filter_localisations.setEnabled(False)

                        n_locs = len(locs)

                        criterion_data = locs[criterion]

                        if subtract_background:
                            bg_criterion = criterion + "_bg"
                            if bg_criterion in columns:
                                bg_values = locs[bg_criterion]
                                criterion_data = criterion_data - bg_values

                        keep_indices = np.where((criterion_data > min_value) &
                                                (criterion_data < max_value))[0]

                        locs = locs[keep_indices]

                        n_filtered = len(locs)

                        if n_filtered < n_locs:

                            n_removed = n_locs - n_filtered

                            loc_dict = self.localisation_dict[dataset_name][channel_name]
                            loc_dict["localisations"] = locs

                            self.draw_localisations(update_vis=True)

            self.update_criterion_ranges()
            print(f"Filtered {n_removed} localisations.")

            self.gui.filter_localisations.setEnabled(True)

        except:
            self.gui.filter_localisations.setEnabled(True)
            print(traceback.format_exc())

    def update_filter_criterion(self, viewer=None):


        try:

            columns = []

            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            selector = self.gui.filter_criterion

            locs = self.get_locs(dataset, channel)

            if len(locs) > 0:

                columns = list(locs.dtype.names)

                columns = [col for col in columns if col not in ["dataset","channel"]]
                columns = [col for col in columns if "_bg" not in col]

            selector.clear()

            if len(columns) > 0:
                selector.addItems(columns)

        except:
            print(traceback.format_exc())


    def update_criterion_ranges(self, viewer=None, plot=True):

        try:

            self.filter_graph_canvas.clear()

            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            criterion = self.gui.filter_criterion.currentText()
            subtract_background = self.gui.filter_subtract_bg.isChecked()

            if "pixel" not in criterion.lower():
                subtract_background = False
                self.gui.filter_subtract_bg.blockSignals(True)
                self.gui.filter_subtract_bg.setChecked(False)
                self.gui.filter_subtract_bg.setEnabled(False)
                self.gui.filter_subtract_bg.blockSignals(False)
            else:
                self.gui.filter_subtract_bg.setEnabled(True)

            locs = self.get_locs(dataset, channel)

            if len(locs) > 0:

                columns = list(locs.dtype.names)

                if criterion in columns:

                    values = locs[criterion]

                    if subtract_background:
                        bg_criterion = criterion + "_bg"
                        if bg_criterion in columns:
                            bg_values = locs[bg_criterion]
                            values = values - bg_values

                    values = values[~np.isnan(values)]

                    if values.dtype in [np.float32, np.float64,
                                        np.int32, np.int64,
                                        np.uint32, np.uint64]:

                        if plot:
                            self.plot_filter_graph(criterion, values)

                        min_value = np.min(values)
                        max_value = np.max(values)

                        self.gui.filter_min.setMinimum(min_value)
                        self.gui.filter_min.setMaximum(max_value)

                        self.gui.filter_max.setMinimum(min_value)
                        self.gui.filter_max.setMaximum(max_value)

                        self.gui.filter_min.setValue(min_value)
                        self.gui.filter_max.setValue(max_value)

        except:
            print(traceback.format_exc())

    def plot_filter_graph(self, criterion = "", values = None):

        try:
            self.filter_graph_canvas.clear()

            if values is not None:

                values = values[~np.isnan(values)]

                if len(values) > 0:
                    ax = self.filter_graph_canvas.addPlot()

                    # Create histogram
                    y, x = np.histogram(values, bins=100)

                    ax.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 75))
                    ax.setLabel('bottom', f"{criterion} values")
                    ax.setLabel('left', 'Frequency')

        except:
            print(traceback.format_exc())



