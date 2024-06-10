import traceback
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint
from shapely.strtree import STRtree

class _loc_filter_utils:


    def remove_seglocs(self, viewer=None):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            segmentations = self.gui.remove_seglocs_segmentation.currentText()
            dataset = self.gui.remove_seglocs_dataset.currentText()
            loc_datasets = list(self.localisation_dict.keys())

            if segmentations not in layer_names:
                print(f"Segmentation {segmentations} not found in viewer.")
                return
            else:
                polygons = self.viewer.layers[segmentations].data.copy()

                if len(polygons) == 0:
                    print(f"No polygons found in segmentation {segmentations}.")
                    return

            if dataset not in loc_datasets:
                print(f"Dataset {dataset} not found in localisation dict.")
                return

            loc_dict = self.localisation_dict[dataset]

            locs = loc_dict["localisations"].copy()
            n_locs = len(locs)

            if n_locs == 0:
                print(f"No localisations found in dataset {dataset}.")
                return

            filtered_locs = []

            polygons = [Polygon(polygon) for polygon in polygons]
            coords = np.stack([locs["x"], locs["y"]], axis=1)
            points = [Point(coord) for coord in coords]

            spatial_index = STRtree(points)

            for polygon_index, polygon in enumerate(polygons):

                possible_points = spatial_index.query(polygon)

                polygon_point_indices = []

                for point_index in possible_points:

                    point = points[point_index]

                    if polygon.contains(point):

                        polygon_point_indices.append(point_index)

                if len(polygon_point_indices) > 0:

                    polygon_locs = locs[polygon_point_indices]

                    seg_name = segmentations[:-1].lower() + "_index"

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

                loc_dict["localisations"] = filtered_locs
                self.localisation_dict[dataset] = loc_dict

                n_filtered = len(filtered_locs)
                n_removed = n_locs - n_filtered

                print(f"Removed {n_removed} localisations.")

                self.draw_localisations()

                self.update_filter_criterion()
                self.update_criterion_ranges()

        except:
            print(traceback.format_exc())

        pass


    def pixseq_filter_localisations(self, viewer=None):

        try:

            dataset = self.gui.picasso_filter_dataset.currentText()
            criterion = self.gui.filter_criterion.currentText()
            min_value = self.gui.filter_min.value()
            max_value = self.gui.filter_max.value()

            if dataset in self.localisation_dict.keys():

                loc_dict = self.localisation_dict[dataset]

                if "localisations" in loc_dict.keys():

                    locs = loc_dict["localisations"].copy()

                    if len(locs) > 0:

                        columns = list(locs.dtype.names)

                        if criterion in columns:

                            self.gui.filter_localisations.setEnabled(False)

                            n_locs = len(locs)

                            locs = locs[locs[criterion] > min_value]
                            locs = locs[locs[criterion] < max_value]

                            n_filtered = len(locs)

                            if n_filtered < n_locs:

                                n_removed = n_locs - n_filtered

                                loc_dict["localisations"] = locs

                                self.localisation_dict[dataset] = loc_dict
                                self.draw_localisations(update_vis=True)

            self.update_criterion_ranges()
            print(f"Filtered {n_removed} localisations.")

            self.gui.filter_localisations.setEnabled(True)

        except:
            self.gui.filter_localisations.setEnabled(True)
            print(traceback.format_exc())


    def update_filter_dataset(self, viewer=None):

        if self.gui.picasso_filter_type.currentText() == "Localisations":
            self.gui.picasso_filter_dataset.setEnabled(True)
            self.gui.picasso_filter_dataset.show()
            self.gui.picasso_filter_dataset_label.show()
        else:
            self.gui.picasso_filter_dataset.setEnabled(False)
            self.gui.picasso_filter_dataset.hide()
            self.gui.picasso_filter_dataset_label.hide()

        self.update_filter_criterion()
        self.update_criterion_ranges()

    def update_filter_criterion(self, viewer=None):


        try:

            columns = []

            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            selector = self.gui.filter_criterion

            if dataset in self.localisation_dict.keys():
                if channel in self.localisation_dict[dataset]:

                    loc_dict = self.localisation_dict[dataset][channel]

                    if "localisations" in loc_dict.keys():

                        locs = loc_dict["localisations"].copy()

                        if len(locs) > 0:

                            columns = list(locs.dtype.names)

                            columns = [col for col in columns if col not in ["dataset","channel"]]

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

            if dataset in self.localisation_dict.keys():
                if channel in self.localisation_dict[dataset]:

                    loc_dict = self.localisation_dict[dataset][channel]

                    if "localisations" in loc_dict.keys():

                        locs = loc_dict["localisations"].copy()

                        if len(locs) > 0:

                            columns = list(locs.dtype.names)

                            if criterion in columns:

                                values = locs[criterion]

                                if values.dtype in [np.float32, np.float64, np.int32, np.int64]:

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



