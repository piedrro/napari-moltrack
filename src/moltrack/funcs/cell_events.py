import numpy as np
import traceback
from moltrack.funcs.compute_utils import Worker
from moltrack.bactfit.fit import BactFit
from moltrack.bactfit.preprocess import data_to_cells
from moltrack.bactfit.fit import BactFit
from functools import partial
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import copy
import random
import string


class _cell_events:


    def moltrack_undo(self, viewer=None, event=None):

        try:

            if hasattr(self, "stored_cells"):

                if len(self.stored_cells) > 1:

                    self.stored_cells.pop(-1)

                    self.cellLayer.events.disconnect(self.update_cells)
                    self.cellLayer.refresh()

                    self.cellLayer.data = copy.deepcopy(self.stored_cells[-1])
                    self.cellLayer.refresh()

                    self.cellLayer.events.data.connect(self.update_cells)
                    self.cellLayer.refresh()

        except:
            print(traceback.format_exc())
            pass

    def initialise_cellLayer(self, shapes = None, shape_types = None, properties = None):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Cells" in layer_names:
            self.viewer.layers.remove("Cells")

        if hasattr(self, "cellLayer"):
            self.cellLayer.events.disconnect(self.update_cells)
            self.cellLayer.refresh()
            del self.cellLayer

        face_color = [1, 1, 1, 0]
        edge_color = [1, 1, 1, 0.5]
        edge_width = 1

        if shapes is not None:
            self.cellLayer = self.viewer.add_shapes(shapes, properties=properties,
                shape_type=shape_types, name="Cells", face_color=face_color,
                edge_color=edge_color, edge_width=edge_width)
        else:
            self.cellLayer = self.viewer.add_shapes(name="Cells", shape_type="polygon",
                face_color=face_color, edge_color=edge_color, edge_width=edge_width)

        self.move_polygons_to_front()

        self.cellLayer.mouse_drag_callbacks.append(self.celllayer_clicked)
        self.cellLayer.mouse_wheel_callbacks.append(self.dilate_cell)
        self.cellLayer.events.data.connect(self.update_cells)
        self.register_shape_layer_keybinds(self.segLayer)

        self.store_cell_shapes(init=True)

        return self.cellLayer


    def celllayer_clicked(self, viewer=None, event=None):

        try:

            if hasattr(self, "segmentation_mode"):

                if self.segmentation_mode == "delete":

                    coords = self.cellLayer.world_to_data(event.position)
                    shape_index = self.cellLayer.get_value(coords)[0]

                    if shape_index is not None:

                        name = self.cellLayer.properties["name"][shape_index]

                        cell = self.get_cell(name)

                        if cell is not None:

                            polygon_index = cell["polygon_index"]
                            midline_index = cell["midline_index"]

                            self.cellLayer.events.data.disconnect(self.update_cells)
                            self.cellLayer.refresh()

                            self.cellLayer.selected_data = [polygon_index, midline_index]
                            self.cellLayer.remove_selected()

                            self.cellLayer.events.data.connect(self.update_cells)
                            self.cellLayer.refresh()

                            self.store_cell_shapes()

        except:
            print(traceback.format_exc())
            pass







    def cellLayer_event_manager(self, mode = "connect"):

        if mode == "connect":
            self.cellLayer.mouse_wheel_callbacks.append(self.dilate_cell)
            self.cellLayer.events.data.connect(self.update_cells)
        else:
            for callback in self.cellLayer.mouse_wheel_callbacks:
                if callback == self.dilate_cell:
                    self.cellLayer.mouse_wheel_callbacks.remove(callback)


    def store_cell_shapes(self, max_stored = 10, init = False):

        try:

            if hasattr(self, "cellLayer"):
                if not hasattr(self, "stored_cells"):
                    self.stored_cells = []

                current_shapes = copy.deepcopy(self.cellLayer.data)

                if init:
                    self.stored_cells = [current_shapes]

                else:

                    if len(current_shapes) > 0:

                        if len(self.stored_cells) == 0:
                            self.stored_cells.append(current_shapes)
                        else:
                            previous_shapes = self.stored_cells[-1]

                            if not np.array_equal(previous_shapes, current_shapes):
                                self.stored_cells.append(current_shapes)

                if len(self.stored_cells) > max_stored:
                    self.stored_cells.pop(0)
        except:
            print(traceback.format_exc())
            pass

    def get_modified_shape_indices(self):

        if not hasattr(self, "stored_cells"):
            return []

        current_data = self.cellLayer.data.copy()
        previous_data = self.stored_cells[-1].copy()

        modified_shapes = []

        for idx, (prev_shape, curr_shape) in enumerate(zip(previous_data, current_data)):

            if not np.array_equal(prev_shape, curr_shape):
                modified_shapes.append(idx)

        return modified_shapes


    def get_cellLayer(self):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Cells" not in layer_names:

            self.cellLayer = self.initialise_cellLayer()

        return self.cellLayer

    def dilate_cell(self, viewer=None, event=None):

        try:
            if 'Control' in event.modifiers:

                coords = self.cellLayer.world_to_data(event.position)
                cell_selection = self.cellLayer.get_value(coords)[0]

                if cell_selection is not None:

                    cell_properties = self.cellLayer.properties.copy()
                    cell_shapes = self.cellLayer.data.copy()

                    cell_name = cell_properties["name"][cell_selection]

                    cell = self.get_cell(cell_name)

                    if cell is not None:

                        midline_coords = cell["midline_coords"]
                        width = cell["width"]

                        midline = LineString(midline_coords)

                        if event.delta[1] > 0:
                            buffer = 0.5
                        else:
                            buffer = -0.5

                        width += buffer

                        polygon = midline.buffer(width)

                        polygon_coords = np.array(polygon.exterior.coords)

                        polygon_coords = polygon_coords[:-1]

                        polygon_index = cell["polygon_index"]
                        midline_index = cell["midline_index"]

                        cell_shapes[polygon_index] = polygon_coords
                        cell_properties["width"][polygon_index] = width
                        cell_properties["width"][midline_index] = width

                        self.cellLayer.data = cell_shapes
                        self.cellLayer.refresh()

                        self.store_cell_shapes()

        except:
            print(traceback.format_exc())
            pass

    def get_cell(self, name):

        cell = None

        try:
                name_list = self.cellLayer.properties["name"].copy()
                width_list = self.cellLayer.properties["width"].copy()

                shape_types = self.cellLayer.shape_type.copy()
                shapes = self.cellLayer.data.copy()

                cell_indices = [i for i, n in enumerate(name_list) if n == name]

                if len(cell_indices) == 2:

                    path_index = [i for i in cell_indices if shape_types[i] == "path"]
                    polygon_index = [i for i in cell_indices if shape_types[i] == "polygon"]

                    if len(path_index) == 1 and len(polygon_index) == 1:

                        midline_coords = shapes[path_index[0]]
                        polygon_coords = shapes[polygon_index[0]]
                        width = width_list[polygon_index[0]]

                        cell = {"midline_coords": midline_coords,
                                "polygon_coords": polygon_coords,
                                "width": width,
                                "midline_index": path_index[0],
                                "polygon_index": polygon_index[0]}
        except:
            print(traceback.format_exc())
            pass

        return cell


    def get_cell_index(self, name, shape_type):

        try:

            name_list = self.cellLayer.properties["name"].copy()
            shape_types = self.cellLayer.shape_type.copy()

            cell_index = [i for i, n in enumerate(name_list) if n == name and shape_types[i] == shape_type]

            if len(cell_index) == 1:
                cell_index = cell_index[0]
            else:
                cell_index = None

        except:
            cell_index = None

        return cell_index


    def move_polygons_to_front(self):

        try:

            if hasattr(self, "cellLayer"):

                shape_types = self.cellLayer.shape_type.copy()

                polygon_indices = [i for i, s in enumerate(shape_types) if s == "polygon"]

                self.cellLayer.selected_data = polygon_indices
                self.cellLayer.move_to_front()
                self.cellLayer.selected_data = []
                self.cellLayer.refresh()

        except:
            pass

    def select_cell_midlines(self):

        try:

            if hasattr(self, "cellLayer"):

                shape_types = self.cellLayer.shape_type.copy()

                path_indices = [i for i, s in enumerate(shape_types) if s == "path"]

                self.cellLayer.selected_data = path_indices
                self.cellLayer.refresh()

        except:
            pass


    def update_cellLayer_shapes(self, shapes, shape_types = None, properties = None):

        try:

            self.cellLayer.events.data.disconnect(self.update_cells)
            self.cellLayer.refresh()

            self.cellLayer.data = shapes

            self.cellLayer.events.data.connect(self.update_cells)
            self.cellLayer.refresh()

        except:
            pass


    def update_cell_model(self, name):

        try:

            cell = self.get_cell(name)

            if cell is not None:

                midline_coords = cell["midline_coords"]
                polygon_coords = cell["polygon_coords"]
                width = cell["width"]
                midline_index = cell["midline_index"]
                polygon_index = cell["polygon_index"]

                bf = BactFit()
                polygon_fit_coords, midline_fit_coords = bf.manual_fit(polygon_coords, midline_coords, width)

                if polygon_fit_coords is not None:
                    shapes = copy.deepcopy(self.cellLayer.data)
                    shapes[polygon_index] = polygon_fit_coords
                    shapes[midline_index] = midline_fit_coords

                    self.update_cellLayer_shapes(shapes)
                    self.store_cell_shapes()

        except:
            pass





    def update_midline_position(self, name):

        try:

            cell = self.get_cell(name)

            if cell is not None:

                midline_coords = cell["midline_coords"]
                polygon_coords = cell["polygon_coords"]
                midline_index = cell["midline_index"]

                cell_polygon = Polygon(polygon_coords)
                cell_midline = LineString(midline_coords)

                polygon_origin = cell_polygon.centroid.coords[0]
                midline_origin = cell_midline.centroid.coords[0]

                x_shift = polygon_origin[0] - midline_origin[0]
                y_shift = polygon_origin[1] - midline_origin[1]

                midline_coords[:, 0] += x_shift
                midline_coords[:, 1] += y_shift

                shapes = copy.deepcopy(self.cellLayer.data)
                shapes[midline_index] = midline_coords

                self.update_cellLayer_shapes(shapes)

                self.store_cell_shapes()

        except:
            pass

    def add_manual_cell(self, last_index, width = 5):

        try:

            shapes = self.cellLayer.data.copy()
            properties = self.cellLayer.properties.copy()
            shape_types = self.cellLayer.shape_type.copy()

            midline_index = last_index

            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

            midline_coords = shapes[last_index]

            midline = LineString(midline_coords)
            polygon = midline.buffer(width)

            polygon_coords = np.array(polygon.exterior.coords)
            polygon_coords = polygon_coords[:-1]

            shapes.append(polygon_coords)
            shape_types.append("polygon")

            properties["name"][midline_index] = name
            properties["width"][midline_index] = width

            self.update_cellLayer_shapes(shapes, shape_types, properties)

            self.cellLayer.events.data.disconnect(self.update_cells)
            self.cellLayer.refresh()

            self.cellLayer.current_properties = {"name": name, "width": width}
            self.cellLayer.add_polygons(polygon_coords)

            self.cellLayer.events.data.connect(self.update_cells)
            self.cellLayer.refresh()

        except:
            print(traceback.format_exc())
            pass


    def update_cells(self, event):

        try:

            if event.action == "changed":

                # modified_indices = list(event.data_indices)

                modified_indices = self.get_modified_shape_indices()
                print(modified_indices)

                if len(modified_indices) == 1:

                    modified_index = modified_indices[0]

                    name_list = self.cellLayer.properties["name"].copy()
                    shape_types = self.cellLayer.shape_type.copy()
                    name = name_list[modified_index]
                    modified_shape_type = shape_types[modified_index]

                    if modified_shape_type == "path":

                        self.update_cell_model(name)

                    if modified_shape_type == "polygon":

                        self.update_midline_position(name)

            if event.action == "added":

                shapes = self.cellLayer.data.copy()
                last_index = len(shapes) - 1

                shape_types = self.cellLayer.shape_type.copy()
                shape_type = shape_types[last_index]

                if shape_type == "path":

                    self.add_manual_cell(last_index)

                self.store_cell_shapes()

        except:
            print(traceback.format_exc())
            pass
