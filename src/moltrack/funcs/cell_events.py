import copy
import random
import string
import traceback

import numpy as np
from bactfit.cell import Cell
from bactfit.utils import manual_fit
from shapely.geometry import LineString, Point, Polygon


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

    def initialise_cellLayer(self, shapes=None,
            shape_types=None, properties=None):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Cells" in layer_names:
            self.viewer.layers.remove("Cells")

        if hasattr(self, "cellLayer"):
            self.cellLayer.events.disconnect(self.update_cells)
            self.cellLayer.refresh()
            del self.cellLayer

        if hasattr(self, "segmentation_image_pixel_size"):
            pixel_size = self.segmentation_image_pixel_size
            scale = [pixel_size, pixel_size]
        elif self.active_dataset is not None:
            pixel_size = self.dataset_dict[self.active_dataset]["pixel_size"]
            scale = [pixel_size, pixel_size]
        else:
            scale = [1, 1]

        face_color = [1, 1, 1, 0]
        edge_color = [1, 1, 1, 0.5]
        edge_width = 1

        if shapes is not None:
            self.cellLayer = self.viewer.add_shapes(
                shapes,
                properties=properties,
                shape_type=shape_types,
                name="Cells",
                face_color=face_color,
                edge_color=edge_color,
                edge_width=edge_width,
            )
        else:
            self.cellLayer = self.viewer.add_shapes(
                name="Cells",
                shape_type="polygon",
                face_color=face_color,
                edge_color=edge_color,
                edge_width=edge_width,
            )

        if self.gui.show_shapes.isChecked() == False:
            if self.cellLayer in self.viewer.layers:
                self.viewer.layers.remove(self.cellLayer)

        self.move_polygons_to_front()

        self.viewer.scale_bar.visible = True
        self.cellLayer.scale = scale
        self.viewer.scale_bar.unit = "nm"

        self.cellLayer.refresh()
        self.viewer.reset_view()

        self.cellLayer.mouse_drag_callbacks.append(self.celllayer_clicked)
        self.cellLayer.mouse_wheel_callbacks.append(self.dilate_cell)
        self.cellLayer.events.data.connect(self.update_cells)
        self.register_shape_layer_keybinds(self.cellLayer)

        self.store_cell_shapes(init=True)

        self.update_segmentation_combos()

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

                            self.remove_cells([polygon_index, midline_index])

                            self.store_cell_shapes()

                    self.segmentation_mode = "panzoom"
                    self.cellLayer.mode = "pan_zoom"

        except:
            print(traceback.format_exc())

    def cellLayer_event_manager(self, mode="connect"):
        if mode == "connect":
            self.cellLayer.mouse_wheel_callbacks.append(self.dilate_cell)
            self.cellLayer.events.data.connect(self.update_cells)
        else:
            for callback in self.cellLayer.mouse_wheel_callbacks:
                if callback == self.dilate_cell:
                    self.cellLayer.mouse_wheel_callbacks.remove(callback)

    def store_cell_shapes(self, max_stored=10, init=False):
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

                            if not np.array_equal(
                                previous_shapes, current_shapes
                            ):
                                self.stored_cells.append(current_shapes)

                if len(self.stored_cells) > max_stored:
                    self.stored_cells.pop(0)
        except:
            print(traceback.format_exc())

    def get_modified_shape_indices(self):
        if not hasattr(self, "stored_cells"):
            return []

        current_data = self.cellLayer.data.copy()
        previous_data = self.stored_cells[-1].copy()

        modified_shapes = []

        for idx, (prev_shape, curr_shape) in enumerate(
            zip(previous_data, current_data)
        ):
            if not np.array_equal(prev_shape, curr_shape):
                modified_shapes.append(idx)

        return modified_shapes

    def get_cellLayer(self):
        layer_names = [layer.name for layer in self.viewer.layers]

        if "Cells" not in layer_names:
            self.cellLayer = self.initialise_cellLayer()

        self.update_segmentation_combos()

        return self.cellLayer

    def dilate_cell(self, viewer=None, event=None):

        try:

            if "Control" in event.modifiers:

                coords = self.cellLayer.world_to_data(event.position)
                cell_selection = self.cellLayer.get_value(coords)[0]

                if cell_selection is not None:
                    properties = copy.deepcopy(self.cellLayer.properties)
                    cell_shapes = copy.deepcopy(self.cellLayer.data)

                    cell_name = properties["name"][cell_selection]

                    cell = self.get_cell(cell_name)

                    if cell is not None:

                        self.cellLayer.events.data.disconnect(
                            self.update_cells
                        )

                        midline_coords = cell["midline_coords"]
                        radius = cell["radius"]

                        midline = LineString(midline_coords)

                        if event.delta[1] > 0:
                            buffer = 0.5
                        else:
                            buffer = -0.5

                        radius += buffer

                        polygon = midline.buffer(radius)

                        polygon_coords = np.array(polygon.exterior.coords)
                        polygon_coords = polygon_coords[:-1]

                        polygon_index = cell["polygon_index"]
                        midline_index = cell["midline_index"]

                        cell_shapes[polygon_index] = polygon_coords
                        properties["cell"][polygon_index]["radius"] = radius
                        properties["cell"][midline_index]["radius"] = radius

                        self.update_cellLayer_shapes(
                            cell_shapes, properties=properties
                        )

                        self.store_cell_shapes()

                        self.cellLayer.events.data.connect(self.update_cells)

        except:
            print(traceback.format_exc())

    def get_cell(self, name, json=False, bactfit=False):
        cell = None

        try:

            name_list = self.cellLayer.properties["name"].copy()
            cell_list = self.cellLayer.properties["cell"].copy()

            cell_list = [cell for cell in cell_list if isinstance(cell, dict)]

            shape_types = self.cellLayer.shape_type.copy()
            shapes = self.cellLayer.data.copy()
            pixel_size = self.cellLayer.scale[0]

            cell_indices = [i for i, n in enumerate(name_list) if n == name]

            if len(cell_indices) == 2:
                path_index = [
                    i for i in cell_indices if shape_types[i] == "path"
                ]
                polygon_index = [
                    i for i in cell_indices if shape_types[i] == "polygon"
                ]

                if len(path_index) == 1 and len(polygon_index) == 1:

                    midline_coords = shapes[path_index[0]]
                    polygon_coords = shapes[polygon_index[0]]
                    cell_properties = cell_list[path_index[0]]

                    if bactfit:
                        midline_coords = np.fliplr(midline_coords)
                        polygon_coords = np.fliplr(polygon_coords)

                    if json is True:
                        midline_coords = midline_coords.tolist()
                        polygon_coords = polygon_coords.tolist()

                        cell_properties["poly_params"] = list(
                            cell_properties["poly_params"]
                        )
                        cell_properties["cell_poles"] = [
                            list(pole)
                            for pole in cell_properties["cell_poles"]
                        ]
                        cell_properties["radius"] = float(
                            cell_properties["radius"]
                        )

                    cell_coords = {
                        "midline_coords": midline_coords,
                        "polygon_coords": polygon_coords,
                        "midline_index": int(path_index[0]),
                        "polygon_index": int(polygon_index[0]),
                        "pixel_size": pixel_size,
                    }

                    cell = {**cell_coords, **cell_properties}

                    if bactfit:
                        cell = Cell(cell)

        except:
            print(traceback.format_exc())

        return cell

    def get_cell_index(self, name, shape_type):
        try:
            name_list = self.cellLayer.properties["name"].copy()
            shape_types = self.cellLayer.shape_type.copy()

            cell_index = [
                i
                for i, n in enumerate(name_list)
                if n == name and shape_types[i] == shape_type
            ]

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

                polygon_indices = [
                    i for i, s in enumerate(shape_types) if s == "polygon"
                ]

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

                path_indices = [
                    i for i, s in enumerate(shape_types) if s == "path"
                ]

                self.cellLayer.selected_data = path_indices
                self.cellLayer.refresh()

        except:
            pass

    def update_cellLayer_shapes(
        self, shapes, shape_types=None, properties=None
    ):
        try:
            self.cellLayer.events.data.disconnect(self.update_cells)
            self.cellLayer.refresh()

            self.cellLayer.data = shapes

            if properties is not None:

                self.cellLayer.properties = properties

            if shape_types is not None:
                self.cellLayer.shape_type = shape_types

            self.cellLayer.events.data.connect(self.update_cells)
            self.cellLayer.refresh()

        except:
            print(traceback.format_exc())

    def update_cell_model(self, name):
        try:
            cell = self.get_cell(name, json=False)

            if cell is not None:
                midline_coords = cell["midline_coords"]
                polygon_coords = cell["polygon_coords"]
                radius = cell["radius"]
                midline_index = cell["midline_index"]
                polygon_index = cell["polygon_index"]

                fit = manual_fit(polygon_coords, midline_coords, radius)
                (
                    polygon_fit_coords,
                    midline_fit_coords,
                    poly_params,
                    cell_radius,
                ) = fit

                if polygon_fit_coords is not None:

                    shapes = copy.deepcopy(self.cellLayer.data)
                    shape_types = copy.deepcopy(self.cellLayer.shape_type)
                    properties = copy.deepcopy(self.cellLayer.properties)

                    shapes[polygon_index] = polygon_fit_coords
                    shapes[midline_index] = midline_fit_coords

                    cell_poles = [
                        midline_fit_coords[0],
                        midline_fit_coords[-1],
                    ]

                    properties["cell"][polygon_index][
                        "poly_params"
                    ] = poly_params
                    properties["cell"][polygon_index]["radius"] = cell_radius
                    properties["cell"][polygon_index][
                        "cell_poles"
                    ] = cell_poles
                    properties["cell"][midline_index][
                        "poly_params"
                    ] = poly_params
                    properties["cell"][midline_index]["radius"] = cell_radius
                    properties["cell"][midline_index][
                        "cell_poles"
                    ] = cell_poles

                    self.update_cellLayer_shapes(
                        shapes, shape_types=shape_types, properties=properties
                    )

        except:
            print(traceback.format_exc())

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

                cell_poles = [midline_coords[0], midline_coords[-1]]

                shapes = copy.deepcopy(self.cellLayer.data)
                properties = copy.deepcopy(self.cellLayer.properties)

                shapes[midline_index] = midline_coords
                properties["cell"][midline_index]["cell_poles"] = cell_poles

                self.update_cellLayer_shapes(shapes, properties=properties)

        except:
            pass

    def find_centerline(self, midline, radius):
        try:

            def resample_line(line, num_points):
                distances = np.linspace(0, line.length, num_points)
                points = [line.interpolate(distance) for distance in distances]
                return LineString(points)

            def extract_end_points(line, num_points=2):
                coords = list(line.coords)
                if len(coords) < num_points:
                    raise ValueError(
                        "The LineString does not have enough points."
                    )
                start_points = coords[:num_points]
                end_points = coords[-num_points:]
                return start_points, end_points

            def extend_away(
                points,
                distance,
            ):
                if len(points) < 2:
                    raise ValueError(
                        "At least two points are required to determine the direction for extension."
                    )

                p1 = Point(points[0])
                p2 = Point(points[1])

                dx = p2.x - p1.x
                dy = p2.y - p1.y
                length = np.hypot(dx, dy)
                factor = distance / length

                # Extend p1 away from p2
                extended_x1 = p1.x - factor * dx
                extended_y1 = p1.y - factor * dy

                # Similarly for the other end
                p3 = Point(points[-1])
                p4 = Point(points[-2])

                dx_end = p4.x - p3.x
                dy_end = p4.y - p3.y
                length_end = np.hypot(dx_end, dy_end)
                factor_end = distance / length_end

                # Extend p3 away from p4
                extended_x2 = p3.x - factor_end * dx_end
                extended_y2 = p3.y - factor_end * dy_end

                return (extended_x1, extended_y1), (extended_x2, extended_y2)

            def concatenate_lines(start_line, centerline, end_line):
                coords = (
                    list(start_line.coords)
                    + list(centerline.coords)
                    + list(end_line.coords)
                )
                return LineString(coords)

            def cut_line_at_intersection(line, intersection):
                if intersection.is_empty:
                    return line
                elif isinstance(intersection, Point):
                    return LineString(
                        [
                            pt
                            for pt in line.coords
                            if Point(pt).distance(intersection) >= 0
                        ]
                    )
                elif isinstance(intersection, LineString):
                    intersection_coords = list(intersection.coords)
                    cropped_coords = [
                        pt
                        for pt in line.coords
                        if Point(pt).distance(Point(intersection_coords[0]))
                        >= 0
                        and Point(pt).distance(Point(intersection_coords[-1]))
                        >= 0
                    ]
                    return LineString(cropped_coords)
                return line

            model = midline.buffer(radius)

            centerline = resample_line(
                midline, 1000
            )  # High resolution with 1000 points

            start_points, end_points = extract_end_points(centerline)

            extension_distance = radius * 3

            extended_start = extend_away(start_points, extension_distance)
            extended_end = extend_away(end_points, extension_distance)

            extended_start_line = LineString(
                [start_points[0], extended_start[0]]
            )
            extended_end_line = LineString([end_points[-1], extended_end[1]])

            outline = LineString(model.exterior.coords)
            intersections_start = outline.intersection(
                extended_start_line
            ).coords[0]
            intersections_end = outline.intersection(extended_end_line).coords[
                0
            ]

            centerline_coords = np.array(centerline.coords)
            centerline_coords = np.insert(
                centerline_coords, 0, intersections_start, axis=0
            )
            centerline_coords = np.append(
                centerline_coords, [intersections_end], axis=0
            )
            centerline = LineString(centerline_coords)

        except:
            print(traceback.format_exc())

        return centerline

    def find_end_cap_centroid(self, midline, radius):
        try:

            def find_nearest_index(coords, point):
                dist = np.linalg.norm(coords - point, axis=1)
                return np.argmin(dist)

            def find_end_points(coords, end):
                start_index = min(end)

                if end[1] < end[0]:
                    n_indices = (len(coords) - end[0]) + end[1]
                    mid_index = end[0] + n_indices
                else:
                    n_indices = end[1] - end[0]
                    mid_index = end[0] + n_indices

                rotated_coords = np.concatenate(
                    (coords[start_index:], coords[:start_index])
                )

                end_point = rotated_coords[n_indices // 2]

                return end_point

            # offset line
            polygon = midline.buffer(radius)
            polygon_coords = np.array(polygon.exterior.coords)

            left_line = midline.parallel_offset(
                radius, side="left", join_style=2
            )
            right_line = midline.parallel_offset(
                radius, side="right", join_style=2
            )

            left_coords = np.array(left_line.coords)
            right_coords = np.array(right_line.coords)

            left_ends = left_coords[[0, -1]]
            right_ends = right_coords[[0, -1]]

            end1 = np.array([left_ends[0], right_ends[0]])
            end2 = np.array([left_ends[1], right_ends[1]])

            end1_indices = [
                find_nearest_index(polygon_coords, end1[0]),
                find_nearest_index(polygon_coords, end1[1]),
            ]
            end2_indices = [
                find_nearest_index(polygon_coords, end2[0]),
                find_nearest_index(polygon_coords, end2[1]),
            ]

            end1_point = find_end_points(polygon_coords, end1_indices)
            end2_point = find_end_points(polygon_coords, end2_indices)

            # plt.plot(*polygon_coords.T)  # plt.scatter(*left_ends.T, c = "b", )  # plt.scatter(*right_ends.T, c = "b", )  # plt.scatter(*end1_point.T, c = "r", marker = "x", s = 100)  # plt.scatter(*end2_point.T, c = "r", marker = "x", s = 100)  # plt.show()

        except:
            print(traceback.format_exc())

    def add_manual_cell(self, last_index):
        try:
            shapes = self.cellLayer.data.copy()
            radius = self.gui.default_cell_radius.value()
            properties = self.cellLayer.properties.copy()
            shape_types = self.cellLayer.shape_type.copy()

            midline_index = last_index

            name = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=10)
            )

            midline_coords = shapes[last_index]
            cell_poles = [midline_coords[0], midline_coords[-1]]

            midline = LineString(midline_coords)
            polygon = midline.buffer(radius)

            polygon_coords = np.array(polygon.exterior.coords)

            fit = manual_fit(polygon_coords, midline_coords, radius)
            polygon_fit_coords, midline_fit_coords, poly_params, cell_radius = (
                fit
            )

            shapes[last_index] = midline_fit_coords

            cell = {
                "name": name,
                "radius": cell_radius,
                "poly_params": poly_params,
                "cell_poles": cell_poles,
            }

            if "name" not in properties.keys():
                properties["name"] = [name]
                properties["cell"] = [cell]
            else:
                properties["name"][last_index] = name
                properties["cell"][last_index] = cell

            self.update_cellLayer_shapes(
                shapes, shape_types=shape_types, properties=properties
            )

            self.cellLayer.events.data.disconnect(self.update_cells)
            self.cellLayer.refresh()

            cell = {
                "name": name,
                "radius": radius,
                "poly_params": poly_params,
                "cell_poles": cell_poles,
            }

            # self.cellLayer.current_properties = {"name": name, "cell": cell}
            self.cellLayer.add_polygons(polygon_fit_coords)
            self.cellLayer.properties["cell"][-1] = cell

            self.cellLayer.events.data.connect(self.update_cells)
            self.cellLayer.refresh()

        except:
            print(traceback.format_exc())

    def update_cells(self, event):

        try:

            if event.action == "changed":
                modified_indices = self.get_modified_shape_indices()

                for modified_index in modified_indices:

                    name_list = self.cellLayer.properties["name"].copy()
                    shape_types = self.cellLayer.shape_type.copy()
                    name = name_list[modified_index]
                    modified_shape_type = shape_types[modified_index]

                    if modified_shape_type == "path":
                        self.update_cell_model(name)

                        self.store_cell_shapes()

                    if modified_shape_type == "polygon":
                        self.update_midline_position(name)

                        self.store_cell_shapes()

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

    def remove_cells(self, indices=[]):
        try:
            if type(indices) == int:
                indices = [indices]

            self.cellLayer.events.data.disconnect(self.update_cells)
            self.cellLayer.refresh()

            self.cellLayer.selected_data = indices
            self.cellLayer.remove_selected()

            self.cellLayer.events.data.connect(self.update_cells)
            self.cellLayer.refresh()

        except:
            pass
