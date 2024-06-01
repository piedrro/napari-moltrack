import time

from napari.utils.notifications import show_info
import traceback
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
import warnings

class _segmentation_events:

    def register_segmentation_keybinds(self, layer):

        layer.bind_key(key='Space', func=lambda event: self.segmentation_modify_mode(mode="add"), overwrite=True)
        layer.bind_key(key='e', func=lambda event: self.segmentation_modify_mode(mode="extend"), overwrite=True)
        layer.bind_key(key='j', func=lambda event: self.segmentation_modify_mode(mode="join"), overwrite=True)
        layer.bind_key(key='s', func=lambda event: self.segmentation_modify_mode(mode="split"), overwrite=True)
        layer.bind_key(key='d', func=lambda event: self.segmentation_modify_mode(mode="delete"), overwrite=True)

    def initialise_segLayer(self, shapes = None):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Segmentations" in layer_names:
            self.viewer.layers.remove("Segmentations")

        if hasattr(self, "segLayer"):
            del self.segLayer

        if shapes is not None:
            self.segLayer = self.viewer.add_shapes(name="Segmentations", shape_type="polygon",
                opacity=0.5, face_color="red", edge_color="black", edge_width=1, data=shapes)
        else:
            self.segLayer = self.viewer.add_shapes(name="Segmentations", shape_type="polygon",
                opacity=0.5, face_color="red", edge_color="black", edge_width=1)

        self.segLayer.mouse_drag_callbacks.append(self.seg_drag_event)
        self.segLayer.mouse_double_click_callbacks.append(self.delete_clicked)
        self.segLayer.events.data.connect(self.update_shapes)
        self.register_segmentation_keybinds(self.segLayer)

        return self.segLayer


    def get_seglayer(self, shapes = None):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Segmentations" not in layer_names:

            self.initialise_segLayer()

        return self.segLayer


    def segmentation_modify_mode(self, viewer=None, mode = "add"):

        try:

            self.segLayer = self.get_seglayer()

            if mode == "add":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "add"
                self.segLayer.mode = "add_polygon_lasso"
                show_info("Add (click/drag to add)")

            if mode == "extend":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "extend"
                self.segLayer.mode = "add_polygon_lasso"
                show_info("Extend (click/drag to extend)")

            if mode == "join":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "join"
                self.segLayer.mode = "add_line"
                show_info("Join (click/drag to join)")

            if mode == "split":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "split"
                show_info("Split (click/drag to split)")

            if mode == "delete":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "delete"

                self.segLayer.mode = "select"

                show_info("Delete (click/drag to delete)")

        except:
            print(traceback.format_exc())
            pass


    def remove_shapes(self, indices):

        if type(indices) == int:
            indices = [indices]

        if len(indices) > 0:

            self.segLayer.mode = "pan_zoom"

            self.segLayer.events.data.disconnect(self.update_shapes)

            self.segLayer.selected_data = indices
            self.segLayer.remove_selected()

            self.segLayer.events.data.connect(self.update_shapes)


    def update_shapes(self, event):

        try:

            if event.action == "added":

                self.segLayer.mode = "pan_zoom"

                if self.segmentation_mode == "join":

                    shapes = self.segLayer.data
                    last_index = len(shapes)-1

                    self.remove_shapes(last_index)

                    if hasattr(self, "join_coords"):

                        if type(self.join_coords) == list:
                            coords1, coords2 = self.join_coords

                            shape_index1 = self.segLayer.get_value(coords1)[0]
                            shape_index2 = self.segLayer.get_value(coords2)[0]

                            if shape_index1 is not None and shape_index2 is not None:

                                shape1 = shapes[shape_index1]
                                shape2 = shapes[shape_index2]

                                union_shape = self.join_shapes(shape1, shape2)

                                if union_shape is not None:

                                    self.remove_shapes([shape_index1, shape_index2])
                                    shapes = self.segLayer.data.copy()
                                    shapes.append(union_shape)
                                    self.segLayer.data = shapes
                                    self.segLayer.refresh()

                    self.join_coords = None

                if self.segmentation_mode == "extend":

                    extend_shapes = False

                    if hasattr(self, "extend_indices"):

                        if type(self.extend_indices) == list:

                            if len(self.extend_indices) == 2:

                                extend_shapes = True

                    if extend_shapes:

                        shape_index, extend_index = self.extend_indices

                        shapes = self.segLayer.data.copy()

                        target = shapes[shape_index].copy()
                        extension = shapes[extend_index].copy()

                        if len(target) > 4 and len(extension) > 4:

                            union_shape = self.join_shapes(target, extension)

                            if union_shape is not None:

                                self.segLayer.events.data.disconnect(self.update_shapes)
                                self.segLayer.selected_data = [shape_index, extend_index]
                                self.segLayer.remove_selected()
                                self.segLayer.add(union_shape, shape_type="polygon")
                                self.segLayer.events.data.connect(self.update_shapes)

                    else:

                        shapes = self.segLayer.data.copy()
                        last_index = len(shapes)-1
                        self.remove_shapes(last_index)

                    self.extend_indices = None

            if event.action in ["added", "changed"]:
                self.segLayer.mode = "pan_zoom"

        except:
            print(traceback.format_exc())
            pass

    def delete_clicked(self, viewer=None, event=None):

        try:

            coords = self.segLayer.world_to_data(event.position)
            shape_index = self.segLayer.get_value(coords)[0]

            if shape_index is not None:
                self.remove_shapes([shape_index])

        except:
            print(traceback.format_exc())
            pass


    def seg_drag_event(self, viwer = None, event = None):

        if hasattr(self, "segmentation_mode"):

            if self.segmentation_mode == "delete":

                coords = self.segLayer.world_to_data(event.position)
                shape_index = self.segLayer.get_value(coords)[0]

                if shape_index is not None:

                    shapes = self.segLayer.data.copy()
                    shapes.pop(shape_index)
                    self.segLayer.data = shapes

            if self.segmentation_mode == "join":

                self.join_coords = None

                canvas_pos = event.position
                coords1 = self.segLayer.world_to_data(canvas_pos)
                shape_index1 = self.segLayer.get_value(coords1)[0]

                shapes = self.segLayer.data.copy()
                shapes.pop(len(shapes)-1)

                if shape_index1 is not None:
                    dragged = False

                    yield
                    while event.type == "mouse_move":
                        dragged = True
                        yield

                    if dragged:

                        coords2 = self.segLayer.world_to_data(event.position)
                        shape_index2 = self.segLayer.get_value(coords2)[0]

                        if shape_index2 is not None:

                            self.join_coords = [coords1, coords2]


            if self.segmentation_mode == "extend":

                canvas_pos = event.position
                coords = self.segLayer.world_to_data(canvas_pos)
                shape_index = self.segLayer.get_value(coords)[0]

                if shape_index is not None:

                    dragged = False

                    yield
                    while event.type == "mouse_move":
                        dragged = True
                        yield

                    if dragged:

                        shapes = self.segLayer.data.copy()
                        last_index = len(shapes)-1

                        self.extend_indices = [shape_index, last_index]


    def join_shapes(self, shape1, shape2, simplify = True, buffer = 1):

        union_shape = None

        try:

            if shape1.shape[1] == 2:

                shape1 = Polygon(shape1)
                shape2 = Polygon(shape2)

                shape1 = shape1.buffer(buffer)
                shape2 = shape2.buffer(buffer)

                if shape1.intersects(shape2):

                    union_polygon = unary_union([shape1, shape2])
                    union_polygon = union_polygon.buffer(-buffer)

                    if simplify == True:
                        union_polygon = union_polygon.simplify(0.1)

                    union_shape = np.array(union_polygon.exterior.coords)

                    union_shape = union_shape[1:]
                    union_shape = union_shape.astype(float)

            else:
                frame_index = shape1[0, 0]
                shape1 = Polygon(shape1[:, 1:])
                shape2 = Polygon(shape2[:, 1:])

                shape1 = shape1.buffer(buffer)
                shape2 = shape2.buffer(buffer)

                if shape1.intersects(shape2):

                    union_polygon = unary_union([shape1, shape2])
                    union_polygon = union_polygon.buffer(-buffer)

                    if simplify == True:
                        union_polygon = union_polygon.simplify(0.1)

                    union_shape = np.array(union_polygon.exterior.coords)
                    union_shape = np.insert(union_shape, 0, frame_index, axis=1)

                    union_shape = union_shape[1:]
                    union_shape = union_shape.astype(float)

        except:
            print(traceback.format_exc())
            pass

        return union_shape











