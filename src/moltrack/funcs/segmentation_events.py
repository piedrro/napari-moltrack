import traceback

import numpy as np
from napari.utils.notifications import show_info
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


class _segmentation_events:

    def dilate_segmentation(self, viewer=None, event=None):
        try:
            if "Control" in event.modifiers:
                coords = self.segLayer.world_to_data(event.position)
                shape_index = self.segLayer.get_value(coords)[0]

                if shape_index is not None:
                    shape_type = self.segLayer.shape_type[shape_index]

                    if shape_type == "polygon":
                        if event.delta[1] > 0:
                            buffer = 0.5
                        else:
                            buffer = -0.5

                        shapes = self.segLayer.data.copy()

                        shape = shapes[shape_index]

                        if shape.shape[1] == 2:
                            shape = Polygon(shape)
                            shape = shape.buffer(buffer)
                            shape = np.array(shape.exterior.coords)
                            shape = shape[1:]
                            shape = shape.astype(float)
                        else:
                            frame_index = shape[0, 0]
                            shape = Polygon(shape[:, 1:])
                            shape = shape.buffer(buffer)
                            shape = np.array(shape.exterior.coords)
                            shape = np.insert(shape, 0, frame_index, axis=1)
                            shape = shape[1:]
                            shape = shape.astype(float)

                        shapes[shape_index] = shape
                        self.segLayer.data = shapes

        except:
            print(traceback.format_exc())

    def register_shape_layer_keybinds(self, layer):
        layer.bind_key("Space", func=lambda event: self.modify_mode(mode="add"), overwrite=True, )
        layer.bind_key("e", func=lambda event: self.modify_mode(mode="extend"), overwrite=True)
        layer.bind_key("j", func=lambda event: self.modify_mode(mode="join"), overwrite=True, )
        layer.bind_key("d", func=lambda event: self.modify_mode(mode="delete"), overwrite=True, )
        layer.bind_key("p", func=lambda event: self.modify_mode(mode="pan_zoom"), overwrite=True, )
        layer.bind_key("m", func=lambda event: self.modify_mode(mode="midline"), overwrite=True, )
        layer.bind_key("s", func=lambda event: self.modify_mode(mode="edit_midlines"), overwrite=True, )
        layer.bind_key("Escape", func=lambda event: self.modify_mode(mode="pan_zoom"), overwrite=True, )

    def initialise_segLayer(self, shapes=None, pixel_size=None, *, model_mask_ids=(),
                            restore_annotations=None, prepared_layer=None, shape_properties=None):
        layer_names = [layer.name for layer in self.viewer.layers]

        if "Segmentations" in layer_names:
            self.viewer.layers.remove("Segmentations")

        if hasattr(self, "segLayer"):
            del self.segLayer

        if pixel_size is None:
            if hasattr(self, "segmentation_image_pixel_size"):
                pixel_size = self.segmentation_image_pixel_size
            elif hasattr(self, "image_layer"):
                pixel_size = self.image_layer.scale[0]
            else:
                pixel_size = 1

        scale = [pixel_size, pixel_size]
        if restore_annotations is not None:
            scale = restore_annotations["scale"]
        ndim = np.asarray(shapes[0]).shape[-1] if shapes is not None and len(shapes) else 2
        units = ("pixel",) * (ndim - 2) + ("nm", "nm")

        if prepared_layer is not None:
            self.segLayer = self.viewer.add_layer(prepared_layer)
        elif shapes is not None:
            self.segLayer = self.viewer.add_shapes(name="Segmentations",
                shape_type=restore_annotations["shape_types"] if restore_annotations is not None else "polygon", properties=restore_annotations.get("properties", {}) if restore_annotations is not None else shape_properties, opacity=0.5, face_color="red", edge_color="black", edge_width=1, data=shapes,
                scale=scale, units=units, )
        else:
            self.segLayer = self.viewer.add_shapes(name="Segmentations",
                shape_type="polygon", opacity=0.5, face_color="red", edge_color="black", edge_width=1,
                scale=scale, units=units, )

        if self.gui.show_shapes.isChecked() == False:
            if self.segLayer in self.viewer.layers:
                self.viewer.layers.remove(self.segLayer)

        self.segLayer.scale = scale
        self.segLayer.refresh()
        self.viewer.reset_view()

        self.segLayer.mouse_drag_callbacks.append(self.seg_drag_event)
        self.segLayer.mouse_double_click_callbacks.append(self.delete_clicked)
        self.segLayer.events.data.connect(self.update_shapes)
        self.segLayer.events.mode.connect(self.segmentation_layer_mode_changed)
        self.segLayer.mouse_wheel_callbacks.append(self.dilate_segmentation)

        self.register_shape_layer_keybinds(self.segLayer)
        from moltrack.model_views import record_layer_annotations, connect_layer_annotations
        if restore_annotations is None:
            record_layer_annotations(self.session, self.segLayer, "segmentations", mask_ids=model_mask_ids)
        connect_layer_annotations(self.session, self.segLayer, "segmentations")
        self.update_segmentation_combos()

        return self.segLayer

    def get_seglayer(self, shapes=None):
        layer_names = [layer.name for layer in self.viewer.layers]

        if "Segmentations" not in layer_names:
            self.initialise_segLayer()

        return self.segLayer

    def modify_mode(self, viewer=None, mode="add"):
        try:
            self._extension_gesture = None
            self.clear_extension_preview()
            if hasattr(self, "segLayer"):
                self.segLayer.mouse_pan = True
            if mode in ["add", "extend", "join"]:
                self.segLayer = self.get_seglayer()

            elif mode in ["midline", "edit_midlines"]:
                self.cellLayer = self.get_cellLayer()

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
                # Capture the drag ourselves: no temporary segmentation and no
                # dependency on napari's lasso finalisation event order.
                self.segLayer.mode = "pan_zoom"
                self.segLayer.mouse_pan = False
                show_info("Extend (E): drag from inside a segmentation, around the extra area, back inside; Escape cancels")

            if mode == "join":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "join"
                self.segLayer.mode = "add_line"
                show_info("Join (click/drag to join)")

            if mode == "delete":
                selected_layer = self.viewer.layers.selection.active

                if selected_layer is not None:
                    if selected_layer.name == "Cells":
                        self.cellLayer.mode = "select"
                        show_info("Delete Cell (click/drag to delete)")
                    if selected_layer.name == "Segmentations":
                        self.segLayer.mode = "select"
                        show_info("Delete Segmentation (click/drag to delete)")

                    self.interface_mode = "segment"
                    self.segmentation_mode = "delete"

            if mode == "pan_zoom":
                selected_layer = self.viewer.layers.selection.active

                if selected_layer is not None:
                    if selected_layer.name == "Cells":
                        self.cellLayer.mode = "pan_zoom"
                    if selected_layer.name == "Segmentations":
                        self.segLayer.mode = "pan_zoom"

                    show_info("Pan/Zoom activated")

                    self.interface_mode = "segment"
                    self.segmentation_mode = "panzoom"

            if mode == "midline":
                self.segmentation_mode = "midline"
                self.viewer.layers.selection.select_only(self.cellLayer)
                self.cellLayer.mode = "add_path"
                self.cellLayer.refresh()

                show_info("Midline (click to add midline)")

            if mode == "edit_midlines":
                self.segmentation_mode = "edit_midlines"
                self.viewer.layers.selection.select_only(self.cellLayer)
                self.cellLayer.mode = "direct"

                self.select_cell_midlines()

                show_info("Edit midlines (click/drag to edit midline)")

        except:
            print(traceback.format_exc())

    def segmentation_layer_mode_changed(self, event):
        """Native toolbar modes also cancel the custom extension tool."""
        if (getattr(self, "segmentation_mode", None) == "extend"
                and self.segLayer.mode != "pan_zoom"):
            self._extension_gesture = None
            self.clear_extension_preview()
            self.segmentation_mode = "panzoom"

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
                    last_index = len(shapes) - 1

                    self.remove_shapes(last_index)

                    if hasattr(self, "join_coords"):
                        if type(self.join_coords) == list:
                            coords1, coords2 = self.join_coords

                            shape_index1 = self.segLayer.get_value(coords1)[0]
                            shape_index2 = self.segLayer.get_value(coords2)[0]

                            if (shape_index1 is not None and shape_index2 is not None):
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

            if event.action in ["added", "changed"]:
                self.segLayer.mode = "pan_zoom"

        except:
            print(traceback.format_exc())

    def delete_clicked(self, viewer=None, event=None):
        try:
            if self.segmentation_mode == "extend":
                return
            coords = self.segLayer.world_to_data(event.position)
            shape_index = self.segLayer.get_value(coords)[0]

            if shape_index is not None:
                self.remove_shapes([shape_index])

        except:
            print(traceback.format_exc())

    def seg_drag_event(self, viwer=None, event=None):
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
                shapes.pop(len(shapes) - 1)

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
                yield from self.extend_segmentation_drag(event)

    def extend_segmentation_drag(self, event):
        """Collect one extension in layer coordinates, then commit an exact union."""
        if getattr(event, "button", 1) != 1:
            return
        layer = self.segLayer
        token = object()
        self._extension_gesture = token
        # Prevent the camera from panning while the custom drawing mode is active.
        event.handled = True
        start = np.asarray(layer.world_to_data(event.position), dtype=float)
        original = [s.copy() for s in layer.data]
        shape_types = list(layer.shape_type)
        target_index = None
        for index in reversed(range(len(original))):
            vertices = original[index]
            if shape_types[index] != "polygon" or not np.isfinite(vertices).all():
                continue
            if vertices.shape[1] != len(start) or not np.all(vertices[:, :-2] == start[:-2]):
                continue
            polygon = Polygon(vertices[:, -2:])
            if polygon.is_valid and polygon.covers(Point(start[-2:])):
                target_index = index
                break
        points = [start]
        try:
            yield
            while event.type == "mouse_move":
                event.handled = True
                if self._extension_gesture is not token:
                    return
                points.append(np.asarray(layer.world_to_data(event.position), dtype=float))
                if target_index is not None and np.isfinite(points).all():
                    self.show_extension_preview(points, layer)
                yield
            if self._extension_gesture is not token:
                return
            points.append(np.asarray(layer.world_to_data(event.position), dtype=float))
            if target_index is None:
                raise ValueError("Start inside a segmentation")
            if len(layer.data) != len(original) or any(
                not np.array_equal(a, b) for a, b in zip(layer.data, original, strict=True)
            ) or list(layer.shape_type) != shape_types:
                raise ValueError("Segmentations changed while drawing; try again")
            points = np.asarray(points)
            if not np.isfinite(points).all() or not np.all(points[:, :-2] == start[:-2]):
                raise ValueError("Keep the extension on the starting frame")
            target = Polygon(original[target_index][:, -2:])
            if not target.covers(Point(points[-1, -2:])):
                raise ValueError("Finish inside the starting segmentation")
            extension = Polygon(points[:, -2:])
            if not extension.is_valid or extension.area <= 0:
                raise ValueError("Draw a non-crossing outline around the extra area")
            union = target.union(extension)
            if (union.geom_type != "Polygon" or not union.is_valid
                    or len(union.interiors) or union.area < target.area):
                raise ValueError("The extension does not form a valid single outline")
            vertices = np.asarray(union.exterior.coords)[:-1]
            if layer.ndim > 2:
                vertices = np.column_stack((np.tile(start[:-2], (len(vertices), 1)), vertices))
            replacement = list(layer.data)
            replacement[target_index] = vertices
            layer.data = replacement
        except (ValueError, TypeError) as error:
            show_info(f"Could not extend segmentation: {error}")
        finally:
            if self._extension_gesture is token:
                self.clear_extension_preview()
                self._extension_gesture = None
                self.segmentation_mode = "panzoom"
                layer.mode = "pan_zoom"
                layer.mouse_pan = True

    def clear_extension_preview(self):
        preview = getattr(self, "_extension_preview", None)
        if preview is not None and preview in self.viewer.layers:
            self.viewer.layers.remove(preview)
        self._extension_preview = None

    def show_extension_preview(self, points, layer):
        # This layer is visual feedback only: it has no annotation callbacks
        # and is removed on release, cancellation, or a mode change.
        preview = getattr(self, "_extension_preview", None)
        if preview is None:
            preview = self.viewer.add_shapes(
                [np.asarray(points)], name="Extension preview", shape_type="path",
                edge_color="yellow", edge_width=2, face_color="transparent",
                scale=layer.scale, translate=layer.translate, rotate=layer.rotate,
                shear=layer.shear, affine=layer.affine.affine_matrix, units=layer.units,
            )
            preview.editable = False
            self._extension_preview = preview
            self.viewer.layers.selection.select_only(layer)
        else:
            preview.data = [np.asarray(points)]

    def join_shapes(self, shape1, shape2, simplify=True, buffer=1):
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

        return union_shape
