import os.path
import traceback
import numpy as np
from functools import partial, wraps
from qtpy.QtWidgets import QSlider, QLabel
import time
from napari.utils.notifications import show_info
import napari
from scipy.ndimage import shift


class _events_utils:

    def populate_dataset_selectors(self):
        try:
            dataset_selectors = ["import_picasso_dataset", "cellpose_dataset",
                                 "moltrack_dataset_selector", "picasso_dataset",
                                 "picasso_filter_dataset", "picasso_render_dataset",
                                 "tracking_dataset", "locs_export_dataset",
                                 "remove_seglocs_dataset", "adc_dataset",
                                 "heatmap_dataset",]

            for selector_name in dataset_selectors:
                dataset_names = list(self.dataset_dict.keys())

                single_dataset_selectors = ["moltrack_dataset_selector", "cellpose_dataset", ]

                if (selector_name not in single_dataset_selectors and len(dataset_names) > 1):
                    dataset_names.append("All Datasets")

                if selector_name == "cellpose_dataset":
                    if hasattr(self, "segmentation_image"):
                        dataset_names.insert(0, "Segmentation Image")

                if hasattr(self.gui, selector_name):
                    getattr(self.gui, selector_name).clear()
                    getattr(self.gui, selector_name).addItems(dataset_names)

        except:
            print(traceback.format_exc())
            pass

    def initialise_channel_selectors(self):
        try:
            channel_selectors = ["import_picasso_channel", "cellpose_channel",
                                 "moltrack_channel_selector", "picasso_channel",
                                 "picasso_filter_channel", "picasso_render_channel",
                                 "tracking_channel", "locs_export_channel",
                                 "remove_seglocs_channel", "adc_channel",
                                 "heatmap_channel",]

            for channel_selector in channel_selectors:
                dataset_selector = channel_selector.replace("channel", "dataset")

                if hasattr(self.gui, dataset_selector) and hasattr(self.gui, channel_selector):
                    dataset_selector = getattr(self.gui, dataset_selector)
                    channel_selector = getattr(self.gui, channel_selector)

                    dataset_selector.currentTextChanged.connect(partial(self.update_channel_selector,
                        dataset_selector=dataset_selector, channel_selector=channel_selector, ))

        except:
            print(traceback.format_exc())
            pass

    def update_channel_selector(self, channel_selector, dataset_selector):
        try:
            single_channel_selectors = ["moltrack_channel_selector", "cellpose_channel", ]

            dataset_name = dataset_selector.currentText()
            channel_names = []

            if dataset_name != "All Datasets":
                if dataset_name in self.dataset_dict.keys():
                    image_dict = self.dataset_dict[dataset_name]["images"]

                    channel_names = list(image_dict.keys())

            else:
                channel_names = []

                for dataset_name in self.dataset_dict.keys():
                    image_dict = self.dataset_dict[dataset_name]["images"]
                    channel_names.append(set(image_dict.keys()))

                channel_names = set.intersection(*channel_names)
                channel_names = list(channel_names)

            channel_selector_name = channel_selector.objectName()

            if (channel_selector_name not in single_channel_selectors and len(channel_names) > 1):
                channel_names.append("All Channels")

            current_channel = channel_selector.currentText()

            channel_selector.blockSignals(True)
            channel_selector.clear()
            channel_selector.addItems(channel_names)

            if current_channel in channel_names:
                selector_index = channel_names.index(current_channel)
                channel_selector.setCurrentIndex(selector_index)

            channel_selector.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass

    def update_import_options(self, event=None):
        try:
            self.gui.import_mode.blockSignals(True)
            self.gui.import_multichannel_mode.blockSignals(True)

            import_mode = self.gui.import_mode.currentText()
            multichannel_mode = self.gui.import_multichannel_mode.currentText()

            if import_mode == "Data (Single Channel)":
                self.gui.import_multichannel_mode.setEnabled(False)
                self.gui.import_multichannel_mode.setCurrentIndex(0)
                self.gui.import_concatenate.setEnabled(True)

            if import_mode == "Data (Multi Channel)":
                self.gui.import_multichannel_mode.setEnabled(True)

                if multichannel_mode != "None":
                    self.gui.import_concatenate.setEnabled(False)
                    self.gui.import_concatenate.setChecked(False)

                else:
                    self.gui.import_concatenate.setEnabled(True)

            if import_mode == "Segmentation Image":
                self.gui.import_multichannel_mode.setEnabled(False)
                self.gui.import_concatenate.setEnabled(False)
                self.gui.import_concatenate.setChecked(False)

            self.gui.import_mode.blockSignals(False)
            self.gui.import_multichannel_mode.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass

    def update_locs_export_options(self, event=None):
        locs_export_data = self.gui.locs_export_data.currentText()

        if locs_export_data == "Localisations":
            export_modes = ["Picasso HDF5", "CSV", "POS.OUT"]
        else:
            export_modes = ["CSV", "POS.OUT"]

        self.gui.locs_export_mode.clear()
        self.gui.locs_export_mode.addItems(export_modes)

    def update_picasso_segmentation_filter(self):
        shapes_layers = [layer.name for layer in self.viewer.layers if layer.name in ["Segmentations", "Cells"]]

        segmentation_layer = self.gui.picasso_segmentation_layer.currentText()

        if segmentation_layer in shapes_layers:
            self.gui.picasso_segmentation_filter.setEnabled(True)

        else:
            self.gui.picasso_segmentation_filter.setEnabled(False)
            self.gui.picasso_segmentation_filter.setChecked(False)

    def update_layer_combos(self):
        try:
            shapes_layers = [layer.name for layer in self.viewer.layers if layer.name in ["Segmentations", "Cells"]]

            self.gui.shapes_export_data.clear()
            self.gui.shapes_export_data.addItems(shapes_layers)

            self.gui.remove_seglocs_segmentation.clear()
            self.gui.remove_seglocs_segmentation.addItems(shapes_layers)

            self.gui.picasso_segmentation_layer.clear()
            self.gui.picasso_segmentation_layer.addItems(shapes_layers)

        except:
            print(traceback.format_exc())
            pass

    def moltrack_progress(self, progress, progress_bar):
        progress_bar.setValue(progress)

        if progress == 100:
            progress_bar.setValue(0)
            progress_bar.setHidden(True)
            progress_bar.setEnabled(False)
        else:
            progress_bar.setHidden(False)
            progress_bar.setEnabled(True)

    def moltrack_notification(self, message):
        show_info(message)

    def update_ui(self, error=None, init=False):
        try:
            if self.verbose:
                print(f"Updating UI, init = {init}")

            controls = ["import_images", "import_picasso",
                        "cellpose_load_model", "segment_active",
                        "dilate_segmentations", "picasso_detect",
                        "picasso_fit", "picasso_detectfit",
                        "filter_localisations", "picasso_render",
                        "link_localisations", "export_localisations",
                        "fit_segmentations", "export_shapes",
                        "remove_seglocs", "export_shapes",
                        "compute_adc", "export_adc", "export_heatmap",
                        "compute_heatmap",]

            progressbars = ["import_progressbar", "cellpose_progressbar",
                            "picasso_progressbar", "export_progressbar",
                            "bactfit_progressbar", "adc_progressbar",
                            "heatmap_progressbar",]

            for progressbar in progressbars:
                if hasattr(self.gui, progressbar):
                    getattr(self.gui, progressbar).setValue(0)

            if init is True:
                for control in controls:
                    getattr(self.gui, control).setEnabled(False)

                self.stop_event.clear()
                self.multiprocessing_active = True

            else:
                for control in controls:
                    getattr(self.gui, control).setEnabled(True)

                self.multiprocessing_active = False

                self.stop_event.clear()
                self.multiprocessing_active = False

            if error is not None:
                print(error)

        except:
            print(traceback.format_exc())
            pass

    def image_layer_auto_contrast(self, image, dataset, channel):
        contrast_limits = None

        try:
            autocontrast = True

            if dataset in self.contrast_dict.keys():
                if channel in self.contrast_dict[dataset].keys():
                    if ("contrast_limits" in self.contrast_dict[dataset][channel].keys()):
                        autocontrast = False

                        contrast_limits = self.contrast_dict[dataset][channel]["contrast_limits"]
                        gamma = self.contrast_dict[dataset][channel]["gamma"]

                        if hasattr(self, "image_layer"):
                            self.image_layer.gamma = gamma
                            self.image_layer.contrast_limits = contrast_limits

            if autocontrast is True:
                full_range = [np.min(image), np.max(image)]

                if max(full_range) > min(full_range):
                    contrast_limits = np.percentile(image[:10].copy(), [0.1, 99.99])

                    gamma = 1.0
                    if contrast_limits[1] > contrast_limits[0]:
                        gamma = np.log(0.5) / np.log((contrast_limits[1] - contrast_limits[0]) / (full_range[1] - full_range[0]))

                    if hasattr(self, "image_layer"):
                        self.image_layer.gamma = gamma
                        self.image_layer.contrast_limits = contrast_limits

        except:
            print(traceback.format_exc())

        return contrast_limits

    def update_contrast_dict(self):
        try:
            dataset = self.active_dataset
            channel = self.active_channel

            if dataset in self.dataset_dict.keys():
                if "images" in self.dataset_dict[dataset].keys():
                    image_dict = self.dataset_dict[dataset]["images"]

                    if channel in image_dict.keys():
                        if dataset not in self.contrast_dict.keys():
                            self.contrast_dict[dataset] = {}
                        if channel not in self.contrast_dict[dataset].keys():
                            self.contrast_dict[dataset][channel] = {}

                        layer_names = [layer.name for layer in self.viewer.layers if dataset in layer.name]

                        image_name = f"{dataset}[{channel}]"

                        if image_name in layer_names:
                            image_layer = self.viewer.layers[image_name]
                            contrast_limits = image_layer.contrast_limits
                            gamma = image_layer.gamma

                            self.contrast_dict[dataset][channel] = {"contrast_limits": contrast_limits, "gamma": gamma, }

        except:
            print(traceback.format_exc())

    def draw_segmentation_image(self, reset_view=True):

        if hasattr(self, "segmentation_image"):
            pixel_size = self.segmentation_image_pixel_size
            scale = [pixel_size, pixel_size]

            if hasattr(self, "segmentation_layer"):
                self.segmentation_layer.data = self.segmentation_image.copy()
                self.segmentation_layer.refresh()
            else:
                self.segmentation_layer = self.viewer.add_image(self.segmentation_image,
                    name="Segmentation Image", visible=True, blending="opaque", )

                self.viewer.reset_view()

            if self.gui.show_data.isChecked() == False:
                if self.segmentation_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.segmentation_layer)

            self.viewer.scale_bar.visible = True
            self.segmentation_layer.scale = scale
            self.viewer.scale_bar.unit = "nm"

            self.segmentation_layer.refresh()

            if reset_view:
                self.viewer.reset_view()

    def update_active_image(self, dataset=None, channel=None, event=None):

        try:

            if dataset == None or dataset not in self.dataset_dict.keys():
                dataset_name = self.gui.moltrack_dataset_selector.currentText()
            else:
                dataset_name = dataset

            if dataset_name in self.dataset_dict.keys():
                if "images" in self.dataset_dict[dataset_name].keys():
                    image_dict = self.dataset_dict[dataset_name]["images"]

                    if channel == None or channel not in image_dict.keys():
                        channel_name = (self.gui.moltrack_channel_selector.currentText())
                    else:
                        channel_name = channel

                    self.update_contrast_dict()

                    self.active_dataset = dataset_name
                    self.active_channel = channel_name

                    image_dict = self.dataset_dict[dataset_name]["images"]
                    pixel_size = float(self.dataset_dict[dataset_name]["pixel_size"])
                    scale = [pixel_size, pixel_size]

                    if channel_name in image_dict.keys():
                        image = image_dict[channel_name]
                        image_name = f"{dataset_name}[{channel_name}]"

                        if hasattr(self, "image_layer") == False:
                            self.image_layer = self.viewer.add_image(image, name=image_name,
                                colormap="gray", blending="opaque", visible=True, scale=scale, )

                            self.viewer.reset_view()

                        else:
                            self.image_layer.data = image
                            self.image_layer.name = image_name
                            self.image_layer.scale = scale

                        if self.gui.show_data.isChecked() == False:
                            if self.image_layer in self.viewer.layers:
                                self.viewer.layers.remove(self.image_layer)

                        self.viewer.scale_bar.visible = True
                        self.image_layer.scale = scale
                        self.viewer.scale_bar.unit = "nm"

                        self.image_layer_auto_contrast(image, dataset_name, channel_name)
                        self.image_layer.refresh()

                        dataset_names = self.dataset_dict.keys()
                        active_dataset_index = list(dataset_names).index(dataset_name)

                        channel_names = list(image_dict.keys())
                        active_channel_index = list(channel_names).index(channel_name)

                        dataset_selector = self.gui.moltrack_dataset_selector
                        channel_selector = self.gui.moltrack_channel_selector

                        dataset_selector.blockSignals(True)
                        dataset_selector.clear()
                        dataset_selector.addItems(dataset_names)
                        dataset_selector.setCurrentIndex(active_dataset_index)
                        dataset_selector.blockSignals(False)

                        channel_selector.blockSignals(True)
                        channel_selector.clear()
                        channel_selector.addItems(channel_names)
                        channel_selector.setCurrentIndex(active_channel_index)
                        channel_selector.blockSignals(False)

            else:
                self.active_dataset = None

            self.draw_localisations()
            self.draw_tracks()
            self.update_overlay_text()

        except:
            print(traceback.format_exc())
            pass

    def update_overlay_text(self):
        try:
            if self.dataset_dict != {}:
                dataset_name = self.gui.moltrack_dataset_selector.currentText()
                channel_name = self.gui.moltrack_channel_selector.currentText()

                if dataset_name in self.dataset_dict.keys():
                    data_dict = self.dataset_dict[dataset_name].copy()

                    path = data_dict["path"]
                    exposure_time_ms = data_dict["exposure_time"]

                    current_step = self.viewer.dims.current_step[0]

                    if type(path) == list:
                        path = path[current_step]

                    file_name = os.path.basename(path)

                    elapsed_time_s = current_step * (exposure_time_ms / 1000)

                    overlay_string = ""
                    overlay_string += f"File: {file_name}\n"
                    overlay_string += f"Channel: {channel_name}\n"
                    overlay_string += f"Elapsed Time: {elapsed_time_s:.2f} s\n"

                    if overlay_string != "":
                        self.viewer.text_overlay.visible = True
                        self.viewer.text_overlay.position = "top_left"
                        self.viewer.text_overlay.text = overlay_string.lstrip("\n")
                        self.viewer.text_overlay.color = "red"
                        self.viewer.text_overlay.font_size = 9
                    else:
                        self.viewer.text_overlay.visible = False

        except:
            print(traceback.format_exc())

    def slider_event(self, viewer=None):
        self.update_overlay_text()
        self.draw_localisations()

    def draw_localisations(self, update_vis=False):

        remove_localisations = True

        if hasattr(self, "localisation_dict"):
            if hasattr(self, "fiducial_layer"):
                show_localisations = self.loc_layer.visible
            else:
                show_localisations = True

            if show_localisations:
                layer_names = [layer.name for layer in self.viewer.layers]

                active_frame = self.viewer.dims.current_step[0]
                dataset_name = self.gui.moltrack_dataset_selector.currentText()
                channel_name = self.gui.moltrack_channel_selector.currentText()

                if dataset_name in self.localisation_dict.keys():
                    if (channel_name in self.localisation_dict[dataset_name].keys()):
                        localisation_dict = self.localisation_dict[dataset_name][channel_name].copy()

                        if "localisations" in localisation_dict.keys():
                            locs = localisation_dict["localisations"]

                            if active_frame in locs.frame:
                                frame_locs = locs[locs.frame == active_frame].copy()
                                render_locs = np.vstack((frame_locs.y, frame_locs.x)).T.tolist()

                                pixel_size = float(self.dataset_dict[dataset_name]["pixel_size"])
                                scale = [pixel_size, pixel_size]

                                vis_mode = (self.gui.picasso_vis_mode.currentText())
                                vis_opacity = float(self.gui.picasso_vis_opacity.currentText())
                                vis_edge_width = float(self.gui.picasso_vis_edge_width.currentText())
                                vis_size = float(self.gui.picasso_vis_size.currentText())

                                vis_size = vis_size / pixel_size

                                if vis_mode.lower() == "square":
                                    symbol = "square"
                                elif vis_mode.lower() == "disk":
                                    symbol = "disc"
                                elif vis_mode.lower() == "x":
                                    symbol = "cross"

                                remove_localisations = False

                                if "localisations" not in layer_names:
                                    if self.verbose:
                                        print("Drawing localisations")

                                    self.loc_layer = self.viewer.add_points(render_locs, ndim=2,
                                        edge_color="red", face_color=[0, 0, 0, 0], opacity=vis_opacity,
                                        name="localisations", symbol=symbol, size=vis_size, edge_width=vis_edge_width,
                                        scale=scale, )

                                    update_vis = True

                                else:
                                    if self.verbose:
                                        print("Updating fiducial data")

                                    self.loc_layer.data = render_locs
                                    self.loc_layer.selected_data = []
                                    self.loc_layer.scale = scale

                                if self.gui.show_locs.isChecked() == False:
                                    if self.loc_layer in self.viewer.layers:
                                        self.viewer.layers.remove(self.loc_layer)

                                self.loc_layer.scale = scale
                                self.viewer.scale_bar.visible = True
                                self.viewer.scale_bar.unit = "nm"

                                if update_vis:
                                    if self.verbose:
                                        print("Updating fiducial visualisation settings")

                                    self.loc_layer.selected_data = list(range(len(self.loc_layer.data)))
                                    self.loc_layer.opacity = vis_opacity
                                    self.loc_layer.symbol = symbol
                                    self.loc_layer.size = vis_size
                                    self.loc_layer.edge_width = vis_edge_width
                                    self.loc_layer.edge_color = "red"
                                    self.loc_layer.selected_data = []

                                    self.loc_layer.refresh()

            if remove_localisations:
                if "localisations" in layer_names:
                    self.viewer.layers["localisations"].data = []

            for layer in layer_names:
                self.viewer.layers[layer].refresh()

    def clear_live_images(self):
        try:
            if self.verbose:
                print("Clearing live images")

            image_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]

            for layer in image_layers:
                frame_shape = layer.data.shape[1:]
                empty_frame = np.zeros(frame_shape, dtype=layer.data.dtype)
                layer.data = empty_frame

        except:
            print(traceback.format_exc())
            pass

    def update_detect_options(self, event=None):
        if self.gui.smlm_detect_mode.currentText() == "Picasso":
            self.gui.picasso_box_size_label.show()
            self.gui.picasso_box_size.show()
            self.gui.picasso_min_net_gradient_label.show()
            self.gui.picasso_min_net_gradient.show()

            self.gui.moltrack_threshold_label.hide()
            self.gui.moltrack_threshold.hide()
            self.gui.moltrack_window_size_label.hide()
            self.gui.moltrack_window_size.hide()

        else:
            self.gui.picasso_box_size_label.hide()
            self.gui.picasso_box_size.hide()
            self.gui.picasso_min_net_gradient_label.hide()
            self.gui.picasso_min_net_gradient.hide()

            self.gui.moltrack_threshold_label.show()
            self.gui.moltrack_threshold.show()
            self.gui.moltrack_window_size_label.show()
            self.gui.moltrack_window_size.show()

    def moltract_translation(self, event=None, direction="left"):

        try:
            translation_target = self.gui.translation_target.currentText()
            size = self.gui.translation_size.value()

            if direction == "up":
                shift_vector = [-size, 0.0]
            elif direction == "down":
                shift_vector = [size, 0.0]
            elif direction == "left":
                shift_vector = [0.0, -size]
            elif direction == "right":
                shift_vector = [0.0, size]

            if translation_target in ["Segmentation Image", "All"]:
                if hasattr(self, "segmentation_image"):
                    image = self.segmentation_image.copy()

                    if len(image.shape) == 2:
                        image = shift(image, shift=shift_vector)
                        self.segmentation_image = image

                    else:
                        for fame_index, frame in enumerate(image):
                            image[fame_index] = shift(frame, shift=shift_vector)
                        self.segmentation_image = image

                    self.segmentation_layer.data = self.segmentation_image
                    self.segmentation_layer.refresh()

            if translation_target in ["Segmentations", "All"]:
                if hasattr(self, "segLayer"):
                    seg_data = self.segLayer.data.copy()

                    for seg_index, seg in enumerate(seg_data):
                        if seg.shape[1] == 2:
                            seg = seg + shift_vector
                            seg_data[seg_index] = seg
                        if seg.shape[1] == 3:
                            seg[:, 1:] = seg[:, 1:] + shift_vector
                            seg_data[seg_index] = seg

                    self.segLayer.data = seg_data

            if translation_target in ["Cells", "All"]:
                if hasattr(self, "cellLayer"):
                    cell_data = self.cellLayer.data.copy()

                    for cell_index, cell in enumerate(cell_data):
                        if cell.shape[1] == 2:
                            cell = cell + shift_vector
                            cell_data[cell_index] = cell
                        if cell.shape[1] == 3:
                            cell[:, 1:] = cell[:, 1:] + shift_vector
                            cell_data[cell_index] = cell

                    self.cellLayer.data = cell_data

        except:
            print(traceback.format_exc())

    def update_active_layers(self, mode="all"):

        if mode == "data":
            if hasattr(self, "image_layer"):
                if self.gui.show_data.isChecked():
                    if self.image_layer not in self.viewer.layers:
                        self.viewer.layers.append(self.image_layer)
                else:
                    if self.image_layer in self.viewer.layers:
                        if self.image_layer in self.viewer.layers:
                            self.viewer.layers.remove(self.image_layer)

            if hasattr(self, "segmentation_layer"):
                if self.gui.show_data.isChecked():
                    if self.segmentation_layer not in self.viewer.layers:
                        self.viewer.layers.append(self.segmentation_layer)
                else:
                    if self.segmentation_layer in self.viewer.layers:
                        if self.segmentation_layer in self.viewer.layers:
                            self.viewer.layers.remove(self.segmentation_layer)

        elif mode == "shapes":
            if hasattr(self, "segLayer"):
                if self.gui.show_shapes.isChecked():
                    if self.segLayer not in self.viewer.layers:
                        self.viewer.layers.append(self.segLayer)
                else:
                    if self.segLayer in self.viewer.layers:
                        if self.segLayer in self.viewer.layers:
                            self.viewer.layers.remove(self.segLayer)

            if hasattr(self, "cellLayer"):
                if self.gui.show_shapes.isChecked():
                    if self.cellLayer not in self.viewer.layers:
                        self.viewer.layers.append(self.cellLayer)
                else:
                    if self.cellLayer in self.viewer.layers:
                        if self.cellLayer in self.viewer.layers:
                            self.viewer.layers.remove(self.cellLayer)

        elif mode == "tracks":
            if hasattr(self, "track_layer"):
                if self.gui.show_tracks.isChecked():
                    if self.track_layer not in self.viewer.layers:
                        self.viewer.layers.append(self.track_layer)
                else:
                    if self.track_layer in self.viewer.layers:
                        if self.track_layer in self.viewer.layers:
                            self.viewer.layers.remove(self.track_layer)

        elif mode == "render":
            if hasattr(self, "render_layer"):
                if self.gui.show_render.isChecked():
                    if self.render_layer not in self.viewer.layers:
                        self.viewer.layers.append(self.render_layer)
                else:
                    if self.render_layer in self.viewer.layers:
                        if self.render_layer in self.viewer.layers:
                            self.viewer.layers.remove(self.render_layer)

        elif mode == "locs":
            if hasattr(self, "loc_layer"):
                if self.gui.show_locs.isChecked():
                    if self.loc_layer not in self.viewer.layers:
                        self.viewer.layers.append(self.loc_layer)
                else:
                    if self.loc_layer in self.viewer.layers:
                        if self.loc_layer in self.viewer.layers:
                            self.viewer.layers.remove(self.loc_layer)

        else:
            pass


