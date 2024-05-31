
import os.path
import traceback
import numpy as np
from functools import partial, wraps
from qtpy.QtWidgets import (QSlider, QLabel)
import time
from napari.utils.notifications import show_info
import napari
from scipy.ndimage import shift

class _events_utils:

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

    def update_ui(self, error=None, init = False):

        try:

            if self.verbose:
                print(f"Updating UI, init = {init}")

            controls = ["import_images",
                        "import_picasso",
                        "cellpose_load_model",
                        "segment_active",
                        "segment_all",
                        "dilate_segmentations",
                        "picasso_detect",
                        "picasso_fit",
                        "picasso_detectfit",
                        "filter_localisations",
                        "picasso_render",
                        "link_localisations",
                        "export_localisations",
                        ]

            progressbars = ["import_progressbar",
                            "cellpose_progressbar",
                            "picasso_progressbar",
                            "export_progressbar",
                            ]

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



    def image_layer_auto_contrast(self, image, dataset):

        contrast_limits = None

        try:
            autocontrast = True

            if dataset in self.contrast_dict.keys():

                autocontrast = False

                contrast_limits = self.contrast_dict[dataset]["contrast_limits"]
                gamma = self.contrast_dict[dataset]["gamma"]

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

            if dataset in self.dataset_dict.keys():

                if dataset not in self.contrast_dict.keys():
                    self.contrast_dict[dataset] = {}

                layer_name = [layer.name for layer in self.viewer.layers if dataset in layer.name]

                if len(layer_name) > 0:

                    image_layer = self.viewer.layers[layer_name[0]]
                    contrast_limits = image_layer.contrast_limits
                    gamma = image_layer.gamma

                    self.contrast_dict[dataset] = {"contrast_limits": contrast_limits,
                                                   "gamma": gamma}

        except:
            print(traceback.format_exc())


    def draw_segmentation_image(self):

        if hasattr(self, "segmentation_image"):

            if hasattr(self, "segmentation_layer"):
                self.segmentation_layer.data = self.segmentation_image.copy()
                self.segmentation_layer.refresh()
            else:
                self.segmentation_layer = self.viewer.add_image(self.segmentation_image,
                    name="Segmentation Image", visible=True)
                self.segmentation_layer.refresh()

    def update_active_image(self, dataset=None, event=None):

        try:

            if dataset == None or dataset not in self.dataset_dict.keys():
                dataset_name = self.gui.moltrack_dataset_selector.currentText()
            else:
                dataset_name = dataset

            if dataset_name in self.dataset_dict.keys():

                self.update_contrast_dict()

                self.active_dataset = dataset_name

                if "data" in self.dataset_dict[dataset_name].keys():

                    image = self.dataset_dict[dataset_name]["data"]

                    if hasattr(self, "image_layer") == False:

                        self.image_layer = self.viewer.add_image(image,
                            name=dataset_name,
                            colormap="gray",
                            blending="additive",
                            visible=True)

                    else:
                        self.image_layer.data = image
                        self.image_layer.name = dataset_name
                        self.image_layer.refresh()

                    self.image_layer_auto_contrast(image, dataset_name)

                    dataset_names = self.dataset_dict.keys()
                    active_dataset_index = list(dataset_names).index(dataset_name)

                    dataset_selector = self.gui.moltrack_dataset_selector

                    dataset_selector.blockSignals(True)
                    dataset_selector.clear()
                    dataset_selector.addItems(dataset_names)
                    dataset_selector.setCurrentIndex(active_dataset_index)
                    dataset_selector.blockSignals(False)

            else:

                self.active_dataset = None

            # self.draw_localisations(update_vis=True)
            self.update_overlay_text()

        except:
            print(traceback.format_exc())
            pass

    def update_overlay_text(self):

        try:

            if self.dataset_dict  != {}:

                dataset_name = self.gui.moltrack_dataset_selector.currentText()

                if dataset_name in self.dataset_dict.keys():

                    data_dict = self.dataset_dict[dataset_name].copy()

                    path = data_dict["path"]

                    if type(path) == list:
                        current_step = self.viewer.dims.current_step[0]
                        path = path[current_step]

                    file_name = os.path.basename(path)

                    overlay_string = ""
                    overlay_string += f"File: {file_name}\n"

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

                if dataset_name in self.localisation_dict.keys():
                    localisation_dict = self.localisation_dict[dataset_name].copy()

                    if "localisations" in localisation_dict.keys():
                        locs = localisation_dict["localisations"]

                        if active_frame in locs.frame:
                            frame_locs = locs[locs.frame == active_frame].copy()
                            render_locs = np.vstack((frame_locs.y, frame_locs.x)).T.tolist()

                            vis_mode = self.gui.picasso_vis_mode.currentText()
                            vis_size = float(self.gui.picasso_vis_size.currentText())
                            vis_opacity = float(self.gui.picasso_vis_opacity.currentText())
                            vis_edge_width = float(self.gui.picasso_vis_edge_width.currentText())

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

                                self.loc_layer = self.viewer.add_points(render_locs,
                                    ndim=2, edge_color="red", face_color=[0, 0, 0, 0],
                                    opacity=vis_opacity, name="localisations", symbol=symbol,
                                    size=vis_size, edge_width=vis_edge_width, )

                                update_vis = True

                            else:
                                if self.verbose:
                                    print("Updating fiducial data")

                                self.loc_layer.data = render_locs
                                self.loc_layer.selected_data = []

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

            self.gui.stormtracker_threshold_label.hide()
            self.gui.stormtracker_threshold.hide()
            self.gui.stormtracker_window_size_label.hide()
            self.gui.stormtracker_window_size.hide()

        else:

            self.gui.picasso_box_size_label.hide()
            self.gui.picasso_box_size.hide()
            self.gui.picasso_min_net_gradient_label.hide()
            self.gui.picasso_min_net_gradient.hide()

            self.gui.stormtracker_threshold_label.show()
            self.gui.stormtracker_threshold.show()
            self.gui.stormtracker_window_size_label.show()
            self.gui.stormtracker_window_size.show()

    def moltract_translation(self, event = None, direction = "left"):

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

            if translation_target in ["Segmentation Image", "Both"]:

                if hasattr(self, "segmentation_image"):

                    image = self.segmentation_image.copy()

                    if len(image.shape) == 2:
                        image = shift(image, shift=shift_vector)
                        self.segmentation_image = image

                    else:

                        for fame_index, frame in enumerate(image):
                            image[fame_index] = shift(frame, shift=shift_vector)
                        self.segmentation_image = image

                    self.draw_segmentation_image()

            if translation_target in ["Segmentations","Both"]:

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









        except:
            print(traceback.format_exc())