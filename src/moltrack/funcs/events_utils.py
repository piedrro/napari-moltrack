
import os.path
import traceback
import numpy as np
from functools import partial, wraps
from qtpy.QtWidgets import (QSlider, QLabel)
import time
from napari.utils.notifications import show_info

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