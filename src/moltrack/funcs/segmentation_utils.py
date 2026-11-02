import os
import traceback
from functools import partial

import cv2
import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog
from shapely.geometry import Polygon

from moltrack.funcs.compute_utils import Worker


class _segmentation_utils:

    def load_cellpose_model(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Open File',
                os.path.expanduser('~/Desktop'), 'Cellpose 4 weights (*)')
        if path and os.path.isfile(path):
            self.cellpose_custom_model_path = path
            self.gui.cellpose_model.setCurrentText('custom')
            show_info(f'Selected Cellpose model: {os.path.basename(path)}')





    def get_segmentation_images(self, mode = "active"):

        images = []

        if self.dataset_dict != {} or hasattr(self, "segmentation_image"):

            dataset = self.gui.cellpose_dataset.currentText()
            channel = self.gui.cellpose_channel.currentText()
            current_frame = self.viewer.dims.current_step[0]

            if dataset == "Segmentation Image":
                if hasattr(self, "segmentation_image"):
                    data = self.segmentation_image.copy()

                    if mode == "active":
                        if len(data.shape) == 3:
                            if data.shape[0] > 1:
                                if current_frame < data.shape[0]:
                                    images = [data[current_frame]]
                            else:
                                images = [data[0]]
                    else:
                        if len(data.shape) == 3:
                            images = [frame for frame in data]
                        else:
                            images = [data]
            else:
                data = self.dataset_dict[dataset]["images"][channel]
                if mode == "active":
                    if len(data.shape) == 3:
                        if current_frame < data.shape[0]:
                            images = [data[current_frame]]
                else:
                    images = data.copy()

                    if len(images.shape) == 3:
                        images = [frame for frame in images]
                    else:
                        images = [images]

        return images


    def initialise_cellpose(self, mode = "active"):

        try:

            images = self.get_segmentation_images(mode = mode)

            if len(images) > 0:

                self.update_ui(init=True)

                self.worker = Worker(self.pixseq_segment, images = images, mode = mode)
                self.worker.signals.result.connect(self.process_cellpose_result)
                self.worker.signals.finished.connect(self.run_cellpose_finished)
                self.worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.cellpose_progressbar))
                self.threadpool.start(self.worker)

        except:
            self.update_ui(init=False)
            print(traceback.format_exc())

    def process_cellpose_result(self, result):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            mode, masks = result

            if mode == "active":
                shapes = self.mask_to_shape(masks[0])
            else:
                shapes = []
                for i, mask in enumerate(masks):
                    layer_shapes = self.mask_to_shape(mask)
                    shapes.extend(layer_shapes)

            if len(shapes) > 0:

                dataset = self.gui.cellpose_dataset.currentText()

                if dataset == "Segmentation Image":
                    pixel_size = self.segmentation_image_pixel_size
                else:
                    pixel_size = self.dataset_dict[dataset]["pixel_size"]

                self.initialise_segLayer(shapes, pixel_size)





        except:
            print(traceback.format_exc())

    def mask_to_shape(self, mask, frame = None):

        """converts mask to napari shapes"""
        mask = mask.copy()
        shapes = []
        try:

            for label in np.unique(mask):
                if label == 0:
                    continue  # Skip background
                # Create a binary mask for the current object
                binary_mask = np.uint8(mask == label)
                # Find contours using OpenCV
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # Squeeze to remove redundant dimensions and convert to float coordinates
                    shape = contour.squeeze().astype(float)
                    shape = shape.astype(int)

                    shape = np.fliplr(shape)

                    if type(frame) == int:
                        shape = np.insert(shape, 0, frame, axis=1)

                    if contour.shape[0] > 0:  # Ensure that the contour has points
                        if len(shape) > 3:
                            shapes.append(shape)
        except:
            pass

        return shapes

    def run_cellpose_finished(self):

        self.update_ui(init=False)
        self.update_segmentation_combos()




    def pixseq_segment(self, images=(), mode='active', progress_callback=None):
        import torch
        from molcore import CellposeSettings
        from moltrack.backend import segment_images

        model = self.gui.cellpose_model.currentText()
        if model == 'custom':
            model = getattr(self, 'cellpose_custom_model_path', '')
            if not os.path.isfile(model):
                raise ValueError('Please select Cellpose 4 custom weights')
        settings = CellposeSettings(pretrained_model=model,
            diameter=float(self.gui.cellpose_diameter_label.text()) or None,
            flow_threshold=float(self.gui.cellpose_flowthresh_label.text()),
            cellprob_threshold=float(self.gui.cellpose_maskthresh_label.text()),
            min_size=int(self.gui.cellpose_minsize_label.text()), gpu=torch.cuda.is_available())
        if self.gui.cellpose_invert_images.isChecked():
            images = [np.max(image) - image for image in images]
        masks = segment_images(images, settings=settings,
            progress_callback=progress_callback, cancel_event=self.stop_event)
        return mode, masks


    def get_segmentation_polygons(self, segmentation_layer = None):

        polygons = []

        layer_names = [layer.name for layer in self.viewer.layers]

        if segmentation_layer in layer_names and filter:

            shapes = self.viewer.layers[segmentation_layer].data.copy()
            shape_types = self.viewer.layers[segmentation_layer].shape_type

            for seg, seg_type in zip(shapes, shape_types):
                if seg_type == "polygon":
                    ndim = seg.shape[1]

                    if ndim == 2:

                        seg = np.fliplr(seg)
                        if len(seg) > 3:
                            poly = Polygon(seg)
                            polygons.append(poly)

                    elif ndim == 3:

                        seg = seg[:, 1:]
                        seg = np.fliplr(seg)
                        if len(seg) > 3:
                            poly = Polygon(seg)
                            polygons.append(poly)

        return polygons

    def dilate_segmentations(self, viewer = None, simplify = True):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            buffer = float(self.gui.dilatation_size.value())

            if "Segmentations" in layer_names:

                self.update_ui(init=True)

                segLayer = self.segLayer

                segmentations = segLayer.data.copy()
                shape_types = segLayer.shape_type

                segmentations = [seg for seg, shape in zip(segmentations, shape_types) if shape == "polygon"]
                segmentations = [seg for seg in segmentations if len(seg) > 4]

                for index, seg in enumerate(segmentations):

                    try:

                        ndim = seg.shape[1]

                        if ndim == 2:

                            seg = np.fliplr(seg)
                            poly = Polygon(seg)
                            poly = poly.buffer(buffer)

                            if simplify == True:
                                poly = poly.simplify(0.1)

                            seg = np.array(poly.exterior.coords)
                            seg = np.fliplr(seg)
                            seg = seg.astype(float)
                            seg = seg[:-1]
                            segmentations[index] = seg

                        elif ndim == 3:

                            frame = int(seg[0, 0])
                            seg = seg[:, 1:]
                            seg = np.fliplr(seg)
                            poly = Polygon(seg)
                            poly = poly.buffer(buffer)

                            if simplify == True:
                                poly = poly.simplify(0.1)

                            seg = np.array(poly.exterior.coords)
                            seg = np.fliplr(seg)
                            seg = seg.astype(float)
                            seg = seg[:-1]
                            seg = np.insert(seg, 0, frame, axis=1)
                            segmentations[index] = seg

                    except:
                        pass

                # update layer
                segLayer.mode = "pan_zoom"
                segLayer.data = segmentations

                self.update_ui()

        except:
            self.update_ui()
            print(traceback.format_exc())
