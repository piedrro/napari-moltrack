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

    def initialise_cellpose_sliders(self):
        for name in ("flowthresh", "maskthresh", "minsize", "diameter"):
            getattr(self.gui, f"cellpose_{name}").valueChanged.connect(
                self.update_cellpose_slider_labels
            )
        self.update_cellpose_slider_labels()

    def update_cellpose_slider_labels(self, value=None):
        for name in ("flowthresh", "maskthresh", "minsize", "diameter"):
            value = getattr(self.gui, f"cellpose_{name}").value()
            if name in ("flowthresh", "maskthresh"):
                value /= 100
            getattr(self.gui, f"cellpose_{name}_label").setText(str(value))

    def _capture_cellpose_settings(self):
        import torch
        from molcore import CellposeSettings
        model = self.gui.cellpose_model.currentText()
        if model == 'custom':
            model = getattr(self, 'cellpose_custom_model_path', '')
            if not os.path.isfile(model):
                raise ValueError('Please select Cellpose 3 custom weights')
        return CellposeSettings(pretrained_model=model,
            diameter=float(self.gui.cellpose_diameter.value()) or None,
            flow_threshold=self.gui.cellpose_flowthresh.value() / 100,
            cellprob_threshold=self.gui.cellpose_maskthresh.value() / 100,
            min_size=int(self.gui.cellpose_minsize.value()), gpu=torch.cuda.is_available(),
            invert=self.gui.cellpose_invert_images.isChecked())

    def _prepare_model_segmentation(self, mode):
        from moltrack.model_processing import ProcessingRequest, SegmentationSettings, prepare
        dataset = self.gui.cellpose_dataset.currentText()
        channel = self.gui.cellpose_channel.currentText()
        if dataset == 'Segmentation Image':
            binding = self.session.annotations['segmentation_image_binding']
            dataset, channel = binding
        frame = int(self.viewer.dims.current_step[0]) if mode == 'active' else None
        if len(self.session._view(dataset, channel)) == 1:
            frame = 0 if mode == 'active' else None
        return prepare(self.session, ProcessingRequest('segmentation', ((dataset, channel),),
            SegmentationSettings(self._capture_cellpose_settings(), frame_index=frame)))

    def load_cellpose_model(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Open File',
                os.path.expanduser('~/Desktop'), 'Cellpose 3 weights (*)')
        if path and os.path.isfile(path):
            self.cellpose_custom_model_path = path
            self.gui.cellpose_model.setCurrentText('custom')
            show_info(f'Selected Cellpose model: {os.path.basename(path)}')


    def initialise_cellpose(self, mode = "active"):

        try:

            from moltrack.model_processing import compute_worker
            prepared = self._prepare_model_segmentation(mode)
            self.update_ui(init=True)
            self.worker = Worker(compute_worker, prepared, self.stop_event)
            self.worker.signals.result.connect(self.process_cellpose_result)
            self.worker.signals.finished.connect(self.run_cellpose_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.worker.signals.progress.connect(partial(self.moltrack_progress,
                progress_bar=self.gui.cellpose_progressbar))
            self.threadpool.start(self.worker)
            return

        except:
            self.update_ui(init=False)
            print(traceback.format_exc())

    def process_cellpose_result(self, result):

        try:

            from moltrack.funcs.compute_utils import publish_model_result
            from moltrack.project_geometry import OUTLINE_SOURCE
            if not publish_model_result(self, result):
                return
            shapes = []
            sources = []
            for dataset, channel, segmentations in result.results:
                view = result.prepared.session._view(dataset, channel)
                for segmentation in segmentations.values():
                    frame = None if result.prepared.request.settings.frame_index is not None else view.channel_index(segmentation.frame)
                    contours = self.mask_to_shape(segmentation.labels, frame=frame)
                    shapes.extend(contours)
                    sources.extend({
                        "acquisition": segmentation.acquisition_id,
                        "channel": segmentation.channel,
                        "frame": segmentation.frame,
                        "space": str(segmentation.space),
                    } for _ in contours)
            dataset = result.prepared.request.partitions[0][0]
            pixel_size = result.prepared.session.metadata(dataset)['pixel_size']
            mask_ids = [mask.id for _, _, masks in result.results for mask in masks.values()]
            self.initialise_segLayer(
                shapes, pixel_size, model_mask_ids=mask_ids,
                shape_properties={OUTLINE_SOURCE: sources},
            )
            return

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
