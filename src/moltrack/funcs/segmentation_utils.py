import traceback
import warnings
from moltrack.funcs.compute_utils import Worker
from napari.utils.notifications import show_info
from torch.cuda import empty_cache
import math
import numpy as np
import os
from functools import partial
import cv2
from qtpy.QtWidgets import QFileDialog
from shapely.geometry import Point, Polygon

class _segmentation_utils:

    def load_cellpose_model(self, path = None):

        self.widget_notifications = True

        cellpose_model_names = ["bact_phase_cp", "bact_fluor_cp", "plant_cp", "worm_cp", "cyto2_omni",
                                "bact_phase_omni", "bact_fluor_omni", "plant_omni", "worm_omni",
                                "worm_bact_omni", "worm_high_res_omni", "cyto", "cyto2", "nuclei", ]

        file_path = os.path.expanduser("~/Desktop")

        if type(path) != str:
            path, _ = QFileDialog.getOpenFileName(self, "Open File", file_path, "Cellpose Models (*)")

        if path != "":
            model_name = os.path.basename(path)

            if "torch" in model_name:
                model_name = model_name.split("torch")[0]

            if ("cellpose_" in model_name or "omnipose_" in model_name or model_name in cellpose_model_names):

                if os.path.isfile(path):
                    self.cellpose_custom_model_path = path
                    self.gui.cellpose_model.setCurrentIndex(13)

                    if "_omni" in model_name or "omnipose_" in model_name:
                        if self.widget_notifications:
                            show_info(f"Selected Omnipose model: {model_name}")
                    else:
                        if self.widget_notifications:
                            show_info(f"Selected Cellpose model: {model_name}")

                else:
                    if self.widget_notifications:
                        show_info("Custom Cellpose model path is invalid")

            else:
                if self.widget_notifications:
                    show_info("Custom Cellpose model path is invalid")

        else:
            if self.widget_notifications:
                show_info("Custom Cellpose model path is invalid")

    def load_cellpose_dependencies(self):

        if self.widget_notifications:
            show_info("Loading Cellpose dependencies [pytorch]")

        import torch

        if self.widget_notifications:
            show_info("Loading Cellpose dependencies [cellpose]")

        from cellpose import models
        from cellpose.dynamics import labels_to_flows

        gpu = False

        if torch.cuda.is_available() and self.cellpose_usegpu.isChecked():
            if self.widget_notifications:
                show_info("Cellpose Using GPU")
            gpu = True
            torch.cuda.empty_cache()
        else:
            if self.widget_notifications:
                show_info("Cellpose Using CPU")

        return gpu, models, labels_to_flows

    def load_omnipose_dependencies(self):
        if self.widget_notifications:
            show_info("Loading Omnipose dependencies [pytorch]")

        import torch

        if self.widget_notifications:
            show_info("Loading Omnipose dependencies [omnipose]")

        from omnipose.core import labels_to_flows

        if self.widget_notifications:
            show_info("Loading Omnipose dependencies [cellpose_omni]")

        from cellpose_omni import models

        gpu = False

        if torch.cuda.is_available() and self.cellpose_usegpu.isChecked():
            if self.widget_notifications:
                show_info("Omnipose Using GPU")
            gpu = True
            torch.cuda.empty_cache()
        else:
            if self.widget_notifications:
                show_info("Omnipose Using CPU")

        return gpu, models, labels_to_flows

    def initialise_segmentation_model(self, model_type="custom", model_path=None,
            diameter=15, mode="eval"):

        try:
            model = None
            gpu = False
            omnipose_model = False
            labels_to_flows = None

            if model_type == "custom":
                if model_path not in ["", None] and os.path.isfile(model_path) == True:
                    model_name = os.path.basename(model_path)

                    if "omnipose_" in model_name and mode == "eval":
                        omnipose_model = True

                        gpu, models, labels_to_flows = self.load_omnipose_dependencies()

                        if self.widget_notifications:
                            show_info(f"Loading Omnipose Model: {os.path.basename(model_path)}")

                        model = models.CellposeModel(pretrained_model=str(model_path),
                            diam_mean=int(diameter), model_type=None, gpu=bool(gpu), net_avg=False, )

                    elif "cellpose_" in model_name:
                        omnipose_model = False

                        gpu, models, labels_to_flows = self.load_cellpose_dependencies()

                        if self.widget_notifications:
                            show_info(f"Loading Cellpose Model: {os.path.basename(model_path)}")

                        model = models.CellposeModel(pretrained_model=str(model_path),
                            diam_mean=int(diameter), model_type=None, gpu=bool(gpu), # net_avg=False,
                        )

                    else:
                        model, gpu, omnipose_model, labels_to_flows = (None, None, True, None,)

                else:
                    if self.widget_notifications:
                        show_info("Please select valid Cellpose Model")

            else:
                if "_omni" in model_type and mode == "eval":
                    omnipose_model = True
                    if self.widget_notifications:
                        show_info(f"Loading Omnipose Model: {model_type}")

                    gpu, models, labels_to_flows = self.load_omnipose_dependencies()
                    model = models.CellposeModel(gpu=bool(gpu), model_type=str(model_type))

                elif "omni" not in model_type:
                    omnipose_model = False

                    gpu, models, labels_to_flows = self.load_cellpose_dependencies()

                    if self.widget_notifications:
                        show_info(f"Loading Cellpose Model: {model_type}")

                    model = models.CellposeModel(diam_mean=int(diameter), model_type=str(model_type), gpu=bool(gpu), # net_avg=False,
                    )

                else:
                    if self.widget_notifications:
                        show_info(f"Could not load model")

        except:
            print(traceback.format_exc())

        return model, gpu, omnipose_model, labels_to_flows


    def get_segmentation_images(self, mode = "active"):

        images = []

        if self.dataset_dict != {} or hasattr(self, "segmentation_image"):

            dataset = self.gui.cellpose_dataset.currentText()
            current_frame = self.viewer.dims.current_step[0]

            if dataset == "Segmentation Image":
                if hasattr(self, "segmentation_image"):
                    images = self.segmentation_image.copy()

                    if mode == "active":
                        images = [images[current_frame]]
                    else:
                        if len(images.shape) == 3:
                            images = [frame for frame in images]
                        else:
                            images = [images]
            else:
                if mode == "active":
                    images = [self.dataset_dict[dataset]["data"][current_frame]]
                else:
                    images = self.dataset_dict[dataset]["data"].copy()

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
            pass

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

                self.initialise_segLayer(shapes)





        except:
            print(traceback.format_exc())
            pass

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
                        shapes.append(shape)
        except:
            pass

        return shapes

    def run_cellpose_finished(self):

        self.update_ui(init=False)

    def get_gpu_free_memory(self):

        import torch
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved

        return f

    def create_image_batches(self, images, gpu = True, batch_size = None):

        if not batch_size:
            seg_batch_size = self.gui.cellpose_batchsize.currentText()

        if seg_batch_size.lower() == "auto":
            if gpu:
                image_nbytes = images[0].nbytes
                gpu_memory_size = self.get_gpu_free_memory()

                batch_size = int(math.floor(gpu_memory_size / image_nbytes))
            else:
                batch_size = 1
        else:
            batch_size = int(seg_batch_size)

        # split list into batches
        batched_images = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

        return batched_images, batch_size


    def pixseq_segment(self, images = [], mode = "active", progress_callback = None):

        try:
            cellpose_masks = []

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)

                flow_threshold = float(self.gui.cellpose_flowthresh_label.text())
                mask_threshold = float(self.gui.cellpose_maskthresh_label.text())
                min_size = int(self.gui.cellpose_minsize_label.text())
                diameter = int(self.gui.cellpose_diameter_label.text())
                model_type = self.gui.cellpose_model.currentText()
                invert = self.gui.cellpose_invert_images.isChecked()

                if hasattr(self, "cellpose_custom_model_path") == False:
                    model_path = None
                else:
                    model_path = self.cellpose_custom_model_path

                self.widget_notifications = True

                (model, gpu, omnipose_model, labels_to_flows,) = self.initialise_segmentation_model(model_type, model_path, diameter)

                if model != None:

                    masks = []

                    n_images = len(images)
                    n_segmented = 0

                    batched_images, batch_size = self.create_image_batches(images, gpu)
                    n_batches = len(batched_images)

                    for batch_index, batch in enumerate(batched_images):

                        try:
                            if omnipose_model:
                                masks, flow, diam = model.eval(batch, channels=[0,0],
                                    diameter=diameter, mask_threshold=mask_threshold,
                                    flow_threshold=flow_threshold, min_size=min_size,
                                    batch_size=batch_size, omni=True, invert = invert,)
                            else:
                                masks, flow, diam = model.eval(batch, channels=[0, 0],
                                    diameter=diameter, flow_threshold=flow_threshold,
                                    cellprob_threshold=mask_threshold, min_size=min_size,
                                    batch_size=batch_size, invert=invert,)

                            for i, mask in enumerate(masks):
                                masks[i] = self._postpocess_cellpose(mask)

                            n_segmented += len(batch)
                            progress = int(((batch_index + 1) / n_batches) * 100)
                            progress_callback.emit(progress)

                            cellpose_masks.extend(masks)

                        except:
                            masks = [np.zeros_like(img) for img in batch]
                            cellpose_masks.extend(masks)

                if gpu:
                    empty_cache()


        except:
            print(traceback.format_exc())
            pass

        return mode, cellpose_masks

    def _postpocess_cellpose(self, mask):

        try:
            min_seg_size = int(self.cellpose_min_seg_size.currentText())

            post_processed_mask = np.zeros(mask.shape, dtype=np.uint16)

            mask_ids = sorted(np.unique(mask))

            for i in range(1, len(mask_ids)):
                cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                cell_mask[mask == i] = 255

                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                cnt = contours[0]

                area = cv2.contourArea(cnt)

                if area > min_seg_size:
                    post_processed_mask[mask == i] = i
        except:
            post_processed_mask = mask

        return post_processed_mask

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
                        poly = Polygon(seg)
                        polygons.append(poly)

                    elif ndim == 3:

                        seg = seg[:, 1:]
                        seg = np.fliplr(seg)
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
