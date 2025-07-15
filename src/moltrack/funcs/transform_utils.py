import json
import math
import os
import traceback
from datetime import datetime
from functools import partial

import cv2
import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog

from moltrack.funcs.compute_utils import Worker


def transform_image(img, transform_matrix,
        progress_callback=None):

    w, h = img.shape[-2:]

    n_frames = img.shape[0]
    n_segments = math.ceil(n_frames / 100)
    image_splits = np.array_split(img, n_segments)

    transformed_image = []

    iter = 0

    transform_matrix = np.array(transform_matrix, dtype=np.float64)

    if transform_matrix.shape == (2, 3):
        transform_mode = "affine"
    elif transform_matrix.shape == (3, 3):
        transform_mode = "homography"
    else:
        show_info("Transformation matrix is wrong shape, should be (2,3) or (3,3)")
        return

    for index, image in enumerate(image_splits):

        image = np.moveaxis(image, 0, -1)

        if transform_mode == "homography":
            image = cv2.warpPerspective(image, transform_matrix, (h, w),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif transform_mode == "affine":
            image = cv2.warpAffine(image, transform_matrix, (h, w),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        transformed_image.append(image)
        iter += 250
        progress = int((iter / n_frames) * 100)

        if progress_callback is not None:
            progress_callback(progress)

    transformed_image = np.dstack(transformed_image)
    transformed_image = np.moveaxis(transformed_image, -1, 0)

    return transformed_image


def transform_locs(locs, transform_matrix, progress_callback=None):

    try:

        n_points = len(locs)
        transformed_locs = np.recarray((n_points,), dtype=locs.dtype)
        transform_matrix = np.array(transform_matrix, dtype=np.float64)

        if transform_matrix.shape == (2, 3):
            transform_mode = "affine"
        elif transform_matrix.shape == (3, 3):
            transform_mode = "homography"
        else:
            raise ValueError("Transformation matrix is wrong shape, should be (2, 3) or (3, 3)")

        for i in range(n_points):
            point = np.array([locs.x[i], locs.y[i]])

            if transform_mode == "homography":
                point_homogeneous = np.append(point, 1)
                transformed_point = np.dot(transform_matrix, point_homogeneous)
                transformed_locs.x[i] = transformed_point[0] / transformed_point[2]
                transformed_locs.y[i] = transformed_point[1] / transformed_point[2]
            elif transform_mode == "affine":
                point_homogeneous = np.append(point, 1)
                transformed_point = np.dot(transform_matrix, point_homogeneous)
                transformed_locs.x[i] = transformed_point[0]
                transformed_locs.y[i] = transformed_point[1]

            if progress_callback is not None:
                progress = int(((i + 1) / n_points) * 100)
                progress_callback(progress)

    except:
        print(traceback.format_exc())
        return None

    return transformed_locs



class _transform_utils:

    def update_fret_transform_target_channel(self):

        try:
            dataset = self.gui.tform_compute_dataset.currentText()
            channel = self.gui.tform_compute_channel.currentText()

            if hasattr(self, "dataset_dict") == False:
                return

            if dataset in self.dataset_dict.keys():
                dataset = self.dataset_dict[dataset]
                channel_list = dataset["images"].keys()
            else:
                return

            channel_list = [c for c in channel_list if c != channel]

            self.gui.tform_compute_target_channel.clear()
            self.gui.tform_compute_target_channel.addItems(channel_list)

        except:
            print(traceback.format_exc())

    def import_fret_transform_matrix(self):

        try:

            self.update_ui(init=True)

            desktop = os.path.expanduser("~/Desktop")
            path, filter = QFileDialog.getOpenFileName(self, "Open Files", desktop, "Files (*.txt)")

            if path != "":
                if os.path.isfile(path) == True:
                    if path.endswith(".txt"):
                        with open(path) as f:
                            transform_matrix = json.load(f)

                    transform_matrix = np.array(transform_matrix, dtype=np.float64)

                    if transform_matrix.shape == (3, 3):
                        self.transform_matrix = transform_matrix

                        show_info("Imported Homography transformation matrix")
                        print(f"{transform_matrix}")

                    elif transform_matrix.shape == (2, 3):
                        self.transform_matrix = transform_matrix

                        show_info("Imported Affine transformation matrix")
                        print(f"{transform_matrix}")

                    else:
                        show_info("Transformation matrix is wrong shape, should be (3,3) or (2,3)")
            else:
                show_info("No file selected")

            self.update_ui(init=False)

        except:
            print(traceback.format_exc())
            self.update_ui(init=False)

    def compute_fret_transform_matrix(self):

        try:
            if self.dataset_dict != {}:

                dataset_name = self.gui.tform_compute_dataset.currentText()
                dst_channel = self.gui.tform_compute_channel.currentText()
                src_channel = self.gui.tform_compute_target_channel.currentText()

                src_locs = self.get_locs(dataset_name, src_channel)
                dst_locs = self.get_locs(dataset_name, dst_channel)

                if len(dst_locs) == 0 or len(src_locs) == 0:
                    missing_channels = []
                    if len(dst_locs) == 0:
                        missing_channels.append(dst_channel)
                    if len(src_locs) == 0:
                        missing_channels.append(src_channel)
                    show_info(f"Missing localisations for channel(s): {missing_channels}")

                if len(dst_locs) > 0 and len(src_locs) > 0:

                    dst_points = [[loc.x, loc.y] for loc in dst_locs]
                    src_points = [[loc.x, loc.y] for loc in src_locs]

                    dst_points = np.array(dst_points).astype(np.float32)
                    src_points = np.array(src_points).astype(np.float32)

                    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

                    matches = bf.match(dst_points, src_points)
                    matches = sorted(matches, key=lambda x: x.distance)

                    dst_points = np.float32([dst_points[m.queryIdx] for m in matches]).reshape(-1, 2)
                    src_points = np.float32([src_points[m.trainIdx] for m in matches]).reshape(-1, 2)

                    if len(dst_points) == 1 or len(dst_points) == 2:

                        dst_point = dst_points[0]
                        src_point = src_points[0]

                        # Calculate translation vector
                        tx = dst_point[0] - src_point[0]
                        ty = dst_point[1] - src_point[1]

                        # Translation matrix
                        self.transform_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                        print(f"Affine Transform Matrix:\n {self.transform_matrix}")

                    elif len(dst_points) == 3:
                        self.transform_matrix = cv2.getAffineTransform(src_points, dst_points)
                        print(f"Affine Transform Matrix:\n {self.transform_matrix}")

                    elif len(dst_points) > 3:
                        self.transform_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
                        print(f"Homography Transform Matrix:\n {self.transform_matrix}")

                    if self.gui.save_tform.isChecked():
                        self.save_transform_matrix()

        except:
            print(traceback.format_exc())


    def _apply_fret_transform_matrix_finished(self):

        self.update_ui()
        self.update_active_image()
        self.gui.fret_tform_progressbar.setValue(0)
        show_info("Transformation complete")

    def _apply_fret_transform_matrix(self, progress_callback):

        show_info("Applying transform matrix...")

        try:
            if self.dataset_dict != {}:

                target_channel = self.gui.tform_apply_target.currentText()

                if "donor" in target_channel.lower():
                    target_channels = ["donor", "dd","da"]
                else:
                    target_channels = ["acceptor", "aa", "ad"]

                target_images = []
                total_frames = 0
                iter = 0

                for dataset_name, dataset_dict in self.dataset_dict.items():
                    image_dict = dataset_dict["images"]

                    for channel_name in image_dict.keys():
                        if channel_name.lower() in target_channels:
                            image = image_dict[channel_name]
                            n_frames = image.shape[0]
                            total_frames += n_frames
                            target_images.append({"dataset_name": dataset_name,"channel_name": channel_name})

                for i in range(len(target_images)):

                    dataset_name = target_images[i]["dataset_name"]
                    channel_name = target_images[i]["channel_name"]



                    image_dict = self.dataset_dict[dataset_name]["images"]
                    img = image_dict[channel_name].copy()

                    def transform_progress(progress):
                        nonlocal iter
                        iter += progress
                        progress = int((iter / total_frames) * 100)
                        progress_callback.emit(progress)

                    img = transform_image(img, self.transform_matrix, progress_callback=transform_progress)
                    self.dataset_dict[dataset_name]["images"][channel_name] = img.copy()

                    locs = self.get_locs(dataset_name, channel_name)

                    if len(locs) > 0:
                        transformed_locs = transform_locs(locs, self.transform_matrix, progress_callback=transform_progress)
                        if transformed_locs is not None:
                            self.localisation_dict[dataset_name][channel_name]["localisations"] = transformed_locs

                    tracks = self.get_tracks(dataset_name, channel_name)

                    if len(tracks) > 0:
                        transformed_tracks = transform_locs(tracks, self.transform_matrix, progress_callback=transform_progress)
                        if transformed_tracks is not None:
                            self.localisation_dict[dataset_name][channel_name]["tracks"] = transformed_tracks

        except:
            print(traceback.format_exc())

    def apply_fret_transform_matrix(self):

        try:
            if self.dataset_dict != {}:

                if hasattr(self, "transform_matrix") == False:
                    show_info("No transform matrix loaded.")
                else:
                    if self.transform_matrix is None:
                        show_info("No transform matrix loaded.")

                    else:
                        self.update_ui(init=True)

                        worker = Worker(self._apply_fret_transform_matrix)
                        worker.signals.progress.connect(partial(self.moltrack_progress,
                            progress_bar=self.gui.fret_tform_progressbar, ))
                        worker.signals.finished.connect(self._apply_fret_transform_matrix_finished)
                        worker.signals.error.connect(self.update_ui)
                        self.threadpool.start(worker)

        except:
            self.update_ui()

            print(traceback.format_exc())


    def save_transform_matrix(self):

        try:

            if self.transform_matrix is not None:

                # get save file name and path
                date = datetime.now().strftime("%y%m%d")
                file_name = f'moltrack_transform_matrix-{date}.txt'

                dataset_name = self.gui.tform_compute_dataset.currentText()

                path = self.dataset_dict[dataset_name]["path"]

                if type(path) == list:
                    path = path[0]

                path_directory = os.path.dirname(path)

                tform_path = os.path.join(path_directory, file_name)

                tform_path = QFileDialog.getSaveFileName(self, 'Save transform matrix', tform_path, 'Text files (*.txt)')[0]

                if tform_path != "":

                    with open(tform_path, 'w') as filehandle:
                        json.dump(self.transform_matrix.tolist(), filehandle)

                    show_info(f"Saved transform matrix to {tform_path}")

        except:
            print(traceback.format_exc())
