import traceback
import numpy as np
import cv2
import os
from functools import partial
from qtpy.QtWidgets import QFileDialog
import math
import json
from datetime import datetime
from moltrack.funcs.compute_utils import Worker
from napari.utils.notifications import show_info

class _transform_utils:


    def update_target_channel(self):

        try:
            dataset = self.gui.tform_compute_dataset.currentText()
            channel = self.gui.tform_compute_channel.currentText()
            target_channel = self.gui.tform_compute_target_channel.currentText()

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
            pass

    def import_transform_matrix(self):

        try:

            self.update_ui(init=True)

            desktop = os.path.expanduser("~/Desktop")
            path, filter = QFileDialog.getOpenFileName(self, "Open Files", desktop, "Files (*.txt)")

            if path != "":
                if os.path.isfile(path) == True:
                    if path.endswith(".txt"):
                        with open(path, 'r') as f:
                            transform_matrix = json.load(f)

                    transform_matrix = np.array(transform_matrix, dtype=np.float64)

                    if transform_matrix.shape == (3, 3):
                        self.transform_matrix = transform_matrix

                        show_info(f"Imported transformation matrix")
                        print(f"{transform_matrix}")

                    else:
                        show_info("Transformation matrix is wrong shape, should be (3,3)")
            else:
                show_info("No file selected")

            self.update_ui(init=False)

        except:
            print(traceback.format_exc())
            self.update_ui(init=False)
            pass

    def compute_transform_matrix(self):

        try:
            if self.dataset_dict != {}:

                dataset_name = self.gui.tform_compute_dataset.currentText()
                target_channel = self.gui.tform_compute_channel.currentText()
                reference_channel = self.gui.tform_compute_target_channel.currentText()

                target_locs = self.get_locs(dataset_name, target_channel)
                reference_locs = self.get_locs(dataset_name, reference_channel)

                if len(reference_locs) == 0 or len(target_locs) == 0:
                    missing_channels = []
                    if len(reference_locs) == 0:
                        missing_channels.append(reference_channel)
                    if len(target_locs) == 0:
                        missing_channels.append(target_channel)
                    show_info(f"Missing localisations for channel(s): {missing_channels}")

                if len(reference_locs) > 0 and len(target_locs) > 0:
                    reference_points = [[loc.x, loc.y] for loc in reference_locs]
                    target_points = [[loc.x, loc.y] for loc in target_locs]

                    reference_points = np.array(reference_points).astype(np.float32)
                    target_points = np.array(target_points).astype(np.float32)

                    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

                    matches = bf.match(reference_points, target_points)
                    matches = sorted(matches, key=lambda x: x.distance)

                    reference_points = np.float32([reference_points[m.queryIdx] for m in matches]).reshape(-1, 2)
                    target_points = np.float32([target_points[m.trainIdx] for m in matches]).reshape(-1, 2)

                    self.transform_matrix, _ = cv2.findHomography(target_points, reference_points, cv2.RANSAC)

                    print(f"Transform Matrix:\n {self.transform_matrix}")

                    if self.gui.save_tform.isChecked():
                        self.save_transform_matrix()

        except:
            print(traceback.format_exc())
            pass

    def apply_transform_matrix(self):
        pass



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
            pass