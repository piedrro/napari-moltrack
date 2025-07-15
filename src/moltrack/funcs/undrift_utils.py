from napari.viewer import Viewer
from typing import TYPE_CHECKING
import traceback
from functools import partial
from moltrack.funcs.compute_utils import Worker


if TYPE_CHECKING:
    from moltrack._widget import MolTrack

class MoltrackUndrifting:
    def __init__(self, viewer: Viewer, parent: "MolTrack"):
        self.viewer = viewer
        self.moltrack = parent

    @property
    def gui(self):
        return self.moltrack.gui

    @property
    def dataset_dict(self):
        return self.moltrack.dataset_dict

    @property
    def localisation_dict(self):
        return self.moltrack.localisation_dict

    def aim_undrift(self):

        try:

            dataset = self.gui.undrift_dataset.currentText()
            channel = self.gui.undrift_channel.currentText()
            segmentation = self.gui.aim_segmentation.value()
            intersect_d = self.gui.aim_intersect_d.value()
            roi_r = self.gui.aim_roi_r.value()

            if dataset == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [dataset]

            undrift_dict = {}

            for dataset in dataset_list:

                loc_dict = self.localisation_dict[dataset][channel]

                if "localisations" not in loc_dict.keys():
                    continue

                locs = loc_dict["localisations"]

                n_locs = len(locs)

                if n_locs > 0 and loc_dict["fitted"] == True:

                    n_frames, height, width = self.dataset_dict[dataset][channel.lower()]["data"].shape
                    pixel_size = self.dataset_dict[dataset][channel.lower()]["pixel_size"]
                    picasso_info = [{'Frames': n_frames, 'Height': height,
                                     'Width': width, 'Pixelsize': pixel_size}, {}]
                    undrift_dict[dataset] = {"loc_dict": loc_dict, "n_locs": n_locs,
                                             "picasso_info": picasso_info,
                                             "channel": channel.lower(), "dataset": dataset}
                else:
                    self.moltrack.molseeq_notification("No fitted localizations found for dataset: " + dataset)

            if undrift_dict != {}:
                self.moltrack.update_ui(init=True)

                self.worker = Worker(self._undrift,
                                     mode = 'AIM',
                                     undrift_dict=undrift_dict,
                                     segmentation=segmentation,
                                     intersect_d = intersect_d,
                                     roi_r = roi_r)
                self.worker.signals.progress.connect(partial(self.moltrack.moltrack_progress, progress_bar=self.gui.aim_progressbar))
                self.worker.signals.finished.connect(self._undrift_images_finished)
                self.moltrack.threadpool.start(self.worker)

        except:
            self.moltrack.update_ui()
            print(traceback.format_exc())
            pass

    def rcc_undrift(self):

        try:

            dataset = self.gui.undrift_dataset_selector.currentText()
            channel = self.gui.undrift_channel_selector.currentText()
            segmentation = self.gui.undrift_segmentation.value()

            if dataset == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [dataset]

            undrift_dict = {}

            for dataset in dataset_list:

                loc_dict = self.localisation_dict[dataset][channel]

                if "localisations" not in loc_dict.keys():
                    continue

                locs = loc_dict["localisations"]

                n_locs = len(locs)

                if n_locs > 0 and loc_dict["fitted"] == True:

                    n_frames,height,width = self.dataset_dict[dataset][channel.lower()]["data"].shape
                    picasso_info = [{'Frames': n_frames, 'Height': height, 'Width': width}, {}]

                    undrift_dict[dataset] = {"loc_dict": loc_dict, "n_locs": n_locs,
                                             "picasso_info": picasso_info,
                                             "channel": channel.lower(), "dataset": dataset}
                else:
                    self.moltrack.moltrack_notification("No fitted localizations found for dataset: " + dataset)

            if undrift_dict != {}:

                self.moltrack.update_ui(init=True)

                self.worker = Worker(self._undrift,
                                     mode = 'RCC',
                                     undrift_dict=undrift_dict,
                                     segmentation=segmentation)
                self.worker.signals.progress.connect(partial(self.moltrack.moltrack_progress,
                                                             progress_bar=self.gui.undrift_progressbar))
                self.worker.signals.finished.connect(self._undrift_images_finished)
                self.moltrack.threadpool.start(self.worker)

        except:
            self.moltrack.update_ui()
            print(traceback.format_exc())
            pass

    def _undrift_images_finished(self):

        try:

            self.moltrack.image_layer.data = self.dataset_dict[self.moltrack.active_dataset][self.moltrack.active_channel]["data"]

            self.undrift_localisations()
            self.moltrack.draw_localisations(update_vis=True)

            for layer in self.viewer.layers:
                layer.refresh()

            self.moltrack.update_ui()

        except:
            print(traceback.format_exc())
            self.moltrack.update_ui()