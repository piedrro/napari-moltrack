"""Existing localization GUI backed by MolCore processing."""
import traceback
from concurrent.futures import CancelledError
from functools import partial

from napari.utils.notifications import show_info

from moltrack.funcs.compute_utils import Worker


def _compute_detection(prepared, cancel_event, progress_callback=None):
    from moltrack.model_processing import compute

    return compute(prepared, progress=progress_callback, cancel_event=cancel_event)


class _picasso_detect_utils:
    def _publish_picasso_result(self, pending):
        from moltrack.model_processing import publish

        try:
            publish(self.session, pending, cancel_event=self.stop_event)
        except CancelledError:
            pass
        except Exception as exc:
            show_info(f"Localization result was not applied: {exc}")

    def _picasso_wrapper_finished(self):

        try:

            dataset_name = self.gui.picasso_dataset.currentText()
            channel_name = self.gui.picasso_channel.currentText()

            if dataset_name == "All Datasets":
                dataset_name = self.active_dataset
            if channel_name == "All Channels":
                channel_name = self.active_channel

            self.update_active_image(dataset=dataset_name,
                channel=channel_name)

            self.draw_localisations()

            if channel_name != "All Channels":
                self.gui.picasso_filter_channel.blockSignals(True)
                self.gui.picasso_filter_channel.setCurrentText(channel_name)
                self.gui.picasso_filter_channel.blockSignals(False)

                self.gui.picasso_render_channel.blockSignals(True)
                self.gui.picasso_render_channel.setCurrentText(channel_name)
                self.gui.picasso_render_channel.blockSignals(False)

                self.gui.tracking_channel.blockSignals(True)
                self.gui.tracking_channel.setCurrentText(channel_name)
                self.gui.tracking_channel.blockSignals(False)

            self.update_filter_criterion()
            self.update_criterion_ranges()

            self.update_ui()

        except:
            print(traceback.format_exc())


    def init_picasso(self, detect = False, fit = False):

        try:
            if bool(self.session.ordinary_datasets()):

                detect_mode = self.gui.smlm_detect_mode.currentText()
                fit_mode = self.gui.smlm_fit_mode.currentText()
                dataset_name = self.gui.picasso_dataset.currentText()
                frame_mode = self.gui.picasso_frame_mode.currentText()
                minimise_ram = self.gui.picasso_minimise_ram.isChecked()

                if dataset_name != "":

                    self.gui.picasso_progressbar.setValue(0)
                    self.gui.picasso_detect.setEnabled(False)
                    self.gui.picasso_fit.setEnabled(False)
                    self.gui.picasso_detectfit.setEnabled(False)

                    self.update_ui(init=True)

                    if minimise_ram == True and frame_mode.lower() != "active":
                        self.clear_live_images()

                    if frame_mode.lower() == "active":
                        frame_index = self.viewer.dims.current_step[0]
                    else:
                        frame_index = None

                    from moltrack.model_processing import DetectionSettings, ProcessingRequest, prepare

                    roi = self.generate_roi()
                    frozen_roi = tuple(
                        ((dataset, channel), tuple(tuple(point) for point in points))
                        for dataset, channels in roi.items()
                        for channel, points in channels.items()
                    )
                    settings = DetectionSettings(
                        detect=detect, fit=fit, frame_index=frame_index,
                        threshold=float(self.gui.smlm_threshold.value()),
                        box_size=int(self.gui.picasso_box_size.value()),
                        roi=frozen_roi,
                        remove_overlapping=self.gui.picasso_remove_overlapping.isChecked(),
                        detect_mode=detect_mode,
                        kernel_size=int(self.gui.moltrack_kernel_size.text()),
                        fit_mode=fit_mode,
                    )
                    request = ProcessingRequest('detect', self.session.selected_partitions(
                        self.gui.picasso_dataset.currentText(), self.gui.picasso_channel.currentText()
                    ), settings)
                    prepared = prepare(self.session, request)
                    stop_event = self.stop_event
                    self.worker = Worker(_compute_detection, prepared, stop_event)
                    self.worker.signals.result.connect(self._publish_picasso_result)

                    self.worker.signals.progress.connect(partial(self.moltrack_progress,
                        progress_bar=self.gui.picasso_progressbar))
                    self.worker.signals.finished.connect(self._picasso_wrapper_finished)
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

        except:
            print(traceback.format_exc())

            self.update_ui()


    def generate_roi(self):

        if self.verbose:
            print("Generating ROI")

        border_width = self.gui.picasso_roi_border_width.text()
        window_cropping = self.gui.picasso_window_cropping .isChecked()

        roi_dict = {}

        try:

            generate_roi = False

            if window_cropping:
                layers_names = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisations"]]

                crop = self.viewer.layers[layers_names[0]].corner_pixels[:, -2:]
                [[y1, x1], [y2, x2]] = crop

                generate_roi = True

            else:

                if type(border_width) == str:
                    border_width = int(border_width)
                    if border_width > 0:
                        generate_roi = True
                elif type(border_width) == int:
                    if border_width > 0:
                        generate_roi = True

            if generate_roi:

                dataset = self.gui.picasso_dataset.currentText()
                channel = self.gui.picasso_channel.currentText()

                if dataset == "All Datasets":
                    dataset_list = list(self.session.ordinary_datasets())
                else:
                    dataset_list = [dataset]

                for dataset_name in dataset_list:

                    if channel == "All Channels":
                        channel_list = list(self.session.ordinary_channels(dataset_name))
                    else:
                        channel_list = [channel]

                    for channel_name in channel_list:

                        if dataset_name not in roi_dict:
                            roi_dict[dataset_name] = {}
                        if channel_name not in roi_dict[dataset_name].keys():
                            roi_dict[dataset_name][channel_name] = {}

                        image_shape = self.session.image(dataset_name, channel_name).shape

                        frame_shape = image_shape[1:]

                        if window_cropping:

                            border_width = int(border_width)

                            if x1 < border_width:
                                x1 = border_width
                            if y1 < border_width:
                                y1 = border_width
                            if x2 > frame_shape[1] - border_width:
                                x2 = frame_shape[1] - border_width
                            if y2 > frame_shape[0] - border_width:
                                y2 = frame_shape[0] - border_width

                            roi = [[y1, x1], [y2, x2]]

                        else:

                            roi = [[int(border_width), int(border_width)],
                                   [int(frame_shape[0] - border_width), int(frame_shape[1] - border_width)]]

                        roi_dict[dataset_name][channel_name] = roi

        except:
            print(traceback.format_exc())

        return roi_dict
