import traceback
from functools import partial
from io import BytesIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info

from moltrack.funcs.compute_utils import Worker


class _pixstats_utils:

    def draw_pixstats_mask(self, mode = "tracks"):

        try:

            if mode == "tracks":

                spot_size = int(self.gui.tracks_pixstats_spot_size.currentText())
                spot_shape = self.gui.tracks_pixstats_spot_shape.currentText()
                background_buffer = int(self.gui.tracks_pixstats_bg_buffer.currentText())
                background_width = int(self.gui.tracks_pixstats_bg_width.currentText())

                canvas = self.tracks_pixstats_canvas

            else:

                spot_size = int(self.gui.locs_pixstats_spot_size.currentText())
                spot_shape = self.gui.locs_pixstats_spot_shape.currentText()
                background_buffer = int(self.gui.locs_pixstats_bg_buffer.currentText())
                background_width = int(self.gui.locs_pixstats_bg_width.currentText())

                canvas = self.locs_pixstats_canvas

            mask, buffer_mask, background_mask = _pixstats_utils.generate_localisation_mask(
                spot_size, spot_shape, background_buffer, background_width)


            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            rgb_mask[:,:,0] = mask * 255
            rgb_mask[:,:,2] = background_mask * 255

            ticks = np.arange(-0.5, mask.shape[1], 1)

            fig, ax = plt.subplots(figsize=(6, 6))

            plt.imshow(rgb_mask, interpolation='none')
            plt.xticks(ticks, [])
            plt.yticks(ticks, [])
            plt.grid(color='black', linestyle='-', linewidth=2)

            plt.tick_params(axis='x', which='both', bottom=False, top=False)
            plt.tick_params(axis='y', which='both', left=False, right=False)

            red_patch = mpatches.Patch(color='red', label='Mask')
            black_patch = mpatches.Patch(color='black', label='Buffer')
            blue_patch = mpatches.Patch(color='blue', label='Background')
            plt.legend(handles=[red_patch, black_patch, blue_patch],
                loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3,
                fontsize=18, columnspacing=0.5)

            buf = BytesIO()
            plt.savefig(buf, format='png',
                bbox_inches='tight', facecolor='black', dpi=300)
            buf.seek(0)
            image = plt.imread(buf)
            plt.close(fig)

            image = np.rot90(image, k=3)
            image = np.fliplr(image)

            canvas.clear()
            canvas.setImage(image)
            canvas.view.autoRange()
            canvas.view.setMouseEnabled(x=False, y=False)

        except:
            print(traceback.format_exc())

    def initialise_pixstats(self, mode="tracks"):

        pixstats_data = []

        if mode == "tracks":

            pixstats_data = self.get_tracks(dataset="All Datasets", channel= "All Channels")

            progressbar = partial(self.moltrack_progress,
                progress_bar=self.gui.tracks_pixstats_progressbar)

            spot_size = int(self.gui.tracks_pixstats_spot_size.currentText())
            spot_shape = self.gui.tracks_pixstats_spot_shape.currentText()
            background_buffer = int(self.gui.tracks_pixstats_bg_buffer.currentText())
            background_width = int(self.gui.tracks_pixstats_bg_width.currentText())

            calculate_fret = self.gui.tracks_pixstats_fret.isChecked()

        if mode == "locs":

            pixstats_data = self.get_locs(dataset="All Datasets", channel= "All Channels")

            progressbar = partial(self.moltrack_progress,
                progress_bar=self.gui.locs_pixstats_progressbar)

            spot_size = int(self.gui.locs_pixstats_spot_size.currentText())
            spot_shape = self.gui.locs_pixstats_spot_shape.currentText()
            background_buffer = int(self.gui.locs_pixstats_bg_buffer.currentText())
            background_width = int(self.gui.locs_pixstats_bg_width.currentText())

            calculate_fret = self.gui.locs_pixstats_fret.isChecked()

        if len(pixstats_data) == 0:
            show_info("No data found")
            return None

        self.update_ui(init=True)

        pixstats_data = pd.DataFrame(pixstats_data)

        channels = pixstats_data["channel"].unique()
        channels = [chan.lower() for chan in channels]

        if calculate_fret:
            if set(["donor", "acceptor"]).issubset(channels):
                calculate_fret = True
                show_info("PixStats FRET calculation enabled")
            elif set(["dd", "da"]).issubset(channels):
                calculate_fret = True
                show_info("PixStats ALEX FRET calculation enabled")
            else:
                show_info("FRET calculation requires Donor+Acceptor or DD+DA channels")
                calculate_fret = False
                self.gui.tracks_pixstats_fret.setChecked(False)

        from moltrack.model_processing import PhotometrySettings, ProcessingRequest, compute_worker, prepare
        partitions = tuple(pixstats_data.groupby(['dataset', 'channel'], sort=False).groups)
        settings = PhotometrySettings(spot_size=spot_size, spot_shape=spot_shape,
            background_buffer=background_buffer, background_width=background_width,
            tracks=mode == 'tracks', calculate_fret=calculate_fret)
        worker = Worker(compute_worker, prepare(self.session,
            ProcessingRequest('photometry', partitions, settings)), self.stop_event)

        worker.signals.progress.connect(progressbar)
        worker.signals.result.connect(self.process_pixstats_result)
        worker.signals.finished.connect(self.compute_pixstats_finished)
        worker.signals.error.connect(self.update_ui)
        self.threadpool.start(worker)


    def compute_pixstats_finished(self):

        self.update_ui()

        self.draw_localisations()

        self.update_filter_criterion()
        self.update_criterion_ranges()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()

        self.update_traces_export_options()

        self.update_trackplot_options()
        self.plot_tracks()


    def process_pixstats_result(self, result):
        from moltrack.funcs.compute_utils import publish_model_result
        publish_model_result(self, result)
        return

    @staticmethod
    def generate_localisation_mask(spot_size, spot_shape="square", buffer_size=0, bg_width=1):

        box_size = spot_size + (bg_width * 2) + (buffer_size * 2)

        # Create a grid of coordinates
        y, x = np.ogrid[:box_size, :box_size]

        # Adjust center based on box size
        if box_size % 2 == 0:
            center = (box_size / 2 - 0.5, box_size / 2 - 0.5)
        else:
            center = (box_size // 2, box_size // 2)

        if spot_shape.lower() == "circle":
            # Calculate distance from the center for circular mask
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Central spot mask
            inner_radius = spot_size // 2
            mask = distance <= inner_radius

            # Buffer mask
            buffer_outer_radius = inner_radius + buffer_size
            buffer_mask = (distance > inner_radius) & (distance <= buffer_outer_radius)

            # Background mask (outside the buffer zone)
            background_outer_radius = buffer_outer_radius + bg_width
            background_mask = (distance > buffer_outer_radius) & (distance <= background_outer_radius)

        elif spot_shape.lower() == "square":
            # Create square mask
            half_size = spot_size // 2
            mask = (abs(x - center[0]) <= half_size) & (abs(y - center[1]) <= half_size)

            # Create square background mask (one pixel larger on each side)
            buffer_mask = (abs(x - center[0]) <= half_size + buffer_size) & (abs(y - center[1]) <= half_size + buffer_size)
            background_mask = (abs(x - center[0]) <= half_size + buffer_size + bg_width) & (abs(y - center[1]) <= half_size + buffer_size + bg_width)
            background_mask = background_mask & ~buffer_mask

        return mask, buffer_mask, background_mask
