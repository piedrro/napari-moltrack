import traceback
import warnings
from functools import partial

import numpy as np
from napari.utils.notifications import show_info
from shapely.geometry import LineString, Polygon

from moltrack.funcs.compute_utils import Worker

warnings.filterwarnings("ignore", category=UserWarning)

class _trackstats_utils:

    def compute_track_stats_finished(self):

        self.update_diffusion_range()
        self.plot_diffusion_graph()

        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()

        self.update_traces_export_options()

        self.update_trackplot_options()
        self.draw_tracks()
        self.plot_tracks()

        self.update_ui()

    def process_track_stats_result(self, track_data):

        if track_data is None:
            return

        from moltrack.model.analysis import publish_analysis
        try:
            return publish_analysis(self.session, track_data)
        except Exception as exc:
            show_info(str(exc))
            return None


    def get_trackstats_polygons(self, type, flipxy=False):

        shape_data = {}

        try:

            layer = None

            if type == "":
                return shape_data

            if type.lower() == "segmentations":
                if hasattr(self, "segLayer"):
                    layer = self.segLayer
            elif type.lower() == "cells":
                if hasattr(self, "cellLayer"):
                    layer = self.cellLayer

            if layer == None:
                return shape_data

            shapes = layer.data.copy()
            shape_type = layer.shape_type.copy()

            if len(shapes) == 0:
                return

            if type.lower() == "segmentations":
                shape_data["polygons"] = []

                for shape_index, shape in enumerate(shapes):
                    if shape_type[shape_index] == "polygon":

                        if flipxy:
                            shape = np.flip(shape, axis=1)

                        shape = Polygon(shape)
                        shape_data["polygons"].append(shape)

            else:
                name_list = self.cellLayer.properties["name"].copy()
                name_list = list(set(name_list))

                if len(name_list) == 0:
                    return

                shape_data["polygons"] = []
                shape_data["midlines"] = []

                for name in name_list:
                    try:
                        cell = self.get_cell(name)
                        if cell is not None:
                            midline = cell["midline_coords"]
                            polygon = cell["polygon_coords"]

                            if flipxy:
                                midline = np.flip(midline, axis=1)
                                polygon = np.flip(polygon, axis=1)

                            midline = LineString(midline)
                            polygon = Polygon(polygon)

                            shape_data["polygons"].append(polygon)
                            shape_data["midlines"].append(midline)

                    except:
                        pass
        except:
            print(traceback.format_exc())

        return shape_data



    def initialise_track_stats(self):
        try:
            from moltrack.model.analysis import prepare_analysis, compute_analysis
            if not len(self.get_tracks("All Datasets", "All Channels")):
                show_info("No tracks found")
                return
            shape_data = None
            if self.tracking_segchannel in ("Segmentations", "Cells"):
                shape_data = self.get_trackstats_polygons(self.tracking_segchannel, flipxy=True)
            prepared = prepare_analysis(self.session,
                fit_lags=self.gui.trackstats_adc_track_length.value(),
                rolling_window=self.gui.trackstats_roll_window_size.value(), shape_data=shape_data,
                cancel_event=self.stop_event)
            self.update_ui(init=True)
            worker = Worker(compute_analysis, prepared)
            worker.signals.result.connect(self.process_track_stats_result)
            worker.signals.finished.connect(self.compute_track_stats_finished)
            worker.signals.progress.connect(partial(self.moltrack_progress, progress_bar=self.gui.track_stats_progressbar))
            self.threadpool.start(worker)
        except Exception:
            print(traceback.format_exc())
            self.update_ui()
