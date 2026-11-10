from molcore import ExecutionContext, PartitionSelection
import traceback
import numpy as np
from napari.utils.notifications import show_info
from shapely.geometry import Point, Polygon
from moltrack.funcs.compute_utils import Worker
from moltrack.model_edits import compute_merge, prepare_merge, publish_merge
from moltrack.model_views import refresh_row_projections
from moltrack.funcs.filter_range_utils import (
    HistogramFilterRange,
    clear_filter_plot,
    reset_filter_limits,
)


class _loc_filter_utils:
    def get_shapes(self, type="segmentations", flipxy=True, polygon=True):
        try:
            shape_data = []
            layer = None
            if type.lower() == "segmentations":
                if hasattr(self, "segLayer"):
                    layer = self.segLayer
            elif type.lower() == "cells":
                if hasattr(self, "cellLayer"):
                    layer = self.cellLayer
            if layer is None:
                return
            shapes = layer.data.copy()
            shape_type = layer.shape_type.copy()
            if len(shapes) == 0:
                return
            for shape_index, shape in enumerate(shapes):
                if shape_type[shape_index] == "polygon":
                    shape_data.append(shape)
            if flipxy:
                shape_data = [np.fliplr(poly) for poly in shape_data]
            if polygon:
                shape_data = [
                    Polygon(poly) for poly in shape_data if len(poly) > 3
                ]
        except:
            print(traceback.format_exc())
        return shape_data

    def remove_seglocs(self, viewer=None):
        try:
            segchannel = self.gui.remove_seglocs_segmentation.currentText()
            dataset = self.gui.remove_seglocs_dataset.currentText()
            channel = self.gui.remove_seglocs_channel.currentText()
            polygons = self.get_shapes(segchannel, flipxy=True, polygon=True)
            if not polygons:
                return
            removed = 0
            column = segchannel[:-1].lower() + "_index"
            for partition in self.session.selected_partitions(
                dataset, channel
            ):
                rows = self.session.query_localizations(*partition)
                labels = [
                    next(
                        (
                            i
                            for i, poly in enumerate(polygons)
                            if poly.contains(Point(x, y))
                        ),
                        -1,
                    )
                    for x, y in zip(rows.x, rows.y, strict=True)
                ]
                keep = np.asarray(labels) >= 0
                self.session.retain_rows(*partition, rows.id.to_numpy()[keep])
                self.session.update_measurements(
                    rows.id.to_numpy()[keep],
                    {column: np.asarray(labels)[keep]},
                )
                removed += int((~keep).sum())
            show_info(f"Removed {removed} localisations.")
            refresh_row_projections(self, localizations=True)
            self.update_filter_criterion()
            self.update_criterion_ranges()
        except Exception:
            print(traceback.format_exc())

    def get_locs(
        self, dataset, channel, return_dict=False, include_metadata=True
    ):
        from moltrack.model_views import gui_query

        return gui_query(
            self.session,
            dataset,
            channel,
            tracks=False,
            return_dict=return_dict,
            include_metadata=include_metadata,
        )

    def pixseq_filter_localisations(self, viewer=None):
        try:
            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            criterion = self.gui.filter_criterion.currentText()
            min_value = self.gui.filter_min.value()
            max_value = self.gui.filter_max.value()
            subtract_background = self.gui.filter_subtract_bg.isChecked()
            if criterion not in self.moltrack_metrics.keys():
                return
            criterion = self.moltrack_metrics[criterion]
            n_removed = 0
            loc_data = self.get_locs(dataset, channel, return_dict=True)
            for dat in loc_data:
                dataset_name = dat["dataset"]
                channel_name = dat["channel"]
                locs = dat["localisations"]
                if len(locs) > 0:
                    columns = list(locs.dtype.names)
                    if criterion in columns:
                        self.gui.filter_localisations.setEnabled(False)
                        n_locs = len(locs)
                        if min_value >= max_value:
                            locs = locs[:0]
                        elif (
                            subtract_background
                            and criterion + "_bg" in locs.dtype.names
                        ):
                            values = locs[criterion] - locs[criterion + "_bg"]
                            locs = locs[
                                np.isfinite(values)
                                & (values > min_value)
                                & (values < max_value)
                            ]
                        else:
                            from molcore import FilterSettings
                            from molcore.filtering import filter_results

                            result = filter_results(
                                self.session.model.snapshot(),
                                PartitionSelection(
                                    *self.session.partition(
                                        dataset_name, channel_name
                                    )
                                ),
                                settings=FilterSettings(
                                    metric=criterion,
                                    minimum=np.nextafter(
                                        float(min_value), np.inf
                                    ),
                                    maximum=np.nextafter(
                                        float(max_value), -np.inf
                                    ),
                                ),
                            )
                            locs = locs[
                                np.isin(locs["id"], result.selected_ids)
                            ]
                        n_filtered = len(locs)
                        if n_filtered < n_locs:
                            n_removed += n_locs - n_filtered
                            self.session.retain_rows(
                                dataset_name, channel_name, locs["id"]
                            )
            if n_removed:
                refresh_row_projections(
                    self, localizations=True, update_vis=True
                )
            self.update_criterion_ranges()
            show_info(f"Filtered {n_removed} localisations.")
            self.gui.filter_localisations.setEnabled(True)
        except:
            self.gui.filter_localisations.setEnabled(True)
            print(traceback.format_exc())

    def update_filter_criterion(self, viewer=None):
        try:
            columns = []
            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            selector = self.gui.filter_criterion
            locs = self.get_locs(dataset, channel)
            if len(locs) > 0:
                for metric_name, metric in self.moltrack_metrics.items():
                    if metric in list(locs.dtype.names):
                        columns.append(metric_name)
            selector.clear()
            if len(columns) > 0:
                selector.addItems(columns)
        except:
            print(traceback.format_exc())

    def update_criterion_ranges(self, viewer=None, plot=True):
        try:
            clear_filter_plot(self.filter_graph_canvas)
            reset_filter_limits([], self.gui.filter_min, self.gui.filter_max)
            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            criterion_name = self.gui.filter_criterion.currentText()
            subtract_background = self.gui.filter_subtract_bg.isChecked()
            if criterion_name not in self.moltrack_metrics.keys():
                return
            criterion = self.moltrack_metrics[criterion_name]
            if "pixel" not in criterion.lower():
                subtract_background = False
                self.gui.filter_subtract_bg.blockSignals(True)
                self.gui.filter_subtract_bg.setChecked(False)
                self.gui.filter_subtract_bg.setEnabled(False)
                self.gui.filter_subtract_bg.blockSignals(False)
            else:
                self.gui.filter_subtract_bg.setEnabled(True)
            locs = self.get_locs(dataset, channel)
            if len(locs) > 0:
                columns = list(locs.dtype.names)
                if criterion in columns:
                    values = locs[criterion]
                    if subtract_background:
                        bg_criterion = criterion + "_bg"
                        if bg_criterion in columns:
                            bg_values = locs[bg_criterion]
                            values = values - bg_values
                    values = values[np.isfinite(values)]
                    if values.dtype in [
                        np.float32,
                        np.float64,
                        np.int32,
                        np.int64,
                        np.uint32,
                        np.uint64,
                    ]:
                        if not len(values):
                            return
                        if plot:
                            self.plot_filter_graph(criterion, values)
                        else:
                            reset_filter_limits(
                                values, self.gui.filter_min, self.gui.filter_max,
                            )
        except:
            print(traceback.format_exc())

    def plot_filter_graph(self, criterion="", values=None):
        try:
            clear_filter_plot(self.filter_graph_canvas)
            values = np.asarray(values) if values is not None else np.array([])
            values = values[np.isfinite(values)]
            reset_filter_limits(
                values, self.gui.filter_min, self.gui.filter_max,
            )
            if len(values) > 0:
                ax = self.filter_graph_canvas.addPlot()
                y, x = np.histogram(values, bins=100)
                ax.plot(
                    x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 75)
                )
                ax.setLabel("bottom", f"{criterion} values")
                ax.setLabel("left", "Frequency")
                HistogramFilterRange(
                    self.filter_graph_canvas, ax,
                    self.gui.filter_min, self.gui.filter_max,
                )
        except:
            print(traceback.format_exc())

    def merge_localisations_finished(self):
        self.update_ui()

    def process_merge_localisations(self, pending):
        try:
            publish_merge(self.session, pending)
            refresh_row_projections(self, localizations=True)
        except Exception as exc:
            show_info(str(exc))

    def merge_localisations(self):
        try:
            dataset = self.gui.merge_locs_dataset.currentText()
            min_distance = self.gui.merge_locs_min_distance.value()
            locs = self.get_locs(dataset, "All Channels")
            if len(locs) == 0:
                return
            self.update_ui(init=True)
            prepared = prepare_merge(
                self.session,
                dataset,
                min_distance,
                cancel_event=self.stop_event,
            )
            worker = Worker(compute_merge, prepared)
            worker.signals.result.connect(self.process_merge_localisations)
            worker.signals.finished.connect(self.merge_localisations_finished)
            self.threadpool.start(worker)
        except:
            print(traceback.format_exc())
            self.update_ui()
