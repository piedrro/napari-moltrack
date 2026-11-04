"""Connect the existing RCC/AIM controls to MolCore drift processing."""
from functools import partial


from napari.utils.notifications import show_info

from moltrack.funcs.compute_utils import Worker
from qtpy.QtCore import QObject


class MoltrackUndrifting(QObject):
    def __init__(self, viewer, parent):
        super().__init__()
        self.viewer = viewer
        self.moltrack = parent

    def aim_undrift(self):
        self._start('aim')

    def rcc_undrift(self):
        self._start('rcc')

    def _start(self, method):
        parent, gui = self.moltrack, self.moltrack.gui
        from molcore import DriftSettings as EstimateSettings
        from moltrack.model_processing import DriftSettings, ProcessingRequest, compute_worker, prepare, selected_partitions
        selected = gui.undrift_dataset.currentText()
        channel = gui.undrift_channel.currentText()
        references = []
        for dataset, chan in selected_partitions(parent.session, selected, channel):
            annotations = parent.session.annotations.get('partitions', {}).get(dataset, {}).get(chan, {})
            if not annotations.get('fitted', False) or not len(parent.session.query_localizations(dataset, chan)):
                show_info(f'No fitted localisations found for dataset: {dataset}')
                continue
            references.append((dataset, chan))
        if not references:
            return
        datasets = {d for d, _ in references}
        partitions = tuple(p for p in selected_partitions(parent.session) if p[0] in datasets)
        settings = DriftSettings(EstimateSettings(method=method,
            segment_frames=int((gui.aim_segmentation if method == 'aim' else gui.undrift_segmentation).value()),
            intersect_distance=float(gui.aim_intersect_d.value()), search_radius=float(gui.aim_roi_r.value())), tuple(references))
        prepared = prepare(parent.session, ProcessingRequest('drift', partitions, settings))
        parent.update_ui(init=True)
        self.worker = Worker(compute_worker, prepared, parent.stop_event)
        bar = gui.aim_progressbar if method == 'aim' else gui.undrift_progressbar
        self.worker.signals.progress.connect(partial(parent.moltrack_progress, progress_bar=bar))
        self.worker.signals.result.connect(self._apply)
        self.worker.signals.error.connect(parent.update_ui)
        self.worker.signals.finished.connect(parent.update_ui)
        parent.threadpool.start(self.worker)
        return
    def _apply(self, pending):
        parent = self.moltrack
        from moltrack.funcs.compute_utils import publish_model_result
        if publish_model_result(parent, pending):
            parent.update_active_image()
            parent.draw_localisations(update_vis=True)
            parent.draw_tracks()
        return
