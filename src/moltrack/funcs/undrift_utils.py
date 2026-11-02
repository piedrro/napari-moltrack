"""Connect the existing RCC/AIM controls to MolCore drift processing."""
from functools import partial
from concurrent.futures import CancelledError

import numpy as np

from napari.utils.notifications import show_info

from moltrack.backend import correct_drift, estimate_drift
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
        if hasattr(parent, 'session'):
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
        selected = gui.undrift_dataset.currentText()
        channel = gui.undrift_channel.currentText()
        datasets = list(parent.dataset_dict) if selected == 'All Datasets' else [selected]
        jobs = []
        for name in datasets:
            entry = parent.localisation_dict.get(name, {}).get(channel, {})
            if not entry.get('fitted') or not len(entry.get('localisations', [])):
                show_info(f'No fitted localisations found for dataset: {name}')
                continue
            images = dict(parent.dataset_dict[name]['images'])
            locs = {c: v.get('localisations') for c, v in parent.localisation_dict.get(name, {}).items()}
            tracks = {c: v.get('tracks') for c, v in parent.tracking_dict.get(name, {}).items()}
            frame_maps = {}
            if parent.dataset_dict[name].get('multichannel_mode') == 'ALEX':
                frame_maps = {c: np.arange(len(movie)) * 2 + (1 if c.upper() in ('DA', 'AA') else 0)
                              for c, movie in images.items()}
            jobs.append((name, channel, images, locs, tracks, frame_maps))
        if not jobs:
            return
        settings = dict(method=method,
            segment_frames=int((gui.aim_segmentation if method == 'aim' else gui.undrift_segmentation).value()),
            intersect_distance=float(gui.aim_intersect_d.value()), search_radius=float(gui.aim_roi_r.value()))
        parent.update_ui(init=True)
        self.worker = Worker(self._compute, jobs, settings)
        bar = gui.aim_progressbar if method == 'aim' else gui.undrift_progressbar
        self.worker.signals.progress.connect(partial(parent.moltrack_progress, progress_bar=bar))
        self.worker.signals.result.connect(self._apply)
        self.worker.signals.error.connect(parent.update_ui)
        self.worker.signals.finished.connect(parent.update_ui)
        parent.threadpool.start(self.worker)

    def _compute(self, jobs, settings, progress_callback=None):
        pending = []
        for name, channel, images, locs, tracks, frame_maps in jobs:
            result = estimate_drift(images[channel], locs[channel], **settings,
                progress_callback=progress_callback, cancel_event=self.moltrack.stop_event)
            output = {}
            for target, movie in images.items():
                options = dict(cancel_event=self.moltrack.stop_event,
                    source_frames=frame_maps.get(channel), target_frames=frame_maps.get(target))
                corrected_movie, corrected_locs = correct_drift(movie, locs.get(target), result, **options)
                corrected_tracks = None
                if tracks.get(target) is not None:
                    _, corrected_tracks = correct_drift(movie, tracks[target], result, images=False, **options)
                output[target] = corrected_movie, corrected_locs, corrected_tracks
            pending.append((name, images, locs, tracks, output, result))
        return pending

    def _apply(self, pending):
        parent = self.moltrack
        from moltrack.model_processing import PendingOperation
        if isinstance(pending, PendingOperation):
            from moltrack.funcs.compute_utils import publish_model_result
            if publish_model_result(parent, pending):
                parent.update_active_image()
                parent.draw_localisations(update_vis=True)
                parent.draw_tracks()
            return
        if parent.stop_event.is_set():
            return
        for name, images, locs, tracks, _, _ in pending:
            if name not in parent.dataset_dict:
                show_info('Dataset removed while estimating drift; results discarded')
                return
            for channel, image in images.items():
                if parent.dataset_dict[name]['images'].get(channel) is not image:
                    show_info('Images changed while estimating drift; results discarded')
                    return
            for channel, rows in locs.items():
                if parent.localisation_dict.get(name, {}).get(channel, {}).get('localisations') is not rows:
                    show_info('Localisations changed while estimating drift; results discarded')
                    return
            for channel, rows in tracks.items():
                if parent.tracking_dict.get(name, {}).get(channel, {}).get('tracks') is not rows:
                    show_info('Tracks changed while estimating drift; results discarded')
                    return
        for name, _, _, _, output, result in pending:
            for channel, (movie, locs, tracks) in output.items():
                parent.dataset_dict[name]['images'][channel] = movie
                if locs is not None:
                    parent.localisation_dict[name][channel]['localisations'] = locs
                if tracks is not None:
                    parent.tracking_dict[name][channel]['tracks'] = tracks
            parent.dataset_dict[name]['drift'] = {'frames': result.frames, 'displacement': result.displacement}
        parent.update_active_image()
        parent.draw_localisations(update_vis=True)
        parent.draw_tracks()
