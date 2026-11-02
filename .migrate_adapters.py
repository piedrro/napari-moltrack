"""One-time mechanical migration of GUI call sites."""
from pathlib import Path
import ast

root = Path('src/moltrack')


def edit(name, replacements, remove=()):
    path = root / name
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in replacements or node.name in remove:
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                changes.append((start, node.end_lineno,
                                replacements.get(node.name, '')))
    for start, end, value in sorted(changes, reverse=True):
        lines[start:end] = [value + '\n'] if value else []
    path.write_text(''.join(lines), encoding='utf-8')


# Bindings only: the UI source and all widgets/layouts stay the same.
for path in root.rglob('*.py'):
    text = path.read_text(encoding='utf-8')
    text = text.replace('from PyQt5', 'from qtpy')
    text = text.replace('from qtpy.QtCore import pyqtSignal, pyqtSlot',
                        'from qtpy.QtCore import Signal, Slot')
    text = text.replace('pyqtSignal(', 'Signal(').replace('@pyqtSlot(', '@Slot(')
    path.write_text(text, encoding='utf-8')


edit('_widget.py', {'check_gpufit_availibility': '''    def configure_picasso_fitting(self):
        from picasso.localize import CUDA_AVAILABLE

        if CUDA_AVAILABLE:
            self.gui.smlm_fit_mode.addItem("Picasso GPU")
            self.gui.smlm_fit_mode.setCurrentIndex(1)'''})
path = root / '_widget.py'
text = path.read_text().replace('self.check_gpufit_availibility()', 'self.configure_picasso_fitting()')
path.write_text(text)

edit('funcs/tracking_utils.py', {'link_locs': '''    def link_locs(self, f, search_range, pos_columns=None,
                  t_column='frame', progress_callback=None, **kwargs):
        from moltrack.backend import link_locs

        if pos_columns not in (None, ['x', 'y'], ['y', 'x']):
            raise ValueError("Tracking requires x/y positions")
        rows = f.rename(columns={t_column: 'frame'}) if t_column != 'frame' else f
        result = link_locs(rows, search_range, memory=kwargs.pop('memory', 0),
                           progress_callback=progress_callback,
                           cancel_event=getattr(self, 'stop_event', None), **kwargs)
        return result.rename(columns={'frame': t_column}) if t_column != 'frame' else result'''})
path = root / 'funcs/tracking_utils.py'
text = path.read_text().replace('import trackpy as tp\n', '')
text = text.replace('                tp.quiet()\n', '')
text = text.replace('progress_callback=progress_callback)\n                    tracks_df',
                    'progress_callback=progress_callback, min_length=min_track_length)\n                    tracks_df')
text = text.replace('memory=memory)\n\n                        particle_array',
                    'memory=memory, min_length=min_track_length)\n\n                        particle_array')
text = text.replace('                tracks_df = tp.filter_stubs(tracks_df, min_track_length).reset_index(drop=True)\n', '')
path.write_text(text)

# Replace numerical transform bodies, preserving public imports used elsewhere.
edit('funcs/transform_utils.py', {}, remove=('transform_image', 'transform_locs'))
path = root / 'funcs/transform_utils.py'
text = path.read_text().replace('import math\n', '')
text = text.replace('from moltrack.funcs.compute_utils import Worker',
                    'from moltrack.funcs.compute_utils import Worker\nfrom moltrack.backend import transform_image, transform_locs\nfrom molcore.alignment import fit_transform')
start = text.index('                    if len(dst_points) == 1 or len(dst_points) == 2:')
end = text.index('                    if self.gui.save_tform.isChecked():', start)
text = text[:start] + '''                    model = 'translation' if len(src_points) < 3 else (
                        'affine' if len(src_points) == 3 else 'homography')
                    self.transform_matrix, _, _ = fit_transform(
                        src_points, dst_points, model=model, ransac_threshold=5.)

''' + text[end:]
text = text.replace('self.localisation_dict[dataset_name][channel_name]["tracks"] = transformed_tracks',
                    'self.tracking_dict[dataset_name][channel_name]["tracks"] = transformed_tracks')
path.write_text(text)

path = root / 'funcs/picasso_render_utils.py'
text = path.read_text().replace('from picasso.render import render', 'from moltrack.backend import render_locs')
text = text.replace('                    viewport=viewport,', '                    image_shape=image_shape,')
text = text.replace('n_rendered_locs, image = render(', 'n_rendered_locs, image = render_locs(')
text = text.replace('                    ang=0,\n', '')
text = text.replace('            h, w = image_shape[-2:]\n\n            viewport = [(float(0), float(0)), (float(h), float(w))]\n', '')
path.write_text(text)

# Keep only GUI entry points; MolCore replaces the detection/fitting worker stack.
path = root / 'funcs/picasso_detect_utils.py'
text = path.read_text()
tree = ast.parse(text)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
keep = {'_picasso_wrapper_finished', 'process_locs', 'init_picasso', 'generate_roi'}
lines = text.splitlines(keepends=True)
methods = [''.join(lines[n.lineno-1:n.end_lineno]) for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in keep]
header = '''"""Existing localization GUI backed by MolCore processing."""
import traceback
from functools import partial

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info

from moltrack.backend import localize_movie
from moltrack.funcs.compute_utils import Worker


class _picasso_detect_utils:
'''
wrapper = '''    def _picasso_wrapper(self, progress_callback, detect, fit,
                         dataset_list=(), channel_list=(), frame_index=None,
                         detect_mode="Picasso", fit_mode="Picasso"):
        box_size = int(self.gui.picasso_box_size.value())
        roi_dict = self.generate_roi()
        pending = []
        for dataset in dataset_list:
            for channel in channel_list:
                movie = self.dataset_dict[dataset]['images'][channel]
                candidates = None
                if not detect:
                    candidates = self.localisation_dict.get(dataset, {}).get(channel, {}).get('localisations')
                    if candidates is None or len(candidates) == 0:
                        continue
                locs = localize_movie(movie,
                    threshold=float(self.gui.smlm_threshold.value()), box_size=box_size,
                    fit=fit, candidates=candidates, frame_index=frame_index,
                    roi=roi_dict.get(dataset, {}).get(channel),
                    remove_overlapping=detect and self.gui.picasso_remove_overlapping.isChecked(),
                    detect_mode=detect_mode, kernel_size=int(self.gui.moltrack_kernel_size.text()),
                    use_gpu=fit_mode == 'Picasso GPU', progress_callback=progress_callback,
                    cancel_event=self.stop_event)
                pending.append((dataset, channel, locs))
        # Commit only after every selected partition has completed successfully.
        for dataset, channel, locs in pending:
            entry = self.localisation_dict.setdefault(dataset, {}).setdefault(channel, {})
            if 'lpx' in locs.dtype.names and 'lpy' in locs.dtype.names:
                data = pd.DataFrame(locs)
                data['sigma'] = np.sqrt((data.lpx ** 2 + data.lpy ** 2) / 2)
                locs = data.to_records(index=False)
            if frame_index is not None and len(entry.get('localisations', [])):
                old = pd.DataFrame(entry['localisations'])
                old = old[old.frame != frame_index]
                locs = pd.concat([old, pd.DataFrame(locs)], ignore_index=True).to_records(index=False)
            entry.update(localisations=locs, fitted=bool(fit), box_size=box_size)
        return pending, bool(fit)
'''
# process_locs was a second, lossy result commit path and is no longer called.
methods = [m for m in methods if not m.startswith('    def process_locs(')]
text = header + '\n\n'.join(methods) + '\n\n' + wrapper
text = text.replace('                smlm_fit_mode = self.gui.smlm_fit_mode.currentText()\n', '')
path.write_text(text)

# Remove the superseded pixel extraction worker and shared-memory plumbing.
edit('funcs/pixstats_utils.py', {'compute_pixstats': '''    def compute_pixstats(self, pixmap_data, mode, calculate_fret, spot_size, spot_shape,
                         background_buffer, background_width, progress_callback=None):
        from moltrack.backend import measure_locs

        measured = []
        for (dataset, channel), rows in pixmap_data.groupby(['dataset', 'channel'], sort=False):
            movie = self.dataset_dict[dataset]['images'][channel]
            measured.append(measure_locs(movie, rows, spot_size=spot_size, spot_shape=spot_shape,
                background_buffer=background_buffer, background_width=background_width,
                progress_callback=progress_callback, cancel_event=self.stop_event))
        results = pd.concat(measured, ignore_index=True) if measured else pd.DataFrame(pixmap_data).copy()
        if calculate_fret:
            results = self.pixstats_calculate_fret(results)
        else:
            results = results.drop(columns=[col for col in results if '_fret' in col])
        return results.to_records(index=False), mode'''},
    remove=('generate_spot_bounds', 'crop_spot_data', 'generate_background_overlap_mask',
            'pixstats_compute_func', 'get_pixstats_compute_jobs'))
path = root / 'funcs/pixstats_utils.py'
text = path.read_text().replace('from concurrent.futures import ProcessPoolExecutor, as_completed\n', '')
text = text.replace('from multiprocessing import Manager, shared_memory\n', '')
path.write_text(text)

readme = Path('README.md')
readme.write_text(readme.read_text().replace('**Picasso**, **GPUfit** and **Trackpy**',
    '**MolCore**, **Picasso** and **Trackpy**'))

# User-authorized deletion, exact files only, no recursive or computed tree deletion.
for name in ('__init__.py', 'version.py', 'gpufit.py', 'Gpufit.dll'):
    path = Path('src/pygpufit') / name
    if path.is_file():
        path.unlink()
