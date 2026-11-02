from pathlib import Path
import ast


def replace_methods(path, replacements, removed=()):
    path = Path(path)
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    changes = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and (node.name in replacements or node.name in removed):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            changes.append((first, node.end_lineno, replacements.get(node.name, '')))
    for first, last, value in sorted(changes, reverse=True):
        lines[first:last] = [value + '\n'] if value else []
    path.write_text(''.join(lines))


replace_methods('src/moltrack/funcs/trackstats_utils.py', {'get_track_stats': '''    @staticmethod
    def get_track_stats(df, min_track_length=4, rolling_window_size=4, shape_data=None):
        from moltrack.backend import track_statistics

        df = df.sort_values('frame', kind='stable').reset_index(drop=True).copy()
        if df.empty:
            return df
        pixel_size = float(df['pixel_size (um)'].iloc[0])
        stats = track_statistics(df, pixel_size=pixel_size,
            time_step=float(df['exposure_time (s)'].iloc[0]),
            fit_lags=min_track_length, rolling_window=rolling_window_size)
        stats['angle'] = _trackstats_utils.calculate_track_angles(df[['x', 'y']].to_numpy())
        stats.update(_trackstats_utils.get_track_shape_stats(df, shape_data, pixel_size))
        for name, value in stats.items():
            df[name] = value
        return df'''}, removed=('calculate_rolling_diffusion',))
path = Path('src/moltrack/funcs/trackstats_utils.py')
path.write_text(path.read_text().replace('import trackpy\n', ''))

replace_methods('src/moltrack/funcs/segmentation_utils.py', {
    'load_cellpose_model': '''    def load_cellpose_model(self, path=None):
        if not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, 'Open File',
                os.path.expanduser('~/Desktop'), 'Cellpose 4 weights (*)')
        if path and os.path.isfile(path):
            self.cellpose_custom_model_path = path
            self.gui.cellpose_model.setCurrentText('custom')
            show_info(f'Selected Cellpose model: {os.path.basename(path)}')''',
    'pixseq_segment': '''    def pixseq_segment(self, images=(), mode='active', progress_callback=None):
        import torch
        from molcore import CellposeSettings
        from moltrack.backend import segment_images

        model = self.gui.cellpose_model.currentText()
        if model == 'custom':
            model = getattr(self, 'cellpose_custom_model_path', '')
            if not os.path.isfile(model):
                raise ValueError('Please select Cellpose 4 custom weights')
        settings = CellposeSettings(pretrained_model=model,
            diameter=float(self.gui.cellpose_diameter_label.text()) or None,
            flow_threshold=float(self.gui.cellpose_flowthresh_label.text()),
            cellprob_threshold=float(self.gui.cellpose_maskthresh_label.text()),
            min_size=int(self.gui.cellpose_minsize_label.text()), gpu=torch.cuda.is_available())
        if self.gui.cellpose_invert_images.isChecked():
            images = [np.max(image) - image for image in images]
        masks = segment_images(images, settings=settings,
            progress_callback=progress_callback, cancel_event=self.stop_event)
        return mode, [self._postpocess_cellpose(mask) for mask in masks]'''},
    removed=('load_cellpose_dependencies', 'load_omnipose_dependencies',
             'initialise_segmentation_model', 'get_gpu_free_memory', 'create_image_batches'))
path = Path('src/moltrack/funcs/segmentation_utils.py')
text = path.read_text().replace('import math\n', '').replace('import warnings\n', '')
text = text.replace('from torch.cuda import empty_cache\n', '')
text = text.replace('self.cellpose_min_seg_size.currentText()', 'self.gui.cellpose_min_seg_size.currentText()')
path.write_text(text)

# Keep the UI source and generated model options synchronized.
import re
models = ['cpsam_v2', 'cpsam', 'cpdino', 'cpdino-vitb', 'custom']
path = Path('src/moltrack/GUI/widget_ui.py')
text = path.read_text()
text = text.replace('        self.cellpose_model.addItem("")\n' * 14,
                    '        self.cellpose_model.addItem("")\n' * len(models))
for index in range(14):
    pattern = rf'^        self\.cellpose_model\.setItemText\({index},.*$'
    value = f'        self.cellpose_model.setItemText({index}, _translate("Frame", "{models[index]}"))' if index < len(models) else ''
    text = re.sub(pattern, value, text, flags=re.M)
path.write_text(text)
path = Path('src/moltrack/GUI/widget_ui.ui')
text = path.read_text()
match = re.search(r'(<widget class="QComboBox" name="cellpose_model">)(.*?)(</widget>)', text, re.S)
if match:
    contents = match.group(2)
    contents = re.sub(r'\s*<item>.*?</item>', '', contents, flags=re.S)
    contents += ''.join(f'\n                <item><property name="text"><string>{model}</string></property></item>' for model in models) + '\n              '
    text = text[:match.start(2)] + contents + text[match.end(2):]
path.write_text(text)

replace_methods('src/moltrack/funcs/segmentation_utils.py', {}, removed=('_postpocess_cellpose',))

# Replace movie loading and metadata inspection with MolCore image sources.
path = Path('src/moltrack/funcs/import_utils.py')
text = path.read_text().replace('from astropy.io import fits\n', '').replace('from PIL import Image\n', '')
text = text.replace('from moltrack.funcs.compute_utils import Worker',
    'from moltrack.funcs.compute_utils import Worker\nfrom moltrack.backend import read_movie, image_source')
start = text.index('        base, ext = os.path.splitext(path)', text.index('def import_image_data'))
end = text.index('        if len(images) > 0:', start)
text = text[:start] + '''        def report(progress):
            progress_dict[index] = progress

        limit = int(import_limit) if str(import_limit).isdigit() else None
        images = read_movie(path, limit=limit,
            crop=lambda frame: crop_frame(frame, crop_mode), progress_callback=report)

''' + text[end:]
path.write_text(text)
replace_methods(path, {'get_image_info': '''    def get_image_info(self, path):
        source = image_source(path)
        try:
            return len(source), source.shape, source.dtype, os.path.getsize(path)
        finally:
            source.close()'''})

path = Path('src/moltrack/funcs/loc_filter_utils.py')
text = path.read_text().replace('from sklearn.cluster import DBSCAN\n', '')
text = text.replace('from moltrack.funcs.compute_utils import Worker',
    'from moltrack.funcs.compute_utils import Worker\nfrom moltrack.backend import filter_locs, merge_locs')
start = text.index('                        criterion_data = locs[criterion]')
end = text.index('                        n_filtered = len(locs)', start)
text = text[:start] + '''                        locs = filter_locs(locs, criterion, min_value, max_value,
                            subtract_background=subtract_background)

''' + text[end:]
path.write_text(text)
replace_methods(path, {'compute_merge_localisations': '''    def compute_merge_localisations(self, dataset, locs, min_distance=1, progress_callback=None):
        data = pd.DataFrame(locs)
        if data.empty or data.channel.nunique() < 2:
            return data
        merged = merge_locs(data, min_distance, progress_callback)
        for channel in self.dataset_dict[dataset]['images']:
            result = merged.copy()
            result['channel'] = channel
            entry = self.localisation_dict[dataset].setdefault(channel,
                self.localisation_dict[dataset][data.channel.iloc[0]].copy())
            entry['localisations'] = result.to_records(index=False)
        return merged'''})

path = Path('src/moltrack/funcs/events_utils.py')
text = path.read_text().replace('import napari\n',
    'import napari\nfrom molcore import ContrastSettings\nfrom molcore.contrast import autocontrast\n')
text = text.replace('np.percentile(image[:10].copy(), [0.1, 99.99])',
    'autocontrast(image[:10], settings=ContrastSettings(percentiles=(0.1, 99.99), max_frames=None, max_pixels_per_frame=None)).limits')
path.write_text(text)

replace_methods('src/moltrack/funcs/compute_utils.py', {},
    removed=('create_shared_image_chunks', 'restore_shared_image_chunks'))
path = Path('src/moltrack/funcs/compute_utils.py')
path.write_text(path.read_text().replace('from multiprocessing import shared_memory\n', ''))
path = Path('src/moltrack/_widget.py')
text = path.read_text().replace('        # self.create_shared_image_chunks()\n', '')
text = text.replace('        # self.restore_shared_image_chunks()\n', '')
path.write_text(text)
