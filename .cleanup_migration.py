from pathlib import Path
import ast


def replace_methods(path, replacements):
    path = Path(path)
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    changes = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in replacements:
            changes.append((node.lineno - 1, node.end_lineno, replacements[node.name]))
    for first, last, value in sorted(changes, reverse=True):
        lines[first:last] = [value + '\n']
    path.write_text(''.join(lines))


replace_methods('src/moltrack/funcs/tracking_utils.py', {
    'detect_tracks': '''    def detect_tracks(self, loc_data, progress_callback=None):
        from concurrent.futures import CancelledError

        segchannel = self.gui.tracking_segmentations.currentText()
        search_range = float(self.gui.trackpy_search_range.value())
        memory = int(self.gui.trackpy_memory.value())
        minimum = int(self.gui.min_track_length.value())
        output, partitions = [], []
        for dat in loc_data:
            if getattr(self, 'stop_event', None) is not None and self.stop_event.is_set():
                raise CancelledError()
            dataset, channel = dat['dataset'], dat['channel']
            partitions.append((dataset, channel))
            if segchannel in ('Segmentations', 'Cells'):
                locs, segcol = self.detect_seglocs(dataset, channel, segchannel)
                self.tracking_segchannel, self.tracking_segcol = segchannel, segcol
            else:
                locs, segcol = dat['localisations'], None
                self.tracking_segchannel = self.tracking_segcol = None
            rows = pd.DataFrame(locs)
            groups = list(rows.groupby(segcol)) if segcol and len(rows) else [(None, rows)]
            linked, offset = [], 0
            for _, group in groups:
                if group.empty:
                    continue
                tracks = self.link_locs(group, search_range=search_range, memory=memory,
                    min_length=minimum, progress_callback=progress_callback)
                if tracks.empty:
                    continue
                tracks['particle'] += offset
                offset = int(tracks.particle.max())
                linked.append(tracks)
            tracks = pd.concat(linked, ignore_index=True) if linked else rows.iloc[:0].copy()
            if 'particle' not in tracks:
                tracks['particle'] = pd.Series(dtype='int64')
            tracks['dataset'], tracks['channel'] = dataset, channel
            metadata = self.dataset_dict[dataset]
            tracks['pixel_size (um)'] = float(metadata['pixel_size']) * 1e-3
            tracks['exposure_time (s)'] = float(metadata['exposure_time']) * 1e-3
            output.append(tracks.sort_values(['particle', 'frame']))
        if getattr(self, 'stop_event', None) is not None and self.stop_event.is_set():
            raise CancelledError()
        self._tracking_partitions = partitions
        return pd.concat(output, ignore_index=True) if output else pd.DataFrame(
            columns=['dataset', 'channel', 'particle', 'frame', 'x', 'y'])''',
    'process_tracking_results': '''    def process_tracking_results(self, track_data):
        if track_data is None or self.stop_event.is_set():
            return
        partitions = getattr(self, '_tracking_partitions', list(
            track_data.groupby(['dataset', 'channel']).groups))
        total_tracks = 0
        for dataset, channel in partitions:
            selected = track_data[(track_data.dataset == dataset) & (track_data.channel == channel)]
            tracks = selected.to_records(index=False)
            self.tracking_dict.setdefault(dataset, {})[channel] = {'tracks': tracks}
            total_tracks += selected.particle.nunique()
            if self.gui.remove_unlinked.isChecked():
                self.localisation_dict[dataset][channel]['localisations'] = tracks.copy()
        if len(track_data):
            self.gui.locs_export_data.clear()
            self.gui.locs_export_data.addItems(['Localisations', 'Tracks'])
            self.gui.heatmap_data.clear()
            self.gui.heatmap_data.addItems(['Localisations', 'Tracks'])
        show_info(f'Tracking complete, {total_tracks} tracks found')'''})

# Whitespace cleanup, leaving UI layout and generated widget definitions intact.
for path in Path('src/moltrack').rglob('*.py'):
    text = path.read_text()
    text = '\n'.join(line.rstrip() for line in text.splitlines()).rstrip() + '\n'
    path.write_text(text)
path = Path('src/moltrack/GUI/widget_ui.ui')
path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
