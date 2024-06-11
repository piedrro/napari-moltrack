import traceback

import numpy as np
import pandas as pd
import trackpy as tp

from moltrack.funcs.compute_utils import Worker


class _tracking_utils:

    def get_tracks(self, dataset, channel, return_dict=False, include_metadata=True):
        track_data = []

        try:
            if dataset == "All Datasets":
                dataset_list = list(self.tracking_dict.keys())
            else:
                dataset_list = [dataset]

            for dataset_name in dataset_list:
                if channel == "All Channels":
                    channel_list = list(self.tracking_dict[dataset_name].keys())
                else:
                    channel_list = [channel]

                for channel_name in channel_list:
                    track_dict = self.tracking_dict[dataset_name][channel_name]

                    if "tracks" in track_dict.keys():
                        tracks = track_dict["tracks"].copy()

                        if include_metadata:
                            tracks = pd.DataFrame(tracks)

                            if "dataset" not in tracks.columns:
                                tracks.insert(0, "dataset", dataset_name)
                            if "channel" not in tracks.columns:
                                tracks.insert(1, "channel", channel_name)

                            tracks = tracks.to_records(index=False)

                        n_tracks = len(tracks)

                        if n_tracks > 0:
                            if return_dict == False:
                                track_data.append(tracks)
                            else:
                                track_dict = {"dataset": dataset_name, "channel": channel_name, "tracks": tracks}
                                track_data.append(track_dict)

        except:
            print(traceback.format_exc())

        if return_dict == False:
            if len(track_data) == 1:
                track_data = track_data[0]
            else:
                track_data = np.hstack(track_data).view(np.recarray).copy()

        return track_data

    def run_tracking(self, locs, progress_callback=None):
        tracks = None

        try:
            search_range = float(self.gui.trackpy_search_range.value())
            memory = int(self.gui.trackpy_memory.value())
            min_track_length = int(self.gui.min_track_length.value())

            columns = list(locs.dtype.names)

            locdf = pd.DataFrame(locs, columns=columns)

            tp.quiet()
            tracks_df = tp.link(locdf, search_range=search_range, memory=memory)

            # Count the frames per track
            track_lengths = tracks_df.groupby("particle").size()

            # Filter tracks by length
            valid_tracks = track_lengths[track_lengths >= min_track_length].index
            tracks_df = tracks_df[tracks_df["particle"].isin(valid_tracks)]

            self.tracks = tracks_df

            track_index = 1
            for particle, group in tracks_df.groupby("particle"):
                tracks_df.loc[group.index, "particle"] = track_index
                track_index += 1

            # tracks_df = tracks_df[['particle', 'frame', 'y', 'x']]
            tracks_df = tracks_df.sort_values(by=["particle", "frame"])
            tracks = tracks_df.to_records(index=False)

        except:
            print(traceback.format_exc())

        return tracks

    def process_tracking_results(self, tracks):
        try:
            if tracks is None:
                return

            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()

            if dataset not in self.tracking_dict.keys():
                self.tracking_dict[dataset] = {}
            if channel not in self.tracking_dict[dataset].keys():
                self.tracking_dict[dataset][channel] = tracks

            n_tracks = np.unique(tracks["particle"])
            print(f"Found {len(n_tracks)} tracks")

            remove_unlinked = self.gui.remove_unlinked.isChecked()
            image_dict = self.dataset_dict[dataset]["images"]
            n_frames = image_dict[channel].shape[0]

            layers_names = [layer.name for layer in self.viewer.layers]

            render_tracks = pd.DataFrame(tracks)
            render_tracks = render_tracks[["particle", "frame", "y", "x"]]
            render_tracks = render_tracks.to_records(index=False)
            render_tracks = [list(track) for track in render_tracks]
            render_tracks = np.array(render_tracks).copy()
            render_tracks[:, 1] = 0

            if "Tracks" not in layers_names:
                self.track_layer = self.viewer.add_tracks(render_tracks,
                    name="Tracks")
            else:
                self.track_layer.data = render_tracks

            self.track_layer.tail_length = n_frames * 2

            self.gui.locs_export_data.clear()
            self.gui.locs_export_data.addItems(["Localisations", "Tracks"])

            if remove_unlinked:
                loc_dict = self.localisation_dict[dataset][channel]

                locs = loc_dict["localisations"]
                n_locs = len(locs)
                n_filtered = len(tracks)

                n_removed = n_locs - n_filtered

                if n_removed > 0:
                    print(f"Removed {n_removed} unlinked localisations")
                    loc_dict["localisations"] = tracks

        except:
            print(traceback.format_exc())

    def tracking_finished(self):
        self.draw_localisations()
        self.update_ui()

    def initialise_tracking(self):
        try:

            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()

            if dataset in self.localisation_dict.keys():
                if channel in self.localisation_dict[dataset].keys():

                    loc_dict = self.localisation_dict[dataset][channel]

                    if "localisations" in loc_dict.keys():
                        locs = loc_dict["localisations"].copy()

                        if len(locs) > 0:
                            self.update_ui(init=True)

                            locs = loc_dict["localisations"].copy()

                            worker = Worker(self.run_tracking, locs)
                            worker.signals.result.connect(self.process_tracking_results)
                            worker.signals.finished.connect(self.tracking_finished)
                            self.threadpool.start(worker)

            else:
                print(f"No localisations found for {dataset}`")

        except:
            print(traceback.format_exc())
            self.update_ui()
