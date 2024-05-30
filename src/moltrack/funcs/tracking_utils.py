import traceback
import pandas as pd
import trackpy as tp
import numpy as np
from moltrack.funcs.compute_utils import Worker

class _tracking_utils:


    def run_tracking(self, locs, progress_callback=None):

        tracks_array = []

        try:

            search_range = float(self.gui.trackpy_search_range.value())
            memory = int(self.gui.trackpy_memory.value())
            min_track_length = int(self.gui.min_track_length.value())

            columns = list(locs.dtype.names)

            locdf = pd.DataFrame(locs, columns=columns)

            tracks_df = tp.link(locdf, search_range=search_range, memory=memory)

            # Count the frames per track
            track_lengths = tracks_df.groupby('particle').size()

            # Filter tracks by length
            valid_tracks = track_lengths[track_lengths >= min_track_length].index
            tracks_df = tracks_df[tracks_df['particle'].isin(valid_tracks)]

            self.tracks = tracks_df

            track_index = 1
            for particle, group in tracks_df.groupby("particle"):
                tracks_df.loc[group.index, "particle"] = track_index
                track_index += 1

            tracks_df = tracks_df[['particle', 'frame', 'y', 'x']]
            tracks_df = tracks_df.sort_values(by=["particle", "frame"])

            tracks_array = tracks_df.to_records(index=False)
            tracks_array = [list(track) for track in tracks_array]
            tracks_array = np.array(tracks_array)

        except:
            print(traceback.format_exc())

        return tracks_array

    def process_tracking_results(self, tracks_array):

        try:

            dataset = self.gui.tracking_dataset.currentText()
            remove_unlinked = self.gui.remove_unlinked.isChecked()

            n_frames = self.dataset_dict[dataset]["data"].shape[0]

            layers_names = [layer.name for layer in self.viewer.layers]

            render_tracks = tracks_array.copy()
            render_tracks[:,1] = 0

            if "Tracks" not in layers_names:
                self.track_layer = self.viewer.add_tracks(render_tracks, name="Tracks")
            else:
                self.track_layer.data = render_tracks

            self.track_layer.tail_length = n_frames * 2

            if remove_unlinked:

                loc_dict = self.localisation_dict[dataset]

                locs = loc_dict["localisations"]
                n_locs = len(locs)

                filtered_locs = pd.DataFrame(tracks_array,
                    columns=["particle", "frame", "y", "x"])
                filtered_locs = filtered_locs.to_records(index=False)

                n_filtered = len(filtered_locs)

                n_removed = n_locs - n_filtered

                if n_removed > 0:
                    print(f"Removed {n_removed} unlinked localisations")

                    loc_dict["localisations"] = filtered_locs



        except:
            print(traceback.format_exc())


    def tracking_finished(self):

        self.draw_localisations()
        self.update_ui()


    def initialise_tracking(self):

        try:

            dataset = self.gui.tracking_dataset.currentText()

            if dataset in self.localisation_dict.keys():

                loc_dict = self.localisation_dict[dataset]

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


