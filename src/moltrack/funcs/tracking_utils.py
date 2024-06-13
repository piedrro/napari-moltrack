import traceback

import numpy as np
import pandas as pd
import trackpy as tp

from moltrack.funcs.compute_utils import Worker


class _tracking_utils:

    def get_tracks(self, dataset, channel, return_dict=False, include_metadata=True):

        order = ["dataset", "channel", "group", "particle",
                 "frame", "cell_index", "segmentation_index",
                 "x", "y", "photons", "bg",
                 "sx", "sy", "lpx", "lpy",
                 "ellipticity", "net_gradient", "iterations", ]

        track_data = []
        group = 0

        try:
            if dataset == "All Datasets":
                dataset_list = list(self.tracking_dict.keys())
            else:
                dataset_list = [dataset]

            for dataset_name in dataset_list:
                if dataset_name not in self.tracking_dict.keys():
                    continue

                if channel == "All Channels":
                    channel_list = list(self.tracking_dict[dataset_name].keys())
                else:
                    channel_list = [channel]

                for channel_name in channel_list:
                    if (channel_name not in self.tracking_dict[dataset_name].keys()):
                        continue

                    track_dict = self.tracking_dict[dataset_name][channel_name]

                    if "tracks" in track_dict.keys():
                        tracks = track_dict["tracks"].copy()

                        tracks = pd.DataFrame(tracks)

                        if include_metadata:
                            if "dataset" not in tracks.columns:
                                tracks.insert(0, "dataset", dataset_name)
                            if "channel" not in tracks.columns:
                                tracks.insert(1, "channel", channel_name)
                            if len(dataset_list) > 1:
                                if "group" not in tracks.columns:
                                    tracks.insert(2, "group", group)

                        mask = []

                        for col in order:
                            if col in tracks.columns:
                                mask.append(col)

                        tracks = tracks[mask]

                        tracks = tracks.to_records(index=False)

                        n_tracks = len(tracks)

                        if n_tracks > 0:
                            if return_dict == False:
                                track_data.append(tracks)
                            else:
                                track_dict = {"dataset": dataset_name, "channel": channel_name, "tracks": tracks, }
                                track_data.append(track_dict)

        except:
            print(traceback.format_exc())

        if return_dict == False:
            if len(track_data) == 0:
                pass
            elif len(track_data) == 1:
                track_data = track_data[0]
            else:
                track_data = np.hstack(track_data).view(np.recarray).copy()

        return track_data

    def run_tracking(self, loc_data, progress_callback=None):
        search_range = float(self.gui.trackpy_search_range.value())
        memory = int(self.gui.trackpy_memory.value())
        min_track_length = int(self.gui.min_track_length.value())

        track_data = []

        for dat in loc_data:
            try:
                dataset = dat["dataset"]
                channel = dat["channel"]
                pixel_size = float(self.dataset_dict[dataset]["pixel_size"])
                exposure_time = float(self.dataset_dict[dataset]["exposure_time"])
                locs = dat["localisations"]

                columns = list(locs.dtype.names)

                locdf = pd.DataFrame(locs, columns=columns)

                tp.quiet()
                tracks_df = tp.link(locdf, search_range=search_range, memory=memory)

                tracks_df.reset_index(drop=True, inplace=True)

                tracks_df = tp.filter_stubs(tracks_df, min_track_length).reset_index(drop=True)

                self.tracks = tracks_df

                track_index = 1
                for particle, group in tracks_df.groupby("particle"):
                    tracks_df.loc[group.index, "particle"] = track_index
                    track_index += 1

                tracks_df = tracks_df.sort_values(by=["particle", "frame"])
                tracks = tracks_df.to_records(index=False)

                track_dict = {"dataset": dataset, "channel": channel, "tracks": tracks, }

                track_data.append(track_dict)

            except:
                print(traceback.format_exc())

        return track_data

    def process_tracking_results(self, track_data):
        try:
            if track_data is None:
                return

            remove_unlinked = self.gui.remove_unlinked.isChecked()
            layers_names = [layer.name for layer in self.viewer.layers]
            total_tracks = 0

            for dat in track_data:
                dataset = dat["dataset"]
                channel = dat["channel"]
                tracks = dat["tracks"]

                if dataset not in self.tracking_dict.keys():
                    self.tracking_dict[dataset] = {}
                if channel not in self.tracking_dict[dataset].keys():
                    self.tracking_dict[dataset][channel] = {}

                self.tracking_dict[dataset][channel] = {"tracks": tracks}

                n_tracks = len(np.unique(tracks["particle"]))
                total_tracks += n_tracks

                if remove_unlinked:
                    loc_dict = self.localisation_dict[dataset][channel]

                    locs = loc_dict["localisations"]
                    n_locs = len(locs)
                    n_filtered = len(tracks)

                    n_removed = n_locs - n_filtered

                    if n_removed > 0:
                        print(f"Removed {n_removed} unlinked localisations")
                        loc_dict["localisations"] = tracks

            if len(track_data) > 0:
                self.gui.locs_export_data.clear()
                self.gui.locs_export_data.addItems(["Localisations", "Tracks"])
                print(f"Tracking complete, {total_tracks} tracks found")

        except:
            print(traceback.format_exc())

    def tracking_finished(self):
        self.draw_localisations()
        self.draw_tracks()
        self.update_ui()

    def initialise_tracking(self):
        try:
            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()

            loc_data = self.get_locs(dataset, channel, return_dict=True, include_metadata=True)

            if len(loc_data) > 0:
                self.update_ui(init=True)

                worker = Worker(self.run_tracking, loc_data)
                worker.signals.result.connect(self.process_tracking_results)
                worker.signals.finished.connect(self.tracking_finished)
                self.threadpool.start(worker)

            else:
                print(f"No localisations found for {dataset}`")

        except:
            print(traceback.format_exc())
            self.update_ui()

    def draw_tracks(self, dataset=None, channel=None):
        try:
            remove_tracks = True

            if hasattr(self, "tracking_dict"):
                if hasattr(self, "tracks_layer"):
                    show_tracks = self.loc_layer.visible
                else:
                    show_tracks = True

            if show_tracks:
                layer_names = [layer.name for layer in self.viewer.layers]

                dataset_name = self.gui.moltrack_dataset_selector.currentText()
                channel_name = self.gui.moltrack_channel_selector.currentText()

                tracks = self.get_tracks(dataset_name, channel_name, return_dict=False, include_metadata=True, )

                if len(tracks) > 0:
                    image_dict = self.dataset_dict[dataset_name]["images"]
                    n_frames = image_dict[channel_name].shape[0]

                    pixel_size = float(self.dataset_dict[dataset_name]["pixel_size"])
                    scale = [pixel_size, pixel_size]

                    remove_tracks = False

                    render_tracks = pd.DataFrame(tracks)
                    render_tracks = render_tracks[["particle", "frame", "y", "x"]]
                    render_tracks = render_tracks.to_records(index=False)
                    render_tracks = [list(track) for track in render_tracks]
                    render_tracks = np.array(render_tracks).copy()
                    render_tracks[:, 1] = 0

                    if "Tracks" not in layer_names:
                        self.track_layer = self.viewer.add_tracks(render_tracks, name="Tracks", blending="opaque", scale=scale, )
                        self.viewer.reset_view()
                    else:
                        self.track_layer.data = render_tracks

                    self.track_layer.selected_data = []
                    self.track_layer.tail_length = n_frames * 2
                    self.track_layer.blending = "opaque"

                    self.track_layer.scale = scale
                    self.viewer.scale_bar.visible = True
                    self.viewer.scale_bar.unit = "um"

            if remove_tracks:
                if "Tracks" in layer_names:
                    self.viewer.layers["Tracks"].data = []

            for layer in layer_names:
                self.viewer.layers[layer].refresh()

        except:
            print(traceback.format_exc())
