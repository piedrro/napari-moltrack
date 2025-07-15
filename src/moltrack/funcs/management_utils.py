import traceback

from napari.utils.notifications import show_info


class _management_utils:

    def get_copy_targets(self, source_dataset, source_channel, mode_index = 0):

        copy_targets = []

        if hasattr(self, "dataset_dict"):

            if mode_index == 0:

                dataset_dict = self.dataset_dict[source_dataset]
                image_dict = dataset_dict["images"]
                channels = list(image_dict.keys())

                for channel in channels:
                    if channel != source_channel:
                        copy_targets.append({"dataset": source_dataset, "channel": channel})

            elif mode_index == 1:

                datasets = list(self.dataset_dict.keys())

                for dataset in datasets:
                    if dataset != source_dataset:
                        copy_targets.append({"dataset": dataset, "channel": source_channel})

            elif mode_index == 2:

                datasets = list(self.dataset_dict.keys())

                for dataset in datasets:
                    channels = list(self.dataset_dict[dataset]["images"].keys())
                    for channel in channels:
                        if dataset != source_dataset and channel != source_channel:
                            copy_targets.append({"dataset": dataset, "channel": channel})

        return copy_targets


    def copy_data(self, mode = "locs"):

        try:

            if mode == "locs":

                dataset = self.gui.copy_locs_dataset.currentText()
                channel = self.gui.copy_locs_channel.currentText()
                mode_index = self.gui.copy_locs_mode.currentIndex()

                locs = self.get_locs(dataset, channel)

                if len(locs) == 0:
                    show_info("No localisations found in selected dataset/channel")
                    return None

                copy_targets = self.get_copy_targets(dataset, channel,
                    mode_index = mode_index)

                if len(copy_targets) == 0:
                    return None

                self.update_ui(init=True)

                for target in copy_targets:
                    dataset = target["dataset"]
                    channel = target["channel"]
                    locs = locs.copy()

                    locs["dataset"] = dataset
                    locs["channel"] = channel

                    if dataset not in self.localisation_dict.keys():
                        self.localisation_dict[dataset] = {}
                    if channel not in self.localisation_dict[dataset].keys():
                        self.localisation_dict[dataset][channel] = {}

                    self.localisation_dict[dataset][channel]["localisations"] = locs
                    show_info(f"Localisations copied to {dataset} - {channel}")

                self.draw_localisations()

            if mode == "tracks":

                dataset = self.gui.copy_tracks_dataset.currentText()
                channel = self.gui.copy_tracks_channel.currentText()
                mode_index = self.gui.copy_tracks_mode.currentIndex()

                tracks = self.get_tracks(dataset, channel)

                if len(tracks) == 0:
                    show_info("No tracks found in selected dataset/channel")
                    return None

                copy_targets = self.get_copy_targets(dataset, channel,
                    mode_index = mode_index)

                if len(copy_targets) == 0:
                    return None

                self.update_ui(init=True)

                for target in copy_targets:
                    dataset = target["dataset"]
                    channel = target["channel"]
                    tracks = tracks.copy()

                    tracks["dataset"] = dataset
                    tracks["channel"] = channel

                    if dataset not in self.tracking_dict.keys():
                        self.tracking_dict[dataset] = {}
                    if channel not in self.tracking_dict[dataset].keys():
                        self.tracking_dict[dataset][channel] = {}

                    self.tracking_dict[dataset][channel]["tracks"] = tracks
                    show_info(f"Tracks copied to {dataset} - {channel}")

                self.draw_tracks()

            self.update_ui()

        except:
            print(traceback.format_exc())
            self.update_ui()

    def delete_data(self, mode = "locs"):

        try:

            deleted_datasets = []
            deleted_channels = []

            if mode == "locs":

                dataset = self.gui.delete_locs_dataset.currentText()
                channel = self.gui.delete_locs_channel.currentText()

                if dataset == "All Datasets":
                    dataset_list = list(self.localisation_dict.keys())
                else:
                    dataset_list = [dataset]

                for dataset in dataset_list:

                    if channel == "All Channels":
                        channel_list = list(self.localisation_dict[dataset].keys())
                    else:
                        channel_list = [channel]

                    for channel in channel_list:
                        if channel in self.localisation_dict[dataset].keys():
                            if self.localisation_dict[dataset][channel] != {}:
                                self.localisation_dict[dataset][channel] = {}
                                show_info(f"Localisations deleted for {dataset} - {channel}")
                                deleted_datasets.append(dataset)
                                deleted_channels.append(channel)

                deleted_datasets = list(set(deleted_datasets))
                deleted_channels = list(set(deleted_channels))

                if len(deleted_datasets) == 0 and len(deleted_channels) == 0:
                    show_info("No localisations found to delete")
                else:
                    show_info(f"Localisations deleted in {len(deleted_datasets)} datasets and {len(deleted_channels)} channels")

                self.draw_localisations()

            if mode == "tracks":

                dataset = self.gui.delete_tracks_dataset.currentText()
                channel = self.gui.delete_tracks_channel.currentText()

                if dataset == "All Datasets":
                    dataset_list = list(self.tracking_dict.keys())
                else:
                    dataset_list = [dataset]

                for dataset in dataset_list:

                    if channel == "All Channels":
                        channel_list = list(self.tracking_dict[dataset].keys())
                    else:
                        channel_list = [channel]

                    for channel in channel_list:
                        if channel in self.tracking_dict[dataset].keys():
                            if self.tracking_dict[dataset][channel] != {}:
                                self.tracking_dict[dataset][channel] = {}
                                show_info(f"Tracks deleted for {dataset} - {channel}")
                                deleted_datasets.append(dataset)
                                deleted_channels.append(channel)

                deleted_datasets = list(set(deleted_datasets))
                deleted_channels = list(set(deleted_channels))

                if len(deleted_datasets) == 0 and len(deleted_channels) == 0:
                    show_info("No tracks found to delete")
                else:
                    show_info(f"Tracks deleted in {len(deleted_datasets)} datasets and {len(deleted_channels)} channels")

                self.draw_tracks()

            self.update_ui()

        except:
            print(traceback.format_exc())
            self.update_ui()
