"""Explicit session operations behind the copy/delete controls."""

import traceback

from napari.utils.notifications import show_info

from moltrack.model_views import refresh_row_projections


class _management_utils:
    def get_copy_targets(self, source_dataset, source_channel, mode_index=0):
        targets = []
        for dataset, channel in self.session.selected_partitions():
            if mode_index == 0:
                keep = dataset == source_dataset and channel != source_channel
            elif mode_index == 1:
                keep = dataset != source_dataset and channel == source_channel
            else:
                keep = dataset != source_dataset and channel != source_channel
            if keep:
                targets.append({"dataset": dataset, "channel": channel})
        return targets

    def copy_data(self, mode="locs"):
        try:
            controls = "copy_tracks" if mode == "tracks" else "copy_locs"
            dataset = getattr(self.gui, controls + "_dataset").currentText()
            channel = getattr(self.gui, controls + "_channel").currentText()
            index = getattr(self.gui, controls + "_mode").currentIndex()
            self.update_ui(init=True)
            for target in self.get_copy_targets(dataset, channel, index):
                self.session.copy_rows(
                    (dataset, channel),
                    (target["dataset"], target["channel"]),
                    tracks=mode == "tracks",
                )
            refresh_row_projections(self, localizations=mode != "tracks")
        except (KeyError, ValueError, TypeError, IndexError):
            print(traceback.format_exc())
        finally:
            self.update_ui()

    def delete_data(self, mode="locs"):
        try:
            controls = "delete_tracks" if mode == "tracks" else "delete_locs"
            dataset = getattr(self.gui, controls + "_dataset").currentText()
            channel = getattr(self.gui, controls + "_channel").currentText()
            for partition in self.session.selected_partitions(
                dataset, channel
            ):
                self.session.delete_rows(*partition, tracks=mode == "tracks")
            refresh_row_projections(self, localizations=mode != "tracks")
            show_info(
                "Selected tracks deleted"
                if mode == "tracks"
                else "Selected localisations deleted"
            )
        except (KeyError, ValueError, TypeError, IndexError):
            print(traceback.format_exc())
        finally:
            self.update_ui()
