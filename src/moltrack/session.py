"""Persistent scientific ownership and explicit detached GUI query boundaries."""

from copy import deepcopy

import numpy as np
import pandas as pd
from molcore import Dataset
from molcore.staging import transact

from .model_views import channel_rows

ORIGIN = "moltrack_origin"
DETECTION = "detection"
IMPORTED_TRACK = "imported_track"


class ProjectSession:
    """Own one Dataset and application labels, never mirrored scientific rows.

    Public row mutation/query methods use GUI channel frame indices and raw
    channel pixel input coordinates. Queries explicitly select coordinate space.
    Annotations contain application-only serializable values; bindings contain
    labels and model references only. Scientific provenance lives in model rows.
    """

    def __init__(self, model: Dataset | None = None):
        self.model = model if model is not None else Dataset()
        self.annotations = {}
        self._partitions = {}

    def datasets(self) -> tuple[str, ...]:
        return tuple(self._partitions)

    def channels(self, dataset: str) -> tuple[str, ...]:
        return tuple(self._partitions[dataset])

    def bind(
        self,
        dataset: str,
        channel: str,
        acquisition_id: str,
        model_channel: str,
    ) -> None:
        if not all(
            isinstance(v, str) and v.strip()
            for v in (dataset, channel, acquisition_id, model_channel)
        ):
            raise ValueError(
                "Partition labels and model identifiers must be nonempty strings"
            )
        if (
            model_channel
            not in self.model.acquisitions[acquisition_id].channels
        ):
            raise ValueError("Unknown model channel")
        self._partitions.setdefault(dataset, {})[channel] = (
            acquisition_id,
            model_channel,
        )

    def partition(self, dataset: str, channel: str) -> tuple[str, str]:
        return self._partitions[dataset][channel]

    def _view(self, dataset, channel):
        acquisition, name = self.partition(dataset, channel)
        return self.model.acquisitions[acquisition].channels[name]

    def _incoming(self, dataset, channel, rows):
        data = deepcopy(pd.DataFrame(rows))
        if ORIGIN in data:
            raise ValueError(f"{ORIGIN} is reserved for session provenance")
        if not len(data):
            for name in ("frame", "x", "y"):
                if name not in data:
                    data[name] = np.array([], dtype=float)
        if not {"frame", "x", "y"} <= set(data):
            raise ValueError("Rows require frame, x and y")
        frames = data.frame.to_numpy()
        view = self._view(dataset, channel)
        if (
            not np.isfinite(frames).all()
            or np.any(frames != frames.astype(np.int64))
            or np.any(frames < 0)
            or np.any(frames >= len(view))
        ):
            raise ValueError("Frame must be a valid integer channel index")
        data["frame"] = view.source_frames[frames.astype(np.int64)]
        # Bindings always determine structural partition columns.
        data = data.drop(
            columns=["acquisition_id", "channel"], errors="ignore"
        )
        return {k: deepcopy(data[k].to_numpy()) for k in data}

    @staticmethod
    def _detections(model, acquisition, channel):
        rows = model.localizations.query(
            acquisition_id=acquisition, channel=channel, space="raw"
        ).to_columns()
        # Existing model tables without MolTrack provenance are detections.
        keep = (
            rows.get(ORIGIN, np.full(len(rows["id"]), DETECTION))
            != IMPORTED_TRACK
        )
        return {k: v[keep] for k, v in rows.items()}

    def replace_localizations(
        self, dataset, channel, rows, *, frame_index=None
    ) -> np.ndarray:
        acquisition, name = self.partition(dataset, channel)
        incoming = self._incoming(dataset, channel, rows)
        camera_frame = None
        if frame_index is not None:
            view = self._view(dataset, channel)
            if not isinstance(
                frame_index, (int, np.integer)
            ) or not 0 <= frame_index < len(view):
                raise ValueError("Invalid preview frame index")
            camera_frame = int(view.source_frames[frame_index])
            keep = incoming["frame"] == camera_frame
            incoming = {k: v[keep] for k, v in incoming.items()}
        incoming[ORIGIN] = np.full(len(incoming["frame"]), DETECTION)

        def replace(model):
            old = self._detections(model, acquisition, name)
            selected = (
                old["id"]
                if camera_frame is None
                else old["id"][old["frame"] == camera_frame]
            )
            supplied = incoming.get("id", np.array([], dtype=np.uint64))
            existing = model.localizations.query(space="raw")["id"]
            if np.any(
                np.isin(supplied, existing) & ~np.isin(supplied, selected)
            ):
                raise ValueError(
                    "Candidate IDs belong to another frame or origin"
                )
            retained = np.intersect1d(supplied, selected)
            if not len(retained):
                model.localizations.delete(selected)
                return model.localizations.add(
                    acquisition_id=acquisition, channel=name, **incoming
                )
            # Candidate fitting edits stable rows through supported table mutators.
            # IDs cannot be deleted and then reused by molcore.
            incoming_ids = incoming["id"]
            existing_rows = model.localizations.query(
                ids=retained, space="raw"
            )
            indices = np.flatnonzero(np.isin(incoming_ids, retained))
            update_ids = incoming_ids[indices]
            numeric = {}
            for key, values in incoming.items():
                if key in ("id", "frame", "x", "y", ORIGIN):
                    continue
                if values.dtype.kind in "iuf":
                    numeric[key] = values[indices]
                else:
                    lookup = (
                        dict(
                            zip(
                                existing_rows["id"],
                                existing_rows[key],
                                strict=True,
                            )
                        )
                        if key in existing_rows
                        else {}
                    )
                    if any(
                        i not in lookup or lookup[i] != values[j]
                        for i, j in zip(update_ids, indices, strict=True)
                    ):
                        raise ValueError(
                            "molcore cannot edit existing nonnumeric measurement columns"
                        )
            frame_lookup = dict(
                zip(existing_rows["id"], existing_rows["frame"], strict=True)
            )
            if any(
                frame_lookup[i] != incoming["frame"][j]
                for i, j in zip(update_ids, indices, strict=True)
            ):
                raise ValueError(
                    "Candidate fitting cannot change source frames"
                )
            model.localizations.delete(selected[~np.isin(selected, retained)])
            model.localizations.move_many(
                update_ids, x=incoming["x"][indices], y=incoming["y"][indices]
            )
            if numeric:
                model.localizations.update_measurements(update_ids, **numeric)
            fresh = ~np.isin(incoming_ids, retained)
            if fresh.any():
                model.localizations.add(
                    acquisition_id=acquisition,
                    channel=name,
                    **{k: v[fresh] for k, v in incoming.items()},
                )
            return incoming_ids.copy()

        return transact(self.model, replace)

    def import_tracks(self, dataset, channel, rows) -> np.ndarray:
        acquisition, name = self.partition(dataset, channel)
        incoming = self._incoming(dataset, channel, rows)
        if "particle" not in incoming:
            if len(incoming["frame"]):
                raise ValueError(
                    "Imported track rows require legacy particle labels"
                )
            incoming["particle"] = np.array([], dtype=np.int64)
        if pd.isna(incoming["particle"]).any():
            raise ValueError("Track labels cannot be missing")
        # Independent imports always allocate fresh scientific identities.
        incoming.pop("id", None)
        incoming.pop("track_id", None)
        incoming[ORIGIN] = np.full(len(incoming["frame"]), IMPORTED_TRACK)

        def replace(model):
            old = model.localizations.query(
                acquisition_id=acquisition, channel=name, space="raw"
            )
            if ORIGIN in old:
                model.localizations.delete(
                    old["id"][old[ORIGIN] == IMPORTED_TRACK]
                )
            ids = model.localizations.add(
                acquisition_id=acquisition, channel=name, **incoming
            )
            tracks = []
            for particle in pd.unique(incoming["particle"]):
                members = ids[incoming["particle"] == particle]
                label = (
                    particle.item()
                    if isinstance(particle, np.generic)
                    else particle
                )
                tracks.append(
                    model.tracks.add(
                        members,
                        metadata={"origin": IMPORTED_TRACK, "particle": label},
                    )
                )
            return np.asarray(tracks, dtype=np.uint64)

        return transact(self.model, replace)

    def query_localizations(
        self, dataset, channel, *, space="corrected"
    ) -> pd.DataFrame:
        acquisition, name = self.partition(dataset, channel)
        rows = self.model.localizations.query(
            acquisition_id=acquisition, channel=name, space=space
        ).to_columns()
        keep = (
            rows.get(ORIGIN, np.full(len(rows["id"]), DETECTION))
            != IMPORTED_TRACK
        )
        data = channel_rows(
            {k: v[keep] for k, v in rows.items()}, self._view(dataset, channel)
        )
        return data.drop(columns=ORIGIN, errors="ignore")

    def query_tracks(
        self, dataset, channel, *, space="corrected"
    ) -> pd.DataFrame:
        acquisition, name = self.partition(dataset, channel)
        rows = self.model.tracks.query(
            acquisition_id=acquisition, channel=name, space=space
        )
        data = channel_rows(rows, self._view(dataset, channel))
        if "particle" not in data:
            data["particle"] = data["track_id"].copy()
        else:
            labels = data["particle"]
            missing = labels.isna() | labels.eq("")
            # molcore fills absent Unicode measurements with empty strings.
            # Object dtype keeps fallback IDs intact alongside string labels.
            if missing.any():
                labels = labels.astype(object).copy()
                labels.loc[missing] = data.loc[missing, "track_id"]
            data["particle"] = labels
        return data.drop(columns=ORIGIN, errors="ignore")
