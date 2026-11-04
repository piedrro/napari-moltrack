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
        self._analysis_cache = {}
        self._project_resources = None

    def close(self):
        """Close lazy sources before releasing their extracted native payload."""
        self.model.close()
        if self._project_resources is not None:
            self._project_resources.cleanup()
            self._project_resources = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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

    def publish_acquisitions(self, acquisitions, bindings, *, clear=False):
        """Publish a complete GUI import, preserving this session and model.

        Source additions and binding validation run against a detached model.
        The prepared registry is assigned only after its scientific transaction
        succeeds; no fallible operation remains after model publication.
        """

        def prepare(model):
            staged = ProjectSession(model)
            staged._partitions = deepcopy(self._partitions)
            annotations = deepcopy(self.annotations)
            if clear:
                # Preserve the standalone segmentation source and any original
                # acquisitions referenced by its derived-source provenance.
                keep = {
                    key
                    for key, acquisition in model.acquisitions.items()
                    if acquisition.metadata.get("role") == "segmentation_image"
                }
                binding = annotations.get("segmentation_image_binding")
                if binding is not None:
                    keep.add(staged.partition(*binding)[0])
                retained_partitions = {
                    (dataset, channel)
                    for dataset, channels in staged._partitions.items()
                    for channel, target in channels.items()
                    if target[0] in keep
                }
                pending = list(keep)
                while pending:
                    metadata = model.acquisitions[pending.pop()].metadata
                    references = [metadata.get("source_acquisition_id")]
                    references.extend(
                        segment.get("acquisition_id")
                        for segment in metadata.get("source_segments", ())
                    )
                    for reference in references:
                        if (
                            reference in model.acquisitions
                            and reference not in keep
                        ):
                            keep.add(reference)
                            pending.append(reference)
                for key in tuple(model.acquisitions):
                    if key not in keep:
                        model.remove_acquisition(key, cascade=True)
                staged._partitions = {
                    dataset: {
                        channel: binding
                        for channel, binding in channels.items()
                        if (dataset, channel) in retained_partitions
                    }
                    for dataset, channels in staged._partitions.items()
                }
                staged._partitions = {
                    dataset: channels
                    for dataset, channels in staged._partitions.items()
                    if channels
                }
                for name in ("partitions", "contrast"):
                    if name in annotations:
                        annotations[name] = {
                            dataset: {
                                channel: value
                                for channel, value in channels.items()
                                if channel
                                in staged._partitions.get(dataset, {})
                            }
                            for dataset, channels in annotations[name].items()
                            if dataset in staged._partitions
                        }
                for name in ("segmentations", "cells"):
                    if name in annotations:
                        annotations[name]["mask_ids"] = [
                            key
                            for key in annotations[name].get("mask_ids", ())
                            if key in model.segmentations
                        ]
            for acquisition in acquisitions:
                model.add_acquisition(acquisition)
            for binding in bindings:
                staged.bind(*binding)
            return staged._partitions, annotations

        partitions, annotations = transact(self.model, prepare)
        self._partitions = partitions
        self.annotations = annotations
        self._analysis_cache.clear()

    def _view(self, dataset, channel):
        acquisition, name = self.partition(dataset, channel)
        return self.model.acquisitions[acquisition].channels[name]

    def image(self, dataset, channel, *, space="corrected"):
        acquisition, name = self.partition(dataset, channel)
        return self.model.transforms.view(acquisition, name, space=space)

    def image_available(self, dataset, channel):
        """Descriptor-only availability query; never probes or fabricates pixels."""
        from molcore import MissingSource

        acquisition, _ = self.partition(dataset, channel)
        source = self.model.acquisitions[acquisition].source
        while hasattr(source, "source"):
            source = source.source
        return not isinstance(source, MissingSource)

    def metadata(self, dataset):
        acquisitions = [
            self.model.acquisitions[a]
            for a, _ in self._partitions[dataset].values()
        ]
        first = acquisitions[0]
        return dict(
            deepcopy(first.metadata),
            pixel_size=first.pixel_size,
            exposure_time=first.metadata.get("exposure_s", 0) * 1000,
        )

    def update_metadata(self, dataset, **values):
        def update(model):
            for acquisition in dict.fromkeys(
                a for a, _ in self._partitions[dataset].values()
            ):
                current = model.acquisitions[acquisition]
                metadata = dict(current.metadata)
                metadata.update(
                    {
                        k: deepcopy(v)
                        for k, v in values.items()
                        if k != "pixel_size"
                    }
                )
                kwargs = {"metadata": metadata}
                if "pixel_size" in values:
                    kwargs["pixel_size"] = values["pixel_size"]
                if "exposure_time" in values:
                    metadata["exposure_s"] = values["exposure_time"] / 1000
                    kwargs["timestamps"] = (
                        np.arange(len(current.source)) * metadata["exposure_s"]
                    )
                model.update_acquisition(acquisition, **kwargs)
            clear_statistics(model)

        transact(self.model, update)
        self._analysis_cache.clear()

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

    def import_tracks(
        self, dataset, channel, rows, *, group_column=None
    ) -> np.ndarray:
        """Import independent tracks, separating grouping identity from labels."""
        acquisition, name = self.partition(dataset, channel)
        incoming = self._incoming(dataset, channel, rows)
        group_column = "particle" if group_column is None else group_column
        if group_column not in incoming:
            if len(incoming["frame"]):
                raise ValueError(
                    f"Imported track rows require grouping column {group_column}"
                )
            incoming[group_column] = np.array([], dtype=np.int64)
        groups = incoming[group_column].copy()
        if pd.isna(groups).any():
            raise ValueError("Track grouping identities cannot be missing")
        if "particle" not in incoming:
            incoming["particle"] = groups.copy()
        if pd.isna(incoming["particle"]).any():
            raise ValueError("Track labels cannot be missing")
        # Independent imports always allocate fresh scientific identities.
        incoming.pop("id", None)
        incoming.pop("track_id", None)
        particles = incoming.pop("particle")
        for group in pd.unique(groups):
            if len(pd.unique(particles[groups == group])) != 1:
                raise ValueError(
                    "A track grouping identity requires one display particle label"
                )
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
            for group in pd.unique(groups):
                selected = groups == group
                members = ids[selected]
                particle = particles[selected][0]
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
        data["particle"] = data["particle"].astype(object)
        # Scientific track IDs are global. Legacy GUI labels remain per partition
        # and may differ from independent imported particle labels.
        for track_id in pd.unique(data["track_id"]):
            parameters = self.model.tracks.metadata[int(track_id)].get(
                "parameters", {}
            )
            if "particle" in parameters:
                data.loc[data.track_id == track_id, "particle"] = parameters[
                    "particle"
                ]
        return data.drop(columns=ORIGIN, errors="ignore")

    def ordinary_datasets(self):
        """Dataset labels with a regular scientific image source."""
        return tuple(d for d in self.datasets() if self.ordinary_channels(d))

    def ordinary_channels(self, dataset):
        return tuple(
            c
            for c in self.channels(dataset)
            if self.model.acquisitions[
                self.partition(dataset, c)[0]
            ].metadata.get("role")
            != "segmentation_image"
        )

    def selected_partitions(
        self, dataset="All Datasets", channel="All Channels"
    ):
        return tuple(
            (d, c)
            for d in self.ordinary_datasets()
            if dataset in ("All Datasets", d)
            for c in self.ordinary_channels(d)
            if channel in ("All Channels", c)
        )

    def delete_rows(self, dataset, channel, *, tracks=False, ids=None):
        """Delete selected localization IDs, or only their track memberships.

        Track deletion never removes detection rows. Imported track-only rows
        are independent localizations and are removed with their membership.
        """
        rows = (self.query_tracks if tracks else self.query_localizations)(
            dataset, channel, space="raw"
        )
        selected = (
            rows.id.to_numpy()
            if ids is None
            else np.asarray(ids, dtype=np.uint64)
        )
        if not np.isin(selected, rows.id).all():
            raise ValueError(
                "IDs must belong to the selected partition and row kind"
            )

        def delete(model):
            if tracks:
                for track_id, group in rows.groupby("track_id", sort=False):
                    remaining = group.id.to_numpy()[
                        ~np.isin(group.id, selected)
                    ]
                    if len(remaining) != len(group):
                        model.tracks.retain_members(int(track_id), remaining)
            else:
                model.localizations.delete(selected)
            if tracks and len(selected):
                native = model.localizations.query(ids=selected, space="raw")
                if ORIGIN in native:
                    imported = native["id"][native[ORIGIN] == IMPORTED_TRACK]
                    model.localizations.delete(imported)
            clear_statistics(model)

        transact(self.model, delete)
        self._analysis_cache.clear()

    def retain_rows(self, dataset, channel, ids, *, tracks=False):
        """Retain exactly these localization IDs within the selected row kind."""
        rows = (self.query_tracks if tracks else self.query_localizations)(
            dataset, channel, space="raw"
        )
        ids = np.asarray(ids, dtype=np.uint64)
        if not np.isin(ids, rows.id).all():
            raise ValueError(
                "IDs must belong to the selected partition and row kind"
            )
        self.delete_rows(
            dataset,
            channel,
            tracks=tracks,
            ids=rows.id.to_numpy()[~np.isin(rows.id, ids)],
        )

    def update_measurements(self, ids, columns):
        """Update numeric measurements by stable identity, atomically."""
        transact(
            self.model,
            lambda model: model.localizations.update_measurements(
                ids, **columns
            ),
        )
        self._analysis_cache.clear()

    def copy_rows(self, source, target, *, tracks=False):
        """Replace target rows with independent copies and fresh scientific IDs."""
        if source == target:
            return
        rows = (self.query_tracks if tracks else self.query_localizations)(
            *source
        )
        binding = self.partition(*target)
        incoming = rows.drop(
            columns=[
                "id",
                "track_id",
                "particle",
                "dataset",
                "channel",
                "group",
            ],
            errors="ignore",
        )
        incoming = self.display_to_raw(*target, incoming)
        columns = self._incoming(*target, incoming)
        columns[ORIGIN] = np.full(
            len(rows), IMPORTED_TRACK if tracks else DETECTION
        )

        def copy(model):
            staged = ProjectSession(model)
            staged.bind(*target, *binding)
            staged.delete_rows(*target, tracks=tracks)
            new_ids = model.localizations.add(
                acquisition_id=binding[0], channel=binding[1], **columns
            )
            if tracks:
                for track_id, group in rows.groupby("track_id", sort=False):
                    meta = deepcopy(
                        self.model.tracks.metadata[int(track_id)]["parameters"]
                    )
                    meta.update(
                        origin=IMPORTED_TRACK, particle=group.particle.iloc[0]
                    )
                    model.tracks.add(
                        new_ids[rows.track_id.to_numpy() == track_id],
                        metadata=meta,
                    )
            clear_statistics(model)

        transact(self.model, copy)
        annotations = deepcopy(self.annotations.get("partitions", {}))
        source_settings = annotations.get(source[0], {}).get(source[1], {})
        annotations.setdefault(target[0], {})[target[1]] = deepcopy(
            source_settings
        )
        self.set_annotations("partitions", annotations)
        self._analysis_cache.clear()

    def set_annotations(self, name: str, value: object):
        """Store detached editable geometry and invalidate scientific dependents.

        Vector annotations remain separate from raster masks. Editing a shape
        invalidates its bound model masks; it never silently rasterizes polygons.
        A model metadata token makes every prepared operation detect app edits.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Annotation names must be nonempty strings")
        detached = deepcopy(value)
        previous = self.annotations.get(name, {})
        mask_ids = set()
        if name in ("segmentations", "cells"):
            if isinstance(previous, dict):
                mask_ids.update(previous.get("mask_ids", ()))
            # Explicit incoming IDs bind newly published masks. Edits omit
            # that binding, invalidating masks derived from the old vertices.
            if isinstance(detached, dict):
                mask_ids.difference_update(detached.get("mask_ids", ()))

        def update(model):
            for mask_id in mask_ids:
                if mask_id in model.segmentations:
                    model.remove_segmentation(mask_id)
            if name in ("segmentations", "cells"):
                clear_statistics(model)
            model.metadata["moltrack_annotation_revision"] = (
                model.metadata.get("moltrack_annotation_revision", 0) + 1
            )

        transact(self.model, update)
        self.annotations[name] = detached
        self._analysis_cache.clear()

    def snapshot(self):
        staged = ProjectSession(self.model.snapshot())
        staged._partitions = deepcopy(self._partitions)
        staged.annotations = deepcopy(self.annotations)
        return staged

    def analyze_tracks(
        self,
        *,
        fit_lags=4,
        rolling_window=4,
        shape_data=None,
        progress_callback=None,
        cancel_event=None,
    ):
        from molcore.staging import state_token

        from .model_analysis import analyze_tracks
        from .model_processing import _cancel

        _cancel(cancel_event)

        geometry = tuple(
            (key, tuple(shape.wkb for shape in value))
            for key, value in (shape_data or {}).items()
            if key in ("polygons", "midlines")
        )
        key = (fit_lags, rolling_window, geometry)
        token = state_token(self.model)
        cached = self._analysis_cache.get(key)
        if cached is None or cached[0] != token:
            rows = analyze_tracks(
                self,
                fit_lags=fit_lags,
                rolling_window=rolling_window,
                shape_data=shape_data,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            self._analysis_cache[key] = (token, rows.copy(deep=True))
        _cancel(cancel_event)
        return self._analysis_cache[key][1].copy(deep=True)

    def display_to_raw(self, dataset, channel, rows):
        """Convert displayed points into the target source channel and frame."""
        from molcore import CoordinateSpace

        data = pd.DataFrame(rows).copy(deep=True)
        columns = self._incoming(dataset, channel, data)
        acquisition, name = self.partition(dataset, channel)
        columns.update(
            acquisition_id=np.full(len(data), acquisition),
            channel=np.full(len(data), name),
        )
        x, y = self.model.transforms.convert_coordinates(
            columns,
            source_space=CoordinateSpace.CORRECTED,
            target_space=CoordinateSpace.RAW,
        )
        data["x"], data["y"] = x, y
        return data

    def translate_image(self, dataset, channel, *, dx, dy):
        """Compose a manual image translation as a source calibration."""
        from molcore import Calibration

        acquisition, name = self.partition(dataset, channel)
        frame = int(self._view(dataset, channel).source_frames[0])

        def update(model):
            matrix = model.transforms.matrix(
                acquisition, name, frame, space="registered"
            )
            translation = np.eye(3)
            translation[:2, 2] = [dx, dy]
            calibration = f"moltrack-translation:{acquisition}:{name}"
            model.transforms.add_calibration(
                Calibration(calibration, translation @ matrix)
            )
            model.transforms.select(acquisition, name, calibration)
            clear_statistics(model)

        transact(self.model, update)
        self._analysis_cache.clear()


def clear_statistics(model):
    """Invalidate GUI-derived numeric statistics after their inputs change."""
    from .model_analysis import STAT_COLUMNS

    rows = model.localizations.query(space="raw")
    columns = {
        name: np.full(len(rows["id"]), np.nan)
        for name in STAT_COLUMNS
        if name in rows
    }
    if len(rows["id"]) and columns:
        model.localizations.update_measurements(rows["id"], **columns)
