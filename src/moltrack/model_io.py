"""Prepared image imports and explicit publication into the persistent model."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from molcore import Acquisition, ArraySource, ChannelSpec


def prepare_image(job, progress_callback=None):
    """Read original camera pixels; never split/repack channel frames in workers."""
    from .backend import read_movie

    result = dict(job)
    limit = (
        int(job["import_limit"])
        if str(job.get("import_limit")).isdigit()
        else None
    )
    result["source_images"] = np.stack(
        read_movie(
            job["path"], limit=limit, progress_callback=progress_callback
        )
    )
    return result


def _acquisition(job):
    images = np.asarray(job["source_images"])
    n, h, w = images.shape
    y0, y1, x0, x1 = 0, h, 0, w
    crop = job.get("import_crop_mode", "").lower()
    if "brightest" in crop:
        crop = (
            "left"
            if images[:, :, : w // 2].mean() > images[:, :, w // 2 :].mean()
            else "right"
        )
    if "left" in crop:
        x1 = w // 2
    elif "right" in crop:
        x0 = w // 2
    mode = job.get("multichannel_mode", "None")
    if mode in ("FRET", "ALEX"):
        if (x1 - x0) % 2:
            raise ValueError("Split channels require even image width")
        mid = (x0 + x1) // 2
        rois = [(y0, y1, x0, mid), (y0, y1, mid, x1)]
        if mode == "FRET":
            specs = dict(
                zip(
                    ("Donor", "Acceptor"),
                    (ChannelSpec(roi=r) for r in rois),
                    strict=False,
                )
            )
        else:
            # Historical GUI mapping: DA is donor odd, AD is acceptor even.
            specs = {
                name: ChannelSpec(
                    roi=rois[side], frame_start=start, frame_step=2
                )
                for name, side, start in [
                    ("DD", 0, 0),
                    ("DA", 0, 1),
                    ("AD", 1, 0),
                    ("AA", 1, 1),
                ]
            }
    else:
        name = (
            Path(job["path"]).name
            if mode == "Multi File"
            else job.get("channel_name", "image")
        )
        specs = {name: ChannelSpec(roi=(y0, y1, x0, x1))}
    metadata = {
        k: deepcopy(v)
        for k, v in job.items()
        if k not in ("source_images", "images")
    }
    exposure_s = float(job.get("exposure_time", 1)) / 1000
    metadata["exposure_s"] = exposure_s
    original = Acquisition(
        ArraySource(images),
        channel_specs=specs,
        pixel_size=float(job["pixel_size"]),
        timestamps=np.arange(n) * exposure_s,
        metadata=metadata,
    )
    if not job.get("frame_averaging"):
        return original, None
    # Averaging is intentionally a derived camera source, with original frame/ROI provenance.
    names = list(specs)
    means = [
        np.asarray(original.channels[name][:]).mean(axis=0) for name in names
    ]
    packed = np.concatenate(means, axis=1)[None]
    offsets = np.cumsum([0] + [a.shape[1] for a in means])
    derived_specs = {
        name: ChannelSpec(
            roi=(0, means[i].shape[0], int(offsets[i]), int(offsets[i + 1]))
        )
        for i, name in enumerate(names)
    }
    metadata.update(
        derived_operation="temporal_mean",
        source_acquisition_id=original.id,
        source_frames={
            name: original.channels[name].source_frames.tolist()
            for name in names
        },
        source_rois={
            name: list(original.channels[name].roi) for name in names
        },
        original_shape=list(images.shape),
    )
    return Acquisition(
        ArraySource(packed),
        channel_specs=derived_specs,
        pixel_size=original.pixel_size,
        timestamps=[0.0],
        metadata=metadata,
    ), original


def import_images(session, jobs, *, concatenate=False, clear=False):
    """Validate descriptors before publication; explicit concatenation is derived.

    Original acquisitions remain in the model, even when a mean or concatenated
    source supplies GUI pixels. Source segment maps resolve every derived frame.
    """
    prepared = [(job["dataset_name"], *_acquisition(job)) for job in jobs]
    originals = [
        original for _, _, original in prepared if original is not None
    ]
    if concatenate and prepared:
        first = prepared[0][1]
        for _, acquisition, _ in prepared:
            if (
                acquisition.pixel_size != first.pixel_size
                or acquisition.metadata["exposure_s"]
                != first.metadata["exposure_s"]
            ):
                raise ValueError(
                    "Concatenation requires matching calibration and exposure"
                )
            if tuple(acquisition.channels) != tuple(first.channels):
                raise ValueError("Concatenation requires matching channels")
            if any(
                acquisition.channels[ch].shape[1:]
                != first.channels[ch].shape[1:]
                for ch in first.channels
            ):
                raise ValueError(
                    "Concatenation requires matching channel geometry"
                )
        derived = []
        for channel in first.channels:
            segments, pixels, offset = [], [], 0
            timestamps, timestamp_offset = [], 0.0
            for _, acquisition, _ in prepared:
                view = acquisition.channels[channel]
                pixels.append(view[:])
                source_times = acquisition.timestamps[view.source_frames]
                timestamps.append(source_times + timestamp_offset)
                segments.append(
                    {
                        "acquisition_id": acquisition.id,
                        "channel": channel,
                        "frame_offset": offset,
                        "timestamp_offset_s": timestamp_offset,
                        "source_timestamps_s": source_times.tolist(),
                        "source_frames": view.source_frames.tolist(),
                        "roi": list(view.roi),
                        "path": acquisition.metadata["path"],
                    }
                )
                offset += len(view)
                # Explicit boundary policy: recordings abut after the last
                # camera exposure, with each channel's source phase preserved.
                timestamp_offset += (
                    float(acquisition.timestamps[-1])
                    + acquisition.metadata["exposure_s"]
                )
            metadata = dict(
                first.metadata,
                derived_operation="concatenate",
                source_segments=segments,
                timestamp_boundary_policy="abut_camera_exposures",
            )
            acquisition = Acquisition(
                ArraySource(np.concatenate(pixels)),
                channel_specs={channel: ChannelSpec()},
                pixel_size=first.pixel_size,
                timestamps=np.concatenate(timestamps),
                metadata=metadata,
            )
            derived.append((prepared[0][0], acquisition, None))
        originals.extend(acquisition for _, acquisition, _ in prepared)
        prepared = derived
    acquisitions, bindings = list(originals), []
    for dataset, acquisition, _ in prepared:
        acquisitions.append(acquisition)
        for name in acquisition.channels:
            bindings.append((dataset, name, acquisition.id, name))
    session.publish_acquisitions(acquisitions, bindings, clear=clear)


def import_rows(session, dataset, channel, rows, *, tracks=False):
    """Import external file rows with fresh IDs, retaining source identities."""
    import pandas as pd

    rows = pd.DataFrame(rows).copy(deep=True)
    group_column = None

    def retain_identity(field, preferred):
        name, suffix = preferred, 2
        while name in rows:
            name = f"{preferred}_{suffix}"
            suffix += 1
        rows[name] = rows.pop(field)
        return name

    if "id" in rows:
        retain_identity("id", "source_localization_id")
    if "localization_id" in rows:
        retain_identity("localization_id", "source_localization_id")
    if "track_id" in rows:
        group_column = retain_identity("track_id", "source_track_id")
        if tracks and "particle" not in rows:
            rows["particle"] = rows[group_column]
    if tracks:
        return session.import_tracks(
            dataset, channel, rows, group_column=group_column
        )
    return session.replace_localizations(dataset, channel, rows)


def select_table_columns(data, columns):
    """Keep chosen fields and the lossless repeated-import identity lineage."""
    import re

    return data[
        [
            name
            for name in data
            if columns is None
            or name in columns
            or re.fullmatch(
                r"source_(?:localization|track)_id(?:_[0-9]+)?", name
            )
        ]
    ]


def save_project(session, path, *, include_images=True):
    from .project_container import save_project as save

    return save(session, path, include_images=include_images)


def load_project(path, *, include_images=True):
    from .project_container import load_project as load

    return load(path, include_images=include_images)


def from_legacy(project, *, include_images=True):
    from .project_legacy import from_legacy as convert

    return convert(project, include_images=include_images)


def table_columns(rows):
    """Detach chosen GUI fields into the typed public file-helper boundary."""
    import pandas as pd

    data = pd.DataFrame(rows)
    result = {}
    for name in data:
        values = data[name].to_numpy(copy=True)
        if values.dtype.kind == "O":
            values = np.asarray(values.tolist())
        if values.ndim != 1 or values.dtype.hasobject:
            raise TypeError(f"Unsupported table column: {name}")
        result[name] = values
    return result
