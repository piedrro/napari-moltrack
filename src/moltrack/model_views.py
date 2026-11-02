"""Detached application projections of model-owned scientific rows.

GUI frame numbers are logical channel indices; model frames are camera frames.
Pixel coordinates remain pixels and acquisition calibration remains nanometres.
"""

from copy import deepcopy

import numpy as np
import pandas as pd


def channel_rows(rows, view):
    """Copy a model query and translate camera frames to GUI channel indices."""
    data = pd.DataFrame({k: deepcopy(v) for k, v in rows.items()})
    data["frame"] = np.asarray(
        [view.channel_index(int(f)) for f in data.frame], dtype=np.int64
    )
    return data


def gui_records(rows):
    """Return detached structured records, including stable identity columns."""
    return deepcopy(pd.DataFrame(rows).to_records(index=False))


def localization_points(rows):
    """Return napari point coordinates in frame, y, x order."""
    return pd.DataFrame(rows).loc[:, ["frame", "y", "x"]].to_numpy(copy=True)


def track_points(rows):
    """Return napari tracks using stable track IDs, then frame, y, x."""
    return (
        pd.DataFrame(rows)
        .loc[:, ["track_id", "frame", "y", "x"]]
        .to_numpy(copy=True)
    )


def legacy_snapshot(session, kind):
    """Read-only detached compatibility projection; never a mutable state store."""
    from types import MappingProxyType

    result = {}
    for dataset in session.datasets():
        channels = {}
        for channel in session.channels(dataset):
            if kind == "images":
                channels[channel] = session.image(dataset, channel).as_dask()
            else:
                rows = (
                    session.query_tracks
                    if kind == "tracks"
                    else session.query_localizations
                )(dataset, channel)
                rows["dataset"] = dataset
                rows["channel"] = channel
                records = gui_records(rows)
                records.flags.writeable = False
                annotations = (
                    session.annotations.get("partitions", {})
                    .get(dataset, {})
                    .get(channel, {})
                )
                channels[channel] = MappingProxyType(
                    dict(annotations, **{kind: records})
                )
        if kind == "images":
            result[dataset] = MappingProxyType(
                dict(
                    session.metadata(dataset),
                    images=MappingProxyType(channels),
                )
            )
        else:
            result[dataset] = MappingProxyType(channels)
    return MappingProxyType(result)


def gui_query(
    session,
    dataset,
    channel,
    *,
    tracks=False,
    return_dict=False,
    include_metadata=True,
):
    import pandas as pd

    frames = []
    result = []
    datasets = session.datasets() if dataset == "All Datasets" else (dataset,)
    group = 0
    for name in datasets:
        if name not in session.datasets():
            continue
        channels = (
            session.channels(name) if channel == "All Channels" else (channel,)
        )
        for ch in channels:
            if ch not in session.channels(name):
                continue
            rows = (
                session.query_tracks if tracks else session.query_localizations
            )(name, ch)
            if not len(rows):
                continue
            if include_metadata:
                rows["dataset"] = name
                rows["channel"] = ch
                if len(datasets) > 1:
                    rows["group"] = group
            else:
                rows = rows.drop(
                    columns=[
                        "dataset",
                        "channel",
                        "cell_index",
                        "segmentation_index",
                    ],
                    errors="ignore",
                )
            group += 1
            if return_dict:
                key = "tracks" if tracks else "localisations"
                item = dict(
                    dataset=name, channel=ch, **{key: gui_records(rows)}
                )
                if not tracks:
                    annotations = (
                        session.annotations.get("partitions", {})
                        .get(name, {})
                        .get(ch, {})
                    )
                    item.update(
                        image_shape=list(session.image(name, ch).shape),
                        fitted=annotations.get("fitted", False),
                        box_size=annotations.get("box_size", 7),
                        net_gradient=annotations.get("net_gradient", 0),
                    )
                result.append(item)
            else:
                frames.append(rows)
    if return_dict:
        return result
    return gui_records(pd.concat(frames, ignore_index=True)) if frames else []
