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
