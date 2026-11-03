"""Detached merging of displayed localizations with atomic publication."""

from copy import deepcopy
from dataclasses import dataclass
from weakref import ref

import numpy as np
import pandas as pd
from molcore.staging import state_token, transact

from .model_processing import _cancel


@dataclass(frozen=True)
class PreparedMerge:
    session: object
    expected: object
    owner: object
    bindings: object
    datasets: tuple
    radius: float
    cancel_event: object = None


def prepare_merge(session, dataset, radius, *, cancel_event=None):
    datasets = (
        session.ordinary_datasets()
        if dataset == "All Datasets"
        else (dataset,)
    )
    return PreparedMerge(
        session.snapshot(),
        state_token(session.model),
        ref(session.model),
        deepcopy(session._partitions),
        datasets,
        float(radius),
        cancel_event,
    )


def compute_merge(prepared, progress_callback=None):
    from molcore import ClusteringSettings
    from molcore.clustering import cluster_points

    _cancel(prepared.cancel_event)
    results = []
    for dataset in prepared.datasets:
        _cancel(prepared.cancel_event)
        data = pd.DataFrame(
            prepared.session.query_localizations(
                dataset, prepared.session.ordinary_channels(dataset)[0]
            )
        ).iloc[:0]
        chunks = [
            prepared.session.query_localizations(dataset, ch)
            for ch in prepared.session.ordinary_channels(dataset)
        ]
        if chunks:
            data = pd.concat(chunks, ignore_index=True)
        merged = []
        groups = list(data.groupby("frame", sort=True))
        for index, (_, group) in enumerate(groups):
            _cancel(prepared.cancel_event)
            # Persistent query order (partition then stable ID) decides which
            # coincident channel observation survives, matching the GUI facade.
            labels, _ = cluster_points(
                group[["x", "y"]].to_numpy(),
                settings=ClusteringSettings(
                    radius=prepared.radius, min_samples=2
                ),
            )
            keep = (labels == -1) | ~pd.Series(labels).duplicated().to_numpy()
            merged.append(group.iloc[np.flatnonzero(keep)])
            if progress_callback is not None:
                progress_callback.emit(int(100 * (index + 1) / len(groups)))
        results.append(
            (dataset, pd.concat(merged, ignore_index=True) if merged else data)
        )
    _cancel(prepared.cancel_event)
    return prepared, results


def publish_merge(session, pending):
    from molcore import StaleResultError

    prepared, results = pending
    _cancel(prepared.cancel_event)
    if (
        session.model is not prepared.owner()
        or session._partitions != prepared.bindings
    ):
        raise StaleResultError("Session bindings changed")

    def commit(model):
        from moltrack.session import ProjectSession

        staged = ProjectSession(model)
        staged._partitions = deepcopy(prepared.bindings)
        for dataset, rows in results:
            _cancel(prepared.cancel_event)
            columns = rows.drop(
                columns=[
                    "id",
                    "acquisition_id",
                    "channel",
                    "particle",
                    "track_id",
                ],
                errors="ignore",
            )
            for channel in staged.ordinary_channels(dataset):
                _cancel(prepared.cancel_event)
                raw = staged.display_to_raw(dataset, channel, columns)
                staged.replace_localizations(dataset, channel, raw)
        from moltrack.session import clear_statistics

        clear_statistics(model)
        _cancel(prepared.cancel_event)

    transact(session.model, commit, expected=prepared.expected)
    session._analysis_cache.clear()
