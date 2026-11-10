"""Explicit legacy baselines: saved corrected pixels/rows are never corrected twice."""

import builtins
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from molcore import Acquisition, ArraySource, ChannelSpec, MissingSource

from .session import ProjectSession


class _LegacyObjects(pickle.Unpickler):
    """Narrow legacy-only adapter for the real application cell property types."""

    def find_class(self, module, name):
        from bactfit.cell import Cell, CellList
        from shapely import from_wkb

        allowed = {
            ("bactfit.cell", "Cell"): Cell,
            ("bactfit.cell", "CellList"): CellList,
            ("shapely.io", "from_wkb"): from_wkb,
            ("shapely", "from_wkb"): from_wkb,
        }
        for namespace in (
            "numpy",
            "numpy.core.multiarray",
            "numpy._core.multiarray",
            "numpy.core.numeric",
            "numpy._core.numeric",
        ):
            for key, value in {
                "ndarray": np.ndarray,
                "dtype": np.dtype,
                "recarray": np.recarray,
                "_reconstruct": np._core.multiarray._reconstruct,
                "scalar": np._core.multiarray.scalar,
                "_frombuffer": np._core.numeric._frombuffer,
            }.items():
                allowed[(namespace, key)] = value
        if module == "builtins" and name in {
            "set",
            "frozenset",
            "slice",
            "complex",
        }:
            return getattr(builtins, name)
        if (module, name) in allowed:
            return allowed[(module, name)]
        raise ValueError(f"Unsupported legacy Python object {module}.{name}")


def read_legacy(path):
    from molcore.legacy import read_moltrack

    try:
        return read_moltrack(path)
    except ValueError:
        # Public records-only migration deliberately rejects Cell/geometry;
        # only these explicit legacy adapters broaden its accepted types.
        with Path(path).open("rb") as stream:
            value = _LegacyObjects(stream).load()
        if not isinstance(value, dict):
            raise ValueError("Expected legacy project dictionary") from None
        return value


def _rows(project, dataset, channel, tracks=False):
    kind, key = (
        ("tracking_dict", "tracks")
        if tracks
        else ("localisation_dict", "localisations")
    )
    value = project.get(kind, {}).get(dataset, {}).get(channel, {})
    return value.get(key) if isinstance(value, dict) else value


def _shape(project, dataset, channel, data, pixels, messages):
    if pixels is not None:
        return tuple(np.asarray(pixels).shape), "embedded"
    record = (
        project.get("localisation_dict", {}).get(dataset, {}).get(channel, {})
    )
    for candidate in (
        record.get("image_shape"),
        data.get("image_shape"),
        data.get("shape"),
    ):
        if isinstance(candidate, dict):
            candidate = candidate.get(channel)
        if candidate is not None and len(candidate) == 3:
            return tuple(int(v) for v in candidate), "saved_descriptor"
    # Header recovery preserves geometry only. Rebinding original pixels to
    # corrected rows requires a complete, known correction mapping.
    path = data.get("path")
    if isinstance(path, str) and Path(path).is_file():
        import tifffile

        try:
            with tifffile.TiffFile(path) as source:
                shape = source.series[0].shape
            if len(shape) == 2:
                shape = (1, *shape)
            mode = str(data.get("multichannel_mode", "None"))
            crop = str(data.get("import_crop_mode", "None")).lower()
            known_mode = mode in {
                "None",
                "Single Channel",
                "Single",
                "Single File",
                "Multi File",
                "FRET",
                "ALEX",
            }
            if len(shape) == 3 and "brightest" not in crop and known_mode:
                n, h, w = shape
                limit = data.get("import_limit")
                if str(limit).isdigit():
                    n = min(n, int(limit))
                if "left" in crop:
                    w //= 2
                elif "right" in crop:
                    w -= w // 2
                if mode in ("FRET", "ALEX"):
                    if w % 2:
                        raise ValueError("Ambiguous legacy split geometry")
                    w //= 2
                if mode == "ALEX":
                    n = len(range(1 if channel in ("DA", "AA") else 0, n, 2))
                if data.get("frame_averaging"):
                    n = 1
                return (n, h, w), "source_header_only"
        except (OSError, ValueError):
            pass
    frames = [
        pd.DataFrame(value)
        for value in (
            _rows(project, dataset, channel),
            _rows(project, dataset, channel, True),
        )
        if value is not None and len(value)
    ]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def extent(column):
        values = (
            np.asarray(table[column], dtype=float) if column in table else []
        )
        return (
            max(1, int(np.ceil(np.nanmax(values))) + 1) if len(values) else 1
        )

    shape = tuple(extent(key) for key in ("frame", "y", "x"))
    messages.append(
        f"{dataset}/{channel}: exact image geometry unavailable; inferred table bounds {shape}; pixels unresolved"
    )
    return shape, "inferred_table_bounds"


def from_legacy(project, *, include_images=True):
    if not isinstance(project, dict) or not any(
        key in project
        for key in ("dataset_dict", "localisation_dict", "tracking_dict")
    ):
        raise ValueError("Unsupported legacy project dictionary")
    session, messages = ProjectSession(), []
    datasets = dict(project.get("dataset_dict", {}))
    for kind in ("localisation_dict", "tracking_dict"):
        for dataset in project.get(kind, {}):
            datasets.setdefault(dataset, {})
    for dataset, data in datasets.items():
        images = data.get("images", {})
        channels = dict.fromkeys(
            [
                *images,
                *project.get("localisation_dict", {}).get(dataset, {}),
                *project.get("tracking_dict", {}).get(dataset, {}),
            ]
        )
        if not channels:
            channels[data.get("channel_name", "image")] = None
        session.annotations.setdefault("legacy_dataset_metadata", {})[
            dataset
        ] = deepcopy({k: v for k, v in data.items() if k != "images"})
        for channel in channels:
            pixels = images.get(channel)
            shape, status = _shape(
                project, dataset, channel, data, pixels, messages
            )
            if len(shape) == 2:
                shape = (1, *shape)
            available = pixels is not None and include_images
            source = (
                ArraySource(np.asarray(pixels).reshape(shape))
                if available
                else MissingSource(
                    shape,
                    np.asarray(pixels).dtype
                    if pixels is not None
                    else data.get("dtype", "float32"),
                    "Legacy image pixels unavailable; explicit source reconciliation required",
                )
            )
            exposure = float(data.get("exposure_time", 1)) / 1000
            metadata = {
                "path": data.get("path", ""),
                "exposure_s": exposure,
                "import_format": "legacy_moltrack",
                "legacy_geometry_status": status,
                "legacy_baseline": "saved_corrected",
                "legacy_channel": channel,
                "legacy_image_binding": "embedded_corrected"
                if available
                else "missing_unresolved",
            }
            acquisition = Acquisition(
                source,
                channel_specs={channel: ChannelSpec()},
                pixel_size=data.get("pixel_size"),
                timestamps=np.arange(shape[0]) * exposure,
                metadata=metadata,
            )
            session.publish_acquisitions(
                [acquisition], [(dataset, channel, acquisition.id, channel)]
            )
            for tracks in (False, True):
                rows = _rows(project, dataset, channel, tracks)
                if rows is not None:
                    from .model_io import import_rows

                    import_rows(session, dataset, channel, rows, tracks=tracks)
            old = (
                project.get("localisation_dict", {})
                .get(dataset, {})
                .get(channel, {})
            )
            if isinstance(old, dict):
                session.annotations.setdefault("partitions", {}).setdefault(
                    dataset, {}
                )[channel] = deepcopy(
                    {k: v for k, v in old.items() if k != "localisations"}
                )
            messages.append(
                f"{dataset}/{channel}: saved corrected rows/pixels establish baseline; historical corrections retained as provenance only"
            )
            if not available:
                messages.append(
                    f"{dataset}/{channel}: original image path was not rebound to corrected rows; pixels unavailable"
                )
    for name in ("segmentations", "cells"):
        if name in project:
            value = project[name]
            shapes = deepcopy(value.get("shapes", value.get("data", [])))
            session.annotations[name] = {
                "data": shapes,
                "shape_types": deepcopy(
                    value.get("shape_types", ["polygon"] * len(shapes))
                ),
                "properties": deepcopy(value.get("properties", {})),
                "scale": deepcopy(
                    value.get("scale", [value.get("pixel_size", 1)] * 2)
                ),
                "mask_ids": [],
            }
    if "segmentation_image" in project:
        value = project["segmentation_image"]
        pixels = np.asarray(value["data"])
        if pixels.ndim == 2:
            pixels = pixels[None]
        # The standalone segmentation image is required project data even when
        # the user omits ordinary microscopy stacks.
        source = ArraySource(pixels)
        acquisition = Acquisition(
            source,
            pixel_size=value.get("pixel_size"),
            metadata={"role": "segmentation_image"},
        )
        binding = ("Segmentation Image", "Segmentation Image")
        session.publish_acquisitions(
            [acquisition], [(*binding, acquisition.id, "image")]
        )
        session.annotations["segmentation_image_binding"] = binding
    session.annotations["legacy_extra"] = deepcopy(
        {
            k: v
            for k, v in project.items()
            if k
            not in {
                "dataset_dict",
                "localisation_dict",
                "tracking_dict",
                "cells",
                "segmentations",
                "segmentation_image",
            }
        }
    )
    session.model.update_metadata({"migration_report": messages})
    from .project_container import validate_session

    validate_session(session)
    return session
