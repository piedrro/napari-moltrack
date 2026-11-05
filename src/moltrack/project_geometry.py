"""Translate project layer geometry at the MolCore persistence boundary."""

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import numpy as np
from molcore import BactFitCell, SegmentationOutline

RECORD_ID = "_molcore_id"
OUTLINE_SOURCE = "_molcore_outline_source"
FIT_SOURCE = "_moltrack_source_outline"
GEOMETRY_LAYERS = ("segmentations", "cells")


def _frame(vertices):
    if vertices.shape[1] == 2:
        return None
    if vertices.shape[1] == 3:
        frame = vertices[0, 0]
        if (
            frame >= 0
            and frame == int(frame)
            and np.all(vertices[:, 0] == frame)
        ):
            return int(frame)
    raise ValueError("Expected spatial vertices with an optional single frame")


def _view(model, outline):
    if outline is not None and outline.acquisition_id is not None:
        return model.acquisitions[outline.acquisition_id].channels[
            outline.channel
        ]
    return None


def _calibration(model, record):
    if isinstance(record, BactFitCell) and record.outline_id is not None:
        record = model.segmentation_outlines[record.outline_id]
    view = (
        _view(model, record)
        if isinstance(record, SegmentationOutline)
        else None
    )
    pixel_size = record.pixel_size
    if pixel_size is None and view is not None:
        pixel_size = view.acquisition.pixel_size
    return view, record.frame, pixel_size


def _spatial_scale(scale, pixel_size):
    if pixel_size is None:
        return np.ones(2)
    return pixel_size / np.asarray(scale[-2:], dtype=float)


def _project(model, record, points, scale):
    view, frame, pixel_size = _calibration(model, record)
    vertices = np.asarray(points)[:, ::-1] * _spatial_scale(scale, pixel_size)
    if len(scale) == 3:
        frame = (
            view.channel_index(frame)
            if frame is not None and view is not None
            else frame
        )
        vertices = np.column_stack(
            (np.full(len(vertices), frame or 0), vertices)
        )
    return vertices


def _coordinates(model, old, vertices, old_points, scale):
    vertices = np.asarray(vertices)
    frame = _frame(vertices)
    if old is None:
        if scale[-2] != scale[-1]:
            raise ValueError("Native geometry requires one spatial pixel size")
        return vertices[:, -2:][:, ::-1], frame, scale[-1]
    view, old_frame, pixel_size = _calibration(model, old)
    projected = _project(model, old, old_points, scale)
    if np.array_equal(vertices, projected) or np.array_equal(
        vertices, projected.astype(np.float32)
    ):
        return old_points, old_frame, pixel_size
    if vertices.shape[1] == 2:
        frame = old_frame
    elif old_frame is None and frame == 0:
        frame = None
    elif frame is not None and view is not None:
        frame = int(view.source_frames[frame])
    points = vertices[:, -2:] / _spatial_scale(scale, pixel_size)
    return points[:, ::-1], frame, pixel_size


def _new_id():
    return f"moltrack-{uuid4().hex}"


def outline_fingerprint(vertices):
    """Match the submitted outline through napari's float32 display conversion."""
    return sha256(np.asarray(vertices, dtype="<f4").tobytes()).hexdigest()


def ensure_record_ids(layer, name):
    """Assign stable identities before shapes are written to a project."""
    props = layer.setdefault("properties", {})
    ids = list(props.get(RECORD_ID, [""] * len(layer["data"])))
    groups = (
        ([index] for index in range(len(ids)))
        if name == "segmentations"
        else _fit_groups(layer, ids)
    )
    seen = set()
    for indices in groups:
        identifier = (
            next(
                (
                    ids[index]
                    for index in indices
                    if ids[index] and ids[index] not in seen
                ),
                None,
            )
            or _new_id()
        )
        seen.add(identifier)
        for index in indices:
            ids[index] = identifier
    props[RECORD_ID] = ids
    return ids


def _fit_groups(layer, ids):
    """Use native identity for existing pairs and names for newly drawn cells."""
    identified = defaultdict(list)
    for index, identifier in enumerate(ids):
        if identifier:
            identified[identifier].append(index)
    paired = set()
    for indices in identified.values():
        if len(indices) == 2 and {
            layer["shape_types"][i] for i in indices
        } == {"polygon", "path"}:
            paired.update(indices)
            yield indices
    by_name = defaultdict(list)
    for index, name in enumerate(layer["properties"].get("name", ())):
        if index not in paired and isinstance(name, str):
            by_name[name].append(index)
    for indices in by_name.values():
        if len(indices) == 2 and {
            layer["shape_types"][i] for i in indices
        } == {"polygon", "path"}:
            yield indices


def _outlines(model, layer, original):
    records = {}
    props = layer["properties"]
    ids = props.get(RECORD_ID, [""] * len(layer["data"]))
    bindings = props.get(OUTLINE_SOURCE, [None] * len(ids))
    for index, (vertices, kind, identifier, binding) in enumerate(
        zip(layer["data"], layer["shape_types"], ids, bindings, strict=True)
    ):
        if kind != "polygon":
            continue
        old = original.get(identifier) if identifier not in records else None
        try:
            points, frame, pixel_size = _coordinates(
                model,
                old,
                vertices,
                old.vertices if old else None,
                layer["scale"],
            )
            if old is not None:
                record = replace(old, vertices=points, frame=frame)
            else:
                source = binding if isinstance(binding, dict) else {}
                acquisition = source.get("acquisition")
                channel = source.get("channel")
                if acquisition not in model.acquisitions:
                    acquisition = channel = None
                if acquisition is not None:
                    view = model.acquisitions[acquisition].channels[channel]
                    frame = (
                        source["frame"]
                        if frame is None
                        else int(view.source_frames[frame])
                    )
                    pixel_size = view.acquisition.pixel_size
                record = SegmentationOutline(
                    identifier or _new_id(),
                    points,
                    acquisition_id=acquisition,
                    channel=channel,
                    frame=frame,
                    pixel_size=pixel_size,
                    space=source.get("space", "raw"),
                )
        except (ValueError, IndexError):
            # Nonplanar and nonuniform legacy shapes remain application annotations.
            continue
        records[record.id] = record
        yield index, record


def _fit_parameters(model, fit, scale):
    _, _, pixel_size = _calibration(model, fit)
    ratio = _spatial_scale(scale, pixel_size)
    return {
        "radius": fit.radius * ratio[0],
        "poly_params": _scale_polynomial(
            fit.polynomial_coefficients, ratio[0]
        ),
        "cell_poles": np.asarray(fit.poles)[:, ::-1] * ratio,
        "fit_error": fit.fit_error,
    }


def _scale_polynomial(coefficients, ratio):
    degree = len(coefficients) - 1
    return tuple(
        value * ratio ** (1 - degree + i)
        for i, value in enumerate(coefficients)
    )


def _fits(model, layer, surviving_outlines, source_outlines):
    props = layer["properties"]
    params = props.get("cell", ())
    ids = props.get(RECORD_ID, [""] * len(layer["data"]))
    for indices in _fit_groups(layer, ids):
        by_kind = {layer["shape_types"][index]: index for index in indices}
        polygon_index, path_index = by_kind["polygon"], by_kind["path"]
        if not len(params) or not all(
            isinstance(params[index], dict) for index in indices
        ):
            continue
        values = params[path_index]
        old = model.bactfit_cells.get(ids[path_index])
        try:
            scale = layer["scale"]
            polygon, polygon_frame, pixel_size = _coordinates(
                model,
                old,
                layer["data"][polygon_index],
                old.polygon if old else None,
                scale,
            )
            midline, frame, _ = _coordinates(
                model,
                old,
                layer["data"][path_index],
                old.midline if old else None,
                scale,
            )
            if frame != polygon_frame:
                continue
            if frame is None and old is None:
                frame = values.get("frame")
            if old is None and frame is not None and frame < 0:
                frame = None
            ratio = _spatial_scale(scale, pixel_size)
            displayed_poles = np.asarray(values["cell_poles"], dtype=float)[
                :, -2:
            ]
            poles = (displayed_poles / ratio)[:, ::-1]
            polynomial = _scale_polynomial(values["poly_params"], 1 / ratio[0])
            radius = values["radius"] / ratio[0]
            if old is not None:
                expected = _fit_parameters(model, old, scale)
                if np.array_equal(
                    values["cell_poles"], expected["cell_poles"]
                ):
                    poles = old.poles
                if np.array_equal(
                    values["poly_params"], expected["poly_params"]
                ):
                    polynomial = old.polynomial_coefficients
                if values["radius"] == expected["radius"]:
                    radius = old.radius
            outline_id = old.outline_id if old is not None else None
            if old is None:
                source = source_outlines.get(values.get(FIT_SOURCE))
                if source is not None:
                    view, source_frame, source_size = _calibration(
                        model, source
                    )
                    if view is not None and source_frame is not None:
                        source_frame = view.channel_index(source_frame)
                    if source_size in (None, pixel_size) and frame in (
                        None,
                        source_frame,
                    ):
                        outline_id = source.id
            old_outline = model.segmentation_outlines.get(outline_id)
            outline = surviving_outlines.get(outline_id)
            fit = BactFitCell(
                ids[path_index] or _new_id(),
                polygon,
                midline,
                poles,
                polynomial,
                radius,
                fit_error=values.get("fit_error"),
                outline_id=outline_id,
                frame=frame if outline_id is None else None,
                pixel_size=(old.pixel_size if old else pixel_size)
                if outline_id is None
                else None,
            )
            if old is not None and outline_id is not None:
                _, old_frame, _ = _calibration(model, old)
                if outline != old_outline or frame != old_frame:
                    if fit == old and frame == old_frame:
                        yield (polygon_index, path_index), None
                        continue
                    fit = replace(
                        fit,
                        outline_id=None,
                        frame=frame,
                        pixel_size=pixel_size,
                    )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        yield (polygon_index, path_index), fit


def _drop_rows(layer, drop):
    for key in ("data", "shape_types"):
        layer[key] = [
            value
            for index, value in enumerate(layer[key])
            if index not in drop
        ]
    for key, values in layer["properties"].items():
        layer["properties"][key] = [
            value for index, value in enumerate(values) if index not in drop
        ]


def encode_geometry(staged, managed):
    """Move supported geometry into the detached model, leaving row references."""
    model = staged.model
    original_outlines = dict(model.segmentation_outlines)
    layers = staged.annotations
    outlines = (
        dict(_outlines(model, layers["segmentations"], original_outlines))
        if "segmentations" in layers
        else {}
    )
    surviving = {
        key: value
        for key, value in original_outlines.items()
        if key not in managed.get("segmentations", ())
    }
    surviving.update((record.id, record) for record in outlines.values())
    source_outlines = {}
    for index, record in outlines.items():
        key = outline_fingerprint(layers["segmentations"]["data"][index])
        source_outlines[key] = None if key in source_outlines else record
    fits = (
        list(
            _fits(
                model,
                layers["cells"],
                surviving,
                source_outlines,
            )
        )
        if "cells" in layers
        else []
    )

    for identifier in original_outlines.keys() - surviving.keys():
        model.remove_segmentation_outline(identifier)
    for record in outlines.values():
        if original_outlines.get(record.id) != record:
            model.add_segmentation_outline(record)
    retained_fits = {fit.id for _, fit in fits if fit is not None}
    for identifier in managed.get("cells", set()) - retained_fits:
        if identifier in model.bactfit_cells:
            model.remove_bactfit_cell(identifier)
    for _, fit in fits:
        if fit is not None and model.bactfit_cells.get(fit.id) != fit:
            model.add_bactfit_cell(fit)

    for name, rows in (
        (
            "segmentations",
            {index: record.id for index, record in outlines.items()},
        ),
        (
            "cells",
            {
                index: fit.id
                for indices, fit in fits
                if fit
                for index in indices
            },
        ),
    ):
        if name not in layers:
            continue
        layer = layers[name]
        layer["data"] = list(layer["data"])
        layer["properties"][RECORD_ID] = [
            rows.get(index, "") for index in range(len(layer["data"]))
        ]
        for index in rows:
            layer["data"][index] = None
        if name == "cells" and rows:
            layer["properties"]["cell"] = [
                {
                    key: value
                    for key, value in params.items()
                    if key
                    not in {
                        "radius",
                        "poly_params",
                        "cell_poles",
                        "fit_error",
                        "frame",
                    }
                }
                if index in rows
                else params
                for index, params in enumerate(layer["properties"]["cell"])
            ]
    if "cells" in layers:
        _drop_rows(
            layers["cells"],
            {
                index
                for indices, fit in fits
                if fit is None
                for index in indices
            },
        )


def restore_geometry(session):
    """Build editable layer rows from native records and MolTrack row references."""
    model = session.model
    managed = {}
    for name, records in (
        ("segmentations", model.segmentation_outlines),
        ("cells", model.bactfit_cells),
    ):
        layer = session.annotations.get(name)
        if layer is None and not records:
            continue
        if layer is None:
            frames_and_sizes = [
                _calibration(model, record)[1:] for record in records.values()
            ]
            pixel_size = next(
                (size for _, size in frames_and_sizes if size is not None), 1.0
            )
            scale = (
                [1.0]
                if any(frame is not None for frame, _ in frames_and_sizes)
                else []
            ) + [pixel_size] * 2
            layer = {
                "data": [],
                "shape_types": [],
                "properties": {},
                "scale": scale,
            }
            session.annotations[name] = layer
        props = layer["properties"]
        layer["data"] = list(layer["data"])
        layer["shape_types"] = list(layer["shape_types"])
        ids = list(props.get(RECORD_ID, [""] * len(layer["data"])))
        props[RECORD_ID] = ids
        restored = set()
        for index, identifier in enumerate(ids):
            if layer["data"][index] is not None:
                continue
            record = records[identifier]
            _restore_row(model, layer, name, index, record)
            restored.add(identifier)
        for identifier, record in records.items():
            if identifier in restored:
                continue
            for kind in (
                ("polygon",)
                if name == "segmentations"
                else ("polygon", "path")
            ):
                index = len(layer["data"])
                for key in ("name", "cell") if name == "cells" else ():
                    props.setdefault(key, [None] * index)
                for key, values in props.items():
                    props[key] = [
                        *values,
                        identifier if key == RECORD_ID else None,
                    ]
                layer["data"].append(None)
                layer["shape_types"].append(kind)
                _restore_row(model, layer, name, index, record)
            restored.add(identifier)
        managed[name] = restored
    session._project_geometry = managed


def _restore_row(model, layer, name, index, record):
    if name == "segmentations":
        points = record.vertices
    else:
        points = (
            record.polygon
            if layer["shape_types"][index] == "polygon"
            else record.midline
        )
        props = layer["properties"]
        names = props["name"]
        label = names[index] if names[index] is not None else record.id
        if not isinstance(names, list):
            props["name"] = names = list(names)
        names[index] = label
        values = props["cell"][index]
        params = dict(values) if isinstance(values, dict) else {}
        params.update(
            _fit_parameters(model, record, layer["scale"]), name=label
        )
        if not isinstance(props["cell"], list):
            props["cell"] = list(props["cell"])
        props["cell"][index] = params
    layer["data"][index] = _project(model, record, points, layer["scale"])
