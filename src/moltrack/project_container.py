"""Atomic ZIP application containers wrapping the public native model format."""

import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
from molcore import Dataset, MissingSource

from .project_codec import AnnotationCodec
from .project_geometry import (
    GEOMETRY_LAYERS,
    RECORD_ID,
    encode_geometry,
    ensure_record_ids,
    restore_geometry,
)
from .session import ProjectSession

FORMAT = "moltrack-project"
VERSION = 1
APP_GROUP = "moltrack"


class _HDF5Annotations:
    """Store the existing pickle-free array encoding as HDF5 byte datasets."""

    def __init__(self, group):
        self.group = group

    def writestr(self, name, data):
        self.group.create_dataset(
            name, data=np.frombuffer(data, dtype=np.uint8)
        )

    def read(self, name):
        group = self.group
        *parents, leaf = name.split("/")
        for parent in parents:
            group = _local_member(group, parent, h5py.Group)
        data = _local_member(group, leaf, h5py.Dataset)
        if data.ndim != 1 or data.dtype != np.dtype("uint8"):
            raise ValueError("Invalid MolTrack annotation payload")
        return data[:].tobytes()


def _local_member(group, name, kind):
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise ValueError(
            f"Missing or nonlocal MolTrack project member {name!r}"
        )
    value = group[name]
    if not isinstance(value, kind):
        raise ValueError(f"Invalid MolTrack project member {name!r}")
    return value


def validate_session(session):
    for dataset in session.datasets():
        for channel in session.channels(dataset):
            acquisition, name = session.partition(dataset, channel)
            if name not in session.model.acquisitions[acquisition].channels:
                raise ValueError("Invalid project partition binding")
    if not isinstance(session.annotations, dict):
        raise ValueError("Project annotations must be a dictionary")
    AnnotationCodec(None).validate(session.annotations)
    for name in ("cells", "segmentations"):
        shapes = session.annotations.get(name)
        if shapes is None:
            continue
        if not isinstance(shapes, dict) or len(shapes.get("data", [])) != len(
            shapes.get("shape_types", [])
        ):
            raise ValueError(f"Invalid {name} shape annotations")
        for vertices, shape_type in zip(
            shapes["data"], shapes["shape_types"], strict=True
        ):
            vertices = np.asarray(vertices)
            if shape_type not in {
                "polygon",
                "path",
                "rectangle",
                "ellipse",
                "line",
            }:
                raise ValueError(f"Invalid {name} shape type")
            if (
                vertices.ndim != 2
                or vertices.shape[1] < 2
                or vertices.dtype.kind not in "iuf"
                or not np.isfinite(vertices).all()
            ):
                raise ValueError(f"Invalid {name} shape vertices")
            if len(vertices) < (3 if shape_type == "polygon" else 2):
                raise ValueError(f"Invalid {name} shape vertex count")
            if shape_type in {"rectangle", "ellipse"} and len(
                vertices
            ) not in {2, 4}:
                raise ValueError(f"Invalid {name} shape vertex count")
            if shape_type == "line" and len(vertices) != 2:
                raise ValueError(f"Invalid {name} shape vertex count")
        properties = shapes.get("properties", {})
        if not isinstance(properties, dict) or any(
            not isinstance(key, str)
            or np.asarray(value).ndim != 1
            or len(value) != len(shapes["data"])
            for key, value in properties.items()
        ):
            raise ValueError(f"Invalid {name} shape properties")
        scale = np.asarray(shapes.get("scale", []), dtype=float)
        if (
            scale.ndim != 1
            or len(scale) < 2
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError(f"Invalid {name} scale")
        dimensions = {
            np.asarray(vertices).shape[1] for vertices in shapes["data"]
        }
        if len(dimensions) > 1 or (dimensions and dimensions != {len(scale)}):
            raise ValueError(f"Incompatible {name} shape/scale dimensions")
        if any(
            mask not in session.model.segmentations
            for mask in shapes.get("mask_ids", [])
        ):
            raise ValueError(f"Invalid {name} native mask binding")
    binding = session.annotations.get("segmentation_image_binding")
    if binding is not None:
        try:
            session.partition(*binding)
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Invalid segmentation image annotation binding"
            ) from error


def save_project(session, path, *, include_images=True):
    staged = session.snapshot()
    validate_session(staged)
    identities = {
        name: ensure_record_ids(staged.annotations[name], name)
        for name in GEOMETRY_LAYERS
        if name in staged.annotations
    }
    encode_geometry(staged, staged._project_geometry)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Everything fallible, including annotation adapters and native storage,
    # completes before replacing the destination on the same filesystem.
    with tempfile.TemporaryDirectory(
        prefix=".moltrack-save-", dir=destination.parent
    ) as work:
        work = Path(work)
        output = work / "model.h5"
        staged.model.save(
            output, image_policy="full" if include_images else "none"
        )
        with h5py.File(output, "r+") as file:
            archive = _HDF5Annotations(file.create_group(APP_GROUP))
            codec = AnnotationCodec(archive)
            annotations = codec.encode(staged.annotations)
            manifest = {
                "format": FORMAT,
                "version": VERSION,
                "bindings": [
                    [d, c, *staged.partition(d, c)]
                    for d in staged.datasets()
                    for c in staged.channels(d)
                ],
                "annotations": annotations,
                "arrays": codec.members,
            }
            archive.writestr("manifest.json", json.dumps(manifest).encode())
        os.replace(output, destination)
    for name, ids in identities.items():
        session.annotations[name].setdefault("properties", {})[RECORD_ID] = ids


def load_project(path, *, include_images=True):
    path = Path(path)
    if h5py.is_hdf5(path):
        return _load_hdf5(path, include_images)
    if zipfile.is_zipfile(path):
        return _load_zip(path, include_images)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        raise ValueError("Invalid MolCore HDF5 project")
    from .project_legacy import from_legacy, read_legacy

    session = from_legacy(read_legacy(path), include_images=include_images)
    try:
        _migrate_geometry(session)
        return session
    except BaseException:
        session.close()
        raise


def _migrate_geometry(session):
    for name in GEOMETRY_LAYERS:
        if name in session.annotations:
            ensure_record_ids(session.annotations[name], name)
    encode_geometry(session, {})
    restore_geometry(session)
    validate_session(session)


def _read_app_state(path):
    with h5py.File(path, "r") as file:
        if APP_GROUP not in file:
            return None, {}
        archive = _HDF5Annotations(_local_member(file, APP_GROUP, h5py.Group))
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("format") != FORMAT
            or type(manifest.get("version")) is not int
            or manifest["version"] != VERSION
        ):
            raise ValueError("Unsupported MolTrack annotation format/version")
        codec = AnnotationCodec(archive)
        codec.members = manifest["arrays"]
        if not isinstance(codec.members, list) or any(
            not isinstance(name, str)
            or not re.fullmatch(r"annotations/[0-9]+\.npy", name)
            for name in codec.members
        ):
            raise ValueError("Invalid MolTrack annotation array names")
        return manifest["bindings"], codec.decode(manifest["annotations"])


def _load_hdf5(path, include_images):
    resource = tempfile.TemporaryDirectory(prefix="moltrack-project-")
    model = None
    try:
        bindings, annotations = _read_app_state(path)
        payload = Path(resource.name) / "model.h5"
        if include_images:
            # Keep lazy readers on a private snapshot if the user exports over
            # the original file, including an export that omits the images.
            shutil.copyfile(path, payload)
        else:
            with Dataset.open(path) as original:
                original.save(payload, image_policy="none")
        model = Dataset.open(payload)
        if include_images:
            _restore_relative_sources(model, path)
        session = ProjectSession(model)
        if bindings is None:
            bindings = [
                (acquisition.id, channel, acquisition.id, channel)
                for acquisition in model.acquisitions.values()
                for channel in acquisition.channels
            ]
            for acquisition in model.acquisitions.values():
                if acquisition.metadata.get("role") == "segmentation_image":
                    annotations["segmentation_image_binding"] = (
                        acquisition.id,
                        next(iter(acquisition.channels)),
                    )
                    break
        for binding in bindings:
            session.bind(*binding)
        session.annotations = annotations
        restore_geometry(session)
        validate_session(session)
        session._project_resources = resource
        return session
    except BaseException:
        if model is not None:
            model.close()
        resource.cleanup()
        raise


def _restore_relative_sources(model, path):
    # An HDF5 payload selected from a directory snapshot may resolve image
    # references relative to its original location.
    unresolved = [
        acquisition
        for acquisition in model.acquisitions.values()
        if isinstance(acquisition.source, MissingSource)
        and acquisition.source.reason.startswith(
            "External images unavailable:"
        )
    ]
    if unresolved:
        with Dataset.open(path) as original:
            for acquisition in unresolved:
                source_acquisition = original.acquisitions[acquisition.id]
                acquisition.source, source_acquisition.source = (
                    source_acquisition.source,
                    acquisition.source,
                )


def _load_zip(path, include_images):
    resource = tempfile.TemporaryDirectory(prefix="moltrack-project-")
    session = None
    model = None
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or any(
                name not in ("manifest.json", "model.h5")
                and not re.fullmatch(r"annotations/[0-9]+\.npy", name)
                for name in names
            ):
                raise ValueError("Unexpected or unsafe project ZIP member")
            manifest = json.loads(archive.read("manifest.json"))
            if (
                manifest.get("format") != FORMAT
                or type(manifest.get("version")) is not int
                or manifest.get("version") != VERSION
            ):
                raise ValueError("Unsupported MolTrack project format/version")
            arrays = manifest.get("arrays", [])
            if not isinstance(arrays, list) or set(names) != {
                "manifest.json",
                "model.h5",
                *arrays,
            }:
                raise ValueError("Invalid project member manifest")
            codec = AnnotationCodec(archive)
            codec.members = arrays
            annotations = codec.decode(manifest["annotations"])
            payload = Path(resource.name) / "model.h5"
            # Never extract arbitrary member paths.
            with (
                payload.open("wb") as target,
                archive.open("model.h5") as source,
            ):
                shutil.copyfileobj(source, target)
        model = Dataset.open(payload)
        if not include_images:
            # Native save performs the lossless descriptor conversion without
            # reading scientific pixels or mutating acquisition identities.
            stripped = Path(resource.name) / "without-images.h5"
            model.save(stripped, image_policy="none")
            model.close()
            model = Dataset.open(stripped)
        session = ProjectSession(model)
        for binding in manifest["bindings"]:
            session.bind(*binding)
        session.annotations = annotations
        validate_session(session)
        _migrate_geometry(session)
        session._project_resources = resource
        return session
    except BaseException:
        if model is not None:
            model.close()
        resource.cleanup()
        raise
