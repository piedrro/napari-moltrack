"""Atomic ZIP application containers wrapping the public native model format."""

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from molcore import Dataset

from .project_codec import AnnotationCodec
from .session import ProjectSession

FORMAT = "moltrack-project"
VERSION = 1


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
    validate_session(session)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Everything fallible, including annotation adapters and native storage,
    # completes before replacing the destination on the same filesystem.
    with tempfile.TemporaryDirectory(
        prefix=".moltrack-save-", dir=destination.parent
    ) as work:
        work = Path(work)
        output = work / "project.zip"
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            codec = AnnotationCodec(archive)
            annotations = codec.encode(session.annotations)
            manifest = {
                "format": FORMAT,
                "version": VERSION,
                "bindings": [
                    [d, c, *session.partition(d, c)]
                    for d in session.datasets()
                    for c in session.channels(d)
                ],
                "annotations": annotations,
                "arrays": codec.members,
            }
            session.model.save(
                work / "model.h5",
                image_policy="full" if include_images else "none",
            )
            archive.write(work / "model.h5", "model.h5")
            archive.writestr(
                "manifest.json", json.dumps(manifest, allow_nan=True)
            )
        os.replace(output, destination)


def load_project(path, *, include_images=True):
    if not zipfile.is_zipfile(path):
        from .project_legacy import from_legacy, read_legacy

        return from_legacy(read_legacy(path), include_images=include_images)
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
                import shutil

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
        session._project_resources = resource
        return session
    except BaseException:
        if model is not None:
            model.close()
        resource.cleanup()
        raise
