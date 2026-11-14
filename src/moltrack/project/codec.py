"""Explicit, pickle-free adapters for application annotations.

Only the named Cell/CellList adapters restore Python instances. Unknown objects
are errors, including attributes nested inside a known cell.
"""

import io

import numpy as np


class AnnotationCodec:
    def __init__(self, archive):
        self.archive = archive
        self.members = []

    def validate(self, value):
        """Check legacy candidates without writing or materializing any arrays."""
        from bactfit.cell import Cell, CellList
        from shapely.geometry.base import BaseGeometry

        if value is None or type(value) in (bool, int, float, str):
            return
        if isinstance(value, (np.generic, np.ndarray)):
            if value.dtype.hasobject:
                if value.dtype.names:
                    raise TypeError(
                        "Unsupported annotation structured object array"
                    )
                for item in np.asarray(value).flat:
                    self.validate(item)
            return
        if type(value) in (Cell, CellList):
            self.validate(vars(value))
            return
        if isinstance(value, BaseGeometry):
            return
        if type(value) is dict:
            for key, item in value.items():
                self.validate(key)
                self.validate(item)
            return
        if type(value) in (list, tuple):
            for item in value:
                self.validate(item)
            return
        raise TypeError(
            f"Unsupported annotation type: {type(value).__module__}.{type(value).__name__}"
        )

    def encode(self, value):
        from bactfit.cell import Cell, CellList
        from shapely.geometry.base import BaseGeometry

        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, np.generic):
            return self.encode(np.asarray(value)) | {"scalar": True}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                if value.dtype.names:
                    raise TypeError(
                        "Unsupported annotation structured object array"
                    )
                return {
                    "type": "object-array",
                    "shape": list(value.shape),
                    "items": [self.encode(v) for v in value.flat],
                }
            name = f"annotations/{len(self.members)}.npy"
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            self.archive.writestr(name, stream.getvalue())
            self.members.append(name)
            return {"type": "array", "member": name}
        if type(value) in (Cell, CellList):
            return {
                "type": type(value).__name__,
                "attributes": self.encode(vars(value)),
            }
        if isinstance(value, BaseGeometry):
            return {"type": "geometry", "wkb": value.wkb_hex}
        if type(value) is dict:
            return {
                "type": "dict",
                "items": [
                    [self.encode(k), self.encode(v)] for k, v in value.items()
                ],
            }
        if type(value) in (list, tuple):
            return {
                "type": type(value).__name__,
                "items": [self.encode(v) for v in value],
            }
        raise TypeError(
            f"Unsupported annotation type: {type(value).__module__}.{type(value).__name__}"
        )

    def decode(self, value):
        from bactfit.cell import Cell, CellList
        from shapely import wkb

        if value is None or type(value) in (bool, int, float, str):
            return value
        if not isinstance(value, dict):
            raise ValueError("Malformed annotation value")
        kind = value.get("type")
        if kind == "array":
            name = value["member"]
            if name not in self.members:
                raise ValueError("Unknown annotation array member")
            data = np.load(
                io.BytesIO(self.archive.read(name)), allow_pickle=False
            )
            return data[()] if value.get("scalar") else data
        if kind == "object-array":
            items = [self.decode(v) for v in value["items"]]
            data = np.empty(len(items), dtype=object)
            data[:] = items
            return data.reshape(value["shape"])
        if kind in ("Cell", "CellList"):
            obj = {"Cell": Cell, "CellList": CellList}[kind].__new__(
                {"Cell": Cell, "CellList": CellList}[kind]
            )
            attributes = self.decode(value["attributes"])
            if not isinstance(attributes, dict) or any(
                not isinstance(k, str) for k in attributes
            ):
                raise ValueError("Invalid cell attributes")
            obj.__dict__.update(attributes)
            return obj
        if kind == "geometry":
            return wkb.loads(value["wkb"], hex=True)
        if kind == "dict":
            return {self.decode(k): self.decode(v) for k, v in value["items"]}
        if kind in ("list", "tuple"):
            items = [self.decode(v) for v in value["items"]]
            return tuple(items) if kind == "tuple" else items
        raise ValueError(f"Unsupported annotation encoding: {kind}")
