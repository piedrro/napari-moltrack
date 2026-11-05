"""Repeatable installed-app checks, available through --smoke-test REPORT.json."""

import json
import multiprocessing
import os
from pathlib import Path


def scientific_probe():
    """Exercise native libraries and JIT compilation inside a spawned worker."""
    import importlib
    import tempfile

    import numba
    import numpy as np
    import tifffile
    import torch
    import torchvision
    for module in ("molcore", "picasso.fitting", "cellpose.models"):
        importlib.import_module(module)

    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    with tempfile.TemporaryDirectory(prefix="moltrack-check-") as directory:
        path = Path(directory) / "image.tif"
        tifffile.imwrite(path, image)
        image_sum = int(tifffile.imread(path).sum())

    # Compile real numerical work rather than merely importing Numba.
    @numba.njit
    def total(values):
        return values.sum()

    result = {
        "pid": os.getpid(), "image_sum": image_sum,
        "compiled_sum": int(total(image)),
        "cpu_tensor_sum": int(torch.arange(16).sum().item()),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
    }
    return result


def run_checks(viewer, report_path):
    """Raise on failure; only write an OK report after every check succeeds."""
    from concurrent.futures import ProcessPoolExecutor

    from npe2 import PluginManager

    from moltrack import __version__

    plugins = [PluginManager.instance().get_manifest(name).name
               for name in ("napari", "napari-moltrack")]

    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        state = manager.dict(status="ready")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            worker = executor.submit(scientific_probe).result(timeout=180)
        manager_value = state["status"]
    if any(worker[name] != 120 for name in ("image_sum", "compiled_sum", "cpu_tensor_sum")):
        raise RuntimeError("Scientific computations returned unexpected results")
    if len(viewer.layers):
        raise RuntimeError("Production viewer unexpectedly loaded data")
    report = {
        "ok": True, "version": __version__, "pid": os.getpid(),
        "layers": len(viewer.layers), "manager_value": manager_value,
        "worker": worker,
        "plugins": plugins,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
