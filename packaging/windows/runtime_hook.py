"""Run in the main process AND frozen workers before application imports."""

import faulthandler
import os
import sys
from pathlib import Path

logs = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MolTrack" / "logs"
logs.mkdir(parents=True, exist_ok=True)

os.environ["QT_API"] = "pyqt6"
# The downloadable Windows application intentionally has no CUDA dependencies.
# Keep Picasso on its CPU path even on hosts that happen to have a CUDA driver.
os.environ["NUMBA_DISABLE_CUDA"] = "1"
for name in ("stdout", "stderr"):
    if getattr(sys, name) is None:
        setattr(sys, name, open(logs / "console.log", "a", encoding="utf-8", buffering=1))  # noqa: SIM115 -- process lifetime

_crash_stream = open(logs / "native-crash.log", "a", encoding="utf-8")  # noqa: SIM115 -- process lifetime
faulthandler.enable(file=_crash_stream, all_threads=True)
