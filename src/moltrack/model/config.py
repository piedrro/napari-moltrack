"""Shared MolTrack processing settings."""

import os

_available_cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
NUM_CPUS = max(1, _available_cpus - 2)
if os.name == "nt":
    # ProcessPoolExecutor supports up to 61 workers on Windows.
    NUM_CPUS = min(NUM_CPUS, 61)
