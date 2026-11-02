"""Console entry point for the development launcher in an editable checkout."""

import runpy
from pathlib import Path


def main():
    launcher = Path(__file__).resolve().parents[2] / "development" / "dev.py"
    if not launcher.is_file():
        raise SystemExit("dev requires an editable napari-moltrack checkout.")
    runpy.run_path(str(launcher), run_name="__main__")
