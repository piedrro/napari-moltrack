"""PyInstaller script; worker dispatch happens inside main before GUI imports."""

from moltrack.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
