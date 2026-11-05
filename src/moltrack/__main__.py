"""Support ``python -m moltrack`` without running the development launcher."""

from moltrack.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
