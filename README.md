# napari-moltrack

MolTrack is a napari plugin for single-molecule localisation microscopy (SMLM)
and single-particle tracking. It provides interactive tools for analysing
single and multichannel microscopy movies in TIFF and FITS formats.

## Core functions

- Detect and fit single-molecule localisations with Picasso, filter them by
  properties such as photon count and spot width, and render super-resolution
  images.
- Correct localisation drift with Picasso.
- Link localisations into trajectories with Trackpy and calculate track
  statistics such as diffusion coefficients.
- Measure fluorescence intensities and calculate FRET efficiencies from FRET
  and ALEX FRET data for individual localisations or tracks.
- Segment cells automatically with Cellpose or edit regions manually. Use
  BactFit to map localisations from rod-shaped cells onto a common cell model
  for renders and heatmaps.
- Save and reopen analysis projects in the `.moltrack.h5` format.

## Installation

### From GitHub

Installing from GitHub requires Python 3.12 and Git. In a Python 3.12 environment,
run the following commands to install and open MolTrack.

```sh
python -m pip install "git+https://github.com/piedrro/napari-moltrack.git"
python -m moltrack
```

### From PyPI

Install the earlier PyPI release, then open **MolTrack** from napari's
**Plugins** menu.

```sh
python -m pip install napari-moltrack
```

### Windows desktop installer

On Windows 10 or 11 (x64), run `napari-moltrack-Setup-<version>-x64.exe` and
open **napari-moltrack** from the Start menu. The installer bundles Python. See
the [Windows desktop distribution guide](docs/windows-installer.md) for how to
obtain a build.
