# napari-moltrack

[![License BSD-3](https://img.shields.io/pypi/l/napari-moltrack.svg?color=green)](https://github.com/piedrro/napari-moltrack/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-moltrack.svg?color=green)](https://pypi.org/project/napari-moltrack)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-moltrack.svg?color=green)](https://python.org)
[![tests](https://github.com/piedrro/napari-moltrack/workflows/tests/badge.svg)](https://github.com/piedrro/napari-moltrack/actions)
[![codecov](https://codecov.io/gh/piedrro/napari-moltrack/branch/main/graph/badge.svg)](https://codecov.io/gh/piedrro/napari-moltrack)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-moltrack)](https://napari-hub.org/plugins/napari-moltrack)

A Napari plugin for single molecule localisation *and* tracking based on **Picasso**, **GPUfit** and **Trackpy**.
This plugin was designed to detect/track single molecules inside cells, but can be used for any other SMLM/tracking application.

All functions are parallelised/GPU accelerated where possible to increase performance.
Multiple datasets can be loaded and processed in parallel.

Single molecule localisations can be filtered by their properties (e.g. photons, width, etc.) and can be rendered as a super resolution image.

Segmentations can be used to exclude regions froHm single molecule localisation and tracking.
Segmentations can be added automatically using Cellpose or can be added manually. Includes tools for editing/modifying segmentations at a sub-pixel resolution.

Compatible with both single and multi channel .tif and .fits files.

napari-moltrack was written by Piers Turner, Kapanidis Group, University of Oxford....

https://www.physics.ox.ac.uk/research/group/gene-machines

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-moltrack` via [pip]:

    pip install napari-moltrack


To install latest development version :

    pip install git+https://github.com/piedrro/napari-moltrack.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-moltrack" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/piedrro/napari-moltrack/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
