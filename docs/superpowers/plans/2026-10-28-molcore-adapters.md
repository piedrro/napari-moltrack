# MolCore adapters implementation plan

Goal: replace duplicated backend computation with MolCore while retaining the existing GUI and application dictionaries. Remove implementations and helpers after replacement; remove the bundled GPU fitting library entirely; upgrade napari to the latest stable release (0.9.2).

Architecture: GUI-independent adapters build transient MolCore datasets from the application's existing arrays and tables. Results are converted back into existing record arrays. MolCore owns detection, CPU fitting, tracking, rendering, registration, drift, photometry and trajectory calculations. A public ProcessorBackend extension uses Picasso's unified Gaussian GPU API. Application-specific cell geometry and plotting remain in MolTrack.

Constraints: preserve the UI layout and workflows; retain custom MolTrack detection; do not modify MolCore; preserve existing project/dependency edits. Use QtPy/PyQt6 for current napari. Processing must retain metadata, frames, input arrays and cancellation.

- [ ] Add headless adapters and regressions for coordinates, frame gaps, previews, measurements and cancellation.
- [ ] Route the existing GUI processing entry points through these adapters and remove replaced code.
- [ ] Remove the obsolete bundled fitting package, availability checks and documentation; expose Picasso GPU fitting in the existing selector.
- [ ] Upgrade napari, synchronize dependencies and test the actual GUI with Qt6.
- [ ] Run focused and integration tests, inspect the diff and build the package.

Review focus: empty results replacing stale data; active-frame processing preserving other frames; cancellation preventing partial commits; preservation of non-coordinate measurements; single/multiple channel geometry and metadata.
