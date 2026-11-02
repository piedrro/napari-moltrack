# Molcore Migration Implementation Plan

Goal: Implement the user-approved full backend migration.
Architecture: One Experiment owned by a GUI-independent backend; GUI adapters query and mutate it through public APIs. Retain application algorithms and legacy import.
Constraints: Python >=3.12,<3.13; molcore GitHub default branch; preserve existing edits; do not modify molcore.

- [x] Model: implement backend acquisitions, localization and track adapters, metadata, snapshots and legacy conversion; verify frame/ID/coordinate invariants.
- [x] Processing: implement detached localisation, alignment, drift and Cellpose operations with cancellation/progress and guarded commit; test real CPU and controlled inference.
- [x] GUI: wire backend ownership, dictionary replacements, processors, controls, imports/exports and retained analyses.
- [x] Packaging: add Git dependency, remove incompatible/obsolete paths, refresh lock and document changes.
- [x] Verification: focused backend and Qt integration tests, full test suite and package build; review final diff.

Review focus: alternating-excitation frames, filtered IDs and memberships, cancellation and stale results, non-destructive transforms, project metadata and missing source images.

Verification: 53 tests passed against GitHub molcore b481f747; static correctness checks and uv lock --check passed; wheel and source distributions built and inspected. GPU inference and pretrained weight downloads were not exercised. Existing upstream Qt/NumPy deprecation warnings remain.

Compatibility: legacy pre-split channel names remain unchanged; original movie imports provide shared-camera provenance. Concatenation records sequences rather than flattening source frames. Measurement edits use stable-ID application metadata because molcore has no public measurement-edit API.
