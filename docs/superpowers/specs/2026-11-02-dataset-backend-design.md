# Dataset-owned MolTrack backend

Status: implemented and independently reviewed; final verification passed on 2026-11-04.

## Intent and scope

The user requested more use of molcore and a cleaner MolTrack backend, and
selected the broader Dataset migration over a focused adapter refactor.
Molcore will own scientific state throughout the application. MolTrack will
own Qt controls, napari layers, selection state, plotting, and application
metadata that molcore cannot represent.

This is a project-sized migration across the widget, backend, import/export,
processing, editing, filtering, and display code. It is complete only when
normal GUI workflows operate against the persistent scientific model.

## Existing implementation

- `MolTrack.initialise_variables` creates independent `dataset_dict`,
  `localisation_dict`, `tracking_dict`, and `segmentation_dict` stores.
- `backend.py` builds a temporary Dataset for each processing call. Tracking,
  localisation, photometry, rendering, segmentation, and drift estimation
  already delegate substantial computation to molcore.
- GUI mixins directly replace nested records and images. Track rows duplicate
  localisation coordinates and measurements. Background drift application
  checks dictionary object identity, then replaces image and row arrays.
- Project persistence pickles scientific dictionaries alongside polygon and
  cell-layer state. Projects may omit image pixels.
- The installed molcore provides stable IDs, acquisition/channel views,
  observations, track memberships, transforms, revisions, detached staging,
  import/export utilities, and native persistence. Use the installed API and
  locked dependency as the implementation baseline, rather than assuming the
  adjacent molcore checkout is identical.

## Approaches

1. Focused adapter refactor: retain authoritative dictionaries and delegate more
   conversions/computation. Smaller change, but repeated model construction and
   duplicated scientific state remain.
2. Dataset migration (selected by the user): persistent scientific models,
   explicit mutation services, and model-backed GUI queries. Larger change with
   broader regression coverage; removes the underlying duplication.

## Ownership and model mapping

Introduce a GUI-independent project/session service that owns one persistent
molcore Dataset for the open MolTrack project. Existing dataset selector entries
map to named acquisitions or explicit acquisition sequences within that model.
Stable model IDs are distinct from display labels and legacy particle numbers.

The service exposes explicit operations for import, replace, copy, delete,
measurement, processing application, and query. GUI handlers must use these
operations; mutable dictionary proxies must not become a second write path.
Temporary compatibility exports may return detached legacy-shaped arrays during
migration. Remove their use as authoritative state before completion.

Scientific ownership:

- Acquisitions own sources, channel geometry, source frame maps, exposure timing,
  and pixel calibration. Preserve camera-frame provenance for ALEX and split
  channels. Convert channel indices at GUI/import boundaries explicitly.
  A GUI dataset may group several acquisitions for Multi File channels; do not
  force independent sources into one acquisition or invent sequence continuity.
  FRET and ALEX channels from one camera source share an acquisition with explicit
  channel specifications. Averaged images are derived sources with recorded
  provenance. GUI pixel sizes are nanometres, as in molcore. GUI exposure times
  are milliseconds and convert to seconds for timestamps. Existing track-statistic
  columns use micrometres and seconds; preserve the displayed time convention.
- Localisations own stable IDs, raw positions, edits, and applicable measurement
  columns. Track membership references those IDs instead of owning copied rows.
- Independently imported track coordinates must remain independent if they
  cannot be matched unambiguously to existing localisations. Never match by row
  order or collapse distinct imported measurements merely because x/y coincide.
  Record their origin so localisation and track selectors preserve their scope.
- Photometry uses observations or the molcore-supported measurement destination
  appropriate to the operation. GUI metric names are translated at query/export
  boundaries. Preserve units and missing-value behavior.
- Registration and drift belong to model transforms. Preserve raw pixels and
  positions; image and point queries explicitly request raw, registered, or
  corrected views. Applying correction twice must not accumulate displacement.
- Segmentations use model masks with explicit acquisition/channel/frame/space
  bindings. Preserve editable polygon vertices, shape types, cell properties,
  and bactfit geometry as application annotations when masks cannot represent
  them losslessly. Do not change strict polygon containment into raster-based
  membership silently.

UI selection, contrast preferences, plot settings, and cell geometry remain
application state. Derived tables and rendered images are disposable caches
keyed by model revisions and query settings.

## Backend and worker boundary

Split backend responsibilities into the session/model service, processing
operations, import/export conversion, and GUI projection helpers. Keep
`moltrack.backend` as a compatibility facade while callers migrate; existing
array-based public functions can remain thin isolated adapters where useful.

Processing consumes persistent model selections or detached model snapshots.
Capture GUI settings on the GUI thread before dispatch. Workers receive no
widgets and do not mutate live application state. Use public molcore staging,
result application, and freshness APIs. Apply successful results on the GUI
thread after cancellation and dependency checks. Multi-selection operations
retain all-or-nothing publication where the current workflow promises it.
Failures, cancellation, or stale results preserve the previous scientific state.

Delegate to molcore's existing localisation, tracking, trajectory, photometry,
rendering, filtering, clustering, transforms, interop, and files APIs wherever
their semantics fit. Keep MolTrack's custom detector and GPU fitter as backend
extensions. Keep narrowly scoped compatibility logic for semantics molcore does
not provide, with explicit tests rather than changing scientific definitions.

## Behavior and compatibility

Keep the current GUI workflow, selectors, metric names, record ordering, extra
columns, pixel/exposure units, ROI edges, overlap handling, and filter bounds.
Preserve per-frame preview replacement and candidate identity through fitting.
Mutations must update or invalidate dependent tracks, measurements, and caches
through model APIs. Copy operations allocate new scientific identities; delete
operations explicitly handle dependents and clear affected displays.

Molcore's track aggregation differs from current GUI behavior (for example,
skipping the first row and sample versus population standard deviation). Preserve
existing behavior in compatibility logic until a separately authorized change.
Likewise, do not alter FRET formulae as part of the ownership refactor.

## Persistence

Keep existing `.moltrack` import and export usable. Add a versioned representation
for model-owned state using molcore persistence, together with lossless MolTrack
annotation/UI metadata. Implementation may use an application container around
the native model payload; expose it through the existing project actions.
Distinguish legacy and versioned formats before decoding.

Legacy import constructs a detached candidate project, validates all bindings,
and publishes it only after conversion succeeds. Use molcore legacy/file helpers
where compatible; extend MolTrack conversion for omitted images, multiple
channels, existing corrected arrays, and cell annotations. Do not reapply saved
drift or registration to legacy arrays already corrected in place. Retain
unrepresentable annotations and report conversion limitations instead of silently
dropping them.

Image-free projects retain geometry, channel/frame mappings, calibration, and
results using missing-source support. Analysis that only needs tables remains
available. Pixel-dependent actions explain the missing source. Saving a project
uses a temporary destination and publishes only after successful serialization;
failed saves must preserve the existing file.

## Delivery and validation

Implement in reviewable stages within this migration:

1. Model/session ownership, conversions, provenance, and stable identity tests.
2. Image/table import and query-driven display, selectors, and metadata updates.
3. Processing and worker publication against persistent models.
4. Copy/delete/filter/edit, masks/cells, plots, and cache invalidation.
5. Versioned persistence, legacy conversion, and removal of dictionary writes.

Validate each stage with focused integration tests, then run backend, display,
and GUI tests using the configured Python 3.12/Qt environment. Add meaningful
coverage for channel timing, copy/delete dependencies, stale/cancelled workers,
raw/corrected consistency, independent imported tracks, image-free projects,
legacy migration, annotation preservation, and save/reload round trips.
Audit remaining dictionary assignments and temporary `_model` creation in normal
GUI paths. Verify representative GUI workflows after the full suite passes.

Design review: a fresh builder reviewer found no important unanswered questions
before planning. Baseline verification: `tests/test_backend.py` passed all 17
tests in the existing Python 3.12 verification environment before code changes.

## Constraints, non-goals, and implementation decisions

User update: molcore now uses Cellpose 3; align controls and verification with
the updated lock (Cellpose 3.0.9, molcore a385ea6 or a later verified revision).
The user also authorized molcore updates when they improve the MolTrack backend.
Use supported public APIs; concrete missing shared capabilities may be added to
molcore with focused tests and independent review. Coordinate dependency revisions
and preserve unrelated changes in either repository.
Do not redesign controls, replace Qt/napari, change scientific formulas, discard
metadata, add mirrored mutable stores, or claim a partial bridge completes the
migration. Exact module names, container layout, and cache granularity are
implementation decisions, subject to the compatibility and ownership rules above.


## Implementation status (2026-11-04)

All six stages are implemented and independently approved, including persistence,
legacy-conversion and final whole-change review fixes. Stage 6 removed dictionary GUI
processing and compatibility snapshot properties. Normal processing uses the
persistent session and detached requests; array compatibility stays in backend.py.
The odd-frame ALEX integration chain exercises detection/fitting, tracking,
photometry, filtering/edits, repeated registration and native/application reload,
including stable identity and raw/corrected consistency.

The exact published MolCore dependency is
`1f6ea9a8b33796424566fd467567efdb00cac651`; both Python 3.12.3 development
runtimes have this revision and Cellpose 3.0.9. Final controller verification passed:
193 repository tests, two default-runtime widget/pipeline smoke tests, core lint,
undefined-name/whitespace/lock checks, and wheel/source builds. Whole-change review
and its scoped fix re-review are approved with no remaining findings.
