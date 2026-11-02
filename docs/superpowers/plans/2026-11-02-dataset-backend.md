# Dataset Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (selected by the user) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a persistent molcore Dataset own MolTrack scientific state across GUI workflows.

**Architecture:** A GUI-independent ProjectSession owns the model and explicit mutation/query services. Processing runs on snapshots and publishes validated results on the GUI thread. Native model persistence plus application annotations replaces dictionary persistence while retaining legacy import.

**Tech Stack:** Python 3.12, installed/locked molcore, NumPy, pandas, napari 0.9.2, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-dataset-backend-design.md`

## Global Constraints

- User authorized concrete molcore improvements when they improve this backend; test/review shared API changes and coordinate the verified dependency revision.
- Use supported public molcore APIs. Align Cellpose controls/settings with the user's Cellpose 3 update and existing updated dependency lock.
- Preserve GUI workflows, scientific formulas, metadata, timing, ordering, ROI/filter semantics, and legacy project usability.
- One authoritative scientific model; no mirrored mutable dictionary stores or mutable compatibility proxies.
- GUI pixel calibration is nm; exposure controls are ms; statistics use um and s.
- Independent sources may share a GUI dataset label without sharing an acquisition or sequence.
- Cancellation, stale results, and failed partitions must not publish partial state.
- Keep array-based backend entry points as compatibility adapters, but remove their use from normal model-based GUI processing.
- Use a fresh implementer and task reviewer per stage; do not run concurrent implementers.

## Review Focus

- Odd ALEX frame counts and split ROIs retain source frame and coordinate provenance (Task 2).
- Coincident independent imported track rows must not merge with detections (Task 1).
- Empty preview results clear only the selected frame, while failed batches preserve all prior partitions (Task 3).
- Deleted partitions and changed calibration invalidate pending workers and cached plots (Tasks 3–4).
- Image-free and already-corrected legacy projects reload without fabricated pixels or double correction (Task 5).

## File boundaries

- Create `src/moltrack/session.py`: ProjectSession, partition registry, identity, query, and mutation API.
- Create `src/moltrack/model_io.py`: acquisition/table conversion, model/application project container, legacy conversion.
- Create `src/moltrack/model_processing.py`: immutable requests, snapshot execution, validated publication, molcore operations.
- Create `src/moltrack/model_views.py`: detached GUI records and napari projections with unit/frame translation.
- Modify `_widget.py` and existing `funcs/*_utils.py`: capture settings, call services, refresh layers and plots.
- Retain `backend.py` array facade and extension backends; extract reusable extensions if needed without circular imports.
- Add focused session, model import, processing, persistence, and GUI integration tests alongside existing tests.

### Task 1: Persistent session, stable identity, and detached queries

**Files:** Create `session.py`, `model_views.py`, `tests/test_session.py`; inspect `backend.py` and installed molcore table APIs.

**Interfaces:** Produce `ProjectSession(model: Dataset | None = None)` with `.model`, `.annotations`, `datasets() -> tuple[str, ...]`, `channels(dataset: str) -> tuple[str, ...]`, `bind(dataset: str, channel: str, acquisition_id: str, model_channel: str) -> None`, and `partition(dataset: str, channel: str) -> tuple[str, str]`.
Produce `replace_localizations(dataset, channel, rows, *, frame_index=None) -> np.ndarray`, `import_tracks(dataset, channel, rows) -> np.ndarray`, `query_localizations(dataset, channel, *, space="corrected") -> pd.DataFrame`, and `query_tracks(dataset, channel, *, space="corrected") -> pd.DataFrame`. Queries carry stable `id`/`track_id` internally and retain legacy particle labels at GUI boundaries.

- [ ] Write tests: `test_detached_queries_preserve_columns_and_ids` mutates returned records and verifies model is unchanged; `test_preview_replacement_retains_other_frame_ids` verifies untouched IDs survive; `test_independent_tracks_do_not_merge_coincident_detections` imports equal x/y with different measurements and verifies distinct scope and membership.
- [ ] Run `.verify312/Scripts/python.exe -m pytest tests/test_session.py -q` and confirm new tests fail before implementation.
- [ ] Implement the interfaces using public tables and explicit detection/track-origin metadata; track memberships reference localizations. Separate application labels from model identifiers. Keep annotations JSON/typed-array serializable.
- [ ] Run session tests; retain all current backend tests. Obtain task review and fix findings before Task 2.

### Task 2: Model-owned imports, metadata, and GUI queries

**Files:** Create import portions of `model_io.py`, `tests/test_model_import.py`; modify `_widget.py`, `import_utils.py`, `events_utils.py`, `loc_filter_utils.py`, `tracking_utils.py`, `model_views.py`, and relevant display tests.

**Interfaces:** Consume Task 1. Produce `import_images(session, jobs) -> None` for detached prepared import descriptors, `import_rows(session, dataset, channel, rows, *, tracks=False) -> None`, `ProjectSession.image(dataset, channel, *, space="corrected")` returning a model-backed image view, `metadata(dataset) -> dict`, and `update_metadata(dataset, **values) -> None`. GUI getters retain their existing outward signatures and return detached session projections.

- [ ] Write tests `test_split_channels_share_source_and_roi`, `test_odd_alex_frame_maps_and_units`, `test_multifile_channels_have_independent_sources`, and `test_average_import_records_derived_source`; assert nm calibration, ms-to-s timing conversion, exact channel/source frame selection, and GUI coordinate round trips. Pin existing AD/DA channel selection instead of renaming channels.
- [ ] Run new import tests and observe failures. Then implement raw-source/channel bindings, independent source grouping, and averaged-source provenance. Keep concatenation explicit and preserve per-source metadata.
- [ ] Initialize `.session` in the widget; migrate import publication, selectors, active images, localisation/track getters, pixel-size updates, and basic display to session queries. No full movie materialization for display-only refreshes.
- [ ] Update test fixtures to construct sessions. Run session/import/display tests and widget initialization. Obtain task review; record any compatibility reads temporarily retained for later tasks.

### Task 3: Model processing and safe worker publication

**Files:** Create `model_processing.py`, `tests/test_model_processing.py`; modify `compute_utils.py`, `picasso_detect_utils.py`, `tracking_utils.py`, `pixstats_utils.py`, `picasso_render_utils.py`, `segmentation_utils.py`, `transform_utils.py`, `undrift_utils.py`, and backend extension locations as needed.

**Interfaces:** Consume persistent session and partition mapping. Produce frozen `ProcessingRequest(operation: str, partitions: tuple[tuple[str, str], ...], settings: object)`, `prepare(session, request) -> PreparedOperation`, `compute(prepared, *, progress=None, cancel_event=None) -> PendingOperation`, and `publish(session, pending, *, cancel_event=None) -> None`. Prepared/pending objects carry public molcore snapshot/freshness information and explicit application scope; workers never retain widget references. Use molcore `Dataset.apply_all`, staging or transactions for atomic publication.

- [ ] Write `test_cancelled_batch_preserves_model`, `test_changed_source_or_rows_reject_pending`, `test_failed_second_partition_preserves_first`, `test_empty_preview_clears_only_selected_frame`, and `test_fit_retains_candidate_identity_and_metadata`. Include deletion while computation is pending.
- [ ] Run new processing tests and verify failures. Implement snapshot preparation on the GUI thread, execution through existing molcore APIs/extensions, and publication on GUI result callbacks; carry all settings in request values.
- [ ] Migrate detection/fitting, tracking, photometry, rendering, segmentation, registration, and drift. Add `test_repeat_drift_does_not_accumulate` and `test_points_and_pixels_use_same_transform`; raw arrays and raw coordinates remain unchanged.
- [ ] Test tracking uses existing detection IDs, remove-unlinked semantics remain intact, and photometry preserves row correspondence/metric names. Run processing/backend/UI tests with fake deterministic expensive backends where appropriate. Obtain task review before editing/filtering migration.

### Task 4: Mutations, analysis, annotations, and remaining GUI consumers

**Files:** Extend `session.py`, `model_views.py`; modify `management_utils.py`, `loc_filter_utils.py`, `track_filter_utils.py`, `trackstats_utils.py`, `diffusion_utils.py`, `trackplot_utils.py`, `cell_heatmap_utils.py`, `cell_events.py`, `segmentation_events.py`, `segmentation_utils.py`, `bactfit_utils.py`, `traces_utils.py`, and `_widget.py` as needed. Add `tests/test_session_mutations.py` and model-backed GUI tests.

**Interfaces:** Produce `copy_rows(source: tuple[str,str], target: tuple[str,str], *, tracks=False) -> None`, `delete_rows(dataset, channel, *, tracks=False, ids=None) -> None`, `retain_rows(dataset, channel, ids, *, tracks=False) -> None`, `update_measurements(ids, columns) -> None`, and `set_annotations(name: str, value: object) -> None` on ProjectSession. Use existing molcore APIs for source mutations and revisions; keep strict geometry logic when no equivalent API exists.

- [ ] Write tests for copying with fresh IDs, deletion cascading to memberships/measurements, track-only deletion retaining detections, clear-all updates, and invalidating revision-keyed cached statistics after calibration edits.
- [ ] Run failures, implement explicit mutations, and replace every GUI scientific dictionary write. Derived statistics reference stable IDs; keep sample-standard-deviation and skipped-first-row behavior for current GUI aggregation.
- [ ] Add tests for overlapping polygons, strict boundary containment, editable polygon/cell-property retention, shape edits invalidating model masks, and empty results clearing existing layers. Bind segmentation masks explicitly; preserve vector annotations independently where needed.
- [ ] Migrate plotting/export-getter metadata reads, heatmaps, traces, cell tools, and statistics to the session. Use molcore trajectory/filtering/clustering APIs where compatible without reconstructing a temporary model per normal operation.
- [ ] Run mutation/display/GUI tests. Audit dictionary usages with `rg -n 'dataset_dict|localisation_dict|tracking_dict|segmentation_dict' src/moltrack` and allow only documented legacy serialization/conversion code. Obtain task review.

### Task 5: Versioned persistence and legacy project compatibility

**Files:** Extend `model_io.py`; modify project actions and table export/import in `export_utils.py`, `import_utils.py`; add `tests/test_model_persistence.py` and fixtures generated in tests.

**Interfaces:** Produce `save_project(session: ProjectSession, path, *, include_images=True) -> None`, `load_project(path, *, include_images=True) -> ProjectSession`, and `from_legacy(project: dict, *, include_images=True) -> ProjectSession`. Use a versioned application container carrying native molcore model payload plus typed-array/JSON application metadata, with an explicit format/version marker and atomic destination replacement. No widget or bactfit Python object in the native model payload.

- [ ] Write round-trip tests for sources, IDs/memberships, arbitrary measurement columns, transforms, frame maps, UI annotations, polygons, shape types, cell properties, and missing sources. Test failed serialization leaves the previous destination unchanged.
- [ ] Run failures, implement model persistence with molcore save/load and application annotation codec. Keep image checkbox behavior; image-free reload permits table analysis and pixel operations raise a clear image-unavailable error.
- [ ] Implement legacy format detection/conversion using molcore helpers where compatible and explicit binding for independent or cropped channel arrays. Test already-corrected legacy coordinates/pixels stay unchanged, independent imported tracks survive, partial/missing images preserve geometry, and conversion failure does not replace the live session.
- [ ] Route GUI project actions through candidate load/save and successful publication. Route compatible table/Picasso I/O through molcore file/interop functions while preserving chosen columns and metadata. Run persistence plus GUI tests and obtain task review.

### Task 6: Whole-application verification and cleanup

**Files:** All changed source/tests, README developer notes, plan/spec status.

**Interfaces:** No new public interfaces; close integration gaps and remove transitional paths.

- [ ] Audit normal GUI paths for temporary Dataset construction, scientific dictionary writes, widget reads from workers, detached-row mutation treated as persistence, and duplicated track positions. Remove bridges left by earlier tasks.
- [ ] Add integration coverage for import → detect/fit → track → measure → filter/edit → correct → save/reload with deterministic backends, checking identity and raw/corrected consistency across the chain.
- [ ] Run `$env:QT_API='pyqt6'; .verify312/Scripts/python.exe -m pytest tests src/moltrack/_tests -q --tb=short --disable-warnings` and applicable lint on changed/new modules; run `git diff --check`. Resolve failures and perform representative GUI verification when available.
- [ ] Obtain fresh whole-change review for spec compliance, identity/provenance, worker publication, and persistence. Resolve review findings and rerun affected verification. Mark completion only when all six tasks and reviews are complete; document any real dependency blocker explicitly rather than claiming a bridge completes the migration.

## Execution and review record

Design approved. User selected subagent implementation and review for each stage.
Plan self-review checked spec coverage, interface consistency, failure-mode ownership,
and staged migration boundaries. Implementation awaits the user's review of this plan,
as required by the writing-plans workflow.
