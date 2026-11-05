# MolTrack Windows Installer Implementation Plan

Approved in chat on 2026-11-04. Implement in the current checkout, preserving
pre-existing uncommitted changes.

## Goal and architecture

Provide a per-user Windows 10/11 x64 setup EXE for nontechnical users. Freeze the
existing napari/PyQt6 application with PyInstaller in folder mode and distribute
it with Inno Setup. Launch MolTrack directly without development data or a console.
Keep Python 3.12, napari 0.9.2, Cellpose 3.0.9 and the pinned MolCore revision.

## Tasks

1. Test and implement a lightweight desktop launcher, GUI script and module entry
   point. Dispatch frozen workers before scientific imports. Make development
   data opt-out explicit without changing the existing plugin/development defaults.
   Log startup tracebacks and show a native failure dialog. Use user cache paths.
2. Create an independent locked Windows build environment with CPU-only
   matching torch/torchvision wheels, PyInstaller collection rules and runtime
   hooks. Test real native numerical work, JIT compilation and spawned processes
   through an executable smoke-test mode.
3. Add a repeatable PowerShell build and per-user Inno Setup configuration with
   shortcuts, stable upgrade identity, uninstaller and package-derived version.
   Build must stop if tests or frozen checks fail. Export dependency evidence
   and SHA256 alongside the setup EXE.
4. Add a manually triggered Windows GitHub Actions artifact build and installation,
   build, diagnostics and release acceptance documentation.

## Validation

Run launcher/diagnostic tests, existing GUI and processing tests, then build and
run the frozen executable. Review the complete change independently. Record
results and deviations in `.superpowers/sdd/windows-installer/progress.md`.
Clean-machine, model download/offline behavior,
interactive scientific workflows, upgrade/uninstall and non-ASCII paths remain
explicit release gates whenever hardware/environment is unavailable here.

## Defaults

Models download on first use; updates use a replacement installer. No automatic
updater, file associations, Windows ARM support or public release publishing.
Initial candidates are unsigned. The compiled app uses CPU only; CUDA remains
available to compatible Python/GitHub installations.

## User revisions

The user subsequently requested crash logs that can be sent manually: provide
log files, an exportable diagnostic ZIP, GUI error capture and native tracebacks
where possible. No automatic transmission.

The final distribution choice is **CPU-only for the compiled Windows app**.
Use official CPU torch/torchvision wheels only in the packaging environment.
Keep CUDA capability unchanged for Python installations from GitHub. CUDA
compiler bundling and real GPU validation therefore no longer apply to this
installer. The release checks in `docs/windows-installer.md` reflect this revision.
