"""Contracts for the installed desktop application."""

import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def isolate_runtime_environment(tmp_path, monkeypatch):
    # configure_runtime intentionally changes the process environment. Register
    # those keys with monkeypatch so later scientific tests never inherit a
    # cache directory belonging to a completed test.
    for name in ("CELLPOSE_LOCAL_MODELS_PATH", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        monkeypatch.setenv(name, str(tmp_path / "runtime" / name))
    import threading
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)


def test_launcher_import_does_not_load_gui_or_scientific_modules():
    result = subprocess.run(
        [sys.executable, "-c", ("import sys; import moltrack.launcher; "
                               "assert not {'napari', 'torch', 'qtpy'} & sys.modules.keys()")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_production_viewer_opens_moltrack_without_dev_data(qtbot, monkeypatch):
    from moltrack._widget import MolTrack
    from moltrack.launcher import create_viewer

    def unexpected_dev_import(self):
        raise AssertionError("Production startup must not import dev data")

    monkeypatch.setattr(MolTrack, "import_dev_data", unexpected_dev_import)
    viewer = create_viewer(show=False)
    try:
        assert len(viewer.layers) == 0
        widgets = viewer.window._qt_window.findChildren(MolTrack)
        assert len(widgets) == 1
        assert widgets[0].viewer is viewer
        assert viewer.title.startswith("napari-moltrack v")
    finally:
        viewer.close()


def test_startup_failure_writes_traceback_and_reports_log(tmp_path, monkeypatch):
    from moltrack import launcher

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    messages = []
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)

    def fail(**kwargs):
        raise RuntimeError("missing test DLL")

    monkeypatch.setattr(launcher, "create_viewer", fail)
    assert launcher.main([]) == 1
    log = tmp_path / "MolTrack" / "logs" / "moltrack.log"
    assert "missing test DLL" in log.read_text(encoding="utf-8")
    assert len(messages) == 1
    assert str(log) in messages[0]


def test_runtime_uses_writable_user_cache(tmp_path, monkeypatch):
    from moltrack.launcher import configure_runtime

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    for name in ("CELLPOSE_LOCAL_MODELS_PATH", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        monkeypatch.delenv(name, raising=False)
    import os

    configure_runtime()
    for name in ("CELLPOSE_LOCAL_MODELS_PATH", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        from pathlib import Path
        path = Path(os.environ[name])
        assert path.is_relative_to(tmp_path / "MolTrack")
        assert path.is_dir()


def test_unhandled_exception_is_exported_with_traceback(tmp_path, monkeypatch):
    import logging
    import zipfile

    from moltrack import launcher

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
    import threading
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    messages = []
    monkeypatch.setattr(launcher, "_show_crash_error", messages.append)
    launcher.configure_runtime()
    try:
        raise ValueError("diagnostic test crash")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    logging.getLogger("molcore").error("test backend diagnostic")
    report = tmp_path / "send-to-developer.zip"
    launcher.export_diagnostics(report)
    with zipfile.ZipFile(report) as archive:
        text = archive.read("moltrack.log").decode("utf-8")
        assert "ValueError: diagnostic test crash" in text
        assert "test backend diagnostic" in text
        assert "Traceback" in text
        assert "system.json" in archive.namelist()
    assert len(messages) == 1


def test_viewer_offers_exportable_diagnostics(qtbot):
    from moltrack.launcher import create_viewer

    viewer = create_viewer(show=False)
    try:
        actions = viewer.window._qt_window.menuBar().actions()
        menu_actions = [item.text() for action in actions if action.menu()
                        for item in action.menu().actions()]
        assert "Save MolTrack diagnostic report…" in menu_actions
        assert "Open MolTrack log folder" in menu_actions
    finally:
        viewer.close()


def test_napari_event_loop_exceptions_are_in_diagnostic_report(tmp_path, monkeypatch, qtbot):
    import zipfile

    from napari.utils.notifications import notification_manager

    from moltrack import launcher

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    launcher.configure_runtime()
    viewer = launcher.create_viewer(show=False)
    try:
        # napari.run enters this context and replaces sys/thread exception hooks.
        with notification_manager:
            try:
                raise RuntimeError("GUI event loop diagnostic error")
            except RuntimeError:
                sys.excepthook(*sys.exc_info())
        report = tmp_path / "gui-error.zip"
        launcher.export_diagnostics(report)
        with zipfile.ZipFile(report) as archive:
            log = archive.read("moltrack.log").decode("utf-8")
        assert "RuntimeError: GUI event loop diagnostic error" in log
        assert "Traceback" in log
    finally:
        viewer.close()
