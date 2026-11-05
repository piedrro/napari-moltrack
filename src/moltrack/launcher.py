"""Desktop entry point. Keep imports lightweight for spawned Windows workers."""

import logging
import multiprocessing
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_fault_stream = None


def log_directory():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MolTrack" / "logs"


def configure_runtime():
    """Put application-owned writable files outside the installation folder."""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MolTrack"
    for variable, directory in (
        ("CELLPOSE_LOCAL_MODELS_PATH", "models"),
        ("NUMBA_CACHE_DIR", "cache/numba"),
        ("MPLCONFIGDIR", "cache/matplotlib"),
    ):
        path = Path(os.environ.setdefault(variable, str(base / directory)))
        path.mkdir(parents=True, exist_ok=True)
    os.environ["QT_API"] = "pyqt6"
    log_dir = log_directory()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "moltrack.log"
    logger = logging.getLogger()
    # Repeated calls (including tests) must close old file handles.
    for handler in logger.handlers[:]:
        if handler.name == "moltrack.desktop":
            logger.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=2,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.set_name("moltrack.desktop")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    import faulthandler
    import threading
    global _fault_stream
    faulthandler.disable()
    if _fault_stream is not None:
        _fault_stream.close()
    _fault_stream = open(log_dir / "native-crash.log", "a", encoding="utf-8")  # noqa: SIM115 -- process lifetime
    faulthandler.enable(file=_fault_stream, all_threads=True)
    sys.excepthook = _handle_exception
    threading.excepthook = _handle_thread_exception
    sys.unraisablehook = _handle_unraisable
    # Preserve print/traceback output from scientific workers in windowed apps.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None or (getattr(sys, "frozen", False) and
                                        getattr(sys, name).name == os.devnull):
            setattr(sys, name, open(log_dir / "console.log", "a", encoding="utf-8", buffering=1))  # noqa: SIM115 -- process lifetime
    return log_path


def _handle_exception(exc_type, value, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, value, tb)
        return
    logging.getLogger("moltrack.desktop").critical(
        "Unhandled application exception", exc_info=(exc_type, value, tb))
    _show_crash_error(str(log_directory()))


def _handle_thread_exception(args):
    logging.getLogger("moltrack.desktop").critical(
        "Unhandled exception in thread %s", args.thread.name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    _show_crash_error(str(log_directory()))


def _handle_unraisable(args):
    logging.getLogger("moltrack.desktop").error(
        "Unraisable exception: %s", args.err_msg,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def _show_crash_error(directory):
    if sys.platform == "win32":
        import ctypes
        answer = ctypes.windll.user32.MessageBoxW(
            None, "MolTrack encountered an unexpected error.\n\n"
            "You can send the developer a diagnostic report using Help > "
            "Save MolTrack diagnostic report, or send the files in:\n"
            f"{directory}\n\nOpen the log folder now?", "MolTrack error", 0x14)
        if answer == 6:  # IDYES
            os.startfile(directory)


def export_diagnostics(destination):
    """Save logs and version information for the user to send manually."""
    import importlib.metadata
    import json
    import platform
    import zipfile

    from moltrack import __version__

    for handler in logging.getLogger().handlers:
        handler.flush()
    if _fault_stream is not None:
        _fault_stream.flush()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    versions = {}
    for package in ("napari", "molcore", "cellpose", "torch", "torchvision", "numba", "PyQt6"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        directory = log_directory()
        for name in ("moltrack.log", "moltrack.log.1", "moltrack.log.2",
                     "native-crash.log", "console.log"):
            path = directory / name
            if path.is_file():
                archive.write(path, name)
        archive.writestr("system.json", json.dumps({
            "moltrack": __version__, "python": sys.version,
            "platform": platform.platform(), "packages": versions,
        }, indent=2))


def _add_diagnostic_menu(viewer):
    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    window = viewer.window._qt_window
    menu = next((action.menu() for action in window.menuBar().actions()
                 if action.text().replace("&", "").lower() == "help"), None)
    if menu is None:
        menu = window.menuBar().addMenu("Help")

    def open_logs():
        directory = log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def save_report():
        path, _ = QFileDialog.getSaveFileName(window, "Save MolTrack diagnostic report",
                                            "MolTrack-diagnostics.zip", "ZIP files (*.zip)")
        if path:
            try:
                export_diagnostics(path)
            except Exception as exc:
                logging.getLogger("moltrack.desktop").exception("Diagnostic export failed")
                QMessageBox.warning(window, "Could not save report", str(exc))

    menu.addAction("Open MolTrack log folder", open_logs)
    menu.addAction("Save MolTrack diagnostic report…", save_report)


def _log_notification(notification):
    """napari replaces exception hooks inside run(); retain those tracebacks."""
    from napari.utils.notifications import ErrorNotification

    logger = logging.getLogger("moltrack.desktop")
    if isinstance(notification, ErrorNotification):
        error = notification.exception
        logger.error("napari error: %s", notification.message,
                     exc_info=(type(error), error, error.__traceback__))
    else:
        severity = str(notification.severity).upper()
        logger.log(getattr(logging, severity, logging.INFO), "%s", notification.message)


def create_viewer(*, show=True):
    """Open the same MolTrack widget used by the napari plugin, with no data."""
    import napari
    from napari.utils.notifications import notification_manager

    from moltrack import MolTrack, __version__

    notification_manager.notification_ready.connect(_log_notification)

    viewer = napari.Viewer(title=f"MolTrack v{__version__}", show=False)
    try:
        widget = MolTrack(viewer, load_dev_data=False)
        viewer.window.add_dock_widget(widget, name=widget.windowTitle(), area="right")
        _add_diagnostic_menu(viewer)
        if show:
            viewer.show()
    except Exception:
        viewer.close()
        raise
    return viewer


def _show_startup_error(message):
    # A native Windows dialog still works when importing Qt itself failed.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "MolTrack could not start", 0x10)
    else:
        print(message, file=sys.stderr)


def main(argv=None):
    # This must precede napari, Qt, torch, argument parsing and viewer creation.
    multiprocessing.freeze_support()
    log_path = None
    args = None
    try:
        log_path = configure_runtime()
        logger = logging.getLogger("moltrack.desktop")
        logger.info("Starting MolTrack")
        import argparse
        parser = argparse.ArgumentParser(description="MolTrack desktop application")
        parser.add_argument("--smoke-test", metavar="REPORT.json",
                            help="check the installed application, write a report and exit")
        args = parser.parse_args(argv)
        viewer = create_viewer(show=not bool(args.smoke_test))
        if args.smoke_test:
            from moltrack.desktop_diagnostics import run_checks
            try:
                run_checks(viewer, args.smoke_test)
            finally:
                viewer.close()
            return 0
        import napari
        napari.run()
        return 0
    except Exception:
        logging.getLogger("moltrack.desktop").exception("MolTrack startup failed")
        location = f"Details were saved to:\n{log_path}" if log_path else (
            "Could not create the application log. Check that your user profile is writable."
        )
        if args is None or not args.smoke_test:
            _show_startup_error(f"MolTrack could not start.\n\n{location}")
        return 1
