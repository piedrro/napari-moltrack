"""Windows folder bundle. Run through build.ps1, never from a dev environment."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo,
    VarStruct, VSVersionInfo,
)
from moltrack import __version__

root = Path(SPECPATH).resolve().parents[1]
datas = [(str(root / "LICENSE"), ".")]
binaries = []
hiddenimports = []
# These packages discover adapters/plugins at runtime; collecting only the
# modules seen in static imports omits functionality from a seemingly valid EXE.
def runtime_module(name):
    if any(part in {"tests", "_tests", "dev", "examples", "benchmarks", "__main__"}
           for part in name.split(".")):
        return False
    return not name.startswith(("picasso.gui", "picasso.server", "cellpose.gui"))


for package in ("moltrack", "molcore", "napari", "napari_resources", "napari_builtins",
                "napari_console", "napari_svg", "ome_types", "npe2", "vispy", "bactfit",
                "picasso", "cellpose"):
    data, binary, imports = collect_all(
        package, include_py_files=False, filter_submodules=runtime_module,
        exclude_datas=["**/tests/**", "**/_tests/**", "**/dev/**", "**/examples/**",
                       "**/benchmarks/**"],
    )
    datas += data
    binaries += binary
    hiddenimports += imports
datas += copy_metadata("napari-moltrack", recursive=True)
datas += collect_data_files("matplotlib")
# Recent SciPy compatibility namespaces load their NumPy adapters dynamically.
hiddenimports += collect_submodules("scipy._external.array_api_compat")
datas += collect_data_files("scipy._external.array_api_compat", include_py_files=True)

version = tuple(int(part) for part in __version__.split("."))
version = (version + (0, 0, 0, 0))[:4]
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=version, prodvers=version, mask=0x3f,
                      flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
    kids=[StringFileInfo([StringTable("040904B0", [
        StringStruct("CompanyName", "MolTrack"),
        StringStruct("FileDescription", "MolTrack single molecule analysis"),
        StringStruct("FileVersion", __version__),
        StringStruct("ProductName", "MolTrack"),
        StringStruct("ProductVersion", __version__),
        StringStruct("OriginalFilename", "MolTrack.exe"),
    ])]), VarFileInfo([VarStruct("Translation", [1033, 1200])])],
)
a = Analysis(
    [str(root / "packaging/windows/entry.py")],
    pathex=[], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[str(root / "packaging/windows/hooks")],
    runtime_hooks=[str(root / "packaging/windows/runtime_hook.py")],
    excludes=["PyQt5", "PySide2", "PySide6", "pytest", "napari._tests",
              "bactfit.dev", "picasso.__main__", "picasso.gui", "picasso.server", "cellpose.gui"],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
    # Numba and inspect.getsource need physical source files inside the bundle.
    module_collection_mode={"napari": "pyz+py", "molcore": "pyz+py", "picasso": "pyz+py",
                            "cellpose": "pyz+py", "bactfit": "pyz+py",
                            "trackpy": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MolTrack",
          icon=str(root / "packaging/windows/moltrack.ico"),
          debug=False, strip=False, upx=False, console=False, version=version_info)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MolTrack")
