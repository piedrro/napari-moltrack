# Windows desktop distribution

MolTrack's desktop installer opens the napari viewer with MolTrack already open.
End users do not install Python, napari, Git or CUDA development tools.
Target: Windows 10 (1809 or later)/11 on Intel/AMD x64 with an OpenGL-capable graphics driver.

## Build the Windows executable and installer

### 1. Install the build tools

Use 64-bit PowerShell on Windows x64. Install uv if `uv --version` fails.
See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/#winget).

```powershell
winget install --id astral-sh.uv -e
```

Install Inno Setup 6 to build the setup EXE. Choose **Install for all users**
if prompted and keep the default Program Files location. Windows may ask for
administrator approval. If WinGet is unavailable, run the installer from the
**Inno Setup 6** section of the [official downloads page](https://jrsoftware.org/isdl.php).

```powershell
winget install --id JRSoftware.InnoSetup -e -s winget -i
```

### 2. Build MolTrack

Open a new 64-bit PowerShell window and change to the root of your
`napari-moltrack` checkout. Then run the build script.

```powershell
.\packaging\windows\build.ps1
```

If the script reports that it cannot find Inno Setup, pass the full path to
`ISCC.exe`. For example, if you already have the compiler in the repository's
`.cache\inno-setup` folder, run this from the repository root.

```powershell
.\packaging\windows\build.ps1 -IsccPath (Resolve-Path '.\.cache\inno-setup\ISCC.exe').Path
```

The script creates an isolated environment under `packaging/windows/.venv`,
installs the separate locked Windows dependency set, runs the tests, freezes the
application, runs a packaged smoke test and compiles the installer. Python 3.12
is provided by uv if needed. Allow several GB for downloads and build outputs.
`-SkipTests` skips the source tests and still runs the packaged smoke check.

### 3. Find the build output

- Run `dist\windows\MolTrack-Setup-<version>-x64.exe` to install MolTrack.
- Run `dist\windows\MolTrack\MolTrack.exe` to launch the bundled app directly.
  Keep the entire `MolTrack` folder together when moving or copying it.

To build just the application folder without Inno Setup, run:

```powershell
.\packaging\windows\build.ps1 -BundleOnly
```

`dist/windows` also contains the installer's SHA256 checksum (for installer
builds), `dependencies.txt`, `uv.lock` and `smoke-test.json`. Builds are unsigned
until a signing identity is configured.

Alternatively, run **Actions → Windows installer → Run workflow** on GitHub and
download the `MolTrack-Windows-x64` artifact. The workflow builds only on demand
and does not publish a release or modify the existing PyPI deployment workflow.

## Dependencies and CPU-only installer

`packaging/windows/pyproject.toml` and its own `uv.lock` control the distribution;
the root development lock is independent. The root package is built from the
current checkout, including local changes. Napari remains 0.9.2, Cellpose 3.0.9,
and MolCore stays at the revision pinned by the root project.

The installer uses the official CPU-only PyTorch 2.9.1 / TorchVision 0.24.1 wheels.
It contains no CUDA runtime or toolkit and runs analysis and segmentation on CPU.
The frozen runtime also disables Numba CUDA, keeping Picasso's fitting selector
on CPU even when the destination computer has an NVIDIA driver.
See [PyTorch's version table](https://docs.pytorch.org/get-started/previous-versions/)
for the matched releases. These pins are confined to the Windows build project.
Installing the Python package from GitHub retains CUDA support when a compatible
NVIDIA GPU, driver and CUDA-enabled scientific dependencies are installed.
Fitting uses Picasso's implementation; the separate Gpufit/pygpufit package is
not required or bundled. Picasso's upstream attribution and license notices
for its ported fitting algorithms are retained.

After intentionally updating dependencies or the root package's metadata, run:

```powershell
uv lock --project packaging/windows --python 3.12
./packaging/windows/build.ps1
```

Commit the updated packaging lock only after testing. A frozen build needs
package metadata, Qt/Vispy resources, scientific DLLs and Python source for JIT
compilation; `packaging/windows/moltrack.spec` defines collection rules.

## Installation and user files

The installer uses `%LOCALAPPDATA%/Programs/MolTrack`, adds a Start menu shortcut,
offers a desktop shortcut and registers an uninstaller. It needs no administrator
rights. Run a newer installer to update the same application. Close MolTrack
before upgrading. Stale bundled libraries are replaced during upgrades.

Models download automatically on first use into `%LOCALAPPDATA%/MolTrack/models`.
An uncached model needs internet access; previously downloaded models can be used
offline. Custom model paths still work. Models, caches and logs survive upgrades
and uninstall. Project save locations are chosen by the user.

Startup failures show a dialog pointing to
`%LOCALAPPDATA%/MolTrack/logs/moltrack.log`. The native dialog also works when Qt
cannot import. A windowed build intentionally has no console window.

For errors while the application is running, choose **Help → Save MolTrack
diagnostic report…** to save a ZIP for the user to send to the developer. It
contains application logs, captured console output, native crash tracebacks
when available, and package/OS versions; it does not include datasets or upload
anything. **Help → Open MolTrack log folder** opens the underlying files.
If the application cannot reopen after a crash, the **MolTrack logs** Start menu
shortcut opens that same folder without starting MolTrack. Python errors are
logged with tracebacks; native crashes are captured where Python's fault handler
can report them. Power loss and forced termination may leave no traceback.

## Release acceptance

A successful CI build is a candidate installer, not a completed release test.
Keep a record of these checks for each release:

- On a clean Windows 10/11 x64 computer without Python: install as a standard
  user, launch from the Start menu and optional desktop shortcut. Confirm an
  empty viewer with MolTrack already open and no extra console or worker windows.
- Import TIFF/FITS movies, including multiple datasets, run localisation and
  tracking, fit multiple BactFit cells, cancel processing and export results.
  Save and reopen a `.moltrack` project, checking data and annotations.
- On a CPU-only computer: run segmentation and normal image analysis.
- Confirm the packaged smoke report has `cuda_build: null`, including when the
  computer happens to have an NVIDIA GPU; this installer intentionally uses CPU.
- Download a built-in Cellpose model, disconnect the network and segment again.
  With an uncached model offline, confirm a readable error and responsive UI.
- Exercise paths containing spaces and non-ASCII characters. Upgrade an older
  installation, verify saved projects still open, then uninstall. Confirm user
  projects, models and logs remain intact.

The installed EXE provides a diagnostic mode. In PowerShell, await it explicitly:

```powershell
$app = "$env:LOCALAPPDATA/Programs/MolTrack/MolTrack.exe"
$report = "$env:TEMP/moltrack-smoke.json"
$process = Start-Process $app -ArgumentList "--smoke-test `"$report`"" -PassThru -Wait
$process.ExitCode
Get-Content $report
```

The check opens a hidden real viewer,
starts a multiprocessing manager and worker, reads a synthetic TIFF, compiles
Numba numerical code and runs a CPU tensor calculation.
It exits and writes an OK report only on success. This supplements
the interactive workflow checks above.
