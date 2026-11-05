[CmdletBinding()]
param(
    [string]$IsccPath = '',
    [switch]$SkipTests,
    [switch]$BundleOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$project = $PSScriptRoot
$python = Join-Path $project '.venv/Scripts/python.exe'
$output = Join-Path $repoRoot 'dist/windows'
$bundle = Join-Path $output 'MolTrack'
$work = Join-Path $repoRoot 'build/windows'
$previousQtApi = $env:QT_API
$previousUvEnvironment = $env:UV_PROJECT_ENVIRONMENT
$previousNumbaCuda = $env:NUMBA_DISABLE_CUDA
$previousPath = $env:PATH

Push-Location $repoRoot
try {
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or
        $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        throw 'Build on 64-bit Intel/AMD Windows using 64-bit PowerShell.'
    }
    $uv = (Get-Command uv -ErrorAction Stop).Source
    if (-not $BundleOnly) {
        if (-not $IsccPath) {
            $compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
            if ($compiler) { $IsccPath = $compiler.Source }
        }
        if (-not $IsccPath) {
            $candidates = @(
                "${env:ProgramFiles(x86)}/Inno Setup 6/ISCC.exe",
                "$env:ProgramFiles/Inno Setup 6/ISCC.exe"
            )
            $IsccPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        }
        if (-not $IsccPath -or -not (Test-Path -LiteralPath $IsccPath)) {
            throw 'Install Inno Setup 6, or pass -IsccPath with the path to ISCC.exe.'
        }
    }
    $env:QT_API = 'pyqt6'
    $env:NUMBA_DISABLE_CUDA = '1'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $project '.venv'
    New-Item -ItemType Directory -Force -Path $output, $work | Out-Null
    Invoke-Checked $uv @('sync', '--project', $project, '--locked', '--no-editable',
        '--reinstall-package', 'napari-moltrack', '--python', '3.12')
    # Single-quoted Python literals survive native argument passing in both
    # Windows PowerShell 5.1 and PowerShell 7.
    $verifyRuntime = @"
import sys, platform, torch, torchvision
assert sys.version_info[:2] == (3, 12)
assert platform.machine() == 'AMD64'
assert torch.version.cuda is None, 'CPU-only PyTorch is required'
print(torch.__version__, torchvision.__version__)
"@
    Invoke-Checked $python @('-c', $verifyRuntime)
    $version = (& $python -c 'from moltrack import __version__; print(__version__)').Trim()
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^\d+\.\d+\.\d+$') {
        throw "Expected a numeric three-part MolTrack version, got: $version"
    }
    if (-not $SkipTests) {
        Invoke-Checked $python @('-m', 'pytest', 'tests', 'src/moltrack/_tests', '-q', '--tb=short', '--disable-warnings')
    }
    # Native dependency discovery must not pick up unrelated DLLs from tools
    # on the caller's PATH (for example Poppler's incompatible ICU build).
    $basePython = (& $python -c 'import sys; print(sys.base_prefix)').Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not locate the Python runtime.' }
    $env:PATH = @((Split-Path $python), $basePython,
        "$env:SystemRoot/System32", $env:SystemRoot,
        "$env:SystemRoot/System32/Wbem") -join [IO.Path]::PathSeparator
    Invoke-Checked $python @('-m', 'PyInstaller', '--noconfirm', '--clean',
        '--distpath', $output, '--workpath', $work, (Join-Path $project 'moltrack.spec'))

    # Windowed executables must be awaited explicitly; & alone can return early.
    $report = Join-Path $output 'smoke-test.json'
    if (Test-Path -LiteralPath $report) { Remove-Item -LiteralPath $report }
    $process = Start-Process -FilePath (Join-Path $bundle 'MolTrack.exe') `
        -ArgumentList @('--smoke-test', ('"' + $report + '"')) -PassThru -WindowStyle Hidden
    if (-not $process.WaitForExit(300000)) {
        Stop-Process -Id $process.Id -Force
        throw 'Packaged smoke test timed out. Inspect %LOCALAPPDATA%/MolTrack/logs/moltrack.log.'
    }
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $report)) {
        throw 'Packaged smoke test failed. Inspect %LOCALAPPDATA%/MolTrack/logs/moltrack.log.'
    }
    $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if (-not $result.ok -or $null -ne $result.worker.cuda_build) {
        throw 'Packaged smoke report did not confirm a CPU-only build.'
    }
    $manifest = & $uv pip freeze --python $python
    if ($LASTEXITCODE -ne 0) { throw 'Failed to export dependency manifest.' }
    $manifest | Set-Content -LiteralPath (Join-Path $output 'dependencies.txt') -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $project 'uv.lock') -Destination (Join-Path $output 'uv.lock')
    Copy-Item -LiteralPath (Join-Path $output 'dependencies.txt') -Destination $bundle
    if ($BundleOnly) {
        Write-Host "Verified application bundle: $bundle"
        return
    }
    Invoke-Checked $IsccPath @("/DAppVersion=$version", "/DBundleDir=$bundle",
        "/DOutputDir=$output", (Join-Path $project 'installer.iss'))
    $installer = Join-Path $output "MolTrack-Setup-$version-x64.exe"
    $hash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($installer))" |
        Set-Content -LiteralPath "$installer.sha256" -Encoding ascii
    Write-Host "Installer: $installer"
}
finally {
    $env:QT_API = $previousQtApi
    $env:UV_PROJECT_ENVIRONMENT = $previousUvEnvironment
    $env:NUMBA_DISABLE_CUDA = $previousNumbaCuda
    $env:PATH = $previousPath
    Pop-Location
}
