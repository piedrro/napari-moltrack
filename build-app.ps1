<#
.SYNOPSIS
Refresh the Windows packaging lockfile and build the desktop application.
.EXAMPLE
.\build-app.ps1
.EXAMPLE
.\build-app.ps1 -BundleOnly
#>
[CmdletBinding()]
param(
    [string]$IsccPath = '',
    [switch]$SkipTests,
    [switch]$BundleOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$project = Join-Path $PSScriptRoot 'packaging/windows'
$uv = (Get-Command uv -ErrorAction Stop).Source
if (-not $IsccPath -and -not $BundleOnly) {
    $cachedCompiler = Join-Path $PSScriptRoot '.cache/inno-setup/ISCC.exe'
    if (Test-Path -LiteralPath $cachedCompiler) {
        $IsccPath = $cachedCompiler
    }
}

Write-Host 'Refreshing the Windows packaging lockfile...'
& $uv lock --project $project --python 3.12
if ($LASTEXITCODE -ne 0) {
    throw "Windows packaging lock update failed with exit code $LASTEXITCODE."
}

& (Join-Path $project 'build.ps1') -IsccPath $IsccPath `
    -SkipTests:$SkipTests -BundleOnly:$BundleOnly
