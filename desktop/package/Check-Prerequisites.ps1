$ErrorActionPreference = 'Stop'
$folder = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $folder 'package-manifest.json'
$exePath = Join-Path $folder 'anime_graph_desktop.exe'

function Fail([string] $message) {
    Write-Error $message
    exit 1
}

if (-not [Environment]::Is64BitOperatingSystem) {
    Fail 'This candidate package requires Windows x64.'
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
    Fail 'The package manifest or desktop EXE is missing. Extract the complete folder.'
}
try {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    Fail 'The package manifest cannot be read. Extract a fresh copy.'
}
if ($manifest.format -ne 'anime-desktop-portable-v1' -or
    $manifest.target -ne 'x86_64-pc-windows-msvc') {
    Fail 'The package manifest format or target is unsupported.'
}
$exeRecord = @($manifest.files | Where-Object { $_.path -eq 'anime_graph_desktop.exe' })
if ($exeRecord.Count -ne 1 -or (Get-Item -LiteralPath $exePath).Length -ne [long] $exeRecord[0].bytes) {
    Fail 'The desktop EXE differs from the package manifest. Extract a fresh copy.'
}
$stream = [System.IO.File]::OpenRead($exePath)
$hasher = [System.Security.Cryptography.SHA256]::Create()
try {
    $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
} finally {
    $stream.Dispose()
    $hasher.Dispose()
}
if ($digest -ne $exeRecord[0].sha256) {
    Fail 'The desktop EXE differs from the package manifest. Extract a fresh copy.'
}

$vcMissing = @('VCRUNTIME140.dll', 'VCRUNTIME140_1.dll') | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $env:WINDIR "System32\$_") -PathType Leaf)
}
if ($vcMissing.Count -gt 0) {
    Fail 'Microsoft Visual C++ Redistributable x64 is required: https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist'
}

$runtimeId = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
$registryPaths = @(
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$runtimeId",
    "HKCU:\Software\Microsoft\EdgeUpdate\Clients\$runtimeId"
)
$webViewVersion = $null
foreach ($registryPath in $registryPaths) {
    $value = (Get-ItemProperty -LiteralPath $registryPath -Name pv -ErrorAction SilentlyContinue).pv
    $parsed = [version] '0.0.0.0'
    if ($value -and [version]::TryParse([string] $value, [ref] $parsed) -and
        $parsed -gt [version] '0.0.0.0') {
        $webViewVersion = [string] $value
        break
    }
}
if (-not $webViewVersion) {
    Fail 'Microsoft Edge WebView2 Evergreen Runtime is required: https://developer.microsoft.com/microsoft-edge/webview2/#download-section'
}
Write-Output "Desktop package prerequisites passed: WebView2 $webViewVersion; Visual C++ x64 runtime present."
