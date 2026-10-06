param(
    [Parameter(Mandatory = $true)] [string] $KitDirectory,
    [switch] $RequireNoCheckout,
    [switch] $ForceKeyboardPicker
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms

function Require([bool] $condition, [string] $message) {
    if (-not $condition) { throw $message }
}

function TextNames($window) {
    $items = $window.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition
    )
    $names = [System.Collections.Generic.List[string]]::new()
    for ($index = 0; $index -lt $items.Count; $index++) {
        $item = $items.Item($index)
        if ($item.Current.ControlType -eq [System.Windows.Automation.ControlType]::Text) {
            $names.Add($item.Current.Name)
        }
    }
    return $names.ToArray()
}

function WaitForText($window, [string] $prefix, [string] $state) {
    $deadline = (Get-Date).AddSeconds(15)
    do {
        $names = @(TextNames $window)
        if (@($names | Where-Object { $_.StartsWith($prefix, [StringComparison]::Ordinal) }).Count -gt 0) {
            return $names
        }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    throw "Desktop smoke timed out waiting for $state."
}

function RequireText([string[]] $names, [string] $fragment, [string] $state) {
    Require (@($names | Where-Object { $_.Contains($fragment) }).Count -gt 0) "Desktop smoke missing $state."
}

function HasStat([string[]] $names, [string] $label, [string] $expected) {
    $index = [Array]::IndexOf($names, $label)
    return ($index -ge 0 -and $index + 1 -lt $names.Count -and $names[$index + 1] -eq $expected)
}

function WaitForCounts($window, [string] $state) {
    $deadline = (Get-Date).AddSeconds(15)
    do {
        $names = @(TextNames $window)
        if ((HasStat $names 'Anime' '8') -and (HasStat $names 'Signed pairs' '11')) {
            return $names
        }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    Write-Host "Desktop count diagnostic ($state): Anime label=$($names -contains 'Anime'); 8=$($names -contains '8'); Signed pairs label=$($names -contains 'Signed pairs'); 11=$($names -contains '11')."
    throw "Desktop smoke timed out waiting for $state counts."
}

function InvokeButton($window, [string] $name) {
    $condition = [System.Windows.Automation.AndCondition]::new(
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Button
        ),
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::NameProperty, $name
        )
    )
    $button = $window.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
    Require ($null -ne $button) "Desktop smoke cannot find $name."
    $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
}

function SelectManifest($root, $window, [int] $processId, [string] $path,
    [bool] $forceKeyboard) {
    InvokeButton $window 'Select local manifest'
    $dialogCondition = [System.Windows.Automation.AndCondition]::new(
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $processId
        ),
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ClassNameProperty, '#32770'
        )
    )
    $deadline = (Get-Date).AddSeconds(15)
    $dialog = $null
    do {
        $dialog = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $dialogCondition)
        if ($null -ne $dialog) { break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    Require ($null -ne $dialog) 'Desktop smoke cannot find the native file picker.'
    $editCondition = [System.Windows.Automation.AndCondition]::new(
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Edit
        ),
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::AutomationIdProperty, '1148'
        )
    )
    $deadline = (Get-Date).AddSeconds(15)
    $edit = $null
    if (-not $forceKeyboard) {
        do {
            $edit = $dialog.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $editCondition)
            if ($null -ne $edit) { break }
            Start-Sleep -Milliseconds 200
        } while ((Get-Date) -lt $deadline)
    }
    if ($null -ne $edit) {
        $edit.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern).SetValue($path)
        $edit.SetFocus()
    } else {
        Require ($dialog.Current.Name -eq 'Select release-manifest.json') `
            'Desktop smoke did not find the expected file picker.'
        Require ($path -match '^[A-Za-z]:\\[A-Za-z0-9_ .\\-]+$') `
            'The invented manifest path cannot be entered by the keyboard picker.'
        [System.Windows.Forms.SendKeys]::SendWait('%n')
        Start-Sleep -Milliseconds 100
        [System.Windows.Forms.SendKeys]::SendWait('^a')
        [System.Windows.Forms.SendKeys]::SendWait($path)
    }
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
}

$kit = [IO.Path]::GetFullPath($KitDirectory)
Require (Test-Path -LiteralPath $kit -PathType Container) 'Desktop smoke kit directory is missing.'
if ($RequireNoCheckout) {
    if (Test-Path -LiteralPath $env:GITHUB_WORKSPACE) {
        $checkoutEntry = Get-ChildItem -LiteralPath $env:GITHUB_WORKSPACE -Force |
            Select-Object -First 1
        Require ($null -eq $checkoutEntry) 'The checkout still exists on the clean smoke runner.'
    }
    Require ([IO.Path]::GetFullPath((Get-Location).Path) -eq $kit) `
        'The app is not launching from the isolated smoke directory.'
}
Require (-not (Test-Path -LiteralPath (Join-Path $kit 'data'))) `
    'The smoke kit unexpectedly contains a relative data directory.'

$innerZip = Join-Path $kit 'anime-graph-desktop-windows-x64.zip'
$bundle = Join-Path $kit 'invented-bundle/release-manifest.json'
$missing = Join-Path $kit 'invented-missing/release-manifest.json'
Require ((Test-Path -LiteralPath $innerZip -PathType Leaf) -and
    (Test-Path -LiteralPath $bundle -PathType Leaf) -and
    (Test-Path -LiteralPath $missing -PathType Leaf)) 'Invented smoke inputs are incomplete.'
$candidate = Join-Path $kit 'extracted-candidate'
Require (-not (Test-Path -LiteralPath $candidate)) 'Candidate extraction directory already exists.'
Expand-Archive -LiteralPath $innerZip -DestinationPath $candidate
$appDirectory = Join-Path $candidate 'anime-graph-desktop'
$exe = Join-Path $appDirectory 'anime_graph_desktop.exe'
Require (-not (Test-Path -LiteralPath (Join-Path $appDirectory 'data'))) `
    'The packaged application unexpectedly contains relative data.'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $appDirectory 'Check-Prerequisites.ps1')
Require ($LASTEXITCODE -eq 0) 'Desktop package prerequisites failed.'

$oldArguments = $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS
Require ([string]::IsNullOrEmpty($oldArguments)) `
    'The runner has unexpected inherited WebView2 browser arguments.'
$env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = '--force-renderer-accessibility'
try {
    $app = Start-Process -FilePath $exe -WorkingDirectory $appDirectory -WindowStyle Normal -PassThru
} finally {
    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = $oldArguments
}
try {
    Require ($app.SessionId -eq [System.Diagnostics.Process]::GetCurrentProcess().SessionId -and
        [Environment]::UserInteractive) 'The packaged app is not in an interactive desktop session.'
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $processCondition = [System.Windows.Automation.PropertyCondition]::new(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $app.Id
    )
    $deadline = (Get-Date).AddSeconds(15)
    $window = $null
    do {
        $app.Refresh()
        if ($app.HasExited) { throw 'The packaged app exited before showing a window.' }
        $topWindows = $root.FindAll([System.Windows.Automation.TreeScope]::Children, $processCondition)
        for ($index = 0; $index -lt $topWindows.Count; $index++) {
            $candidateWindow = $topWindows.Item($index)
            if ($candidateWindow.Current.Name -eq 'Dioxus App' -and
                $candidateWindow.Current.ClassName -eq 'Window Class') {
                $window = $candidateWindow
                break
            }
        }
        if ($null -ne $window) { break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    Require ($null -ne $window) 'The packaged app has no accessible window.'

    $names = WaitForText $window 'No data selected.' 'No data'
    RequireText $names 'No graph to show.' 'empty initial graph'
    RequireText $names 'does not rank recommendations or import viewing history' 'desktop scope'
    Write-Output 'GUI state passed: No data; no graph.'

    InvokeButton $window 'Open invented demo'
    $names = WaitForText $window 'Invented demo.' 'Invented demo'
    $names = WaitForCounts $window 'Invented demo'
    RequireText $names 'graph-compact-v3' 'aggregate-only graph format'
    RequireText $names 'limited to 300 connected titles and 1,400 pairs' 'overview limit'
    RequireText $names 'not similarity scores' 'pair semantics'
    RequireText $names 'graph keeps no user rows' 'aggregate-only graph limit'
    Require (-not ($names -contains 'No graph to show.')) 'Demo has no graph.'
    Write-Output 'GUI state passed: Invented demo; 8 anime; 11 pairs; limits and pair semantics.'

    SelectManifest $root $window $app.Id $bundle $ForceKeyboardPicker.IsPresent
    $names = WaitForText $window 'Selected local data.' 'selected local data'
    $names = WaitForCounts $window 'selected local data'
    RequireText $names 'graph-compact-v3' 'selected graph format'
    RequireText $names 'data-vsynthetic-desktop-v1' 'invented manifest tag'
    RequireText $names 'other assets and source permissions are not checked here' 'source-permission limit'
    Require (-not ($names -contains 'No graph to show.')) 'Selected local data has no graph.'
    Write-Output 'GUI state passed: selected invented local bundle; 8 anime; 11 pairs; source limits.'

    SelectManifest $root $window $app.Id $missing $ForceKeyboardPicker.IsPresent
    $names = WaitForText $window 'Load failed. No graph is active.' 'missing asset failure'
    RequireText $names 'graph.compact.json: missing or unreadable' 'missing graph error'
    RequireText $names 'No graph to show.' 'cleared graph after failure'
    Require (-not ($names | Where-Object { $_.Contains('Selected manifest tag') })) `
        'A failed selection retained the prior local graph.'
    Write-Output 'GUI state passed: missing graph fails and clears the prior graph.'

    $os = Get-CimInstance Win32_OperatingSystem
    Write-Output "Clean GUI smoke passed on $($os.Caption) $($os.Version); app session $($app.SessionId)."
    Write-Output "Candidate package SHA-256: $((Get-FileHash -LiteralPath $innerZip -Algorithm SHA256).Hash.ToLowerInvariant())"
} finally {
    if ($null -ne $app -and -not $app.HasExited) { Stop-Process -Id $app.Id -Force }
}
