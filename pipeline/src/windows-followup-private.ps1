param(
    [Parameter(Mandatory = $true)][ValidateSet('Inspect', 'Reserve')][string]$Action,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$ScopeSha256,
    [string]$RecordBase64
)
# No network, deletion, ACL mutation, directory creation or pilot-content reads.
$ErrorActionPreference = 'Stop'
$handles = [System.Collections.Generic.List[Microsoft.Win32.SafeHandles.SafeFileHandle]]::new()
$stream = $null
$stage = 'initialization'
try {
    Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class FollowupDirectoryPins {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern SafeFileHandle CreateFileW(string name, uint access, uint share,
        IntPtr security, uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern uint GetFinalPathNameByHandleW(SafeFileHandle handle, StringBuilder path, uint size, uint flags);
}
'@
    $stage = 'path-syntax'
    $studyId = 'wikidata-followup-101-200-v1'
    if ($RepositoryRoot -notmatch '^[A-Za-z]:\\' -or $RepositoryRoot.Length -gt 240 -or
        $RepositoryRoot.Substring(2) -match '[:*?"<>|]') { throw 'Path refused' }
    $stage = 'path-normalization'
    if ([IO.Path]::GetFullPath($RepositoryRoot) -cne $RepositoryRoot) { throw 'Path refused' }
    $stage = 'path-component'
    foreach ($component in $RepositoryRoot.Substring(3).Split('\')) {
        if (-not $component -or $component.EndsWith('.') -or $component.EndsWith(' ') -or
            $component -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Path refused' }
    }
    $stage = 'derived-path'
    $privateParent = [IO.Path]::Combine([IO.Path]::GetDirectoryName($RepositoryRoot), 'Anime-private')
    $target = [IO.Path]::Combine($privateParent, $studyId)
    $prior = [IO.Path]::Combine($privateParent, 'wikidata-pilot-2026-10-07')
    $reservation = [IO.Path]::Combine($privateParent, $studyId + '.reserved.json')
    if ($target.Length -gt 240 -or $reservation.Length -gt 240 -or $ScopeSha256 -cnotmatch '^[a-f0-9]{64}$') { throw 'Path or scope refused' }
    $stage = 'ancestor-enumeration'
    $paths = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($leaf in @($RepositoryRoot, $privateParent)) {
        $cursor = $leaf
        while ($cursor) {
            [void]$paths.Add($cursor)
            $cursor = [IO.Path]::GetDirectoryName($cursor)
        }
    }
    # Open without FILE_SHARE_DELETE and with OPEN_REPARSE_POINT before checking each child.
    # Pins stay open through reserve/flush; same-owner/administrator sabotage is outside this policy.
    foreach ($directory in @($paths | Sort-Object Length, { $_ })) {
        $stage = 'directory-pin'
        $handle = [FollowupDirectoryPins]::CreateFileW($directory, 0x80, 3, [IntPtr]::Zero, 3, 0x02200000, [IntPtr]::Zero)
        if ($handle.IsInvalid) { $handle.Dispose(); throw 'Directory pin refused' }
        $handles.Add($handle)
        $stage = 'directory-attributes'
        $attributes = [IO.File]::GetAttributes($directory)
        if (-not ($attributes -band [IO.FileAttributes]::Directory) -or ($attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Directory refused' }
        $buffer = [Text.StringBuilder]::new(1024)
        $stage = 'directory-resolution'
        $size = [FollowupDirectoryPins]::GetFinalPathNameByHandleW($handle, $buffer, 1024, 0)
        if ($size -eq 0 -or $size -ge 1024 -or -not $buffer.ToString().StartsWith('\\?\') -or
            $buffer.ToString().Substring(4) -ine $directory) { throw 'Resolved path refused' }
    }
    $stage = 'private-access'
    $ownerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    function Test-PrivateAcl([string]$location, [bool]$isDirectory) {
        $acl = Get-Acl -LiteralPath $location
        if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $ownerSid.Value -or
            ($isDirectory -and -not $acl.AreAccessRulesProtected)) { return $false }
        $full = [Security.AccessControl.FileSystemRights]::FullControl
        $inherited = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
        $ownerFull = $false
        foreach ($rule in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
            if ($rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow -or
                $rule.IdentityReference.Value -notin @($ownerSid.Value, 'S-1-5-18', 'S-1-5-32-544')) { return $false }
            if ($rule.IdentityReference.Value -eq $ownerSid.Value -and ($rule.FileSystemRights -band $full) -eq $full -and
                $rule.PropagationFlags -eq [Security.AccessControl.PropagationFlags]::None -and
                (-not $isDirectory -or ($rule.InheritanceFlags -band $inherited) -eq $inherited)) { $ownerFull = $true }
        }
        return $ownerFull
    }
    function Test-AnyEntry([string]$location) {
        try { [void][IO.File]::GetAttributes($location); return $true }
        catch [IO.FileNotFoundException] { return $false }
        catch [IO.DirectoryNotFoundException] { return $false }
    }
    $facts = [ordered]@{ repoRoot = $RepositoryRoot; privateParent = $privateParent; resolvedPrivateParent = $privateParent;
        targetPath = $target; ancestorsHaveReparsePoints = $false; priorPilotExists = (Test-AnyEntry $prior);
        targetExists = (Test-AnyEntry $target); reservationExists = (Test-AnyEntry $reservation);
        access = 'unknown' }
    if (Test-PrivateAcl $privateParent $true) { $facts.access = 'verified-owner-only' }
    if ($Action -eq 'Inspect') { [Console]::Out.WriteLine(($facts | ConvertTo-Json -Compress)); exit 0 }
    $stage = 'reservation-preflight'
    # A fresh pinned inspection closes the gap between the parent's initial inspection and reservation.
    if ($facts.access -ne 'verified-owner-only' -or $facts.priorPilotExists -or $facts.targetExists) { throw 'Reservation refused' }
    if ($facts.reservationExists) { [Console]::Out.WriteLine('{"reserved":false}'); exit 0 }
    $stage = 'reservation-record'
    if (-not $RecordBase64 -or $RecordBase64.Length -gt 6000) { throw 'Record refused' }
    $bytes = [Convert]::FromBase64String($RecordBase64)
    if ($bytes.Length -gt 4096) { throw 'Record refused' }
    $json = [Text.UTF8Encoding]::new($false, $true).GetString($bytes)
    $record = $json | ConvertFrom-Json
    $keys = @('format', 'studyId', 'scopeSha256', 'approvalSha256', 'state', 'startedAt', 'expiresAt', 'publicArtifacts')
    if (@($record.PSObject.Properties).Count -ne $keys.Count -or
        @($record.PSObject.Properties.Name | Where-Object { $_ -notin $keys }).Count -ne 0 -or
        $record.format -cne 'wikidata-followup-reservation-v1' -or $record.studyId -cne $studyId -or
        $record.state -cne 'started' -or $record.scopeSha256 -cne $ScopeSha256 -or
        $record.approvalSha256 -cnotmatch '^[a-f0-9]{64}$' -or $record.publicArtifacts -isnot [bool] -or $record.publicArtifacts) { throw 'Record refused' }
    $dateStyle = [Globalization.DateTimeStyles]::AssumeUniversal -bor [Globalization.DateTimeStyles]::AdjustToUniversal
    $started = [DateTimeOffset]::ParseExact($record.startedAt, 'yyyy-MM-ddTHH:mm:ss.fffZ', [Globalization.CultureInfo]::InvariantCulture, $dateStyle)
    $expires = [DateTimeOffset]::ParseExact($record.expiresAt, 'yyyy-MM-ddTHH:mm:ss.fffZ', [Globalization.CultureInfo]::InvariantCulture, $dateStyle)
    if ($started -gt [DateTimeOffset]::UtcNow -or $expires -le [DateTimeOffset]::UtcNow -or $expires -le $started -or ($expires - $started).TotalDays -gt 7) { throw 'Record refused' }
    $stage = 'reservation-create'
    try { $stream = [IO.FileStream]::new($reservation, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write,
        [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough) }
    catch [IO.IOException] {
        if (Test-AnyEntry $reservation) { [Console]::Out.WriteLine('{"reserved":false}'); exit 0 }
        throw
    }
    # Any failure after exclusive creation leaves this marker present and therefore consumed.
    $stage = 'reservation-access'
    if (-not (Test-PrivateAcl $reservation $false)) { throw 'Reservation access refused' }
    $stage = 'reservation-flush'
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Flush($true)
    [Console]::Out.WriteLine('{"reserved":true}')
} catch {
    # Only a local fixed stage code crosses the boundary; never exception messages or paths.
    [Console]::Out.WriteLine(('{"code":"windows-private-preflight-failed","stage":"' + $stage + '"}'))
    exit 1
} finally {
    if ($stream) { $stream.Dispose() }
    foreach ($handle in $handles) { $handle.Dispose() }
}
