param(
    [Parameter(Mandatory = $true)][ValidateSet('Inspect', 'Reserve', 'Save')][string]$Action,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$ScopeSha256,
    [string]$RecordBase64
)
# No network, deletion, existing ACL mutation or pilot-content reads. Save creates only fresh private output.
$ErrorActionPreference = 'Stop'
$handles = [System.Collections.Generic.List[Microsoft.Win32.SafeHandles.SafeFileHandle]]::new()
$stream = $null
$markerStream = $null
$stage = 'initialization'
try {
    Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class FollowupDirectoryPins {
    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityAttributes { public int length; public IntPtr descriptor; public int inherit; }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern bool CreateDirectoryW(string name, ref SecurityAttributes security);
    public static bool CreatePrivateDirectory(string name, byte[] descriptor) {
        IntPtr memory = Marshal.AllocHGlobal(descriptor.Length);
        try {
            Marshal.Copy(descriptor, 0, memory, descriptor.Length);
            var security = new SecurityAttributes { length = Marshal.SizeOf(typeof(SecurityAttributes)), descriptor = memory, inherit = 0 };
            return CreateDirectoryW(name, ref security);
        } finally { Marshal.FreeHGlobal(memory); }
    }
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
            -not $acl.AreAccessRulesProtected) { return $false }
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
    if ($Action -eq 'Reserve') {
        $stage = 'reservation-preflight'
        # A fresh pinned inspection closes the gap between the parent's initial inspection and reservation.
        if ($facts.access -ne 'verified-owner-only' -or $facts.priorPilotExists -or $facts.targetExists) { throw 'Reservation refused' }
        if ($facts.reservationExists) { [Console]::Out.WriteLine('{"reserved":false}'); exit 0 }
    }
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
    # Elevated Windows tokens can default new-file ownership to Administrators. Supply
    # the strict current-owner security descriptor atomically, never repair an existing ACL.
    $stage = 'reservation-security'
    $security = [Security.AccessControl.FileSecurity]::new()
    $security.SetOwner($ownerSid)
    $security.SetAccessRuleProtection($true, $false)
    $security.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($ownerSid,
        [Security.AccessControl.FileSystemRights]::FullControl, [Security.AccessControl.AccessControlType]::Allow))
    $rights = [Security.AccessControl.FileSystemRights]::Write -bor [Security.AccessControl.FileSystemRights]::ReadPermissions
    if ($Action -eq 'Save') {
        $stage = 'output-preflight'
        if ($facts.access -ne 'verified-owner-only' -or $facts.priorPilotExists -or $facts.targetExists -or -not $facts.reservationExists) { throw 'Output refused' }
        $stage = 'output-marker'
        if (([IO.File]::GetAttributes($reservation) -band [IO.FileAttributes]::ReparsePoint) -or -not (Test-PrivateAcl $reservation $false)) { throw 'Marker refused' }
        # Lock the exact existing consumption marker without sharing, through every output flush.
        $markerStream = [IO.FileStream]::new($reservation, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
        $buffer = [Text.StringBuilder]::new(1024)
        $size = [FollowupDirectoryPins]::GetFinalPathNameByHandleW($markerStream.SafeFileHandle, $buffer, 1024, 0)
        if ($size -eq 0 -or $size -ge 1024 -or $buffer.ToString() -ine ('\\?\' + $reservation) -or $markerStream.Length -ne $bytes.Length) { throw 'Marker refused' }
        $held = [byte[]]::new($bytes.Length)
        $offset = 0
        while ($offset -lt $held.Length) {
            $read = $markerStream.Read($held, $offset, $held.Length - $offset)
            if ($read -le 0) { throw 'Marker refused' }; $offset += $read
        }
        if ([Convert]::ToBase64String($held) -cne [Convert]::ToBase64String($bytes)) { throw 'Marker refused' }
        $stage = 'output-payload'
        $inputBuilder = [Text.StringBuilder]::new()
        $characters = [char[]]::new(8192)
        while (($read = [Console]::In.Read($characters, 0, $characters.Length)) -gt 0) {
            if ($inputBuilder.Length + $read -gt 12 * 1024 * 1024) { throw 'Payload refused' }
            [void]$inputBuilder.Append($characters, 0, $read)
        }
        $payload = $inputBuilder.ToString() | ConvertFrom-Json
        if (@($payload.PSObject.Properties).Count -ne 3 -or $payload.format -cne 'private-followup-output-payload-v1' -or
            ($payload.reservation | ConvertTo-Json -Compress) -cne ($record | ConvertTo-Json -Compress)) { throw 'Payload refused' }
        $names = @('source-projection.json', 'definition-labels.json', 'inventory.json', 'receipt.json', 'completed.json')
        if ($payload.files -isnot [Array] -or $payload.files.Count -ne 5) { throw 'Payload refused' }
        $decoded = [System.Collections.Generic.List[byte[]]]::new()
        for ($i = 0; $i -lt 5; $i++) {
            $file = $payload.files[$i]
            $maximum = 256 * 1024; if ($i -lt 2) { $maximum = 4 * 1024 * 1024 }
            if (@($file.PSObject.Properties).Count -ne 4 -or $file.name -cne $names[$i] -or
                $file.bytes -isnot [int] -or $file.bytes -lt 0 -or $file.bytes -gt $maximum -or
                $file.sha256 -cnotmatch '^[a-f0-9]{64}$' -or $file.base64 -isnot [string] -or
                $file.base64.Length -gt 4 * [Math]::Ceiling($maximum / 3)) { throw 'Payload refused' }
            $data = [Convert]::FromBase64String($file.base64)
            $hasher = [Security.Cryptography.SHA256]::Create()
            try { $digest = [BitConverter]::ToString($hasher.ComputeHash($data)).Replace('-', '').ToLowerInvariant() } finally { $hasher.Dispose() }
            if ($data.Length -ne $file.bytes -or $digest -cne $file.sha256 -or [Convert]::ToBase64String($data) -cne $file.base64) { throw 'Payload refused' }
            $decoded.Add($data)
        }
        $completion = [Text.UTF8Encoding]::new($false, $true).GetString($decoded[4]) | ConvertFrom-Json
        if (@($completion.PSObject.Properties).Count -ne 11 -or $completion.format -cne 'private-followup-completion-v1' -or
            $completion.studyId -cne $record.studyId -or $completion.scopeSha256 -cne $record.scopeSha256 -or
            $completion.approvalSha256 -cne $record.approvalSha256 -or $completion.startedAt -cne $record.startedAt -or
            $completion.expiresAt -cne $record.expiresAt -or $completion.state -cne 'completed' -or
            $completion.mappingReviewed -isnot [bool] -or $completion.mappingReviewed -or
            $completion.publicArtifacts -isnot [bool] -or $completion.publicArtifacts -or $completion.files.Count -ne 4) { throw 'Completion refused' }
        $completedAt = [DateTimeOffset]::ParseExact($completion.completedAt, 'yyyy-MM-ddTHH:mm:ss.fffZ', [Globalization.CultureInfo]::InvariantCulture, $dateStyle)
        if ($completedAt -lt $started -or $completedAt -gt [DateTimeOffset]::UtcNow -or $completedAt -ge $expires) { throw 'Completion refused' }
        for ($i = 0; $i -lt 4; $i++) {
            $entry = $completion.files[$i]
            if (@($entry.PSObject.Properties).Count -ne 3 -or $entry.name -cne $payload.files[$i].name -or
                $entry.bytes -ne $payload.files[$i].bytes -or $entry.sha256 -cne $payload.files[$i].sha256) { throw 'Completion refused' }
        }
        $stage = 'output-create'
        foreach ($name in $names + @('completed.partial.json')) {
            if ([IO.Path]::Combine($target, $name).Length -gt 240) { throw 'Output path refused' }
        }
        if (Test-AnyEntry $target) { throw 'Output refused' }
        if ([DateTimeOffset]::UtcNow -ge $expires) { throw 'Expired' }
        $directorySecurity = [Security.AccessControl.DirectorySecurity]::new()
        $directorySecurity.SetOwner($ownerSid); $directorySecurity.SetAccessRuleProtection($true, $false)
        $inherit = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
        $directorySecurity.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($ownerSid, 'FullControl', $inherit, 'None', 'Allow'))
        if (-not [FollowupDirectoryPins]::CreatePrivateDirectory($target, $directorySecurity.GetSecurityDescriptorBinaryForm())) { throw 'Output creation refused' }
        $handle = [FollowupDirectoryPins]::CreateFileW($target, 0x80, 3, [IntPtr]::Zero, 3, 0x02200000, [IntPtr]::Zero)
        if ($handle.IsInvalid) { $handle.Dispose(); throw 'Output pin refused' }; $handles.Add($handle)
        $stage = 'output-access'
        $attributes = [IO.File]::GetAttributes($target)
        $buffer = [Text.StringBuilder]::new(1024)
        $size = [FollowupDirectoryPins]::GetFinalPathNameByHandleW($handle, $buffer, 1024, 0)
        if (($attributes -band [IO.FileAttributes]::ReparsePoint) -or -not (Test-PrivateAcl $target $true) -or
            $size -eq 0 -or $size -ge 1024 -or $buffer.ToString() -ine ('\\?\' + $target)) { throw 'Output access refused' }
        for ($i = 0; $i -lt 5; $i++) {
            $stage = 'output-files'
            if ([DateTimeOffset]::UtcNow -ge $expires) { throw 'Expired' }
            $name = $names[$i]; if ($i -eq 4) { $name = 'completed.partial.json' }
            $location = [IO.Path]::Combine($target, $name)
            $outputRights = $rights -bor [Security.AccessControl.FileSystemRights]::ReadData
            $stream = [IO.FileStream]::new($location, [IO.FileMode]::CreateNew, $outputRights, [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough, $security)
            if (-not (Test-PrivateAcl $location $false)) { throw 'File access refused' }
            $stream.Write($decoded[$i], 0, $decoded[$i].Length); $stream.Flush($true)
            $stream.Position = 0; $hasher = [Security.Cryptography.SHA256]::Create()
            try { $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() } finally { $hasher.Dispose() }
            if ($stream.Length -ne $payload.files[$i].bytes -or $digest -cne $payload.files[$i].sha256) { throw 'Written bytes refused' }
            $stream.Dispose(); $stream = $null
        }
        $stage = 'output-complete'
        if ([DateTimeOffset]::UtcNow -ge $expires) { throw 'Expired' }
        [IO.File]::Move([IO.Path]::Combine($target, 'completed.partial.json'), [IO.Path]::Combine($target, 'completed.json'))
        if ([DateTimeOffset]::UtcNow -ge $expires) { throw 'Expired' }
        [Console]::Out.WriteLine('{"saved":true}'); exit 0
    }
    $stage = 'reservation-create'
    try { $stream = [IO.FileStream]::new($reservation, [IO.FileMode]::CreateNew, $rights,
        [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough, $security) }
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
    if ($markerStream) { $markerStream.Dispose() }
    foreach ($handle in $handles) { $handle.Dispose() }
}
