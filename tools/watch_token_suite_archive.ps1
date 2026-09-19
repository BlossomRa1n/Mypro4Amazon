param(
    [string]$HostName = "connect.weste.seetacloud.com",
    [int]$Port = 43050,
    [string]$RemoteRun = "/root/autodl-tmp/future_window_token_suite_20260920",
    [string]$LocalRun = "D:\MyPro-Amazon\server_snapshot\2026-09-20\token_suite_final_20260920",
    [int]$PollSeconds = 900,
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
$remote = "root@$HostName"
$options = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=D:/MyPro-Amazon/.ssh_known_hosts')
$sshArgs = @('-p', $Port) + $options
$scpArgs = @('-P', $Port) + $options
New-Item -ItemType Directory -Force -Path $LocalRun | Out-Null

do {
    try {
        & ssh @sshArgs $remote "test -s '$RemoteRun/evidence.sha256'"
        if ($LASTEXITCODE -eq 0) {
            foreach ($file in @('evidence.tar.gz', 'evidence.sha256')) {
                & scp @scpArgs "$remote`:$RemoteRun/$file" (Join-Path $LocalRun $file)
                if ($LASTEXITCODE -ne 0) { throw "Download failed: $file" }
            }
            $expected = (Get-Content (Join-Path $LocalRun 'evidence.sha256') -Raw).Trim()
            $actual = (Get-FileHash (Join-Path $LocalRun 'evidence.tar.gz') -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected) { throw 'Archive checksum mismatch' }
            & tar -xzf (Join-Path $LocalRun 'evidence.tar.gz') -C $LocalRun
            if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed' }
            $checksums = Get-Content (Join-Path $LocalRun 'archive_manifest.json') -Raw | ConvertFrom-Json -AsHashtable
            foreach ($name in $checksums.Keys) {
                $hash = (Get-FileHash (Join-Path $LocalRun $name) -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($hash -ne $checksums[$name]) { throw "Evidence checksum mismatch: $name" }
            }
            $completed = Get-Content (Join-Path $LocalRun 'COMPLETED.json') -Raw | ConvertFrom-Json
            if ($completed.status -ne 'complete') { throw 'Suite is not complete' }
            Set-Content -Encoding utf8 (Join-Path $LocalRun 'ARCHIVED_AT.txt') (Get-Date -Format o)
            & scp @scpArgs (Join-Path $LocalRun 'evidence.sha256') "$remote`:$RemoteRun/ARCHIVE_ACK.sha256"
            if ($LASTEXITCODE -ne 0) { throw 'Archive acknowledgement failed' }
            Write-Output 'Evidence downloaded, checked, and acknowledged; shutdown may proceed.'
            exit 0
        }
    } catch {
        Add-Content -Encoding utf8 (Join-Path $LocalRun 'watch_errors.log') ("{0} {1}" -f (Get-Date -Format o), $_.Exception.Message)
        if ($Once) { throw }
    }
    if (!$Once) { Start-Sleep -Seconds $PollSeconds }
} while (!$Once)
exit 2
