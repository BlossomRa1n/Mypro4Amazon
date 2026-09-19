param(
    [string]$HostName = "connect.weste.seetacloud.com",
    [int]$Port = 43050,
    [string]$RemoteRun = "/root/autodl-tmp/future_window_token_suite_20260920",
    [string]$LocalRun = "D:\MyPro-Amazon\server_snapshot\2026-09-20\token_suite_final_20260920",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$remote = "root@$HostName"
$sshArgs = @("-p", $Port, "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no")
New-Item -ItemType Directory -Force -Path $LocalRun | Out-Null

while ($true) {
    try {
        $marker = & ssh @sshArgs $remote "test -s '$RemoteRun/COMPLETED.json'"
        if ($LASTEXITCODE -eq 0) {
            $files = @(
                "suite_manifest.json",
                "results.json",
                "COMPLETED.json",
                "monitor_status.json",
                "baseline_screen_metrics.json",
                "baseline_test_metrics.json",
                "concat_screen_final_metrics.json",
                "concat_test_final_metrics.json",
                "semantic_concat_epoch1_metrics.json",
                "semantic_concat_epoch2_metrics.json",
                "semantic_concat_epoch3_metrics.json",
                "semantic_rankmixer_epoch1_metrics.json",
                "semantic_rankmixer_epoch2_metrics.json",
                "semantic_rankmixer_epoch3_metrics.json"
            )
            foreach ($file in $files) {
                & scp @sshArgs "$remote`:$RemoteRun/$file" (Join-Path $LocalRun $file)
            }
            & ssh @sshArgs $remote "tail -n 200 '$RemoteRun/suite.log'" | Set-Content -Encoding utf8 (Join-Path $LocalRun "suite_tail.log")
            Set-Content -Encoding utf8 (Join-Path $LocalRun "ARCHIVED_AT.txt") (Get-Date -Format o)
            break
        }
    } catch {
        Add-Content -Encoding utf8 (Join-Path $LocalRun "watch_errors.log") ("{0} {1}" -f (Get-Date -Format o), $_.Exception.Message)
    }
    Start-Sleep -Seconds $PollSeconds
}
