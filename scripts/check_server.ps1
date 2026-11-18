param(
    [Parameter(Mandatory = $true)]
    [uri]$BaseUrl
)

$base = $BaseUrl.AbsoluteUri.TrimEnd('/')
$health = Invoke-RestMethod -Method Get -Uri "$base/health" -TimeoutSec 10
if ($health.status -ne "ready") {
    throw "PathSnap API is not ready: $($health | ConvertTo-Json -Compress)"
}
$page = Invoke-WebRequest -UseBasicParsing -Uri "$base/" -TimeoutSec 10
if ($page.StatusCode -ne 200 -or $page.Content -notmatch "Route reconstruction") {
    throw "PathSnap UI did not return the expected page"
}
Write-Host "PathSnap API and UI are reachable at $base"
