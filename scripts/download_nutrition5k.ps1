param(
    [string]$Destination = "..\data\raw\nutrition5k_dataset"
)

$ErrorActionPreference = "Stop"

$gcloudCommand = Get-Command gcloud.cmd -ErrorAction SilentlyContinue
if (-not $gcloudCommand) {
    $knownPath = "D:\Google\google-cloud-sdk\bin\gcloud.cmd"
    if (Test-Path -LiteralPath $knownPath) {
        $gcloudExecutable = $knownPath
    }
    else {
        throw "gcloud.cmd is not available. Reopen PowerShell after installing Google Cloud CLI."
    }
}
else {
    $gcloudExecutable = $gcloudCommand.Source
}

$destinationPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Destination))
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null

$metadataDestination = Join-Path $destinationPath "metadata"
$dishIdsDestination = Join-Path $destinationPath "dish_ids"
$rgbDestination = Join-Path $destinationPath "imagery\realsense_overhead"

New-Item -ItemType Directory -Force -Path $metadataDestination | Out-Null
New-Item -ItemType Directory -Force -Path $dishIdsDestination | Out-Null
New-Item -ItemType Directory -Force -Path $rgbDestination | Out-Null

& $gcloudExecutable storage rsync `
    "gs://nutrition5k_dataset/nutrition5k_dataset/metadata" `
    $metadataDestination `
    --recursive
if ($LASTEXITCODE -ne 0) { throw "Failed to download Nutrition5k metadata." }

& $gcloudExecutable storage rsync `
    "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids" `
    $dishIdsDestination `
    --recursive
if ($LASTEXITCODE -ne 0) { throw "Failed to download Nutrition5k split files." }

# Keep only overhead RGB files. Depth images are not used by this baseline.
& $gcloudExecutable storage rsync `
    "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" `
    $rgbDestination `
    --recursive `
    '--exclude=.*depth.*[.]png$'
if ($LASTEXITCODE -ne 0) { throw "Failed to download Nutrition5k overhead RGB images." }

Write-Host "Nutrition5k metadata, splits, and overhead RGB downloaded to: $destinationPath"
