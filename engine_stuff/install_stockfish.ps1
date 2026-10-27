# Run from any folder: powershell -File engine_stuff/install_stockfish.ps1
# Installs only inside this project. Keep the official license/source files.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$release = Invoke-RestMethod 'https://api.github.com/repos/official-stockfish/Stockfish/releases/tags/sf_19'
$assetName = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    'stockfish-windows-arm64-universal.zip'
} else { 'stockfish-windows-x86-64-universal.zip' }
$asset = $release.assets | Where-Object name -eq $assetName
if ($null -eq $asset) { throw "Official release has no $assetName" }
$destination = Join-Path $PSScriptRoot 'bin'
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$download = Join-Path $destination $assetName
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $download -UseBasicParsing
$hash = (Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash.ToLower()
if (-not $asset.digest -or $asset.digest -ne "sha256:$hash") {
    throw 'Archive does not match the official SHA256 digest; installation stopped'
}
Expand-Archive -LiteralPath $download -DestinationPath $destination -Force
# Delete just the verified download, not the installed engine/license/source.
Remove-Item -LiteralPath $download
Write-Output "Installed $($release.tag_name) in $destination; SHA256 $hash"
