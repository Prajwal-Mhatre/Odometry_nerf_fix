param(
    [string]$InVideo = 'samples\alley_night\input.mp4',
    [string]$BakeDir = 'samples\alley_night\bake',
    [string]$OutVideo = 'samples\alley_night\fixed.mp4',
    [int]$Crf = 18
)

$ErrorActionPreference = 'Stop'

Write-Host '=== FieldFixer demo ==='
Write-Host "Input video : $InVideo"
Write-Host "Bake output : $BakeDir"
Write-Host "Output video: $OutVideo"

$cli = Get-Command fieldfixer -ErrorAction SilentlyContinue
if (-not $cli) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
    if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
    if (-not $py) {
        Write-Error "Cannot find fieldfixer or python on PATH. Activate your environment or run 'pip install -e .'"
        exit 1
    }
    $cli = "$($py.Source) -m fieldfixer.cli"
} else {
    $cli = 'fieldfixer'
}

if (-not (Test-Path $InVideo)) {
    Write-Error "Input video not found at $InVideo"
    exit 1
}

New-Item -ItemType Directory -Path $BakeDir -Force | Out-Null

Write-Host '-> Baking identity sidecars (stub pipeline)...'
& $cli bake --in $InVideo --out $BakeDir --modules rsnerf --modules deblurnerf --modules rawnerf --modules nerfw

Write-Host '-> Applying sidecars to input video...'
& $cli apply --in $InVideo --bake $BakeDir --out $OutVideo --crf $Crf

Write-Host 'Demo complete. Output written to' $OutVideo
