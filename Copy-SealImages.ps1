[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$SourceRoot = 'W:\VisionPC\Machine Shipped Database Backup\APS',
    [string]$DestinationRoot = $PSScriptRoot,
    [ValidateRange(1, 128)]
    [int]$ThrottleLimit = 12
)

$ErrorActionPreference = 'Stop'
$imageExtensions = @(
    '.bmp', '.gif', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
)

if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Source folder does not exist or is unavailable: $SourceRoot"
}

if (-not (Test-Path -LiteralPath $DestinationRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
}

$machineFolders = Get-ChildItem -LiteralPath $SourceRoot -Directory |
    Where-Object { $_.Name -match '^Z206i-(\d+)' } |
    Sort-Object Name

$copiedCount = 0
$missingCount = 0
$copyJobs = [System.Collections.Generic.List[object]]::new()

foreach ($machineFolder in $machineFolders) {
    $machineNumber = [regex]::Match($machineFolder.Name, '^Z206i-(\d+)').Groups[1].Value
    $sealFolder = foreach ($buyoffName in @('BuyOff', 'Buy Off')) {
        foreach ($saveImageName in @('50 Save Image', '50 Save Images')) {
            Join-Path $machineFolder.FullName "$buyoffName\$saveImageName\Seal"
        }
    }
    $sealFolder = $sealFolder |
        Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
        Select-Object -First 1

    if (-not $sealFolder) {
        Write-Warning "Seal folder not found for $($machineFolder.Name) under the supported BuyOff/Buy Off and Save Image/Save Images paths."
        $missingCount++
        continue
    }

    $destinationFolder = Join-Path $DestinationRoot $machineNumber
    $images = Get-ChildItem -LiteralPath $sealFolder -File -Recurse |
        Where-Object { $imageExtensions -contains $_.Extension.ToLowerInvariant() }

    foreach ($image in $images) {
        $relativePath = $image.FullName.Substring($sealFolder.Length).TrimStart('\')
        $destinationFile = Join-Path $destinationFolder $relativePath
        $destinationDirectory = Split-Path -Parent $destinationFile

        if ($PSCmdlet.ShouldProcess($image.FullName, "Copy to $destinationFile")) {
            $copyJobs.Add([pscustomobject]@{
                Source               = $image.FullName
                Destination          = $destinationFile
                DestinationDirectory = $destinationDirectory
            })
        }
    }

    Write-Host "$($machineFolder.Name): $($images.Count) image(s)"
}

if ($copyJobs.Count -gt 0) {
    Write-Host "Copying $($copyJobs.Count) image(s) with up to $ThrottleLimit parallel workers..."

    $results = $copyJobs | ForEach-Object -Parallel {
        try {
            New-Item -ItemType Directory -Path $_.DestinationDirectory -Force | Out-Null
            Copy-Item -LiteralPath $_.Source -Destination $_.Destination -Force
            [pscustomobject]@{ Success = $true; Source = $_.Source; Error = $null }
        }
        catch {
            [pscustomobject]@{ Success = $false; Source = $_.Source; Error = $_.Exception.Message }
        }
    } -ThrottleLimit $ThrottleLimit

    $copiedCount = @($results | Where-Object Success).Count
    $failures = @($results | Where-Object { -not $_.Success })
    foreach ($failure in $failures) {
        Write-Warning "Failed to copy '$($failure.Source)': $($failure.Error)"
    }
}
else {
    $failures = @()
}

Write-Host "Finished. Copied $copiedCount image(s); $($failures.Count) failed; $missingCount machine(s) had no Seal folder."
