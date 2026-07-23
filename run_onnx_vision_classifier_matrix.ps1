[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $Dataset,

    [ValidateRange(1, 100000)]
    [int] $Repeat = 1,

    [ValidateRange(0, 86400)]
    [double] $GapSeconds = 0,

    [string] $OutputCsv = "runs\onnx-vision-classifier-matrix.csv",

    [int] $Seed = 0,

    # Optional custom entries in the form: "path\runner.exe|provider"
    [string[]] $CommandSpec
)

$ErrorActionPreference = "Stop"

function Resolve-InputPath([string] $Path) {
    $ExecutionPath = Join-Path (Get-Location) $Path
    if (-not (Test-Path -LiteralPath $ExecutionPath -PathType Leaf) -and
        -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "File does not exist: $Path"
    }
    return [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).Path)
}

function Resolve-DirectoryPath([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Directory does not exist: $Path"
    }
    return [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).Path)
}

function New-DefaultCommandSpec {
    @(
        "onnx-vision\bin\openvino\Release\net461\win7-x64\OnnxVision.OpenVino.exe|openvino-gpu",
        "onnx-vision\bin\openvino\Release\net461\win7-x64\OnnxVision.OpenVino.exe|openvino-cpu",
        "onnx-vision\bin\openvino\Release\net461\win7-x64\OnnxVision.OpenVino.exe|cpu",
        "onnx-vision\bin\directml\Release\net461\win7-x64\OnnxVision.DirectML.exe|directml",
        "onnx-vision\bin\directml\Release\net461\win7-x64\OnnxVision.DirectML.exe|cpu",
        "onnx-vision\bin\Release\net461\win7-x64\OnnxVision.exe|cpu"
    )
}

function Get-ParsedValue([string] $Text, [string] $Pattern, [int] $Group = 1) {
    $Match = [regex]::Match($Text, $Pattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($Match.Success) { return $Match.Groups[$Group].Value }
    return $null
}

$ModelPath = Resolve-InputPath $Model
$DatasetPath = Resolve-DirectoryPath $Dataset
$Specs = if ($null -eq $CommandSpec -or $CommandSpec.Count -eq 0) {
    New-DefaultCommandSpec
} else {
    $CommandSpec
}

$Commands = foreach ($Spec in $Specs) {
    $Parts = $Spec -split "\|", 2
    if ($Parts.Count -ne 2 -or [string]::IsNullOrWhiteSpace($Parts[0]) -or [string]::IsNullOrWhiteSpace($Parts[1])) {
        throw "Invalid CommandSpec '$Spec'. Use: executable-path|provider"
    }
    [pscustomobject]@{
        Executable = Resolve-InputPath $Parts[0].Trim()
        Provider = $Parts[1].Trim()
    }
}

$Jobs = for ($Iteration = 1; $Iteration -le $Repeat; $Iteration++) {
    foreach ($Command in $Commands) {
        [pscustomobject]@{
            Iteration = $Iteration
            Executable = $Command.Executable
            Provider = $Command.Provider
        }
    }
}

$Random = [Random]::new($Seed)
$Jobs = @($Jobs | Sort-Object { $Random.Next() })
$CsvPath = [IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputCsv))
$CsvDirectory = Split-Path -Parent $CsvPath
if ($CsvDirectory) { New-Item -ItemType Directory -Force -Path $CsvDirectory | Out-Null }

if (Test-Path -LiteralPath $CsvPath) {
    Remove-Item -LiteralPath $CsvPath -Force
}

$JobNumber = 0
foreach ($Job in $Jobs) {
    $JobNumber++
    $Started = Get-Date
    $Stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $Output = ""
    $ErrorOutput = ""
    $ExitCode = -1

    Write-Host ("[{0}/{1}] {2} ({3}), repeat {4}" -f $JobNumber, $Jobs.Count, (Split-Path $Job.Executable -Leaf), $Job.Provider, $Job.Iteration)

    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 can turn stderr from a native executable into
        # a terminating error when ErrorActionPreference is Stop. The runners
        # may legitimately write warnings to stderr, so keep the process alive
        # and capture that stream together with stdout.
        $ErrorActionPreference = "Continue"
        $Output = & $Job.Executable $ModelPath $DatasetPath $Job.Provider 2>&1 | Out-String
        $ExitCode = $LASTEXITCODE
    } catch {
        $ErrorOutput = ($_ | Out-String)
        $Output = $ErrorOutput
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    $Stopwatch.Stop()

    $CombinedOutput = $Output + "`n" + $ErrorOutput
    $Row = [ordered]@{
        run = $JobNumber
        repeat = $Job.Iteration
        executable = $Job.Executable
        provider = $Job.Provider
        exit_code = $ExitCode
        success = ($ExitCode -eq 0)
        wall_seconds = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 3)
        accuracy = Get-ParsedValue $CombinedOutput 'Accuracy:\s*\d+/\d+\s*\(([^)]+)\)'
        accuracy_fraction = Get-ParsedValue $CombinedOutput 'Accuracy:\s*(\d+/\d+)'
        flipped_recall = Get-ParsedValue $CombinedOutput 'Flipped recall:\s*\d+/\d+\s*\(([^)]+)\)'
        normal_recall = Get-ParsedValue $CombinedOutput 'Normal recall:\s*\d+/\d+\s*\(([^)]+)\)'
        end_to_end_ms_per_image = Get-ParsedValue $CombinedOutput 'End-to-end:\s*([\d.]+)\s*ms/image'
        images_per_second = Get-ParsedValue $CombinedOutput 'End-to-end:\s*[\d.]+\s*ms/image\s*\(([\d.]+)\s*images/s\)'
        preprocess_ms_per_image = Get-ParsedValue $CombinedOutput 'Preprocess:\s*([\d.]+)\s*ms/image'
        inference_ms_per_image = Get-ParsedValue $CombinedOutput '(?:Inference|ONNX graph \(preprocess \+ inference\)):\s*([\d.]+)\s*ms/image'
        started_at = $Started.ToString("o")
        output = ($CombinedOutput.Trim() -replace "`r?`n", " | ")
    }

    [pscustomobject]$Row | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Append
    Write-Host ("  exit={0}, wall={1:N3}s, end-to-end={2} ms/image" -f $ExitCode, $Stopwatch.Elapsed.TotalSeconds, $Row.end_to_end_ms_per_image)

    if ($GapSeconds -gt 0 -and $JobNumber -lt $Jobs.Count) {
        Start-Sleep -Milliseconds ([int]($GapSeconds * 1000))
    }
}

Write-Host "CSV written to $CsvPath"
