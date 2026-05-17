param(
    [string]$CondaEnv = "",
    [string]$OutDir = "",
    [string]$StrategyPreset = "pair_first_robust_fast_v1",
    [string]$ExecutionMode = "direct-first",
    [double]$InitialAssemblyOffsetX = 0.0,
    [double]$InitialAssemblyOffsetY = 0.0,
    [double]$InitialAssemblyOffsetZ = 0.0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "repro_runs\v0_3_smoke"
}

$scriptPath = Join-Path $repoRoot "standard_four_wall_retry_build.py"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$commonArgs = @(
    "--out-dir", $OutDir,
    "--no-unique-out-dir",
    "--strategy-preset", $StrategyPreset,
    "--execution-mode", $ExecutionMode,
    "--initial-assembly-offset-x", ([string]$InitialAssemblyOffsetX),
    "--initial-assembly-offset-y", ([string]$InitialAssemblyOffsetY),
    "--initial-assembly-offset-z", ([string]$InitialAssemblyOffsetZ)
)

if ([string]::IsNullOrWhiteSpace($CondaEnv)) {
    Write-Host "[run-v0.3-smoke] python $scriptPath $($commonArgs -join ' ')"
    python $scriptPath @commonArgs
} else {
    $condaBase = (conda info --base).Trim()
    $pythonExe = Join-Path $condaBase "envs\$CondaEnv\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "[run-v0.3-smoke] missing python executable: $pythonExe"
    }
    Write-Host "[run-v0.3-smoke] $pythonExe $scriptPath $($commonArgs -join ' ')"
    & $pythonExe $scriptPath @commonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "[run-v0.3-smoke] failed: exit=$LASTEXITCODE"
    }
}
