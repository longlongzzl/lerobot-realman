param(
    [string]$EnvName = "test_env",
    [string]$ManiSkillEditableRoot = "D:\Project\Scaling\ManiSkill-main\ManiSkill-main",
    [switch]$UseFullSpec,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$envDir = Join-Path $repoRoot "repro\env"
$fullSpec = Join-Path $envDir "sim2real_curobo_v0.3_full.yml"
$historySpec = Join-Path $envDir "sim2real_curobo_v0.3_from_history.yml"
$pipRuntime = Join-Path $envDir "pip_runtime_v0.3.txt"
$specPath = if ($UseFullSpec) { $fullSpec } else { $historySpec }

function Invoke-NativeStep {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    Write-Host "[setup-test-env] $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "[setup-test-env] failed: $Label (exit=$LASTEXITCODE)"
    }
}

function Get-CondaEnvPython {
    param(
        [string]$Name
    )

    $condaBase = (conda info --base).Trim()
    if (-not $condaBase) {
        throw "[setup-test-env] cannot resolve conda base"
    }
    $pythonExe = Join-Path $condaBase "envs\$Name\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "[setup-test-env] missing python executable: $pythonExe"
    }
    return $pythonExe
}

$envExists = $false
$envList = conda env list
foreach ($line in $envList) {
    if ($line -match "^\s*$([regex]::Escape($EnvName))\s+") {
        $envExists = $true
        break
    }
}

if ($envExists -and $Recreate) {
    Invoke-NativeStep "remove existing env $EnvName" { conda env remove -n $EnvName -y }
    $envExists = $false
}

if (-not $envExists) {
    Invoke-NativeStep "create $EnvName from $specPath" { conda env create -n $EnvName -f $specPath }
} else {
    Write-Host "[setup-test-env] reuse existing env $EnvName"
}

$pythonExe = Get-CondaEnvPython -Name $EnvName

Invoke-NativeStep "install torch stack with conda" {
    conda install -n $EnvName pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
}

Invoke-NativeStep "install pip runtime requirements from $pipRuntime" {
    & $pythonExe -m pip install --no-deps -r $pipRuntime
}

Invoke-NativeStep "install editable mani_skill from $ManiSkillEditableRoot" {
    & $pythonExe -m pip install --no-deps -e $ManiSkillEditableRoot
}

Invoke-NativeStep "apply repo patch" {
    & $pythonExe (Join-Path $repoRoot "repro\apply_maniskill_patch.py") --target-root $ManiSkillEditableRoot
}

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Invoke-NativeStep "verify runtime" {
    & $pythonExe (Join-Path $repoRoot "repro\verify_runtime_setup.py") --target-root $ManiSkillEditableRoot
}

Write-Host "[setup-test-env] done"
