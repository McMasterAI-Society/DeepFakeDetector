param(
    [string]$ImageTag,
    [string]$EnvPath
)

$ErrorActionPreference = "Stop"

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Key
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    foreach ($line in Get-Content -Path $Path) {
        if ($line -match "^\s*$Key\s*=\s*(.*)\s*$") {
            $value = $Matches[1]
            # Remove trailing inline comments for unquoted values.
            if ($value -notmatch '^\s*".*"\s*$' -and $value -notmatch "^\s*'.*'\s*$") {
                $value = ($value -replace "\s+#.*$", "")
            }
            return $value.Trim().Trim('"').Trim("'")
        }
    }

    return $null
}

function Set-EnvValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Value
    )

    $newLine = "$Key=$Value"

    if (-not (Test-Path $Path)) {
        $newLine | Set-Content -Path $Path -Encoding UTF8
        return
    }

    $lines = Get-Content -Path $Path
    $updated = $false

    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$Key\s*=") {
            $lines[$i] = $newLine
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        $lines += $newLine
    }

    $lines | Set-Content -Path $Path -Encoding UTF8
}

function Test-GhcrTag {
    param([string]$Tag)
    return ($Tag -match '^ghcr\.io\/[a-zA-Z0-9._-]+\/[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+$')
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

if ([string]::IsNullOrWhiteSpace($EnvPath)) {
    $EnvPath = Join-Path $scriptDir ".env"
}

Write-Host "DeepFake Detector - GHCR Base Image Updater" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed or not available on PATH."
}

$currentEnvTag = Get-EnvValue -Path $EnvPath -Key "HF_BASE_IMAGE"

if ([string]::IsNullOrWhiteSpace($ImageTag)) {
    if (-not [string]::IsNullOrWhiteSpace($currentEnvTag) -and (Test-GhcrTag -Tag $currentEnvTag)) {
        $ImageTag = $currentEnvTag
        Write-Host "Using HF_BASE_IMAGE from .env: $ImageTag" -ForegroundColor Green
    }
}

if ([string]::IsNullOrWhiteSpace($ImageTag)) {
    Write-Host "No valid GHCR image tag found in .env (HF_BASE_IMAGE)." -ForegroundColor Yellow
    $ImageTag = Read-Host "Enter GHCR image tag (example: ghcr.io/lukhsaankumar/deepfake-detector-backend-base:latest). Leave blank for guided setup"

    if ([string]::IsNullOrWhiteSpace($ImageTag)) {
        Write-Host "" 
        Write-Host "Guided setup:" -ForegroundColor Cyan
        Write-Host "1) Create a GitHub PAT with scopes: write:packages, read:packages" -ForegroundColor Gray
        Write-Host "2) Run: docker login ghcr.io" -ForegroundColor Gray
        Write-Host "3) Use image tag format: ghcr.io/<github-username>/deepfake-detector-backend-base:latest" -ForegroundColor Gray
        Write-Host ""

        $ghUser = Read-Host "Enter your GitHub username"
        if ([string]::IsNullOrWhiteSpace($ghUser)) {
            throw "GitHub username is required to build an image tag."
        }

        $defaultTag = "ghcr.io/$ghUser/deepfake-detector-backend-base:latest"
        $manualTag = Read-Host "Image tag to use [Press Enter for $defaultTag]"
        if ([string]::IsNullOrWhiteSpace($manualTag)) {
            $ImageTag = $defaultTag
        }
        else {
            $ImageTag = $manualTag.Trim()
        }
    }
}

$ImageTag = $ImageTag.Trim()

if (-not (Test-GhcrTag -Tag $ImageTag)) {
    throw "Invalid GHCR image tag: '$ImageTag'. Expected format: ghcr.io/<user>/<name>:<tag>"
}

if ($currentEnvTag -ne $ImageTag) {
    Set-EnvValue -Path $EnvPath -Key "HF_BASE_IMAGE" -Value $ImageTag
    Write-Host "Updated HF_BASE_IMAGE in $EnvPath" -ForegroundColor Green
}
else {
    Write-Host "HF_BASE_IMAGE already set in $EnvPath" -ForegroundColor Green
}

$shouldLogin = Read-Host "Run 'docker login ghcr.io' now? (Y/n)"
if ([string]::IsNullOrWhiteSpace($shouldLogin) -or $shouldLogin -match '^(y|yes)$') {
    & docker login ghcr.io
    if ($LASTEXITCODE -ne 0) {
        throw "Docker login failed."
    }
}

Push-Location $repoRoot
try {
    Write-Host ""
    Write-Host "Building base image: $ImageTag" -ForegroundColor Cyan
    & docker build -f backend/Dockerfile.base -t $ImageTag backend
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed."
    }

    Write-Host ""
    Write-Host "Pushing base image: $ImageTag" -ForegroundColor Cyan
    & docker push $ImageTag
    if ($LASTEXITCODE -ne 0) {
        throw "Docker push failed."
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Success. Base image updated in GHCR." -ForegroundColor Green
Write-Host "HF_BASE_IMAGE is now set to: $ImageTag" -ForegroundColor Green
Write-Host ""
Write-Host "Next step:" -ForegroundColor Cyan
Write-Host "bash ./backend/deploy-to-hf.sh" -ForegroundColor Gray
