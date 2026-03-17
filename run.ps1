$ErrorActionPreference = "Stop"

if (-not $env:GEMINI_API_KEY) {
    Write-Host "GEMINI_API_KEY is not set. Please set it before running." -ForegroundColor Yellow
    Write-Host 'Example: $env:GEMINI_API_KEY="YOUR_KEY_HERE"' -ForegroundColor Yellow
    exit 1
}

$env:GEMINI_GENERATION_ENABLED = "true"
$env:GEMINI_VALIDATION_ENABLED = "true"
$env:GEMINI_VALIDATE_FAIL_OPEN = "false"
$env:GEMINI_VALIDATE_MIN_ACCEPT_RATIO = "0.7"
$env:GEMINI_VALIDATE_MAX_REJECTS = "3"
$env:GEMINI_REPAIR_MAX_ATTEMPTS = "6"
$env:DEBUG_GENERATION = "true"

python app.py
