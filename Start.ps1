param(
    [string]$Python = "",
    [int]$Port = 8765
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 or newer, then rerun Start.ps1." }
        & $Python -m pip install -r requirements-lock.txt
        if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
    }
}
if (-not (Test-Path -LiteralPath "frontend\dist\index.html")) {
    throw "Build the frontend first: cd frontend; npm ci; npm run build. See RUNNING.txt."
}
# Optional settings (e.g. a language model for the coach), KEY=VALUE lines: a git-ignored .env file here, then
# %LOCALAPPDATA%\Courtside\courtside.env, which stays outside synced folders such as OneDrive (use it for API keys).
$envFiles = @(".env")
if ($env:LOCALAPPDATA) { $envFiles += (Join-Path $env:LOCALAPPDATA "Courtside\courtside.env") }
foreach ($envFile in $envFiles) {
    if (-not (Test-Path -LiteralPath $envFile)) { continue }
    foreach ($line in Get-Content -LiteralPath $envFile) {
        if ($line -match '^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$' -and $line -notmatch '^\s*#') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"').Trim("'"), "Process")
        }
    }
}
# Gemini settings saved as Windows user environment variables count even if this terminal opened before they were set.
if ($env:OS -eq "Windows_NT") {
    foreach ($name in "GEMINI_API_KEY", "COURTSIDE_GEMINI_MODEL") {
        if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
            $value = [Environment]::GetEnvironmentVariable($name, "User")
            if ($value) { [Environment]::SetEnvironmentVariable($name, $value, "Process") }
        }
    }
}
Write-Host "Courtside: http://127.0.0.1:$Port"
Write-Host "Local research app. Press Ctrl+C to stop."
& $Python -m uvicorn backend.app:app --host 127.0.0.1 --port $Port
if ($LASTEXITCODE -ne 0) { throw "Courtside exited with an error." }
