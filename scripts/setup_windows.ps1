param(
  [string]$PythonCmd = "py"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Create virtual environment (.venv)"
if ($PythonCmd -eq "py") {
  & py -3 -m venv .venv
} else {
  & $PythonCmd -m venv .venv
}

Write-Host "[2/4] Activate virtual environment"
& .\.venv\Scripts\Activate.ps1

Write-Host "[3/4] Upgrade pip"
python -m pip install -U pip

Write-Host "[4/4] Install requirements"
python -m pip install -r requirements.txt

Write-Host "Done"
