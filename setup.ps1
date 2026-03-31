# Local Windows setup (training is usually on Linux + CUDA; this preps the repo).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

& $py -m venv .venv
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "On a Linux GPU box, use setup.sh and install the CUDA torch wheel there."
Write-Host "FlashAttention is optional and often awkward on Windows."
