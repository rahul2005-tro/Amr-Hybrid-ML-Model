$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$hybridPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $hybridPython)) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
}
& $hybridPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $hybridPython -m streamlit run app.py --server.port=8503 --server.address=127.0.0.1
