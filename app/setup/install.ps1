# osu!taiko recommender setup.
# Everything goes into the installation folder: .\runtime (private embeddable Python + numpy/PySide6)
# and the launcher. Nothing is written outside it (no shortcuts, no temp files, no pip cache).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)          # the installation folder
$runtime = Join-Path $root 'runtime'
$pyVersion = '3.12.10'
$pyZip = "python-$pyVersion-embed-amd64.zip"
$env:PIP_NO_CACHE_DIR = '1'                                             # no pip cache in %LOCALAPPDATA%
$env:PYTHONNOUSERSITE = '1'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'                                # much faster Invoke-WebRequest

if (-not (Test-Path (Join-Path $runtime 'python.exe'))) {
    New-Item -ItemType Directory -Force -Path $runtime | Out-Null
    $tmp = Join-Path $root $pyZip                                       # downloaded inside the folder, then deleted
    Write-Host "downloading Python $pyVersion ..."
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$pyVersion/$pyZip" -OutFile $tmp -UseBasicParsing
    Expand-Archive -Path $tmp -DestinationPath $runtime -Force
    Remove-Item $tmp
    # enable site-packages and put the app folder on the path
    $pth = Get-ChildItem $runtime -Filter 'python*._pth' | Select-Object -First 1
    $lines = Get-Content $pth.FullName | ForEach-Object { if ($_ -match '^#\s*import site') { 'import site' } else { $_ } }
    $lines += '..\app'
    Set-Content -Path $pth.FullName -Value $lines -Encoding ASCII
    Write-Host "installing pip ..."
    $getpip = Join-Path $runtime 'get-pip.py'
    Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getpip -UseBasicParsing
    & (Join-Path $runtime 'python.exe') $getpip --no-warn-script-location --no-cache-dir
    if ($LASTEXITCODE -ne 0) { throw "pip installation failed" }
    Remove-Item $getpip
}

Write-Host "installing libraries (numpy, PySide6) - this can take a few minutes ..."
& (Join-Path $runtime 'python.exe') -m pip install --no-warn-script-location --no-cache-dir --upgrade -r (Join-Path $root 'app\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "library installation failed" }

$launcher = Join-Path $root 'osu!taiko recommender.bat'
Set-Content -Path $launcher -Encoding ASCII -Value "@echo off`r`nstart `"`" `"%~dp0runtime\pythonw.exe`" `"%~dp0app\main.py`""

Write-Host ""
Write-Host "done. start the app with 'osu!taiko recommender.bat' in this folder."
Write-Host "(your data is kept in the 'data' folder here; to uninstall, delete this folder.)"
