[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $repoRoot 'build'
$repoTmpDir = Join-Path $buildDir 'tmp'
$simPath = Join-Path $buildDir 'lpu_smoke.vvp'
$archSimPath = Join-Path $buildDir 'lpu_arch.vvp'

if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
  throw 'iverilog was not found on PATH. Install Icarus Verilog before running the smoke test.'
}

New-Item -ItemType Directory -Force -Path $buildDir, $repoTmpDir | Out-Null
$env:TEMP = $repoTmpDir
$env:TMP = $repoTmpDir

Push-Location $repoRoot
try {
  & iverilog -g2012 -s lpu_smoke_tb -o $simPath -f sim/filelist.f
  if ($LASTEXITCODE -ne 0) {
    throw "Icarus Verilog compilation failed with exit code $LASTEXITCODE."
  }

  & vvp $simPath
  if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed with exit code $LASTEXITCODE."
  }

  & iverilog -g2012 -s lpu_arch_tb -o $archSimPath -f sim/filelist.f
  if ($LASTEXITCODE -ne 0) {
    throw "Icarus Verilog architecture-test compilation failed with exit code $LASTEXITCODE."
  }

  & vvp $archSimPath
  if ($LASTEXITCODE -ne 0) {
    throw "Architecture test failed with exit code $LASTEXITCODE."
  }

}
finally {
  Pop-Location
}
