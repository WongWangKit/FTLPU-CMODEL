[CmdletBinding()]
param()

# Native Icarus diagnostics can be emitted on stderr even when compilation
# succeeds. Native process status is checked explicitly below.
$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path -Parent $PSScriptRoot
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
  'ftlpu_final_regression_' + [Guid]::NewGuid().ToString('N')
)

$regressions = @(
  [pscustomobject]@{ Name = 'SRF'; Filelist = 'sim/filelist_srf.f'; Marker = 'VMODEL_SRF_PORT TEST_PASS' },
  [pscustomobject]@{ Name = 'MEM-SRF'; Filelist = 'sim/filelist_mem_srf.f'; Marker = 'VMODEL_MEM_SRF_INTEGRATION TEST_PASS' },
  [pscustomobject]@{ Name = 'MEM-SRF-SXM'; Filelist = 'sim/filelist_mem_srf_sxm_roundtrip.f'; Marker = 'VMODEL_MEM_SRF_SXM_ROUNDTRIP TEST_PASS' },
  [pscustomobject]@{ Name = 'C2C CREDIT'; Filelist = 'sim/filelist_c2c_vector_credit_serializer.f'; Marker = 'C2C_VECTOR_CREDIT_SERIALIZER TEST_PASS' },
  [pscustomobject]@{ Name = 'C2C TX'; Filelist = 'sim/filelist_c2c_tx_peer_path.f'; Marker = 'C2C_TX_PEER_PATH TEST_PASS' },
  [pscustomobject]@{ Name = 'C2C RX PAIR'; Filelist = 'sim/filelist_c2c_rx_issue_pair.f'; Marker = 'C2C_RX_ISSUE_PAIR TEST_PASS' },
  [pscustomobject]@{ Name = 'C2C RX SRF'; Filelist = 'sim/filelist_c2c_rx_srf_integration.f'; Marker = 'C2C_RX_SRF_INTEGRATION TEST_PASS' },
  [pscustomobject]@{ Name = 'C2C E2E'; Filelist = 'sim/filelist_c2c_srf_peer_e2e.f'; Marker = 'C2C_SRF_PEER_E2E TEST_PASS' },
  [pscustomobject]@{ Name = 'DMA STORE'; Filelist = 'sim/filelist_dma_store_vector_sink.f'; Marker = 'DMA_STORE_VECTOR_SINK TEST_PASS' },
  [pscustomobject]@{ Name = 'DMA CONTEXT'; Filelist = 'sim/filelist_dma_store_lane_context.f'; Marker = 'DMA_STORE_LANE_CONTEXT TEST_PASS' }
)

function Write-ToolOutput {
  param([object[]]$Lines)

  foreach ($line in $Lines) {
    Write-Host $line
  }
}

function Invoke-Regression {
  param([pscustomobject]$Test, [string]$OutputDirectory)

  $filelistPath = Join-Path $repoRoot $Test.Filelist
  $outputName = ($Test.Name -replace '[^A-Za-z0-9]+', '_') + '.vvp'
  $vvpPath = Join-Path $OutputDirectory $outputName

  Write-Host "[RUN ] $($Test.Name)"

  if (-not (Test-Path -LiteralPath $filelistPath)) {
    Write-Host "[FAIL] $($Test.Name) compile: missing filelist $($Test.Filelist)"
    return [pscustomobject]@{ Name = $Test.Name; Pass = $false; Stage = 'compile'; Detail = 'missing filelist' }
  }

  $compileOutput = @(& iverilog -g2012 -o $vvpPath -f $Test.Filelist 2>&1)
  $compileExitCode = $LASTEXITCODE
  Write-ToolOutput $compileOutput
  if ($compileExitCode -ne 0) {
    Write-Host "[FAIL] $($Test.Name) compile exit=$compileExitCode command=iverilog -g2012 -o <temp> -f $($Test.Filelist)"
    return [pscustomobject]@{ Name = $Test.Name; Pass = $false; Stage = 'compile'; Detail = "exit=$compileExitCode" }
  }

  $simulationOutput = @(& vvp $vvpPath 2>&1)
  $simulationExitCode = $LASTEXITCODE
  Write-ToolOutput $simulationOutput
  $allOutput = (@($compileOutput) + @($simulationOutput) | Out-String -Width 4096)
  $markerFound = $allOutput.Contains($Test.Marker)
  $testFailFound = $allOutput.Contains('TEST_FAIL')

  if (($simulationExitCode -ne 0) -or -not $markerFound -or $testFailFound) {
    $detail = "simulation_exit=$simulationExitCode marker_found=$markerFound test_fail_found=$testFailFound"
    Write-Host "[FAIL] $($Test.Name) simulation $detail marker=$($Test.Marker)"
    return [pscustomobject]@{ Name = $Test.Name; Pass = $false; Stage = 'simulation'; Detail = $detail }
  }

  Write-Host "[PASS] $($Test.Name)"
  return [pscustomobject]@{ Name = $Test.Name; Pass = $true; Stage = 'complete'; Detail = $Test.Marker }
}

if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
  Write-Host 'ERROR: iverilog was not found on PATH.'
  exit 1
}

if (-not (Get-Command vvp -ErrorAction SilentlyContinue)) {
  Write-Host 'ERROR: vvp was not found on PATH.'
  exit 1
}

$results = @()
$allPass = $true
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

Push-Location $repoRoot
try {
  Write-Host '========================================'
  Write-Host 'FTLPU FINAL REGRESSION'
  Write-Host '========================================'
  Write-Host "Simulator: $((& iverilog -V 2>&1 | Select-Object -First 1))"

  foreach ($test in $regressions) {
    $result = Invoke-Regression -Test $test -OutputDirectory $tempRoot
    $results += $result
    if (-not $result.Pass) {
      $allPass = $false
    }
  }

  Write-Host '========================================'
  Write-Host 'REGRESSION SUMMARY'
  Write-Host '========================================'
  foreach ($result in $results) {
    $status = if ($result.Pass) { 'PASS' } else { 'FAIL' }
    Write-Host ('{0,-24} {1}' -f $result.Name, $status)
  }

  if ($allPass) {
    Write-Host 'ALL KEY REGRESSIONS PASS'
  } else {
    Write-Host 'REGRESSION HAS FAILURES'
    foreach ($result in ($results | Where-Object { -not $_.Pass })) {
      Write-Host ("FAILURE: {0} stage={1} {2}" -f $result.Name, $result.Stage, $result.Detail)
    }
  }
}
finally {
  Pop-Location
  Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($allPass) {
  exit 0
}

exit 1
