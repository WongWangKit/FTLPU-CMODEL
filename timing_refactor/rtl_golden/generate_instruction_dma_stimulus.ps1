param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$rows = [System.Collections.Generic.List[object]]::new()
$cycle = 0

function Add-DmaCycle {
    param(
        [int]$Scenario,
        [int]$ResetN = 1,
        [int]$CommandValid = 0,
        [UInt64]$CommandAddress = 0,
        [UInt32]$CommandLength = 0,
        [int]$CommandId = 0,
        [int]$RequestReady = 1,
        [int]$ResponseValid = 0,
        [Int64]$ResponseKeep = 0,
        [int]$ResponseLast = 0,
        [int]$ResponseId = 0,
        [int]$ResponseError = 0,
        [int]$ReassemblerReady = 0
    )
    $byte = '{0:x2}' -f ($script:cycle % 256)
    $script:rows.Add([pscustomobject]@{
        cycle = $script:cycle
        scenario = $Scenario
        reset_n = $ResetN
        cmd_valid = $CommandValid
        cmd_address = $CommandAddress
        cmd_length = $CommandLength
        cmd_id = $CommandId
        req_ready = $RequestReady
        rsp_valid = $ResponseValid
        rsp_data = $byte * 32
        rsp_keep = if ($ResponseKeep -lt 0) { 'ffffffff' }
            else { '{0:x8}' -f ([UInt32]$ResponseKeep) }
        rsp_last = $ResponseLast
        rsp_id = $ResponseId
        rsp_error = $ResponseError
        reasm_ready = $ReassemblerReady
    })
    ++$script:cycle
}

# 0: initial/reset; 1: one command, request pause, two response beats.
Add-DmaCycle -Scenario 0 -ResetN 0 -RequestReady 0
Add-DmaCycle -Scenario 0 -ResetN 0 -RequestReady 0
Add-DmaCycle -Scenario 1 -CommandValid 1 -CommandAddress 4096 -CommandLength 64 -CommandId 1 -RequestReady 0
Add-DmaCycle -Scenario 1 -CommandValid 1 -CommandAddress 4096 -CommandLength 64 -CommandId 1
Add-DmaCycle -Scenario 1 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseId 1
Add-DmaCycle -Scenario 1 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 1 -ReassemblerReady 1
Add-DmaCycle -Scenario 1 -ReassemblerReady 1

# 2: continuous commands, outstanding=16, rejection at full, ordered returns.
Add-DmaCycle -Scenario 2 -ResetN 0
Add-DmaCycle -Scenario 2 -ResetN 0
for ($id = 0; $id -lt 16; ++$id) {
    Add-DmaCycle -Scenario 2 -CommandValid 1 -CommandAddress (8192 + 256 * $id) -CommandLength 32 -CommandId $id
}
Add-DmaCycle -Scenario 2 -CommandValid 1 -CommandAddress 16384 -CommandLength 32 -CommandId 99
for ($id = 0; $id -lt 16; ++$id) {
    Add-DmaCycle -Scenario 2 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId $id -ReassemblerReady 1
}
Add-DmaCycle -Scenario 2 -ReassemblerReady 1

# 3: RX FIFO full, blocked response, full+dequeue replacement and recovery.
Add-DmaCycle -Scenario 3 -ResetN 0
Add-DmaCycle -Scenario 3 -ResetN 0
Add-DmaCycle -Scenario 3 -CommandValid 1 -CommandAddress 24576 -CommandLength 576 -CommandId 42
for ($beat = 0; $beat -lt 16; ++$beat) {
    Add-DmaCycle -Scenario 3 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseId 42
}
Add-DmaCycle -Scenario 3 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseId 42
Add-DmaCycle -Scenario 3 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseId 42 -ReassemblerReady 1
Add-DmaCycle -Scenario 3 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 42 -ReassemblerReady 1
for ($drain = 0; $drain -lt 16; ++$drain) {
    Add-DmaCycle -Scenario 3 -ReassemblerReady 1
}

# 4: simultaneous ID mismatch + response error. ID has priority. The
# triggering beat still enters RX; after fault, only existing RX may drain.
Add-DmaCycle -Scenario 4 -ResetN 0
Add-DmaCycle -Scenario 4 -ResetN 0
Add-DmaCycle -Scenario 4 -CommandValid 1 -CommandAddress 32768 -CommandLength 32 -CommandId 7
Add-DmaCycle -Scenario 4 -CommandValid 1 -CommandAddress 33024 -CommandLength 32 -CommandId 8
Add-DmaCycle -Scenario 4 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 99 -ResponseError 1
Add-DmaCycle -Scenario 4 -CommandValid 1 -CommandAddress 33280 -CommandLength 32 -CommandId 9 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 8 -ReassemblerReady 1
Add-DmaCycle -Scenario 4 -ReassemblerReady 1

# 5: response without any outstanding command.
Add-DmaCycle -Scenario 5 -ResetN 0
Add-DmaCycle -Scenario 5 -ResetN 0
Add-DmaCycle -Scenario 5 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 55
Add-DmaCycle -Scenario 5 -ReassemblerReady 1
Add-DmaCycle -Scenario 5 -ReassemblerReady 1

# 6: matching response error and sticky fault after the response drains.
Add-DmaCycle -Scenario 6 -ResetN 0
Add-DmaCycle -Scenario 6 -ResetN 0
Add-DmaCycle -Scenario 6 -CommandValid 1 -CommandAddress 36864 -CommandLength 32 -CommandId 9
Add-DmaCycle -Scenario 6 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 9 -ResponseError 1
Add-DmaCycle -Scenario 6 -ReassemblerReady 1
Add-DmaCycle -Scenario 6 -CommandValid 1 -CommandAddress 37120 -CommandLength 32 -CommandId 10 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 10 -ReassemblerReady 1

# 7: final clean ordered sequence and complete drain.
Add-DmaCycle -Scenario 7 -ResetN 0
Add-DmaCycle -Scenario 7 -ResetN 0
Add-DmaCycle -Scenario 7 -CommandValid 1 -CommandAddress 40960 -CommandLength 32 -CommandId 20 -RequestReady 0
Add-DmaCycle -Scenario 7 -CommandValid 1 -CommandAddress 40960 -CommandLength 32 -CommandId 20
Add-DmaCycle -Scenario 7 -CommandValid 1 -CommandAddress 41216 -CommandLength 32 -CommandId 21
Add-DmaCycle -Scenario 7 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 20
Add-DmaCycle -Scenario 7 -ResponseValid 1 -ResponseKeep 0xffffffff -ResponseLast 1 -ResponseId 21 -ReassemblerReady 1
Add-DmaCycle -Scenario 7 -ReassemblerReady 1
Add-DmaCycle -Scenario 7 -ReassemblerReady 1

$directory = Split-Path -Parent $OutputPath
if ($directory -and -not (Test-Path -LiteralPath $directory)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}
$header = 'cycle,scenario,reset_n,cmd_valid,cmd_address,cmd_length,cmd_id,req_ready,rsp_valid,rsp_data,rsp_keep,rsp_last,rsp_id,rsp_error,reasm_ready'
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add($header)
foreach ($row in $rows) {
    $lines.Add(($row.cycle, $row.scenario, $row.reset_n, $row.cmd_valid,
        $row.cmd_address, $row.cmd_length, $row.cmd_id, $row.req_ready,
        $row.rsp_valid, $row.rsp_data, $row.rsp_keep, $row.rsp_last,
        $row.rsp_id, $row.rsp_error, $row.reasm_ready) -join ',')
}
Set-Content -LiteralPath $OutputPath -Value $lines -Encoding ascii
Write-Output "wrote $($rows.Count) instruction DMA stimulus cycles to $OutputPath"
