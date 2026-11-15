param(
    [Parameter(Mandatory = $true)]
    [string]$RtlTrace,
    [Parameter(Mandatory = $true)]
    [string]$CModelTrace
)

$rtl = @(Import-Csv -LiteralPath $RtlTrace)
$cmodel = @(Import-Csv -LiteralPath $CModelTrace)
$fields = @(
    'cycle','scenario','reset_n','cmd_valid','cmd_address','cmd_length',
    'cmd_id','req_ready','rsp_valid','rsp_data','rsp_keep','rsp_last',
    'rsp_id','rsp_error','reasm_ready','cmd_ready_pre','req_valid_pre',
    'req_address_pre','req_length_pre','req_id_pre','cmd_fire',
    'outstanding_enq_fire','outstanding_deq_fire','rsp_ready_pre',
    'rsp_input_fire','resp_valid_pre','resp_data_pre','resp_keep_pre',
    'resp_last_pre','resp_id_pre','resp_error_pre','reasm_fire',
    'outstanding_pre','rx_pre','fault_comb_pre','fault_code_comb_pre',
    'outstanding_post','rx_post','fault_state_post','fault_code_post','idle_post'
)
$hexFields = @('rsp_data','rsp_keep','resp_data_pre','resp_keep_pre')
$mismatches = [System.Collections.Generic.List[object]]::new()

if ($rtl.Count -ne $cmodel.Count) {
    $mismatches.Add([pscustomobject]@{
        row = -1; field = 'row_count'; rtl = $rtl.Count; cmodel = $cmodel.Count
    })
}

$count = [Math]::Min($rtl.Count, $cmodel.Count)
for ($row = 0; $row -lt $count; ++$row) {
    foreach ($field in $fields) {
        $rtlValue = $rtl[$row].$field
        $cmodelValue = $cmodel[$row].$field
        if ($hexFields -contains $field) {
            $rtlValue = $rtlValue.ToLower().TrimStart('0')
            $cmodelValue = $cmodelValue.ToLower().TrimStart('0')
            if ($rtlValue -eq '') { $rtlValue = '0' }
            if ($cmodelValue -eq '') { $cmodelValue = '0' }
        }
        if ($rtlValue -ne $cmodelValue) {
            $mismatches.Add([pscustomobject]@{
                row = $row; field = $field
                rtl = $rtl[$row].$field; cmodel = $cmodel[$row].$field
            })
        }
    }
}

if ($mismatches.Count -gt 0) {
    $first = $mismatches[0]
    Write-Output "FIRST_MISMATCH row=$($first.row) field=$($first.field) RTL=$($first.rtl) CModel=$($first.cmodel)"
    if ($first.row -ge 0) {
        $begin = [Math]::Max(0, $first.row - 3)
        $end = [Math]::Min($count - 1, $first.row + 3)
        for ($index = $begin; $index -le $end; ++$index) {
            Write-Output ("CONTEXT row={0} cycle={1} scenario={2} out={3}/{4} rx={5}/{6} fault={7}/{8}" -f
                $index, $rtl[$index].cycle, $rtl[$index].scenario,
                $rtl[$index].outstanding_post, $cmodel[$index].outstanding_post,
                $rtl[$index].rx_post, $cmodel[$index].rx_post,
                $rtl[$index].fault_code_post, $cmodel[$index].fault_code_post)
        }
    }
}

$semanticFailures = [System.Collections.Generic.List[string]]::new()
function Require-Coverage([bool]$Condition, [string]$Name) {
    if (-not $Condition) { $script:semanticFailures.Add($Name) }
}

if ($rtl.Count -gt 0) {
    $maximumOutstanding = ($rtl | Measure-Object -Property outstanding_post -Maximum).Maximum
    $maximumRx = ($rtl | Measure-Object -Property rx_post -Maximum).Maximum
    Require-Coverage ($maximumOutstanding -eq 16) 'outstanding_reaches_16'
    Require-Coverage ($maximumRx -eq 16) 'rx_fifo_reaches_16'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '2' -and $_.cmd_id -eq '99' -and
        $_.cmd_valid -eq '1' -and $_.cmd_ready_pre -eq '0' -and
        $_.outstanding_pre -eq '16'
    }).Count -gt 0) 'outstanding_full_rejects_command'
    Require-Coverage (@($rtl | Where-Object {
        $_.cmd_valid -eq '1' -and $_.req_ready -eq '0' -and
        $_.req_valid_pre -eq '1' -and $_.cmd_ready_pre -eq '0'
    }).Count -gt 0) 'request_ready_pause'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '2' -and $_.rsp_input_fire -eq '1' -and
        $_.outstanding_deq_fire -eq '1'
    }).Count -eq 16) 'ordered_multi_outstanding_responses'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '3' -and $_.rx_pre -eq '16' -and
        $_.rsp_valid -eq '1' -and $_.reasm_ready -eq '0' -and
        $_.rsp_ready_pre -eq '0' -and $_.rsp_input_fire -eq '0'
    }).Count -gt 0) 'rx_full_backpressure'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '3' -and $_.rx_pre -eq '16' -and
        $_.rsp_input_fire -eq '1' -and $_.reasm_fire -eq '1' -and
        $_.rx_post -eq '16'
    }).Count -gt 0) 'rx_full_dequeue_replacement'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '3' -and $_.idle_post -eq '1' -and
        $_.rx_post -eq '0' -and $_.outstanding_post -eq '0'
    }).Count -gt 0) 'rx_recovery_and_drain'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '4' -and $_.rsp_id -eq '99' -and
        $_.rsp_input_fire -eq '1' -and $_.fault_code_post -eq '2' -and
        $_.rx_post -eq '1' -and $_.outstanding_post -eq '1'
    }).Count -gt 0) 'id_mismatch_trigger_beat_is_enqueued'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '4' -and $_.cmd_id -eq '9' -and
        $_.cmd_ready_pre -eq '0' -and $_.rsp_ready_pre -eq '0' -and
        $_.reasm_fire -eq '1' -and $_.fault_code_post -eq '2'
    }).Count -gt 0) 'fault_sticky_blocks_inputs_but_drains_rx'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '5' -and $_.rsp_id -eq '55' -and
        $_.rsp_input_fire -eq '1' -and $_.fault_code_post -eq '2'
    }).Count -gt 0) 'response_without_outstanding_fault'
    Require-Coverage (@($rtl | Where-Object {
        $_.scenario -eq '6' -and $_.rsp_error -eq '1' -and
        $_.rsp_input_fire -eq '1' -and $_.fault_code_post -eq '3'
    }).Count -gt 0) 'response_error_fault'
    Require-Coverage (@($rtl | Where-Object {
        $_.rsp_last -eq '1' -and $_.rsp_input_fire -eq '1' -and
        $_.outstanding_deq_fire -eq '1'
    }).Count -gt 0) 'last_beat_releases_outstanding'
    Require-Coverage (@($rtl | Where-Object {
        $_.reset_n -eq '0' -and $_.fault_state_post -eq '0' -and
        $_.outstanding_post -eq '0' -and $_.rx_post -eq '0'
    }).Count -ge 2) 'reset_clears_protocol_state'
    $last = $rtl[-1]
    Require-Coverage ($last.idle_post -eq '1' -and
        $last.outstanding_post -eq '0' -and $last.rx_post -eq '0' -and
        $last.fault_state_post -eq '0') 'final_complete_drain'
}

Write-Output "mismatches=$($mismatches.Count) semantic_failures=$($semanticFailures.Count) rows=$($rtl.Count)"
if ($semanticFailures.Count -gt 0) {
    Write-Output ("SEMANTIC_FAILURES=" + ($semanticFailures -join ','))
}
if ($mismatches.Count -ne 0 -or $semanticFailures.Count -ne 0) {
    exit 1
}
Write-Output 'PASS: instruction DMA RTL and integrated CModel protocol core match cycle by cycle'
