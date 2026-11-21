param(
    [Parameter(Mandatory = $true)]
    [string]$RtlTrace,
    [Parameter(Mandatory = $true)]
    [string]$CModelTrace
)

$rtl = @(Import-Csv -LiteralPath $RtlTrace)
$cmodel = @(Import-Csv -LiteralPath $CModelTrace)

if ($rtl.Count -ne $cmodel.Count) {
    throw "trace row count differs: RTL=$($rtl.Count) CModel=$($cmodel.Count)"
}

$fields = @(
    'cycle',
    'enqueue_valid',
    'enqueue_data',
    'dequeue_ready',
    'enqueue_ready_pre',
    'dequeue_valid_pre',
    'enqueue_fire',
    'dequeue_fire',
    'dequeue_data_pre',
    'level_pre',
    'write_ptr_pre',
    'read_ptr_pre',
    'level_post',
    'write_ptr_post',
    'read_ptr_post',
    'dequeue_valid_post',
    'dequeue_data_post'
)

for ($row = 0; $row -lt $rtl.Count; ++$row) {
    foreach ($field in $fields) {
        if ($rtl[$row].$field -ne $cmodel[$row].$field) {
            throw "trace mismatch row=$row field=$field RTL=$($rtl[$row].$field) CModel=$($cmodel[$row].$field)"
        }
    }
}

Write-Output "PASS: $($rtl.Count) FIFO cycles match across RTL and CModel"
