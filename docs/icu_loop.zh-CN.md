# ICU Loop 指令

`Loop` 是队列局部控制指令，用于回放此前连续发射的一组功能指令。它与
`Repeat` 互补：`Repeat` 只能回放紧邻的一条指令，`Loop` 可以回放多条指令。

## 语义

`Loop(window_size, count, interval, address_stride)` 将前面的
`window_size` 条功能指令额外执行 `count` 轮。Loop 控制指令的发射周期同时
用于发射第一条回放指令。`interval` 表示相邻两轮起始周期的距离，且不能小于
`window_size`。

对于 MEM bank 队列，ICU 将 `address_stride` 乘以回放轮次后，加到 bank-local
row address 上。其他类型队列要求地址步进为 0。Loop 窗口不能包含控制
指令，也暂不允许嵌套。

## 32 位编码

| 位段 | 字段 |
| --- | --- |
| 1:0 | opcode = 3 |
| 7:2 | window_size，范围 1..63 |
| 15:8 | count，范围 1..255 |
| 23:16 | interval，范围 1..255 |
| 31:24 | 有符号 address_stride，范围 -128..127 |

ICU 保存回放窗口在本地 i-MEM 中的 PC，并在每轮执行时重新取指。因此 runtime
只需提交一条 Loop 控制指令，不需要在 Host 侧展开重复指令序列。
