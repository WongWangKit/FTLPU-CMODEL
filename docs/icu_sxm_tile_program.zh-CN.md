# SXM TILE_PROGRAM

`SXM_TILE_PROGRAM` 将一条完整的原生 SXM 配置与一到三维绝对 cycle 迭代
空间组合起来。配置会完整保留 opcode、源/目标 stream 列表、row/tile 选择器
以及 32-lane permutation map。

```text
cycle = start_cycle + sum(i[d] * cycle_stride[d])
```

每个半球的 transpose 和 permute 仍属于两条独立 ICU 队列。tile program 不会
把两个端口串行化，也不会改变两者原有的并行关系。典型 lowering 用一条
transpose program 表示较长的 token/block repeat，并用四条 permute program
表示循环出现的四种 tile-row map。

当前版本不对 SXM 指令内部字段做归纳。非规则 map 使用不同 tile program，
并通过队列的 next-issue calendar 交错发射。

