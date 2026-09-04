# MEM ICU Slice Program

## 目的

`MEM_SLICE_PROGRAM` 用一个 MEM ICU FIFO entry 描述某个物理 SRAM slice
上的一小段程序。它把遍历同一个一到三维启动域的多项 MEM 操作放在一起，
同时保留每次原生 read/write 的精确发射 cycle 和地址。

所有 body entry 共享以下启动域：

```text
start_cycle
rank                                  // 1..3
count[3]
cycle_stride[3]
```

一条 program 可以包含 1 到 16 个 body entry，每项包含：

```text
cycle_offset
address_stride[3]
native MemInstruction                 // read/write、地址、SR、方向
```

对启动坐标 `i[d]` 和 body entry `b`，展开后的操作为：

```text
cycle(b, i) = start_cycle + cycle_offset[b]
              + sum(i[d] * cycle_stride[d])
address(b, i) = native[b].address
                + sum(i[d] * address_stride[b][d])
```

不同 body 可以分别做 read 或 write，也可以使用不同 stream register；它们
能够合成一条 program 的依据是 launch count 和 cycle stride 相同。

## ICU 执行方式

整条 program 在 i-MEM/IQ 中只占一个 entry，并共享一个 program counter。
CModel 只校验一次 program，然后为各 body 建立队列本地的 active cursor。
这相当于对 MEM ICU 本地微序列器做功能建模，同时保留逐 cycle 仲裁和报错。

以下情况会立即报错：body 为空或超过 16 项、N 维启动域非法、地址归纳超出
原生 MEM 地址空间、指令到达时已经错过发射 cycle，以及同一个 MEM queue
在同一 cycle 有两个 body/program cursor 到期。

## Binary 原型

软件验证格式使用 `MSPG` magic、版本 1 和 ICU `Extended` opcode。扩展
payload 为：

```text
header: magic, version, start_cycle, rank, body_count,
        count[0], cycle_stride[0], ... count[2], cycle_stride[2]
body:   cycle_offset, address_stride[0..2], native_lo, native_hi
```

header 占 11 个 32-bit word，每个 body 占 6 个 word。对 program 做 binding
relocation 时，runtime 会修改每个 body 中原生 MEM 指令的基地址。当前采用
可变长软件 envelope；最终 RTL 可以把 body store 与固定宽 FIFO descriptor
分开实现。

## 硬件实现方向

MEM ICU 可以只保存一份 launch counter，并在本地 program RAM 中保存最多
16 个紧凑 body record。每项需要 next-issue offset、三个有符号地址步长和
一条原生 MEM 操作，再由小型 calendar 选择本 cycle 到期的 body。16 项上限
属于 ISA revision 约束，不包含模型或 projection 的专用语义。

