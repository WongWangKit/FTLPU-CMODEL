# ICU Repeat2D 指令

`Repeat2D` 是队列局部、阻塞式的二维单指令迭代控制。紧邻它之前的功能指令已经作为
坐标 `(0, 0)` 发射；描述符按 outer-major、inner-minor 顺序产生其余坐标。

```text
cycle(o, i) = base_cycle + o * outer_interval + i * inner_interval
delta(o, i) = o * outer_stride + i * inner_stride
```

两个 count 均包含基点。执行期间只有当前本地队列被阻塞，其他 ICU 队列继续执行。

## 96-bit 编码

| 位段 | 字段 |
| --- | --- |
| 1:0 | 扩展控制 opcode，复用 Loop 类别 |
| 11:2 | inner_count |
| 21:12 | outer_count |
| 37:22 | inner_interval |
| 53:38 | outer_interval |
| 69:54 | 有符号 inner_stride |
| 85:70 | 有符号 outer_stride |
| 87:86 | induction target |
| 91:88 | 扩展 subtype = 1 |
| 95:92 | 保留 |

字段范围为 count `1..1023`、interval `1..65535`、stride
`-32768..32767`。`outer_interval` 必须大于一个 inner wave 的最后发射 offset。

归纳目标：`None`、`MemAddress`、`MxmWeightColumn`。MEM 归纳所选 bank 的 local row
address；MXM 只允许归纳 `IW` weight column。实现保存上一条已解码功能指令，不重新读取
i-MEM，并在每个目标 cycle 复制和修改该指令。
