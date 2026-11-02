# 共享 QCTL 控制池

当前 main 已接入共享池，并适配 ICU JSON 参数化和定时 DMA 的 QCTL 提交/PRIME。以下隔离基点记录为历史证据；本次 main 集成验证见 SOFTWARE 的 `docs/queue_control_shared_pool_main.zh-CN.md`。

本 worktree 使用一个每芯片 512 条的共享控制池、每队列 4 条缓存，单队列记录上限 16 条。共享读端口为 1、读延迟 2 拍、请求启动间隔 1 拍。控制操作和原生状态机保持不变，Native 绑定不消耗控制池记录。

`IcuQueueControlPool` 保存唯一的完整记录数组及有限分区缓存。`InstructionControlUnit` 在每个物理 dispatch 和 PRIME 拍统一推进池；队列前端持有目录句柄，不自行推进共享 RAM 时钟。返回位置在请求时预留，缓存加在途信用不超过4。双槽响应流水及轮转仲裁由明确边沿推进，不使用无延迟 RAM 旁路。

任务重置清理池和响应流水、更新内部 generation。已经发出的 FU 数据流水、Native 通知语义、MEM 包执行和实际 SRAM 提交逻辑保持原实现。

独立 RAM 后端仍用于 storage_version=0 的兼容目标。新池使用 storage_version=1，三个新硬件字段由 SOFTWARE v40 容器及 ABI 携带。目标能力不匹配时拒绝；不可把旧每队列端口解释成新的全芯片端口。

定向测试已覆盖四队列各16条记录、4条缓存下48次运行期补充、单端口和两拍延迟、无额外控制拍、缓存耗尽拒绝及旧任务句柄隔离。SOFTWARE 真实 seq32/seq128 在25,600 MB/s下的数值、逐拍轨迹和最终镜像对照通过，主执行分别为110,978与417,796拍；新 PRIME 均为296拍。

详细供给证明、空间估算、性能和原始证据见 SOFTWARE 同一 `codex/qctl-shared-pool` worktree 的 `docs/queue_control_shared_pool.zh-CN.md`。本实现基于 `c0988ab`，未覆盖 main 上同时进行的 ICU 参数化及 DMA 完整程序装载修改；尚未进行该装载路径集成或 RTL 综合。修改未提交、未推送。
