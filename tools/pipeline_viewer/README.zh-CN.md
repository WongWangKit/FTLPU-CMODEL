# FTLPU Pipeline Viewer

Pipeline Viewer 是一个无外部依赖的 Canvas 波形工作台，用于查看 runtime
schedule trace。浏览器直接打开 `index.html`，然后载入
`write_schedule_trace_csv()` 生成的 CSV。

通过 HTTP 启动仓库目录时，也可以用 URL 直接打开某个 trace：

```text
http://127.0.0.1:8765/tools/pipeline_viewer/?trace=/docs/traces/smollm2_attention_schedule.csv
```

输入格式：

```csv
start,end,resource,detail
0,32,"MEM.E.Read","slice=0 addr=0 stream=E0"
32,64,"MXM.E0.Compute","Compute buffer=0 act=E0 out=W0"
```

操作：

- 鼠标滚轮：以指针所在 cycle 为中心缩放。
- 鼠标拖拽：水平移动可见时间窗口。
- Shift + 滚轮：水平移动。
- Ctrl/Alt + 滚轮：垂直滚动功能单元。
- 选择 A/B 后单击：放置测量游标。
- 单击 Overview：快速跳转到对应时间区域。
- 搜索框和功能单元菜单：筛选资源与指令详情。

远景下查看器会自动启用 LOD，把每个可见资源行的绘制数量限制在合理范围；
放大后自动恢复逐 event 绘制。状态栏中的 `LOD` 数字表示当前实际绘制的 event
数量，点击命中和详情查询始终使用完整原始 trace。

现有 runtime 测试会生成可直接载入的 trace，例如：

```text
build-ftlpu-vs2026/compiler/ftlpu_lower/
  smollm2_135m_ffn_seq128_pipeline/ffn.runtime.csv
  smollm2_attention_pipeline/attention.runtime.csv
  smollm2_decoder_layer_binary_runtime/decoder_layer.runtime.csv
```

查看器只绘制当前可见的资源行和 cycle 区间，因此长调度不再需要生成超宽
SVG。事件按资源建立时间索引，Overview 使用缓存的有界采样，因此几十万条
event 的整层 trace 仍可流畅缩放和拖动。原有 SVG/PNG 脚本继续用于文档中的
静态快照。
