# MuxWise 调度机制对照（2026-11 优化轮）

参考实现：`ykcombat/sglang` `slo_config` 分支 `eeac514`，
`python/sglang/srt/multiplex/multiplexing.py`（`event_loop_pdmux`）与 `pdmux_readme.md`。
下表按参考实现的实际代码逐项对照，不按论文描述推断。

| MuxWise 机制（参考代码） | MuxPD 对应 | 状态 |
|---|---|---|
| D 与 P 在两条 stream 上同时提交：D 整批、P 提交 `forward_count` 层 | `decoupled` + `stream_mode=overlap` + `isolated` 通信组；D 为独立 ACL graph | 覆盖（设备重叠程度未用 profiler 验证） |
| 每轮只等 D（`decode_stream.synchronize()`），P 完成用 event `query()` 非阻塞探测并跨 TP allreduce 标志 | worker 在 P 最后一层后随采样同步完成 | 近似：P 最后一片仍在该轮内同步 |
| D 连续运行，不因新 P 到来而停顿（无 CPU overlap 调度，`--disable-overlap-schedule`） | 纯 D 阶段使用原生异步队列（`async_decode_pipeline`）；P 过渡前排空 | 本轮修正：只在等待的 prompt 确实可接纳时才排空（见下） |
| `split_forward_token_budget // extend_num_tokens` 决定每轮 P 层数 | `prefill_layer_token_budget` | 覆盖；本轮允许与异步 D 组合 |
| 一次只有一个 split prefill batch，完成后合入 running batch | 一次一个 P cohort，完成后进入 D | 一致 |
| 新 P 批只在上一批完成后由 `get_new_batch_prefill()` 组成（可包含多个等待请求） | 同；`max_prefill_requests` 限制批内请求数 | 一致 |
| 按 decode batch size 选择 SM 分区（green context / `manual_divisions`） | 无 | 受限：Ascend 未接入 Cube/Vector 配额；当前 runner 开启内部格式，不满足 stream 资源限制前提 |
| 每个 SM 分区录制一套 decode CUDA graph | 单套 D graph | 受限（同上） |
| pynccl 使用当前 stream | 独立 HCCL 通信组（`isolated`） | 等效替代 |
| 无显式 TTFT/TPOT 预测器（artifact 中 SLO 仅体现在配置与分区表） | `adaptive_layers` / `prefill_first` / `decode_budget_guard`（均默认关） | MuxPD 额外启发式，本轮推荐配置不使用 |

## 本轮调度修正

1. **异步 D 队列的排空条件**（`async_scheduler.py`）。原实现只要 `waiting` 或
   `skipped_waiting` 非空就停止填队列。当所有 running 槽位已占满时，这些请求无法被接纳，
   排空只会把 D 变成同步执行，直到有请求结束。这在突发和高负载下整段移除了流水。
   MuxWise 中 D 不会因为等待队列而停顿。现在仅当等待请求可以立即开始 P（有空槽、未暂停）时才排空。
   队列中的 D 输出若结束请求，其释放的槽位在退休后的下一次 `schedule_batch` 中生效。
2. **token-work 层配额与异步 D 可组合**（`config.py`）。对应 MuxWise 的
   `split_forward_token_budget`，该配额只依赖 cohort token 数，不依赖延迟样本。
   实测在 fenced 模式下把 9.7K P 切为 6 层一片时，每轮 D 间隔约 105 ms，优于不切片，
   但差于 isolated overlap q4（最大间隔约 85–115 ms），因此未进入推荐配置。

## 推荐配置（TP2 已测）

`async_decode_pipeline=true`、`stream_mode=overlap`、`overlap_communication=isolated`、
`decode_graph=true`、`layers_per_slice=4`，其余开关关闭。启动脚本见
[`serve-qwen35-muxwise.sh`](../scripts/serve-qwen35-muxwise.sh)。性能结果、代价与适用场景见
评测报告；质量门（#23）尚未在该配置下重做。

## 仍与 MuxWise 不一致的点

- P 的最后一片和采样仍在该轮同步完成；MuxWise 用 event `query()` 延后探测。
- P 过渡前仍排空异步 D 队列；P 存在期间 D 为同步步进。
- 无空间算力分区，无按 decode batch size 的分区选择。
- P 引入的 TP 归约布局与原生混合批不同，输出不与原生逐 token 一致（#23）。
