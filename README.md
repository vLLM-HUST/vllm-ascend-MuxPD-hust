# MuxPD for HUST vLLM Ascend

将 MuxWise 的 P/D 解耦机制适配到现有 HUST vLLM，使用原生 KV 分配、抢占、
取消、采样与 OpenAI API。安装包本身不修改 vLLM；通过显式 scheduler/worker
类启用，无需改动基线源码。

**开发状态：集成与实机验证中。** 当前实现包含完整 prefill、独立 P/D 模型
batch；串行策略具有在线延迟估计和等待阈值。实验 worker 使用独立输入/metadata 与共享
权重/KV，通过 transformer 层边界切换。完整层 event fence 是正确性起点，
不代表设备 kernel 重叠或论文中的空间资源分区。`decoupled` 模式将完整 prefill
保留在 worker 中，允许多轮 decode 在其完成前独立提交结果；取消时等待 worker
确认停止访问 KV 后再释放缓存。

| 模式 | P/D 执行与结果提交 |
|---|---|
| `serial` | 每轮只有完整 P 或 D，作为参考实现 |
| `layered` | 独立 P/D 模型 batch，按层切换，同轮提交结果 |
| `decoupled` | 完整 P 跨轮保存，D 每轮独立完成，延迟提交 P 结果 |

`stream_mode=fenced` 对两流建立完整事件依赖；实验性 `overlap` 保留共享初始化、
同组 collective 和采样完成所需依赖，取消普通层边界的跨流等待，并隔离两条
lane 的临时 workspace。设备并行程度必须由
NPU profiler 验证，不能由 Python 协程或调度 trace 推断。

## 固定运行基线

| 组件 | 验证目标 |
|---|---|
| vllm-hust | `d0f22d2bda562156e4dbf433ce645e1769b4f804` |
| vllm-ascend-hust | `03766ac696fde5ab1980d80ca0b8543d3580c989` |
| 模型 / 设备 | Qwen3.5-35B-A3B / Ascend 910B2，BF16，TP2+EP |
| 执行 | V1 NPU runner，eager，同步，纯文本 |
| 底层依赖 | 复用已运行环境，不重新解析 torch / torch-npu / CANN |

参考正式 [ASPLOS 2026 论文 v3](https://arxiv.org/html/2504.14489v3) 与作者指定
[`slo_config` 分支](https://github.com/ykcombat/sglang/tree/eeac5148f456428a05823c330e1d40afb0ef2438)。
实现与参考差异见 [设计](docs/design.md) 和 [来源](docs/references.md)。

## 安装与启用

以下命令针对服务器现有 `/root/workspace/runtime/env.sh`；其他路径可设置
`MUXPD_WORKSPACE`。使用 `--no-deps` 保持已验证的 Ascend 软件栈。

```bash
source /root/workspace/runtime/env.sh
uv pip install --python /root/workspace/.venv/bin/python --no-deps -e .

# 串行参考实现：每批只做 P 或 D，不切分 prompt。
ASCEND_RT_VISIBLE_DEVICES=2,3 scripts/serve-qwen35.sh \
  --batch-admission-policy-config '{"trace_path":"local-results/serial-{pid}.jsonl"}'
```

默认监听 `127.0.0.1:8002`。示例使用本机逻辑 NPU `2,3`（物理 `6,7`），
原服务使用逻辑 `0,1`；启动前需停止占用所选设备的旧进程。脚本显式关闭 chunked prefill、
async scheduling 和 prefix caching，并将 token budget 设置为完整 16K 上下文。
**Qwen3.5 的 Mamba `align` 模式本身会拆 prefill**，因此使用 `none`。
这会改变缓存条件，性能比较必须保留相同缓存设置的对照组。

```bash
# 实验分层执行：两套 runner 输入与元数据，模型权重/KV 共用。
uv pip install --python /root/workspace/.venv/bin/python 'greenlet>=3.1,<4'
ASCEND_RT_VISIBLE_DEVICES=2,3 VLLM_USE_V2_MODEL_RUNNER=0 scripts/serve-qwen35.sh \
  --scheduler-cls vllm_muxpd.multiplex_scheduler.LayeredMuxPDScheduler \
  --worker-cls vllm_muxpd.worker.MuxPDNPUWorker \
  --batch-admission-policy-config \
  '{"execution":"layered","layers_per_slice":1,"trace_path":"local-results/layered-{pid}.jsonl"}'

# 跨轮执行：完整 prompt 不切 token，decode 在 P 完成前持续推进。
ASCEND_RT_VISIBLE_DEVICES=2,3 VLLM_USE_V2_MODEL_RUNNER=0 scripts/serve-qwen35.sh \
  --scheduler-cls vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker \
  --batch-admission-policy-config \
  '{"execution":"decoupled","stream_mode":"fenced","layers_per_slice":8,"trace_path":"local-results/decoupled-{pid}.jsonl"}'
```

显式 `scheduler_cls` 是插件入口；不依赖 `vllm.general_plugins` 全局 monkey patch。
不要仅将 `MuxPDPolicy` 加到原生 scheduler：必须和本仓库 scheduler 配对。
不满足配置契约会报错，不能静默退回 mixed batch。

## 验证与性能记录

```bash
source /root/workspace/runtime/env.sh
/root/workspace/.venv/bin/python -m pytest tests -q
/root/workspace/.venv/bin/python -m ruff check .

/root/workspace/.venv/bin/python scripts/smoke_serving.py \
  --base-url http://127.0.0.1:8002 --output local-results/smoke.json
/root/workspace/.venv/bin/python scripts/benchmark_serving.py \
  --label muxpd-serial --output local-results/benchmark.json
/root/workspace/.venv/bin/python scripts/check_trace.py local-results/serial-*.jsonl
```

CPU 测试覆盖策略与协程；原生 Scheduler 测试覆盖 KV 容量、抢占重算、取消、
完整 prompt 和调度入口。实机脚本覆盖流式/并发生成、2305-token prompt 和取消后
继续服务。benchmark 保存合成负载的原始 SSE 时间、TTFT、平均 TPOT、吞吐和
给定 SLO 下的 goodput；SSE 事件间隔不会被冒充为单 token 延迟。

## 配置和支持边界

`--batch-admission-policy-config` 接收严格 JSON。主要字段：

| 字段 | 默认值 | 作用 |
|---|---:|---|
| `max_decode_steps` | 8 | 串行模式下连续 D 的上限 |
| `max_prefill_wait_ms` | 100 | 串行 P 等待阈值，仍受 KV 可用容量约束 |
| `ttft_slo_ms` / `tpot_slo_ms` | 1000 / 100 | 策略目标，不是服务保证 |
| `max_prefill_requests` | 8 | 单批 P 请求数上限 |
| `max_prefill_batch_tokens` | 0 | 软 batching 上限；0 使用完整预算 |
| `ewma_alpha` | 0.2 | 串行模式的调度至完成延迟在线估计权重 |
| `layers_per_slice` | 1 | 实验 worker 每次提交的完整 transformer 层数 |
| `execution` | serial | 必须与 scheduler/worker 类匹配 |
| `stream_mode` | fenced | `overlap` 是需独立验证的双流执行后端 |
| `adaptive_layers` | false | decoupled 按本机耗时和 TTFT 预算放大层配额 |
| `trace_path` | null | 可选 JSONL 文件；支持 `{pid}`，不记录 prompt 内容 |

`adaptive_layers=true` 只在获得完整 P-only 和 D-only 样本后调整配额；未校准时
保留固定配额。它优先减少 P 的首 token 等待，可能增加个别 D 的响应间隔，不保证
TPOT 达标，也不进行硬件资源分区。详细算法见 [设计](docs/design.md)。

初始延迟系数是可配置启动值，不能作为 MuxWise 论文 contention predictor 的复现。
PP/DP/CP、speculative decoding、KV/EC transfer、streaming-input continuation
不在当前范围。实验 worker 进一步限制 Qwen3.5、A2、eager、BF16、无 LoRA、
无结构化输出/logprobs、无 FlashComm/共享专家侧流/EPLB。decoupled 拒绝 MFU
指标选项，避免拆分结果退休后向原生统计模块传入不完整的 FLOPs 元数据。
`decoupled` 当前只允许一个未完成的 P cohort，保留其 KV 与请求槽位；不支持
存在未完成 P 时的 `keep` 暂停，需先 drain 或 abort。完整 prompt 过长或 KV
不足仍会造成等待，层调度不能消除内存容量限制。

回滚：停止本插件服务，使用原 `/root/workspace/runtime/serve-qwen35.sh`。
基线源码与原启动脚本不由安装过程修改。
