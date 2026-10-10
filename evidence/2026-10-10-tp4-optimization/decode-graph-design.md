# 只捕获 Decode 的 Ascend ACL graph 接入设计

记录日期：2026-10-10。状态：只读设计，未运行 graph capture、NPU 请求或重 CPU 测试。基线插件 commit `c9969f8ea6483b839a17765a8b841512be90a7b9`；HUST Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`；HUST vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`。本地文件行号对应调查时的上述源码；同分支其他 agent 的未提交实现应另外验证。

## 范围及证据

最小目标：新增显式 opt-in `decode_graph`。仅允许 `execution=decoupled`、`stream_mode=fenced`；D 使用原生 FULL_DECODE_ONLY capture/replay/metadata 更新，P 仍执行完整 prompt 并保留跨迭代层边界。保持一份权重和 KV tensor；D 独占 graph entries、静态输入、metadata builders、workspace、update stream。

原论文/代码要求 D 整网 replay，见 [`cuda_graph_runner.py:875`](https://github.com/ykcombat/sglang/blob/eeac5148f456428a05823c330e1d40afb0ef2438/python/sglang/srt/model_executor/cuda_graph_runner.py#L875)。本机支持入口包括：

| 本地源码 | 已核实行为 |
| --- | --- |
| `/root/workspace/vllm-ascend-hust/vllm_ascend/ops/gdn_attn_builder.py:193` | GDN metadata builder 声明 `UNIFORM_BATCH`。 |
| `/root/workspace/vllm-hust/vllm/v1/attention/backends/gdn_attn.py:513` | GDN full graph capture 明确限 decode；捕获大小必须覆盖请求/实际 token 数。 |
| `/root/workspace/vllm-ascend-hust/vllm_ascend/ops/gdn_attn_builder.py:704` | Decode 的 state indices/query start loc 拷入 builder 持有的静态 buffer。 |
| `/root/workspace/vllm-ascend-hust/vllm_ascend/worker/model_runner_v1.py:3620` | 原生 load_model 创建 ACLGraphWrapper 和 update_stream。 |
| `/root/workspace/vllm-ascend-hust/vllm_ascend/worker/model_runner_v1.py:4777` | 原生 capture_model 适配后复用 GPUModelRunner.capture_model。 |
| `/root/workspace/vllm-ascend-hust/vllm_ascend/worker/model_runner_v1.py:2653` | FULL replay 后调用原生 update_full_graph_params；ENPU 调用顺序不同。 |

本机 `/usr/local/Ascend/driver/version.info` 只读显示 `26.0.rc1`，已记录 CANN 9.1.0、torch-npu 2.10.0.post4。没有发现明确的“旧 HDK <25.5.1”阻断，但这不是 graph 可运行证明。首次有限 capture 仍可能暴露 stream 资源、kernel 或混合 attention 问题。

## 必须解决的隔离点

1. **独立 wrapper。** 现 `src/vllm_muxpd/worker.py:79` 的 attach_model 直接引用 owner.model，会连 graph entries 一并共用。应提取 raw model，让 P 持有 raw；D 新建 ACLGraphWrapper(raw, config, runtime_mode=FULL) 及 update_stream。不能重复加载参数，也不能沿用 owner 的图输入地址。
2. **P 永远 eager。** 仅 unwrap 不够：Qwen3.5 内层 `@support_torch_compile` 仍可能编译。P `_determine_batch_execution_and_padding` 强制 `force_eager=True`，`_model_forward` 强制已有 forward context 的 runtime 为 NONE、skip_compiled=True、capturing=False。`vllm/compilation/decorators.py:509` 提供 skip_compiled 分支。一个 token 的新 P 也必须执行这条路径；原生 `model_runner_v1.py:2783` 在无 speculative 时可能按 shape 将其视为 uniform decode。
3. **GraphParams 只归 D。** `vllm_ascend/compilation/acl_graph.py:307` 为进程全局；`:310` 第二次 set 必抛错。owner `use_aclgraph=False` 可抑制其 `model_runner_v1.py:4772` 注册，而 D 正常注册。不要就地把共享 model_config/compilation_config 改成 NONE：D 和模型 compile decorator 仍需要原配置。P runtime NONE、capturing=False 时不会写 graph attention 参数。`ascend_forward_context.py:406` 的 _EXTRA_CTX 是 forward-context 代理，随既有 context 切换。
4. **仅 D capture。** `worker/worker.py:684` 当前 warmup/capture 调用 owner；插件需定向 D，保留原生 ATB warmup、CPU binding、seed reset、内存统计尾部行为。D attach 不经原生 load_model，必须补上自己的 update_stream。
5. **固定捕获 stream。** `model_runner_v1.py:222` 的 graph_capture 在 `:236` 另建 NPU stream，外层 `with decode_stream` 不足。另安装包 `/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/npu/graphs.py:847–856` 中，`torch.npu.graph(stream=None)` 使用类级 default_capture_stream，而 ACLGraphWrapper:187 没有显式传 stream。因此 D capture 作用域需同时绑定 native context factory 和 torch.npu.graph.default_capture_stream，使用已有 decode stream，完成或异常后恢复两层及 GPU module 的 factory。复用 native capture 主体及 input prep 同步协议，不复制大段 runner 实现。补丁只允许启动阶段、单 host thread、无在途请求。
6. **固定 workspace。** P 使用 native profile 已预热的 manager；D 新建独立 manager。fenced lane state/backend 也需保存和恢复 manager。D warmup/capture/replay/sampler 使用同一个 D manager。`GPUModelRunner.capture_model:6703` 会 lock_workspace；若锁到 P/default manager，完整 prompt 可能遇到按 decode 尺寸锁死的 workspace。
7. **保留原生元数据更新和同步。** `ACLGraphWrapper:263` 当前在 FULL replay 前同步当前 stream；第一版不移除。GraphParams 的 events/handles/workspaces 属于 D，attention update stream 仍按原生 external-event 协议工作。动态 seq_len/block_table/GDN state 更新全部复用 native builder。
8. **LayerInterleaver 不伪造 D 层数。** 捕获/预热时 `_interleaver=None`；回放时没有 Python layer hooks，D greenlet 返回后 `done=True` 即完成。严格层进度 receipt 仅验证 P，继续使用真实 P checkpoints。

## 生命周期建议

1. 校验 opt-in 和 FULL_DECODE_ONLY 配置；初始化原有 worker/streams。
2. 创建 D 独立 workspace；为 P 安装局部 eager 防护，保证 owner 不注册 GraphParams。
3. 加载一次 raw model；P 持有 raw，D 独立 wrapper。保留独立 input batches/builders。
4. 初始化 P KV，再让 D 复用 KV 并注册唯一 GraphParams；禁止已有另一个主图/草稿图 owner。
5. 在 D stream + D workspace 下执行 native warmup/capture；确保 capture descriptors 全为 FULL 且 uniform，只覆盖受限 decode batch sizes。
6. 启动后 P 永远 NONE；D 按原生 dispatcher 选择已捕获图。未覆盖 batch 可执行明确计数的 eager D fallback，不得热路径 lazy capture。
7. 记录 graph capture 数量、replay/fallback 数量、真实 capture stream 和 D 静态 buffer 地址。shutdown 前同步，仍由原 owner 释放共享模型/KV。

## 首版 fail-closed 限制

- 仅 decoupled + fenced；暂拒绝 overlap，即便实验 isolated communicator 已实现也须另行验证图捕获时的组绑定。
- 只接受 FULL_DECODE_ONLY；拒绝 FULL mixed、PIECEWISE、speculative、LoRA、权重热更新、sleep、异步调度、ubatching/DBO、PP/DP/CP、encoder、量化等现有未支持路径。
- 第一版拒绝 ENPU、xlite、自定义模型 offload 等改变 capture/更新/拷贝顺序的路径。
- P graph 或 GraphParams 双 owner、D/P 静态输入 alias、错误 stream/manager、capture 后意外新图均直接失败。
- capture 部分失败后终止启动并重启干净进程，不盲目清空全局 GraphParams 后静默 fallback。
- 当前 shared-group CollectiveGates 是 Python collective wrapper；graph replay 不再经过它，捕获依赖不能自动变成后续 P 依赖。因此不能仅删除 eager guard后开放 overlap。

## 验证要求（尚未运行）

CPU 验证：选项组合；P force-eager及一 token P；D-only wrapper/GraphParams 注册；context factory异常后恢复；workspace manager切换及恢复；没有 layer hook 的 D 完成协议；fallback计数；拒绝 graph+overlap。

设备验证窗口由主 agent 协调：小范围 capture，核对 stream/manager/静态地址；D batch 从 1 到上限变化及 padding；长 P 跨迭代期间连续 D replay；P 完成/取消/重用KV后greedy结果；同 TP4、同 graph 配置原生对照；独立纯 D 和混合负载重复测量。验证前不能声明已获得 graph 加速或支持本机完整 workload。
