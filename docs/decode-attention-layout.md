# 可选解码注意力布局（`decode_attention_layout`）

默认 `tnd`：不安装任何补丁，行为与未设置此项完全相同。其余取值都是显式启用的质量缓解或诊断手段，
不是性能预设。

## 问题

原生 Ascend 解码路径以 TND 布局调用 `npu_fused_infer_attention_score`。在 910B2、Qwen3.5-35B-A3B
TP2 的形状下实测：同一行的输出会随同批其他行的数量和 KV 长度改变（算子切分随批组成变化）；
把解码批切成 2/4/8 行一段仍不能消除，只有每行单独调用才与单请求一致。一次 prefill 中放多个
prompt 也会改变数值。原生 vLLM-Ascend 在并发下同样存在这两点；MuxPD 的 P/D 分离只是改变了
请求落入的批组成，因此两者的并发输出不同。

## 取值

| 取值 | 做法 | 性质 |
|---|---|---|
| `tnd`（默认） | 原生批量 TND | 数值随批组成变化 |
| `bsh` | 纯 D 批改用 BSH（S=1） | 对批大小不变，开销接近 TND；但数值与原生 TND 不同，不等于原生单请求结果 |
| `row` | 每个 D 行一次原生 TND M=1 调用（与原生图路径相同的 mask/sparse/window 参数） | 每行与该请求单独运行时逐位一致；开销很大 |

配合 `max_prefill_requests: 1`（每次 prefill 只含一个 prompt）时，`row` 使并发服务中每个请求的输出
与它单独运行时相同。预设：`scripts/serve-qwen35-muxwise-row.sh`。

## 适用范围（fail-closed）

- 仅 `execution: decoupled`；其他执行模式配置校验直接拒绝。
- 只替换 `DecodeOnly` 状态、单 token 查询的基础 FIA 调用；sinks、滑动窗口、C8 量化、非因果、
  非 decoder、draft 模型以及所有 prefill/chunked prefill 调用一律原样交给原生实现。
- 图模式：同一捕获尺寸内若混有原生与 row 参数则报错；重放元数据不是对应大小的解码批时报错。
- 已在设备上验证的范围：910B2、Qwen3.5-35B-A3B BF16、**TP2+EP**、max-num-seqs 16、图尺寸
  [1,2,4,8,16]、`scripts/serve-qwen35-muxwise.sh` 策略。TP4/TP8、其他模型、其他 CANN/torch_npu
  版本未验证；不同版本的算子切分可能不同，启用前应重新验证。
- 其余算子（GDN、MoE、HCCL）在上述配置下经验证未引入批依赖；其他配置未证实。

## 代价

图模式下每个全注意力层、每一行都需要一次 host 侧任务更新（模型 40 层中 10 层为全注意力），
另外每次 prefill 只处理一个 prompt。实测中输出吞吐、TPOT、TTFT 均显著劣于默认 `tnd`，并发越高差距
越大；两处代价各自占比尚未拆分。具体数值只发布在 benchmark 仓库报告中（组织数据策略）。

适合：需要并发输出与单请求输出一致的评测、复现和问题定位。不适合：以吞吐或延迟为目标的生产服务。
它保证的是“与单请求一致”，不是“答案更正确”。
