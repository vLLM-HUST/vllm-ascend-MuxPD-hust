# ECPA 安装与显式启用

Bundle ID 为 `org.vllm-hust.muxpd`，版本为 `0.2.0a1`。
这是 Qwen3.5-35B-A3B / 910B2 / BF16 / TP2 或 TP4 + EP 的固定软件栈入口。
完整 prompt 保持在 P lane，D lane 使用独立图与通信组，每轮独立提交输出。
默认采用现有 raw-P 四层切片 streaming 配置。默认关闭的 group-FX 选项及固定
适配约束见 [实验入口](experimental-prefill-group-fx.md)，不会由安装自动启用。

## 软件边界

| 组件 | 部署入口检查的版本或源码 |
|---|---|
| vllm-hust | `d0f22d2bda562156e4dbf433ce645e1769b4f804` editable checkout |
| vllm-ascend-hust | `03766ac696fde5ab1980d80ca0b8543d3580c989` editable checkout |
| torch / torch-npu | `2.10.0+cpu` / `2.10.0.post4` |
| greenlet | `>=3.1,<4` |
| Extension Manager | 含 scheduler carrier 和 module evidence 的 `0.3.1.dev1` 开发版本 |

`vllm-muxpd doctor` 检查软件版本、实际模块来源、Git commit 和已跟踪修改，不初始化设备，
也不能替代服务验证。Ascend checkout 仅允许已记录的 `csrc/build_aclnn.sh`
构建修改，其 SHA256 为
`b089d52475c6cb36fffd733bb5be3fbe4158b6845074751268938e98a1d0042c`。
其他 native wheel、源码修改或版本需要另行验证，不能仅凭 admission API `1.1`
认定兼容。容器还需具备原环境的 CANN、ATB、通信和模型配置。

兼容层另有针对 2026-10-25 上游源码的签名、输出字段和 KV 回收契约检查。
该上游依赖 Torch/NPU 2.13；这些 CPU 检查不是当前服务器上的设备验证。
Manifest 限制 host 版本；部署命令和 Manager 选择的 scheduler/worker 入口在
分配原生资源前另行核对上述源码与依赖边界，拒绝 PYTHONPATH 指向其他原生源码。

## 安装与配置

使用 [Releases](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/releases)
中版本对应的 wheel 和校验和，以及发布说明指定的 Manager 开发 wheel/源码 commit。
先安装指定 Manager，再在现有 Ascend Python 环境安装本插件，避免重新解析底层依赖：

```bash
python -m pip install --no-deps /path/to/vllm_ascend_muxpd-0.2.0a1-py3-none-any.whl
vllm-muxpd doctor
vllm-hust-ext extension inspect org.vllm-hust.muxpd
vllm-hust-ext extension validate org.vllm-hust.muxpd
```

若环境使用 uv，可将安装命令换为 `uv pip install --python /path/to/python --no-deps ...`。
Manager 0.3.0 不支持这三个入口的完整投影，不能用于本部署路径。
`[ecpa]` extra 声明依赖范围；发布说明中的精确开发构建是可复现安装依据，
不表示 Manager 已正式发布稳定版 0.3.1。

使用独立配置文件可避免和其他服务的启用状态互相影响：

```bash
export VLLM_HUST_EXT_CONFIG="$PWD/muxpd-state/config.json"
vllm-muxpd configuration --output muxpd-configuration.json
vllm-hust-ext extension configure org.vllm-hust.muxpd --file muxpd-configuration.json
vllm-hust-ext extension enable org.vllm-hust.muxpd
vllm-hust-ext extension plan org.vllm-hust.muxpd
vllm-hust-ext extension render org.vllm-hust.muxpd
```

安装、inspect、validate、configure、enable 和 plan 均不运行模型。
启用 Bundle 只保存意图；实际运行由下一次 Manager launch 决定。
三组件独占 scheduler、worker 和 admission policy 资源；与其他占用相同入口的
Bundle 同时启用会报冲突。不要在命令行另加 scheduler、worker 或 policy 选择器。

## 启动与状态

在原 Ascend 环境中显式选择当前容器的逻辑设备编号。下面的 `0,1,2,3`
仅是四张可见卡的示例，不能用旧容器的物理卡号代替逻辑编号：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
vllm-muxpd serve --model /path/to/Qwen3.5-35B-A3B --tensor-parallel-size 4 --dry-run
vllm-muxpd serve --model /path/to/Qwen3.5-35B-A3B --tensor-parallel-size 4
```

`--dry-run` 展示 Manager 最终投影，不启动服务。默认 HTTP 地址为
`127.0.0.1:8002`。模型路径必须显式提供。服务关闭 chunked prefill、async
scheduling、prefix caching，使用 Mamba `none`、完整 16K token budget 和
1/2/4/8 decode 图形状。并发上限为 8，不支持随意追加模型参数绕过此配置。

`vllm-hust-ext extension status org.vllm-hust.muxpd` 在实际成功输出后才可出现
`runtime_effective`。Manager 将 manifest SHA、组件 class、launch ID 和
进程 PID/出生身份绑定；scheduler 与 admission 在验证并提交输出后报告，worker
在成功采样后报告。三个组件都需有回执，任一已观察进程退出会撤销状态。
该状态不是所有 TP rank 的独立设备完成证明，也不证明文本正确性或性能收益。

需要调度生命周期 trace 时，生成配置和启动命令必须使用相同的路径：

```bash
vllm-muxpd configuration --trace-path '/tmp/muxpd-trace-{pid}.jsonl' --output muxpd-traced.json
vllm-hust-ext extension configure org.vllm-hust.muxpd --file muxpd-traced.json
vllm-muxpd serve --model /path/to/Qwen3.5-35B-A3B --trace-path '/tmp/muxpd-trace-{pid}.jsonl'
```

trace 默认关闭；开启后会增加记录开销，不应把它与无 trace 的性能窗口混用。
测量回执应存入本地临时目录，发布到中央 benchmark 仓库，而不是本代码仓库。

## 实验性确定性通信配置

`--deterministic-collectives` 显式选择 `HCCL_DETERMINISTIC=true`。
这项配置仍属实验性。source98 已有单独的
[普通 worker 与 Manager 服务桥接](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025/device-evidence-v3)；
配对质量、性能及 profile 的完整处置见
[当前报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ebddb585390c333ac1f8c6ea40cfeead3c4321af/reports/muxpd-followup-20261025/quality-reference-v5)。该桥接实际运行源码为
`98eac12dc2595a8742a81b58e96b899fed905194`。本次文档更新未改动其运行时代码，
文档 HEAD 不是新的实测源码或新 wheel 验收。
它不表示任意输入、任意 batch 或所有算子都具有确定性，也不自动启用
`VLLM_BATCH_INVARIANT`、`strict` 模式或 Torch 全局确定性设置。
不传此选项时，插件保留现有环境和默认运行方式。

配置、查看计划和启动必须使用相同选项：

```bash
vllm-muxpd configuration --deterministic-collectives --output muxpd-deterministic.json
vllm-hust-ext extension configure org.vllm-hust.muxpd --file muxpd-deterministic.json
vllm-muxpd plan --model /path/to/Qwen3.5-35B-A3B --deterministic-collectives
vllm-muxpd serve --model /path/to/Qwen3.5-35B-A3B --deterministic-collectives
```

同时选择 group-FX 时，三个命令都使用相同组合：

```bash
vllm-muxpd configuration --experimental-prefill-groups --deterministic-collectives \
  --output muxpd-group-deterministic.json
vllm-hust-ext extension configure org.vllm-hust.muxpd --file muxpd-group-deterministic.json
vllm-muxpd plan --model /path/to/Qwen3.5-35B-A3B --tensor-parallel-size 4 \
  --experimental-prefill-groups --deterministic-collectives
vllm-muxpd serve --model /path/to/Qwen3.5-35B-A3B --tensor-parallel-size 4 \
  --experimental-prefill-groups --deterministic-collectives --dry-run
```

该示例沿用前文显式 enable 和设备选择步骤；确认投影后，实际启动时去掉 `--dry-run`。
group flag 选择所需 fusion 后端；deterministic flag 本身不改变普通 raw 后端默认值。
因此普通 Manager raw/group 激活桥接不是仅改变 P 分组的后端匹配比较，
也不提供 token 等价证据。配对数值和性能比较的配置须以 canonical 报告为准。

保存配置在 `batch_admission_policy_config` 中记录严格布尔值
`require_hccl_deterministic=true`；Manager 将其连同三个组件选择器投影到启动命令。
`serve` 在调用 Manager 前设置通信环境，因此 TP 子进程及其中创建的独立 P
通信组都继承它。若调用方已经显式设置其他值（包括 `strict`），或启用了
`VLLM_BATCH_INVARIANT`，启动会报冲突，不会默默覆盖。
worker 在原生设备和通信初始化前检查这个要求；缺失或不匹配会直接失败。

当前 Manager 不支持条件环境声明，只允许覆盖 manifest 已声明的静态环境项。
本插件没有添加默认 HCCL 环境项，以免改变普通启动或覆盖调用方已有设置。
因此 `vllm-muxpd plan` 单独展示 `requested_environment`；Manager 的 `--dry-run`
只展示激活增量，不能用于证明继承的 HCCL 值。绕过 `vllm-muxpd serve` 使用
通用 `vllm-hust-ext run -- ...` 时，调用方须在新进程启动前显式设置
`HCCL_DETERMINISTIC=true`，并保存匹配的 policy 要求。

显式启用后，每个 TP rank 在原生和独立通信组初始化完成时输出一条
`MuxPD collective profile startup` JSON 回执，记录 rank/PID、请求值与环境读回、
Torch 状态、Manager launch/plan ID（如存在）和独立 P 组的名称/成员。
该回执不改变采样、执行图或算子，也不升级 Manager 的 `runtime_effective`
语义；环境读回不是设备实际算法、文本质量或性能收益的证明。
source98 桥接的退出状态及 native abort/清理警告按原样保存在中央证据中；
设备清空和外层监督器未升级清理不表示所有内部组件都执行了无信号的优雅 drain。

比较性能时，对照组和插件组必须使用相同通信设置。带完整 logits 观察器的
数值诊断不用于吞吐或延迟测量。移除该配置时，重新保存不带选项的普通配置，
并在新服务进程中启动；运行中的进程不会切换通信模式。
本轮性能和生命周期数据见 [canonical v4](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ef849ef49577c491fee68317ffc96d52796cd379/reports/muxpd-followup-20261025/performance-lifecycle-v4)。
实用 native 默认后端参考与实际混合 profile 已完成，结果及全部失败见上方当前报告。
原严格无截断门槛与补充答案不退化门槛均未通过。f7 group-P 保留了固定题集上
raw-P 的完整输出，也保留了相对原生控制的新增错误；整体正确数提高不抵消这些错误。
该配置改善特定长 prefill 场景的最坏出词停顿，同时损失吞吐并增加首字延迟与常规尾部间隔。
服务桥接、原始输出一致或性能测量均不能替代答案质量验收，已知限制由
[#23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23) 继续跟踪。

## 停止、停用与回滚

先停止接收新请求，等待现有请求完成；前台服务用 Ctrl-C 停止 Manager 监督的
进程树。取消未完成 P 时，KV 必须等 worker 确认不再访问后才释放。
关闭失败保留待释放资源以便重试，不把失败伪装成关闭成功。

```bash
vllm-hust-ext extension disable org.vllm-hust.muxpd
vllm-hust-ext extension status org.vllm-hust.muxpd
```

disable 不修改仍在运行的服务。停止旧服务后，用原生 vLLM 启动命令启动一个新
进程；不要继续使用要求 Bundle 已启用的 `vllm-muxpd serve`。可用
`vllm-hust-ext run -- <原生启动命令>` 验证停用后的下一次启动无 MuxPD 选择器。
原生服务按自身配置恢复 chunked prefill / mixed batch。
插件不修改基线源码，卸载前应先停止并停用。Ascend 图全局状态不支持同一 worker
进程内销毁后重建；服务重启必须创建新的 worker 进程。

本次安装包验收与中央证据链接跟踪在
[Issue #21](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/21)。
现有特定交互场景的 Goodput 收益和代价见
[TP4 Goodput 报告](performance-tp4-goodput.md)，不能外推为通用吞吐加速。
