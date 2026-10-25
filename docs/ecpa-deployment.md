# ECPA 安装与显式启用

Bundle ID 为 `org.vllm-hust.muxpd`，版本为 `0.2.0a1`。
这是 Qwen3.5-35B-A3B / 910B2 / BF16 / TP2 或 TP4 + EP 的固定软件栈入口。
完整 prompt 保持在 P lane，D lane 使用独立图与通信组，每轮独立提交输出。
默认采用现有 raw-P 四层切片 streaming 配置；不是 group-FX 实验入口。

## 软件边界

| 组件 | 部署入口检查的版本或源码 |
|---|---|
| vllm-hust | `d0f22d2bda562156e4dbf433ce645e1769b4f804` editable checkout |
| vllm-ascend-hust | `03766ac696fde5ab1980d80ca0b8543d3580c989` editable checkout |
| torch / torch-npu | `2.10.0+cpu` / `2.10.0.post4` |
| greenlet | `>=3.1,<4` |
| Extension Manager | 含 scheduler carrier 和 module evidence 的 `0.3.1.dev1` 开发版本 |

`vllm-muxpd doctor` 仅检查软件版本、Git commit 和已跟踪修改，不初始化设备，
也不能替代服务验证。Ascend checkout 仅允许已记录的 `csrc/build_aclnn.sh`
构建修改，其 SHA256 为
`b089d52475c6cb36fffd733bb5be3fbe4158b6845074751268938e98a1d0042c`。
其他 native wheel、源码修改或版本需要另行验证，不能仅凭 admission API `1.1`
认定兼容。容器还需具备原环境的 CANN、ATB、通信和模型配置。

兼容层另有针对 2026-10-25 上游源码的签名、输出字段和 KV 回收契约检查。
该上游依赖 Torch/NPU 2.13；这些 CPU 检查不是当前服务器上的设备验证。
Manifest 的 host 范围和部署入口均保留上述固定软件边界。

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
