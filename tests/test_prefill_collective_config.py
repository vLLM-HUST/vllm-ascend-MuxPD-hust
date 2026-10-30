"""Prefix opt-in rejects unsupported engine/deployment combinations on CPU."""

import json
from types import SimpleNamespace as NS

import pytest
from test_policy import engine_config

from vllm_muxpd import deployment
from vllm_muxpd.config import MuxPDConfig, validate_engine_config
from vllm_muxpd.prefill_collective_layout import POLICY_KEY


def policy():
    return deployment.configuration(
        experimental_prefill_prefix_layout=True, deterministic_collectives=True
    )["launch_options"]["batch_admission_policy_config"]


def config():
    cfg = engine_config()
    cfg.scheduler_config.batch_admission_policy_config = policy()
    cfg.scheduler_config.scheduler_cls = (
        "vllm_muxpd.decoupled_scheduler.DecoupledMuxPDScheduler"
    )
    cfg.scheduler_config.max_num_seqs = 8
    cfg.model_config.dtype = "torch.bfloat16"
    cfg.model_config.enforce_eager = False
    cfg.model_config.hf_text_config = NS(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        vocab_size=248320,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    cfg.parallel_config.tensor_parallel_size = 4
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.worker_cls = (
        "vllm_muxpd.decoupled_worker.DecoupledMuxPDNPUWorker"
    )
    cfg.compilation_config = NS(
        mode=NS(name="VLLM_COMPILE"),
        cudagraph_mode=NS(name="FULL_DECODE_ONLY"),
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
    )
    return cfg


def test_default_off_preserves_launch_recipe_and_explicit_off_is_valid():
    assert MuxPDConfig().experimental_prefill_collective_layout == "off"
    assert (
        POLICY_KEY
        not in deployment.configuration()["launch_options"][
            "batch_admission_policy_config"
        ]
    )
    assert deployment.configuration() == deployment.configuration(
        experimental_prefill_prefix_layout=False
    )
    assert deployment.host_command("/model") == deployment.host_command(
        "/model", experimental_prefill_prefix_layout=False
    )
    cfg = engine_config()
    cfg.scheduler_config.batch_admission_policy_config = {POLICY_KEY: "off"}
    validate_engine_config(cfg)
    validate_engine_config(config())


@pytest.mark.parametrize(
    "value", [True, False, None, 0, 1, "auto", "suffix", "copy_only"]
)
def test_layout_mode_requires_literal_supported_string(value):
    with pytest.raises(ValueError, match=POLICY_KEY):
        MuxPDConfig.from_mapping({POLICY_KEY: value})
    cfg = engine_config()
    cfg.scheduler_config.batch_admission_policy_config = {POLICY_KEY: value}
    with pytest.raises(ValueError, match=POLICY_KEY):
        validate_engine_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("execution", "serial"),
        ("stream_mode", "fenced"),
        ("overlap_communication", "shared"),
        ("decode_graph", False),
        ("layers_per_slice", 8),
        ("adaptive_layers", True),
        ("prefill_first", True),
        ("decode_budget_guard", True),
        ("prefill_layer_token_budget", 1024),
        ("compile_full_prefill", True),
        ("compile_prefill_groups", True),
        ("async_decode_pipeline", True),
        ("require_hccl_deterministic", False),
    ],
)
def test_incompatible_policy_combinations_fail_closed(key, value):
    with pytest.raises(ValueError, match="prefix layout"):
        MuxPDConfig.from_mapping({**policy(), key: value})


@pytest.mark.parametrize(
    "owner,name,value",
    [
        ("scheduler_config", "async_scheduling", True),
        (
            "scheduler_config",
            "scheduler_cls",
            "vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler",
        ),
        (
            "parallel_config",
            "worker_cls",
            "vllm_muxpd.atomic_worker.AtomicDecoupledMuxPDNPUWorker",
        ),
        (
            "parallel_config",
            "worker_cls",
            "vllm_muxpd.async_worker.AsyncDecoupledMuxPDNPUWorker",
        ),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", False),
        ("parallel_config", "use_sequence_parallel_moe", True),
        ("parallel_config", "enable_dbo", True),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "mamba_cache_mode", "align"),
        ("model_config", "dtype", "torch.float16"),
        ("model_config", "enforce_eager", True),
        ("compilation_config", "cudagraph_capture_sizes", [1, 2, 4]),
        ("scheduler_config", "enable_chunked_prefill", True),
    ],
)
def test_incompatible_engine_profiles_fail(owner, name, value):
    cfg = config()
    setattr(getattr(cfg, owner), name, value)
    with pytest.raises(ValueError):
        validate_engine_config(cfg)


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_concurrent_batches", 2),
        ("use_v2_model_runner", True),
        ("speculative_config", object()),
        ("quant_config", object()),
        ("lora_config", object()),
        ("kv_transfer_config", object()),
        ("ec_transfer_config", object()),
    ],
)
def test_unsupported_runtime_features_fail(name, value):
    cfg = config()
    setattr(cfg, name, value)
    with pytest.raises(ValueError):
        validate_engine_config(cfg)


def test_model_type_string_alone_does_not_expand_architectural_profile():
    cfg = config()
    cfg.model_config.hf_text_config.num_hidden_layers = 60
    with pytest.raises(ValueError, match="architectural profile"):
        validate_engine_config(cfg)


def test_cli_requires_determinism_and_is_mutually_exclusive():
    with pytest.raises(ValueError, match="deterministic"):
        deployment.configuration(experimental_prefill_prefix_layout=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        deployment.configuration(
            experimental_prefill_prefix_layout=True,
            experimental_prefill_groups=True,
            deterministic_collectives=True,
        )
    with pytest.raises(ValueError, match="TP4"):
        deployment.host_command(
            "/model", tensor_parallel_size=2, experimental_prefill_prefix_layout=True
        )


def test_cli_plan_projects_prefix_and_disables_other_compile_backends(capsys):
    assert (
        deployment.main(
            [
                "plan",
                "--model",
                "/model",
                "--experimental-prefill-prefix-layout",
                "--deterministic-collectives",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (
        result["configuration"]["launch_options"]["batch_admission_policy_config"][
            POLICY_KEY
        ]
        == "prefix"
    )
    assert result["requested_environment"] == {"HCCL_DETERMINISTIC": "true"}
    command = result["host_command"]
    config = json.loads(command[command.index("--additional-config") + 1])
    assert config["ascend_compilation_config"] == {
        "enable_npugraph_ex": False,
        "enable_static_kernel": False,
    }
