import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.probe_fixed_token_numerics import digest, plan, run, runtime_profile


def test_fixed_plan_has_fully_drained_warm_control_phases():
    value = plan()
    assert value == plan()
    assert value["requests"] == sum(len(p["prompts"]) for p in value["phases"]) == 15
    assert [(p["name"], p["warm_repeat"]) for p in value["phases"]] == [
        ("single-p128", 0),
        ("single-p128", 1),
        ("single-p128", 2),
        ("four-p128", 0),
        ("four-p128", 1),
        ("four-p128", 2),
    ]
    assert all(len(tokens) == 128 for p in value["phases"] for tokens in p["prompts"])
    assert all(p["max_tokens"] == 8 for p in value["phases"])
    body = {k: v for k, v in value.items() if k != "plan_sha256"}
    assert digest(body) == value["plan_sha256"]
    changed = copy.deepcopy(body)
    changed["phases"][0]["prompts"][0][0] += 1
    assert digest(changed) != value["plan_sha256"]


def install_fake_engine(monkeypatch, *, never_finishes=False, per_token=False):
    class InprocClient:
        def shutdown(self):
            self.stopped = True

    class Engine:
        def __init__(self):
            self.engine_core = InprocClient()
            self.engine_core.engine_core = SimpleNamespace(
                scheduler=SimpleNamespace(requests={})
            )
            self.output_processor = SimpleNamespace(request_states={})
            self.pending = []

        def has_unfinished_requests(self):
            return bool(self.pending)

        def step(self):
            if never_finishes:
                return []
            outputs = []
            pending = []
            for internal, external, tokens in self.pending:
                request = self.engine_core.engine_core.scheduler.requests[internal]
                request.num_output_tokens += 1 if per_token else 8
                if request.num_output_tokens < 8:
                    pending.append((internal, external, tokens))
                    continue
                outputs.append(
                    SimpleNamespace(
                        request_id=external,
                        finished=True,
                        prompt_token_ids=tokens,
                        outputs=[
                            SimpleNamespace(
                                token_ids=list(range(8)),
                                finish_reason="length",
                                stop_reason=None,
                            )
                        ],
                    )
                )
                del self.output_processor.request_states[internal]
                del self.engine_core.engine_core.scheduler.requests[internal]
            self.pending = pending
            return outputs

    class LLM:
        def __init__(self, **kwargs):
            self.llm_engine = Engine()
            self.count = 0

        def enqueue(self, inputs, params, use_tqdm):
            ids = []
            for entry in inputs:
                external = str(self.count)
                internal = external + "-random-native-suffix"
                self.count += 1
                self.llm_engine.output_processor.request_states[internal] = (
                    SimpleNamespace(external_req_id=external)
                )
                self.llm_engine.pending.append(
                    (internal, external, entry["prompt_token_ids"])
                )
                self.llm_engine.engine_core.engine_core.scheduler.requests[internal] = (
                    SimpleNamespace(num_output_tokens=0)
                )
                ids.append(internal)
            return ids

    vllm = ModuleType("vllm")
    vllm.LLM = LLM
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    core = ModuleType("vllm.v1.engine.core_client")
    core.InprocClient = InprocClient
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core_client", core)
    for key in (
        "MUXPD_NUMERIC_OUTPUT",
        "MUXPD_NUMERIC_CAPTURE_STEPS",
        "MUXPD_NUMERIC_RANKS",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
    ):
        monkeypatch.setenv(key, "test-before")


def arguments(tmp_path):
    frozen = tmp_path / "plan.json"
    frozen.write_text(json.dumps(plan()))
    engine = tmp_path / "engine.json"
    engine.write_text(
        json.dumps(
            {
                "tensor_parallel_size": 4,
                "async_scheduling": False,
                "enable_prefix_caching": False,
                "max_num_seqs": 8,
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
            }
        )
    )
    return SimpleNamespace(
        plan=frozen,
        plan_sha256=plan()["plan_sha256"],
        engine_args=engine,
        role="native",
        output_dir=tmp_path / "out",
    )


def test_internal_external_ids_bound_before_native_completion(monkeypatch, tmp_path):
    install_fake_engine(monkeypatch)
    args = arguments(tmp_path)
    assert run(args) == 0
    receipt = json.loads((args.output_dir / "run.json").read_text())
    assert receipt["passed"] and len(receipt["phases"]) == 6
    for phase in receipt["phases"]:
        assert set(phase["internal_to_external"]) == set(phase["request_ids"])
        assert set(phase["internal_to_external"].values()) == set(
            phase["external_request_ids"]
        )
        assert {row["request_id"] for row in phase["outputs"]} == set(
            phase["external_request_ids"]
        )
        assert not set(phase["request_ids"]) & set(phase["external_request_ids"])


def test_step_watchdog_preserves_failure_and_calls_shutdown(monkeypatch, tmp_path):
    install_fake_engine(monkeypatch, never_finishes=True)
    args = arguments(tmp_path)
    assert run(args) == 1
    receipt = json.loads((args.output_dir / "run.json").read_text())
    assert not receipt["passed"] and receipt["shutdown_returned"]
    assert len(receipt["phases"][0]["steps"]) == 500
    assert "exceeded 500" in receipt["errors"][0]


def test_staggered_gate_uses_live_scheduler_when_client_is_final_only(
    monkeypatch, tmp_path
):
    install_fake_engine(monkeypatch, per_token=True)
    args = arguments(tmp_path)
    frozen = plan(staggered=True)
    args.plan.write_text(json.dumps(frozen))
    args.plan_sha256 = frozen["plan_sha256"]
    engine = json.loads(args.engine_args.read_text())
    engine["enable_chunked_prefill"] = True
    args.engine_args.write_text(json.dumps(engine))
    assert run(args) == 0
    receipt = json.loads((args.output_dir / "run.json").read_text())
    assert receipt["passed"]
    for phase in receipt["phases"][3:]:
        assert phase["initial_request_count"] == 1
        assert phase["steps"][0]["outputs"] == []
        assert len(phase["admissions"]) == 2
        initial, peers = phase["admissions"]
        assert len(initial["request_ids"]) == 1
        assert len(peers["request_ids"]) == 3
        assert peers["seed_num_output_tokens"] == 1
        assert peers["seed_live_in_scheduler"] is True
        assert peers["after_step"] == initial["after_step"] + 1
        assert len(phase["outputs"]) == 4


def test_staggered_plan_keeps_original_control_plan_identity():
    original = plan()
    assert original["plan_sha256"] == (
        "f24b55775a6a754c61c8edc79aec1bca20877ed09114d291cfe949e5d6196c4c"
    )
    staggered = plan(staggered=True)
    assert staggered["requests"] == 15
    assert staggered["plan_sha256"] != original["plan_sha256"]
    assert [p["prompts"] for p in original["phases"]] == [
        p["prompts"] for p in staggered["phases"]
    ]


def test_group_role_binds_explicit_policy_and_global_backend():
    engine = {
        "batch_admission_policy_config": {"compile_prefill_groups": True},
        "additional_config": {
            "ascend_compilation_config": {
                "enable_npugraph_ex": False,
                "enable_static_kernel": False,
            }
        },
    }
    result = runtime_profile("group", engine)
    assert result["compile_prefill_groups"] is True
    assert (
        result["ascend_compilation_config"]
        == engine["additional_config"]["ascend_compilation_config"]
    )
    with pytest.raises(ValueError, match="disagree"):
        runtime_profile("raw", engine)
    engine["batch_admission_policy_config"]["compile_prefill_groups"] = False
    assert runtime_profile("raw", engine)["compile_prefill_groups"] is False
    with pytest.raises(ValueError, match="disagree"):
        runtime_profile("group", engine)
    engine["batch_admission_policy_config"]["compile_prefill_groups"] = 0
    with pytest.raises(ValueError, match="boolean"):
        runtime_profile("raw", engine)


def test_group_role_cannot_hide_default_or_partial_backend():
    engine = {"batch_admission_policy_config": {"compile_prefill_groups": True}}
    with pytest.raises(ValueError, match="fusion backend"):
        runtime_profile("group", engine)
    engine["additional_config"] = {
        "ascend_compilation_config": {"enable_npugraph_ex": False}
    }
    with pytest.raises(ValueError, match="fusion backend"):
        runtime_profile("group", engine)
