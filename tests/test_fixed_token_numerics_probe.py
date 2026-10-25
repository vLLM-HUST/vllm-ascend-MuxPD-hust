import copy

from scripts.probe_fixed_token_numerics import digest, plan


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
