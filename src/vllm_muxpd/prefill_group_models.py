"""Distinct source/code identities for ten P-only AOT entry points.

Imported only for the explicit experimental opt-in. These modules reference
loaded layers; they neither load weights nor replace the shared model forward.
"""

import torch
from vllm.compilation.decorators import support_torch_compile


class _GroupBase(torch.nn.Module):
    def __init__(self, *, layers, vllm_config):
        super().__init__()
        if len(layers) != 4:
            raise ValueError("Prefill FX group must contain four actual layers")
        self.layers = torch.nn.ModuleList(layers)

    def run_layers(self, positions, hidden_states, residual):
        for layer in self.layers:
            # Bypass only nn.Module hooks. The loaded forward remains the
            # authority for HUST vs release implementations and custom ops.
            # Parent validation excludes every hook except our progress hooks.
            hidden_states, residual = layer.forward(
                positions=positions, hidden_states=hidden_states, residual=residual
            )
        return hidden_states, residual


class PrefillGroup00(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup01(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup02(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup03(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup04(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup05(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup06(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup07(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup08(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


class PrefillGroup09(_GroupBase):
    def forward(self, positions, hidden_states, residual):
        return self.run_layers(positions, hidden_states, residual)


GROUP_CLASSES = tuple(
    support_torch_compile(
        dynamic_arg_dims={
            "positions": -1,
            "hidden_states": 0,
            "residual": 0,
        }
    )(cls)
    for cls in (
        PrefillGroup00,
        PrefillGroup01,
        PrefillGroup02,
        PrefillGroup03,
        PrefillGroup04,
        PrefillGroup05,
        PrefillGroup06,
        PrefillGroup07,
        PrefillGroup08,
        PrefillGroup09,
    )
)
