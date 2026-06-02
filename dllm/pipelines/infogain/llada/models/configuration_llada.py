"""
LLaDA Info-Gain configuration wrapper.
Registers a distinct model_type for AutoConfig while reusing LLaDA hyperparameters.

Run (from repo root): python -c "from dllm.pipelines.infogain.llada.models import InfoGainLLaDAConfig; print(InfoGainLLaDAConfig)"
"""

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig


class InfoGainLLaDAConfig(LLaDAConfig):
    model_type = "infogain_llada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
