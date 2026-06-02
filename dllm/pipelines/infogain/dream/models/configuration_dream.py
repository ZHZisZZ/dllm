"""
Dream Info-Gain configuration wrapper.

Run (from repo root): python -c "from dllm.pipelines.infogain.dream.models import InfoGainDreamConfig; print(InfoGainDreamConfig)"
"""

from dllm.pipelines.dream.models.configuration_dream import DreamConfig


class InfoGainDreamConfig(DreamConfig):
    """Thin wrapper with a distinct ``model_type`` for HuggingFace auto-registration."""

    model_type = "infogain_dream"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
