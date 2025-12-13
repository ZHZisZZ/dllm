# from .models.gpt2.modeling_gpt2 import (
#     A2DGPT2Config,
#     A2DGPT2LMHeadModel,
# )
import transformers

from .models.llama.modeling_llama import A2DLlamaConfig, A2DLlamaLMHeadModel
from .models.qwen2.modeling_qwen2 import A2DQwen2Config, A2DQwen2LMHeadModel
from .models.qwen3.modeling_qwen3 import A2DQwen3Config, A2DQwen3LMHeadModel

A2D_CONFIG_MAP = {
    # transformers.GPT2Config: A2DGPT2Config,
    transformers.LlamaConfig: A2DLlamaConfig,
    transformers.Qwen2Config: A2DQwen2Config,
    transformers.Qwen3Config: A2DQwen3Config,
}
