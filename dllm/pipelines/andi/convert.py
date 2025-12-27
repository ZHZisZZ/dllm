from dataclasses import dataclass

import transformers
import tyro

import dllm

AnD_CONFIG_MAP = {
    "llama": dllm.pipelines.andi.AnDLlamaConfig,
    "qwen2": dllm.pipelines.andi.AnDQwen2Config,
    "qwen3": dllm.pipelines.andi.AnDQwen3Config,
}


@dataclass
class ScriptArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = "models/andi/Qwen2.5-0.5B"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def main():

    args = tyro.cli(ScriptArguments)
    dllm.utils.print_args(args)

    # Load source model
    src_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype="bfloat16",
    )
    src_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    src_config = src_model.config

    # Remove unused HF fields
    for k in ["auto_map", "architectures"]:
        if hasattr(src_config, k):
            delattr(src_config, k)

    # Select corresponding AnD config class
    base_type = src_config.model_type
    tgt_config_cls = AnD_CONFIG_MAP[base_type]

    # Build AnD config from source config dict
    cfg_dict = src_config.to_dict()
    cfg_dict.pop("model_type", None)
    tgt_config = tgt_config_cls(**cfg_dict)

    with dllm.utils.init_device_context_manager():
        # Build AnD model
        tgt_model = transformers.AutoModel.from_config(tgt_config)
        # Direct weight copy (models match exactly)
        tgt_model.load_state_dict(src_model.state_dict())

        # Save model and config
        tgt_model.save_pretrained(args.output_dir)
        tgt_config.save_pretrained(args.output_dir)
        src_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
