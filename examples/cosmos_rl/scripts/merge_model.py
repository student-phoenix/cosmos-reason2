"""
Merge LoRA adapter weights into the base model and save the result.

Usage:
    python merge_model.py \
        --adapter_path outputs/nvidia_uber_cosmos_sft_fumseck/20260507155513/safetensors/step_21 \
        --output_path merged_model \
        [--base_model nvidia/Cosmos-Reason2-2B] \
        [--dtype float16]
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the directory containing adapter_config.json and adapter_model.safetensors",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where the merged model will be saved",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name or path (defaults to base_model_name_or_path from adapter_config.json)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype to load and save the model in (default: bfloat16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)

    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_id = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model_id:
        raise ValueError("Could not determine base model: set --base_model or ensure adapter_config.json has base_model_name_or_path")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Base model : {base_model_id}")
    print(f"Adapter    : {adapter_path}")
    print(f"Output     : {output_path}")
    print(f"Dtype      : {args.dtype}")

    print("\nLoading base model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # PEFT 0.18.0 crashes if alpha_pattern/rank_pattern are None (calls .keys() unconditionally).
    # Patch null pattern dicts to {} and load from a temp dir to avoid modifying the saved adapter.
    for key in ("alpha_pattern", "rank_pattern", "r_pattern"):
        if adapter_config.get(key) is None:
            adapter_config[key] = {}

    print("Loading LoRA adapter...")
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        for f in adapter_path.iterdir():
            if f.name != "adapter_config.json":
                shutil.copy2(f, tmp / f.name)
        (tmp / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
        model = PeftModel.from_pretrained(model, str(tmp))

    print("Merging weights...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(str(output_path), safe_serialization=True)

    print("Saving processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)
    processor.save_pretrained(str(output_path))

    print(f"\nDone. Merged model saved to: {output_path}")


if __name__ == "__main__":
    main()

