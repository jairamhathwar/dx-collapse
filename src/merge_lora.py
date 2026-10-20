"""Merge a saved LoRA adapter into its base model and save a full bf16 model for vLLM evaluation.
Used to move trained runs between clusters by copying only the adapter (~140 MB vs ~3.4 GB).

    python merge_lora.py --base Qwen/Qwen3-1.7B --adapter RUN_DIR --out RUN_DIR/merged
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True, help="dir with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    print("merged ->", a.out)


if __name__ == "__main__":
    main()
