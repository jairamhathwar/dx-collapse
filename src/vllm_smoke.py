"""Minimal vLLM check for a new environment/cluster: load a small model and generate.

    python vllm_smoke.py [MODEL]
"""
import sys


def main():
    from vllm import LLM, SamplingParams
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.6)
    out = llm.generate(["The most common cause of chest pain is"], SamplingParams(max_tokens=16, temperature=0))
    print("VLLM OK:", out[0].outputs[0].text.strip()[:80])


if __name__ == "__main__":  # required: vLLM workers use spawn, which re-imports this module
    main()
