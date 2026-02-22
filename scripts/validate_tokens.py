"""
Token-level validation of megakernel talker decoder vs HuggingFace/Qwen3-TTS.

This script:
  - Loads the official Qwen3TTSForConditionalGeneration model
  - Loads the megakernel-backed TalkerDecoder
  - Runs both on a small set of prompts and compares token IDs step-by-step
"""

from typing import List

import torch

from qwen_tts.core.models import Qwen3TTSForConditionalGeneration

from qwen_tts_megakernel.backend.talker_decoder import load_talker_decoder


TEST_PROMPTS = [
    "Hello, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "Megakernel integration for Qwen3-TTS.",
]


def extract_talker_ref_ids(
    tts: Qwen3TTSForConditionalGeneration,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
) -> List[int]:
    """
    Use the official TTS model to generate talker token IDs for a prompt.

    Because internal APIs may differ, this function is intentionally loose:
    it uses the standard generate() to produce output token IDs.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(tts.device)
    with torch.no_grad():
        out = tts.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # Take only the generated continuation
    gen_ids = out[0, input_ids.shape[1] :]
    return gen_ids.tolist()


def extract_talker_megakernel_ids(
    decoder,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
) -> List[int]:
    """
    Use the megakernel-backed TalkerDecoder to generate token IDs for a prompt.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    ids_tensor = torch.tensor(input_ids, device=decoder._device, dtype=torch.long)
    gen_ids = decoder.generate_ids(ids_tensor, max_new_tokens=max_new_tokens)
    # Exclude the last prompt token to match the reference continuation behavior
    return gen_ids.tolist()[1:]


def main():
    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    print("[*] Loading reference Qwen3-TTS model...")
    tts = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    tokenizer = getattr(tts, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen3-TTS tokenizer not found on model")

    print("[*] Loading megakernel talker decoder...")
    decoder, mk_tokenizer = load_talker_decoder(model_name=model_name, device="cuda")

    # Ideally tokenizer and mk_tokenizer are the same
    if mk_tokenizer is not tokenizer:
        print("[!] Warning: tokenizer instances differ; using reference tokenizer")

    for prompt in TEST_PROMPTS:
        print(f"\n=== Prompt: {prompt!r} ===")
        ref_ids = extract_talker_ref_ids(tts, tokenizer, prompt)
        mk_ids = extract_talker_megakernel_ids(decoder, tokenizer, prompt)

        min_len = min(len(ref_ids), len(mk_ids))
        mismatch_idx = None
        for i in range(min_len):
            if ref_ids[i] != mk_ids[i]:
                mismatch_idx = i
                break

        if mismatch_idx is None:
            print(f"OK: first {min_len} tokens match")
        else:
            print("MISMATCH:")
            print(f"  index: {mismatch_idx}")
            print(f"  ref: {ref_ids[mismatch_idx]}")
            print(f"  mk : {mk_ids[mismatch_idx]}")
            raise SystemExit(1)

    print("\nAll prompts matched for compared token ranges.")


if __name__ == "__main__":
    main()