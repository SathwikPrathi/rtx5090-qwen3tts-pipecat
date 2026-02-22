import time
from typing import Tuple

import numpy as np

from qwen_tts_megakernel.backend.talker_decoder import load_talker_decoder
from qwen_tts_megakernel.backend.tts_pipeline import QwenMegakernelTTSPipeline


def bench_megakernel_decode(
    num_tokens: int = 256,
    warmup_tokens: int = 32,
) -> Tuple[float, float]:
    """
    Benchmark megakernel decode throughput.

    Returns:
        tokens_per_second, ms_per_step
    """
    decoder, _ = load_talker_decoder()
    decoder.reset()

    # use some arbitrary BOS token; 1 is typical for many Qwen configs
    token_id = 1

    # Warmup
    for _ in range(warmup_tokens):
        token_id = decoder.step(token_id)

    # Timed decode
    t0 = time.time()
    for _ in range(num_tokens):
        token_id = decoder.step(token_id)
    t1 = time.time()

    dt = t1 - t0
    tokens_per_second = num_tokens / dt
    ms_per_step = 1000.0 * dt / num_tokens
    print(f"[Megakernel] {tokens_per_second:.2f} tok/s, {ms_per_step:.3f} ms/step")

    return tokens_per_second, ms_per_step


def bench_tts_utterance(text: str = "This is a latency test."):
    """
    Benchmark TTFC, RTF, and end-to-end latency for a single TTS utterance.
    """
    pipe = QwenMegakernelTTSPipeline()
    sr = pipe.sample_rate

    chunks = []
    t0 = time.time()
    ttfc_ms = None

    for i, chunk in enumerate(pipe.stream_tts(text)):
        now = time.time()
        if i == 0:
            ttfc_ms = (now - t0) * 1000.0
        chunks.append(chunk)

    t1 = time.time()

    audio = np.concatenate(chunks)
    duration_s = audio.shape[0] / sr
    compute_s = t1 - t0
    rtf = compute_s / duration_s if duration_s > 0 else float("inf")

    print(f"[TTS] TTFC: {ttfc_ms:.1f} ms")
    print(f"[TTS] RTF: {rtf:.3f}")
    print(f"[TTS] Total compute latency: {compute_s*1000:.1f} ms for {duration_s:.2f} s audio")

    return ttfc_ms, rtf, compute_s, duration_s


if __name__ == "__main__":
    print("=== Megakernel decode benchmark ===")
    bench_megakernel_decode()

    print("\n=== TTS benchmark ===")
    bench_tts_utterance()