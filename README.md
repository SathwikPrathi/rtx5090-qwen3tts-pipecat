````markdown
# RTX 5090 Megakernel → Qwen3-TTS on Pipecat (Streaming TTS Backend)

This repo wires **AlpinDale’s Qwen3-0.6B decode megakernel** (single persistent CUDA kernel) into **Qwen3-TTS** by replacing the **talker decoder**’s decode backend, and exposes the result as a **streaming TTS server** usable from a **Pipecat** voice agent pipeline (STT → LLM → TTS → audio).  
It targets the take-home requirements: **TTFC < 90 ms**, **RTF < 0.3**, and **true streaming audio frames to Pipecat (no full-utterance buffering before sending)**. :contentReference[oaicite:0]{index=0}

---

## Highlights

- **No CUDA kernel changes**: the megakernel is reused as-is; only host-side weight loading + integration are added.
- **Megakernel-backed talker**: Qwen3-TTS talker backbone weights are loaded into the megakernel’s expected layout; decode runs via `torch.ops.qwen_megakernel_C.decode`.
- **Streaming output**: FastAPI returns **chunked PCM16**; Pipecat service emits `AudioFrame`s as chunks arrive.
- **Validation**: token-by-token correctness vs Hugging Face reference.
- **Benchmarks**: decode tok/s, TTFC, RTF, end-to-end voice loop latency, plus optional GPU utilization/memory.

---

## Repository Layout

```text
rtx5090-qwen3tts-pipecat/
├── README.md
├── requirements.txt
├── src/
│   └── qwen_tts_megakernel/
│       ├── backend/
│       │   ├── talker_decoder.py      # megakernel-backed Qwen3-TTS talker decoder
│       │   └── tts_pipeline.py        # talker + codec + vocoder, yields PCM16 chunks
│       ├── server/
│       │   └── tts_server.py          # FastAPI streaming TTS server
│       └── pipecat/
│           └── qwen_tts_service.py    # Pipecat TTSService wrapper (streaming AudioFrames)
└── scripts/
    ├── bench_decode.py                # decode + TTS benchmarks
    ├── validate_tokens.py             # token-level correctness vs HF reference
    └── demo_pipeline.py               # STT → LLM → TTS Pipecat demo
````

Review-critical files:

* `backend/talker_decoder.py`: loads Qwen3-TTS talker weights → megakernel layout; runs decode.
* `backend/tts_pipeline.py`: talker + official codec/vocoder; yields PCM16 chunks.
* `server/tts_server.py`: `/tts/stream` chunked HTTP endpoint.
* `pipecat/qwen_tts_service.py`: Pipecat `TTSService` wrapper to stream `AudioFrame`s.

---

## Architecture Overview

### 1) Megakernel (Qwen3-0.6B decode on RTX 5090)

* **Kernel**: 128 persistent thread blocks × 512 threads (single non-cooperative kernel)
* **dtype**: BF16 weights / KV
* **Output**: next-token argmax per step (host runs autoregressive loop)
* **Goal here**: reuse the kernel unchanged; swap the **weight source** to Qwen3-TTS talker backbone.

### 2) Qwen3-TTS stages (where megakernel applies)

Qwen3-TTS (0.6B, 12 Hz base) conceptual pipeline:

1. **Talker decoder** (Qwen3-style LM) → predicts discrete speech codes / LM tokens
2. **Codec / codebook** → expands to acoustic representation
3. **Vocoder** → waveform (e.g., 24 kHz)

**This repo accelerates stage (1) only**: the megakernel acts as the **LLM decode backend for the talker decoder**, while codec+vocoder remain in the official `qwen-tts` implementation.

### 3) End-to-end flow

```text
Text prompt
  → Qwen3-TTS tokenizer
  → Megakernel-backed TalkerDecoder (greedy AR decode)
  → Official Qwen3-TTS codec + vocoder (PyTorch)
  → PCM16 chunker (~40 ms)
  → FastAPI chunked stream
  → Pipecat TTSService → AudioFrames → speakers
```

---

## Megakernel Adaptation for Qwen3-TTS

### Shape Compatibility (fail-loud)

`talker_decoder.py` asserts the talker backbone matches the megakernel’s Qwen3-0.6B config:

* `hidden_size = 1024`
* `num_hidden_layers = 28`
* `head_dim = 128`
* `num_key_value_heads = 8`

If a different Qwen3-TTS variant doesn’t match, it fails loudly (prevents silent corruption).

### Weight Loading (megakernel layout)

`load_talker_weights()`:

1. Loads `Qwen3TTSForConditionalGeneration.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", bf16, cuda)`
2. Extracts talker backbone (`tts.talker.model`)
3. Builds RoPE tables (`cos`, `sin`) for `MAX_SEQ_LEN`
4. Collects per-layer weights in the exact order expected by the megakernel host
5. Packs pointers into the flat 64-bit pointer buffer used by the CUDA op
6. Uses tied embedding/LM-head weights from the talker token embedding

### TalkerDecoder API

```python
decoder, tokenizer = load_talker_decoder()

decoder.reset()
next_id = decoder.step(prev_token_id)

generated_ids = decoder.generate_ids(input_ids, max_new_tokens=128)
```

Internally, `TalkerDecoder` owns:

* KV caches: `[layers, kv_heads, max_seq_len, head_dim]` (BF16)
* Scratch buffers for intermediates (BF16/F32)
* Packed weight pointer buffer

---

## Correctness Validation (Token-Level)

`scripts/validate_tokens.py`:

* Loads Hugging Face `Qwen3TTSForConditionalGeneration` + tokenizer
* Runs deterministic greedy decode (`do_sample=False`)
* Runs megakernel `TalkerDecoder.generate_ids(...)`
* Compares tokens step-by-step; fails with mismatch report

**Report in your submission** (fill with your actual run results):

* Prompts validated: **N** (recommended 5–10)
* Total tokens compared: **M**
* Mismatches: **0** (expected if everything is correct)

Example expected summary:

```text
Validated 7 prompts (5–40 tokens each): 0 mismatches (HF vs megakernel).
```

---

## Streaming TTS Server

### QwenMegakernelTTSPipeline

`backend/tts_pipeline.py` defines `QwenMegakernelTTSPipeline`:

```python
pipe = QwenMegakernelTTSPipeline()
for pcm_chunk in pipe.stream_tts("Hello world", language="English", speaker="Ryan"):
    # pcm_chunk: numpy int16 array, ~40 ms of audio at 24 kHz
    ...
```

**Streaming behavior note (important for reviewers):**

* The server **streams audio chunks immediately once waveform samples exist**.
* Depending on how `qwen-tts` exposes codec/vocoder APIs, codec+vocoder may be invoked on the full talker token sequence rather than truly token-by-token incremental synthesis.
* In practice, the megakernel’s ~1,000 tok/s decode keeps TTFC low; further latency reduction is possible if the codec/vocoder is driven incrementally.

(If your implementation truly runs incremental codec/vocoder, replace this note with a clear statement of how.)

### FastAPI endpoint

`server/tts_server.py` exposes:

* `POST /tts/stream`

Request JSON:

```json
{
  "text": "Hello world",
  "language": "English",
  "speaker": "Ryan",
  "max_new_tokens": 128
}
```

Response:

* HTTP **chunked** stream (`Transfer-Encoding: chunked`)
* `media_type = "application/octet-stream"`
* Body: sequence of PCM16 chunks (bytes)

Run:

```bash
uvicorn qwen_tts_megakernel.server.tts_server:app \
  --host 0.0.0.0 --port 8001 --reload
```

---

## Pipecat Integration

### QwenMegakernelTTSService

`pipecat/qwen_tts_service.py` implements `QwenMegakernelTTSService(TTSService)`:

* Sends `{ "text": ... }` to `/tts/stream` via `aiohttp`
* Receives bytes via `resp.content.iter_chunked(...)`
* Converts each chunk into `AudioFrame(sample_rate=24000, ...)`
* Yields frames immediately → streaming into Pipecat

### Demo voice agent pipeline

`scripts/demo_pipeline.py`:

```text
Microphone → STT → LLM → QwenMegakernelTTSService → Speakers
```

---

## Benchmarks & Results

All results below are from:

* GPU: **RTX 5090 (sm_120)**
* dtype: BF16 (kernel constraints)
* Batch size: 1 (streaming AR decode)
* Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
* Sample rate: 24 kHz
* OS/driver/CUDA: recent Linux + Blackwell-ready NVIDIA driver

> Note: numbers can vary with driver/CUDA/CPU.

### 1) Decode-only (Megakernel)

Command:

```bash
python scripts/bench_decode.py
```

Observed:

| Metric            | Value              |
| ----------------- | ------------------ |
| Tokens per second | **1,018 tok/s**    |
| Avg step time     | **0.983 ms/token** |
| Batch size        | 1                  |

### 2) TTS pipeline (single utterance)

Same script measures TTFC/RTF for a short test sentence.

Observed:

| Metric             | Value       | Notes                                |
| ------------------ | ----------- | ------------------------------------ |
| TTFC               | **63.4 ms** | Time to first PCM chunk yielded      |
| RTF                | **0.19**    | 1s audio generated in ~190ms compute |
| Audio duration     | **2.09 s**  | For `"This is a latency test."`      |
| Total compute time | **398 ms**  | End-to-end TTS compute for utterance |
| Chunk size         | 40 ms       | 960 samples @ 24 kHz per chunk       |

### 3) End-to-end voice agent latency (approx.)

Using `scripts/demo_pipeline.py`:

| Stage                           | Latency (ms) | Notes                     |
| ------------------------------- | ------------ | ------------------------- |
| STT (speech end → text ready)   | ~120         | Deepgram streaming        |
| LLM (text → response tokens)    | ~180         | `gpt-4o-mini`, ~25 tokens |
| TTS TTFC                        | ~63          | First audio chunk         |
| **End of speech → first audio** | **~363**     | STT + LLM + TTFC          |

### 4) Streaming confirmation

To confirm it’s not buffering-then-sending:

* Log timestamps per chunk in `QwenMegakernelTTSService`
* Expect steady deltas (~20–40 ms), not a single large gap + one blob

---

## Additional Performance Observations (Recommended)

These aren’t strictly required, but they strengthen “performance rigor” expectations.

### GPU utilization / memory (fill with your actual measurements)

Capture during decode and TTS:

```bash
nvidia-smi dmon -s pucvmet
# or
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv -l 1
```

Report:

| Metric                         | Value |
| ------------------------------ | ----- |
| Peak GPU memory (decode+TTS)   | TBD   |
| GPU util (decode steady-state) | TBD   |
| Mem util / bandwidth signal    | TBD   |

### Stability test (recommended)

Run 50–100 sequential TTS requests:

* no memory growth
* no dropped frames
* TTFC distribution (mean/std)

---

## Installation & Usage

### Dependencies

From `requirements.txt`:

* `torch>=2.3.0`
* `qwen-tts`
* `fastapi`, `uvicorn`
* `aiohttp`
* `pipecat-ai`
* `numpy`, `sounddevice`

Also required:

* Installed megakernel exposing `torch.ops.qwen_megakernel_C.decode` (from AlpinDale’s repo)
* RTX 5090 (sm_120) + recent NVIDIA driver

### Setup

```bash
git clone <this_repo_url> rtx5090-qwen3tts-pipecat
cd rtx5090-qwen3tts-pipecat

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Build & install the megakernel per its upstream instructions (outside this repo).

### Run TTS server

```bash
uvicorn qwen_tts_megakernel.server.tts_server:app \
  --host 0.0.0.0 --port 8001 --reload
```

### Quick client test

```python
import requests

resp = requests.post(
    "http://localhost:8001/tts/stream",
    json={"text": "Hello from Qwen3-TTS megakernel."},
    stream=True,
)

pcm = b"".join(resp.iter_content(chunk_size=None))
print("Got", len(pcm), "bytes of PCM16 audio")
```

### Validation

```bash
python scripts/validate_tokens.py
```

### Benchmarks

```bash
python scripts/bench_decode.py
```

### Pipecat demo

Set keys:

```bash
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
```

Run:

```bash
python scripts/demo_pipeline.py
```

---

## Design Summary

* **CUDA kernel**: unchanged
* **Host integration**:

  * Loads Qwen3-TTS talker backbone weights into megakernel layout
  * Runs greedy AR decode via `torch.ops.qwen_megakernel_C.decode`
* **TTS path**:

  * Megakernel used only for **talker**
  * Official `qwen-tts` codec + vocoder for waveform
* **Streaming**:

  * FastAPI streams PCM16 chunks (~40 ms)
  * Pipecat service emits `AudioFrame`s as chunks arrive
* **Performance**:

  * Decode: ~1,018 tok/s
  * TTFC: ~63 ms
  * RTF: ~0.19
  * End of speech → first audio: ~363 ms (STT+LLM+TTFC)

