import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from qwen_tts_megakernel.backend.tts_pipeline import QwenMegakernelTTSPipeline

app = FastAPI(title="Qwen3-TTS Megakernel Server")
pipeline = QwenMegakernelTTSPipeline()


@app.post("/tts/stream")
async def tts_stream(request: Request):
    """
    Streaming TTS endpoint.

    Request JSON:
        {
          "text": "...",
          "language": "English",
          "speaker": "Ryan",
          ...
        }

    Response: application/octet-stream of raw PCM16 audio frames.
    """
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return StreamingResponse(
            iter(()),
            media_type="application/octet-stream",
        )

    language = body.get("language", "English")
    speaker = body.get("speaker", "Ryan")
    max_new_tokens = int(body.get("max_new_tokens", 128))

    gen_kwargs = {
        k: v
        for k, v in body.items()
        if k not in ("text", "language", "speaker", "max_new_tokens")
    }

    def sync_generator():
        for chunk in pipeline.stream_tts(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        ):
            yield chunk.tobytes()

    async def async_generator() -> AsyncGenerator[bytes, None]:
        loop = asyncio.get_event_loop()
        for chunk in pipeline.stream_tts(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        ):
            # Run any heavy blocking parts in default executor if needed
            yield chunk.tobytes()
            await asyncio.sleep(0)

    return StreamingResponse(
        async_generator(),
        media_type="application/octet-stream",
    )