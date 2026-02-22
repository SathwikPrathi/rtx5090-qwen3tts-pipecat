import asyncio
import os

from pipecat.pipeline import Pipeline
from pipecat.transports.local.audio import LocalAudioInput, LocalAudioOutput
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

from qwen_tts_megakernel.pipecat import QwenMegakernelTTSService


async def main():
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not deepgram_key or not openai_key:
        raise RuntimeError("Please set DEEPGRAM_API_KEY and OPENAI_API_KEY environment variables")

    stt = DeepgramSTTService(api_key=deepgram_key)
    llm = OpenAILLMService(api_key=openai_key, model="gpt-4o-mini")

    tts = QwenMegakernelTTSService(
        server_url="http://localhost:8001",
        sample_rate=24000,
    )

    mic = LocalAudioInput()
    speaker = LocalAudioOutput()

    pipeline = Pipeline(
        stages=[
            mic.input(),
            stt,
            llm,
            tts,
            speaker.output(),
        ]
    )

    print("Starting Pipecat voice loop. Speak into your microphone...")
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())