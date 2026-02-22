import aiohttp

from pipecat.services.tts_service import TTSService
from pipecat.frames import AudioFrame


class QwenMegakernelTTSService(TTSService):
    """
    Pipecat TTS service that streams audio from the local Qwen3-TTS megakernel server.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_url = server_url.rstrip("/")
        self.sample_rate = sample_rate

    async def run_tts(self, text: str, context_id: str):
        """
        Implement the Pipecat TTS interface:

            - text: text to synthesize
            - context_id: pipeline context

        Yields AudioFrame objects with raw PCM16 bytes.
        """
        url = f"{self.server_url}/tts/stream"
        payload = {"text": text}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                async for chunk in resp.content.iter_chunked(4096):
                    if not chunk:
                        break
                    frame = AudioFrame(
                        audio=bytes(chunk),
                        sample_rate=self.sample_rate,
                        context_id=context_id,
                    )
                    yield frame