"""
Simple integration example showing how to add filler filtering to an existing agent.

This demonstrates the minimal code changes needed to add the FillerFilterPlugin
to any existing LiveKit agent implementation.
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.voice import FillerFilterPlugin  # <-- Add this import
from livekit.plugins import deepgram, openai, cartesia, silero

logger = logging.getLogger("simple-integration")
load_dotenv()


class SimpleAgent(Agent):
    """Your existing agent class - no changes needed here."""

    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant."
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Your existing entrypoint function."""
    
    # Your existing session setup
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )
    
    # ========================================================================
    # ADD THESE 3 LINES TO ENABLE FILLER FILTERING
    # ========================================================================
    filler_filter = FillerFilterPlugin()  # Create filter with defaults
    filler_filter.attach_to_session(session)  # Attach to session
    logger.info("✅ Filler filter enabled")
    # ========================================================================
    
    # Your existing agent start code
    await session.start(
        agent=SimpleAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
