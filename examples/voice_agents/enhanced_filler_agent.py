"""
Enhanced filler detection agent with multi-layered intelligence.
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.agents.voice.filler_detection import (
    SemanticFillerDetector,
    TemporalAnalyzer,
    AudioFeatureFilter,
    ConfidenceScorer,
    FillerAwareSTT,
)
from livekit.plugins import deepgram, openai, cartesia, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("enhanced-filler-agent")
load_dotenv()


class AgentStateTracker:
    """Simple tracker for agent state."""
    def __init__(self):
        self.agent_speaking = False


class EnhancedAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "Keep responses concise and natural. "
                "No emojis or special characters."
            )
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info("=== Initializing Enhanced Filler Detection System ===")
    
    # 1. Initialize all detection components
    semantic_detector = SemanticFillerDetector(
        filler_threshold=0.75,
        command_margin=0.1,
    )
    
    temporal_analyzer = TemporalAnalyzer(
        window_size=10,
        window_duration=3.0,
        rapid_threshold=2.0,
        sustained_threshold=3.0,
    )
    
    # Audio feature configuration via environment variables
    enable_audio = os.getenv('ENABLE_AUDIO_FEATURES', 'false').lower() in ('1', 'true', 'yes')
    audio_fast_mode = os.getenv('AUDIO_FEATURES_FAST_MODE', 'true').lower() in ('1', 'true', 'yes')

    audio_filter = None
    if enable_audio:
        audio_filter = AudioFeatureFilter(
            sample_rate=16000,
            short_duration_threshold=0.3,
            low_energy_threshold=0.1,
            fast_mode=audio_fast_mode,
        )
    
    # 2. Create multi-factor confidence scorer
    filler_words = set(os.getenv('FILLER_WORDS', 'uh,um,hmm,er,ah,haan').split(','))
    
    confidence_scorer = ConfidenceScorer(
        semantic_detector=semantic_detector,
        temporal_analyzer=temporal_analyzer,
        audio_filter=audio_filter,
        filler_words=filler_words,
    )
    
    logger.info("✅ All detection components initialized")
    
    # 3. Create agent state tracker
    state_tracker = AgentStateTracker()
    
    # 4. Wrap base STT with filtering layer
    base_stt = deepgram.STT()
    
    wrapped_stt = FillerAwareSTT(
        base_stt=base_stt,
        confidence_scorer=confidence_scorer,
        agent_state_tracker=state_tracker,
        filter_interim=True,
        filter_final=True,
    )
    
    logger.info("✅ STT wrapper configured with multi-layered filtering")
    
    # 5. Create agent session with wrapped STT
    session = AgentSession(
        stt=wrapped_stt,
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        min_interruption_duration=0.5,
        resume_false_interruption=True,
        false_interruption_timeout=1.5,
        preemptive_generation=True,
    )
    
    # 6. Hook agent state tracking
    @session.on("agent_state_changed")
    def track_agent_state(ev):
        state_tracker.agent_speaking = (ev.new_state == "speaking")
        logger.debug(f"Agent state: {ev.old_state} -> {ev.new_state}")
    
    # 7. Log filtering statistics periodically
    async def log_stats():
        stats = wrapped_stt.get_stats()
        logger.info(f"Filtering stats: {stats}")
    
    ctx.add_shutdown_callback(log_stats)
    
    logger.info("=== Starting Enhanced Agent Session ===")
    
    # 8. Start session
    await session.start(
        agent=EnhancedAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable debug for filler detection
    logging.getLogger("livekit.agents.voice.filler_detection").setLevel(logging.DEBUG)
    
    cli.run_app(server)
