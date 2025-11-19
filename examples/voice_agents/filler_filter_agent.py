"""
Filler Word Filter Demo Agent

This example demonstrates how to use the FillerFilterPlugin to intelligently
filter out filler words during agent speech while allowing them as valid
speech when the agent is quiet.

Features demonstrated:
- Basic filler filter setup and configuration
- Event-based integration with AgentSession
- Dynamic configuration via environment variables
- Logging of filter decisions
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
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import FillerFilterPlugin
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("filler-filter-agent")

load_dotenv()


class FillerFilterDemoAgent(Agent):
    """Demo agent showcasing filler word filtering capabilities."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly, a helpful voice assistant. "
                "Keep your responses concise and conversational. "
                "Do not use emojis, asterisks, markdown, or other special characters. "
                "You are friendly, patient, and have a sense of humor. "
                "Sometimes users might say 'umm' or 'uh' while you're speaking - "
                "these are just fillers and you should continue speaking unless they "
                "say something meaningful like 'wait', 'stop', or ask a real question."
            ),
        )

    async def on_enter(self):
        """When the agent is added to the session, generate initial greeting."""
        self.session.generate_reply()

    @function_tool
    async def get_current_time(self, context: RunContext):
        """
        Get the current time.
        
        Call this when the user asks what time it is.
        """
        from datetime import datetime

        current_time = datetime.now().strftime("%I:%M %p")
        logger.info(f"Providing current time: {current_time}")
        return f"The current time is {current_time}."

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """
        Get weather information for a location.
        
        Args:
            location: The city or region name
            latitude: The latitude (estimate if not provided by user)
            longitude: The longitude (estimate if not provided by user)
        """
        logger.info(f"Looking up weather for {location}")
        return f"The weather in {location} is sunny with a temperature of 72 degrees Fahrenheit."


server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm resources before agent starts."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entry point for the agent session."""
    # Configure logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # ============================================================================
    # FILLER FILTER CONFIGURATION
    # ============================================================================
    
    # Load configuration from environment variables
    filler_words_env = os.getenv('FILLER_WORDS', 'uh,uhh,um,umm,hmm,er,ah,haan')
    filler_words = set(word.strip().lower() for word in filler_words_env.split(','))
    
    confidence_threshold = float(os.getenv('FILLER_CONFIDENCE_THRESHOLD', '0.6'))
    min_word_count = int(os.getenv('FILLER_MIN_WORD_COUNT', '1'))
    
    # Initialize the filler filter plugin
    filler_filter = FillerFilterPlugin(
        filler_words=filler_words,
        confidence_threshold=confidence_threshold,
        min_word_count=min_word_count,
        case_sensitive=False,
    )
    
    logger.info(
        f"Filler filter configured: {len(filler_words)} words, "
        f"confidence threshold: {confidence_threshold}, "
        f"min words: {min_word_count}"
    )
    
    # ============================================================================
    # AGENT SESSION SETUP
    # ============================================================================
    
    session = AgentSession(
        # Speech-to-text configuration
        stt=os.getenv('STT_MODEL', 'deepgram/nova-3'),
        
        # Language model configuration
        llm=os.getenv('LLM_MODEL', 'openai/gpt-4.1-mini'),
        
        # Text-to-speech configuration
        tts=os.getenv('TTS_MODEL', 'cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc'),
        
        # Turn detection and VAD
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        
        # Interruption handling
        allow_interruptions=True,
        min_interruption_duration=0.5,
        min_interruption_words=1,
        
        # False interruption handling
        # The filler filter works in conjunction with these settings
        resume_false_interruption=True,
        false_interruption_timeout=1.5,
        
        # Preemptive generation for better responsiveness
        preemptive_generation=True,
    )
    
    # ============================================================================
    # ATTACH FILLER FILTER TO SESSION
    # ============================================================================
    
    # This is the key integration step - attach the filter to the session
    # The filter will now monitor agent state and user input events
    filler_filter.attach_to_session(session)
    
    logger.info("✅ Filler filter attached to session and monitoring events")
    
    # ============================================================================
    # METRICS AND LOGGING
    # ============================================================================
    
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """Log metrics as they are collected."""
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        """Log final usage statistics when session ends."""
        summary = usage_collector.get_summary()
        logger.info(f"Session Usage Summary: {summary}")
        
        # Log filler filter statistics
        filter_stats = filler_filter.get_stats()
        logger.info(f"Filler Filter Stats: {filter_stats}")

    ctx.add_shutdown_callback(log_usage)
    
    # ============================================================================
    # OPTIONAL: Additional event handlers to demonstrate filter behavior
    # ============================================================================
    
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev):
        """Log when agent state changes."""
        logger.info(f"🔄 Agent state: {ev.old_state} → {ev.new_state}")
    
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        """Log user input (already handled by filler filter, but shown for demo)."""
        if ev.is_final:
            logger.info(f"💬 User said: '{ev.transcript}'")
    
    # ============================================================================
    # START THE AGENT SESSION
    # ============================================================================
    
    await session.start(
        agent=FillerFilterDemoAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Uncomment to enable noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    # Set up logging to show filter decisions
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # You can set DEBUG level for more detailed filter logs
    # logging.getLogger("livekit.agents.voice.filler_filter").setLevel(logging.DEBUG)
    
    cli.run_app(server)
