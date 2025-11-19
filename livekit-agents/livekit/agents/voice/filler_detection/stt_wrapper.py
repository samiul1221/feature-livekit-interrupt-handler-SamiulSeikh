"""
STT wrapper that intercepts and filters transcription events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator, Optional

if TYPE_CHECKING:
    from .confidence_scorer import ConfidenceScorer
    from livekit import rtc
    from livekit.agents import stt

logger = logging.getLogger(__name__)


class FillerAwareSTT:
    """
    Wraps a base STT plugin to filter filler words at the pipeline level.
    Intercepts both interim and final transcripts before they reach the session.
    """
    
    def __init__(
        self,
        base_stt,
        confidence_scorer: 'ConfidenceScorer',
        agent_state_tracker=None,
        filter_interim: bool = True,
        filter_final: bool = True,
    ):
        self.base_stt = base_stt
        self.confidence_scorer = confidence_scorer
        self.agent_state_tracker = agent_state_tracker
        self.filter_interim = filter_interim
        self.filter_final = filter_final
        
        self._stats = {
            'total_events': 0,
            'filtered_events': 0,
            'forwarded_events': 0,
            'interim_filtered': 0,
            'final_filtered': 0,
        }
    
    @property
    def agent_speaking(self) -> bool:
        """Check if agent is currently speaking."""
        if self.agent_state_tracker is None:
            return False
        return getattr(self.agent_state_tracker, 'agent_speaking', False)
    
    async def _quick_filler_check(self, text: str) -> bool:
        """Quick string-based filler check for interim results."""
        if not text or not text.strip():
            return False
        
        # Use simple string matching for speed
        normalized = text.lower().strip()
        words = set(normalized.split())
        
        filler_words = self.confidence_scorer.filler_words
        
        # Check if all words are known fillers
        return bool(words) and words.issubset(filler_words)
    
    async def _full_filter_check(
        self,
        text: str,
        confidence: float = 1.0,
    ) -> tuple[bool, dict]:
        """Full pipeline check using all detectors."""
        should_suppress, result = self.confidence_scorer.should_suppress(
            text=text,
            agent_speaking=self.agent_speaking,
            asr_confidence=confidence,
        )
        return should_suppress, result
    
    async def recognize(
        self,
        *,
        buffer: 'rtc.AudioStream',
        language: Optional[str] = None,
    ) -> 'stt.SpeechEvent':
        """
        Recognize speech from audio stream with filtering.
        Intercepts events from base STT and filters before returning.
        """
        async for event in self.stream(buffer=buffer, language=language):
            # For the recognize() method, we want the final result
            # So we keep yielding until we get a final event that passes filter
            if event.is_final:
                return event
    
    async def stream(
        self,
        *,
        buffer: 'rtc.AudioStream',
        language: Optional[str] = None,
    ) -> AsyncIterator['stt.SpeechEvent']:
        """
        Stream recognition results with real-time filtering.
        """
        # Get events from base STT
        async for event in self.base_stt.stream(buffer=buffer, language=language):
            self._stats['total_events'] += 1
            
            # Extract text and confidence
            text = event.alternatives[0].text if event.alternatives else ""
            confidence = event.alternatives[0].confidence if event.alternatives else 1.0
            
            # Apply filtering based on event type
            should_filter = False
            filter_reason = None
            
            # INTERIM events - quick filtering
            if not event.is_final and self.filter_interim:
                if self.agent_speaking:
                    is_filler = await self._quick_filler_check(text)
                    if is_filler:
                        should_filter = True
                        filter_reason = 'interim_filler_quick_check'
                        self._stats['interim_filtered'] += 1
            
            # FINAL events - full pipeline filtering
            elif event.is_final and self.filter_final:
                should_suppress, result = await self._full_filter_check(text, confidence)
                
                if should_suppress:
                    should_filter = True
                    filter_reason = result.get('reason', 'full_pipeline_filter')
                    self._stats['final_filtered'] += 1
                    
                    logger.info(
                        f"🚫 Filtered: '{text}' "
                        f"(conf: {result['confidence']:.2f}, "
                        f"class: {result['classification']})"
                    )
                else:
                    logger.debug(
                        f"✅ Forwarded: '{text}' "
                        f"(conf: {result['confidence']:.2f})"
                    )
            
            # Forward or suppress event
            if should_filter:
                self._stats['filtered_events'] += 1
                logger.debug(f"Filtered event: '{text}' (reason: {filter_reason})")
                continue  # Don't yield this event
            else:
                self._stats['forwarded_events'] += 1
                yield event
    
    def get_stats(self) -> dict:
        """Get filtering statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            'total_events': 0,
            'filtered_events': 0,
            'forwarded_events': 0,
            'interim_filtered': 0,
            'final_filtered': 0,
        }
    
    # Delegate other methods to base STT
    def __getattr__(self, name):
        """Delegate unknown attributes to base STT."""
        return getattr(self.base_stt, name)
