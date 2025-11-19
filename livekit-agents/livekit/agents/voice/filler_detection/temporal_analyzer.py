"""
Temporal pattern analysis for distinguishing fillers from genuine speech.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

logger = logging.getLogger(__name__)


@dataclass
class SpeechEvent:
    """Represents a single speech event for temporal analysis."""
    text: str
    timestamp: float
    word_count: int
    confidence: float


class TemporalAnalyzer:
    """
    Analyze temporal patterns to detect fillers.
    Tracks speech rate, repetition, and timing patterns.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        window_duration: float = 3.0,
        rapid_threshold: float = 2.0,
        sustained_threshold: float = 3.0,
        repetition_threshold: int = 3,
    ):
        self.window_size = window_size
        self.window_duration = window_duration
        self.rapid_threshold = rapid_threshold  # words/sec for fillers
        self.sustained_threshold = sustained_threshold  # words/sec for genuine
        self.repetition_threshold = repetition_threshold
        
        self.events: Deque[SpeechEvent] = deque(maxlen=window_size)
    
    def add_event(self, text: str, confidence: float = 1.0):
        """Add a speech event to the temporal window."""
        words = text.strip().split()
        event = SpeechEvent(
            text=text,
            timestamp=time.time(),
            word_count=len(words),
            confidence=confidence
        )
        self.events.append(event)
    
    def _clean_old_events(self):
        """Remove events outside the time window."""
        current_time = time.time()
        while self.events and (current_time - self.events[0].timestamp) > self.window_duration:
            self.events.popleft()
    
    def calculate_speech_rate(self) -> float:
        """Calculate words per second in the current window."""
        self._clean_old_events()
        
        if not self.events:
            return 0.0
        
        total_words = sum(e.word_count for e in self.events)
        time_span = self.events[-1].timestamp - self.events[0].timestamp
        
        if time_span == 0:
            return 0.0
        
        return total_words / time_span
    
    def detect_repetition(self, text: str) -> bool:
        """Detect if the same word is being repeated rapidly."""
        self._clean_old_events()
        
        text_lower = text.lower().strip()
        
        # Count occurrences of this text in recent events
        recent_count = sum(
            1 for e in self.events
            if e.text.lower().strip() == text_lower
        )
        
        return recent_count >= self.repetition_threshold
    
    def detect_rapid_single_words(self) -> bool:
        """Detect pattern of rapid single-word utterances (filler pattern)."""
        self._clean_old_events()
        
        if len(self.events) < 3:
            return False
        
        # Check if most recent events are single words
        recent_events = list(self.events)[-5:]
        single_word_events = sum(1 for e in recent_events if e.word_count == 1)
        
        if single_word_events < 3:
            return False
        
        # Check if speech rate is low (characteristic of fillers)
        speech_rate = self.calculate_speech_rate()
        return speech_rate < self.rapid_threshold
    
    def detect_sustained_speech(self) -> bool:
        """Detect sustained speech pattern (genuine interruption)."""
        self._clean_old_events()
        
        if len(self.events) < 2:
            return False
        
        # Check total word count in window
        total_words = sum(e.word_count for e in self.events)
        
        if total_words < 5:
            return False
        
        # Check if speech rate is high (characteristic of genuine speech)
        speech_rate = self.calculate_speech_rate()
        return speech_rate > self.sustained_threshold
    
    def analyze(self, text: str, confidence: float = 1.0) -> dict:
        """
        Perform comprehensive temporal analysis.
        
        Returns:
            Dict with analysis results and is_filler classification
        """
        # Add current event
        self.add_event(text, confidence)
        
        # Perform all analyses
        speech_rate = self.calculate_speech_rate()
        is_repetitive = self.detect_repetition(text)
        is_rapid_single = self.detect_rapid_single_words()
        is_sustained = self.detect_sustained_speech()
        
        # Filler indicators:
        # - Repetitive patterns
        # - Rapid single words
        # - Low speech rate
        # - NOT sustained speech
        
        filler_score = 0.0
        
        if is_repetitive:
            filler_score += 0.4
        
        if is_rapid_single:
            filler_score += 0.3
        
        if speech_rate < self.rapid_threshold:
            filler_score += 0.2
        
        if is_sustained:
            filler_score -= 0.5  # Strong indicator of genuine speech
        
        is_filler = filler_score > 0.5 and not is_sustained
        
        return {
            'is_filler': is_filler,
            'filler_score': max(0.0, min(1.0, filler_score)),
            'speech_rate': speech_rate,
            'is_repetitive': is_repetitive,
            'is_rapid_single': is_rapid_single,
            'is_sustained': is_sustained,
            'event_count': len(self.events),
        }
    
    def reset(self):
        """Clear the event window."""
        self.events.clear()
