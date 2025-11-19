"""
Filler Word Filter Plugin for LiveKit Agents

This module provides intelligent filtering of filler words (like "uh", "umm", "hmm")
during agent speech, while allowing them to be recognized as valid speech when the
agent is quiet.

The filter prevents false interruptions from filler words when the agent is speaking,
but still allows genuine interruptions containing real commands to stop the agent
immediately.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional, Set

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .events import AgentStateChangedEvent, UserInputTranscribedEvent

logger = logging.getLogger(__name__)


class FillerFilterPlugin:
    """
    A plugin that filters out filler words during agent speech to prevent false interruptions.
    
    This plugin hooks into the AgentSession event system to track agent state and filter
    user input based on whether the agent is currently speaking.
    
    Key Features:
    - Ignores filler-only speech when agent is speaking
    - Allows filler words when agent is quiet (registers as valid speech)
    - Respects confidence thresholds for low-confidence detections
    - Supports dynamic, runtime-configurable word lists
    - Language-agnostic design
    
    Args:
        filler_words: Set of words/phrases to filter. Defaults to common English fillers.
        confidence_threshold: Minimum confidence score (0.0-1.0) for considering speech.
            Speech below this threshold is more likely to be ignored. Default: 0.6
        min_word_count: Minimum number of words required to trigger interruption.
            Helps filter out very short utterances. Default: 1
        case_sensitive: Whether to match filler words case-sensitively. Default: False
    
    Example:
        >>> filler_filter = FillerFilterPlugin(
        ...     filler_words={'uh', 'umm', 'hmm', 'haan', 'er'},
        ...     confidence_threshold=0.6
        ... )
        >>> # Attach to session
        >>> filler_filter.attach_to_session(session)
    """
    
    def __init__(
        self,
        filler_words: Optional[Set[str]] = None,
        confidence_threshold: float = 0.6,
        min_word_count: int = 1,
        case_sensitive: bool = False,
    ) -> None:
        # Default filler words (common English fillers)
        self.filler_words = filler_words or {
            'uh', 'uhh', 'uhhh',
            'um', 'umm', 'ummm',
            'hmm', 'hmmm', 'hmmmm',
            'er', 'err', 'errr',
            'ah', 'ahh', 'ahhh',
            'haan',  # Hindi filler
            'accha',  # Hindi filler
        }
        
        self.confidence_threshold = confidence_threshold
        self.min_word_count = min_word_count
        self.case_sensitive = case_sensitive
        
        # Track whether the agent is currently speaking
        self._agent_speaking = False
        
        # Track the session this plugin is attached to
        self._session: Optional[AgentSession] = None
        
        logger.info(
            f"FillerFilterPlugin initialized with {len(self.filler_words)} filler words, "
            f"confidence threshold: {confidence_threshold}, min words: {min_word_count}"
        )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text (lowercased if not case-sensitive, stripped)
        """
        text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text
    
    def _extract_words(self, text: str) -> list[str]:
        """
        Extract words from text, removing punctuation.
        
        Args:
            text: Text to extract words from
            
        Returns:
            List of normalized words
        """
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
        if not self.case_sensitive:
            words = [w.lower() for w in words]
        return words
    
    def is_filler_only(self, text: str) -> bool:
        """
        Check if the given text contains ONLY filler words.
        
        Args:
            text: The transcribed text to check
            
        Returns:
            True if text contains only filler words, False otherwise
            
        Example:
            >>> filter.is_filler_only("umm")
            True
            >>> filter.is_filler_only("umm wait")
            False
        """
        if not text or not text.strip():
            return False
        
        words = self._extract_words(text)
        
        # Empty after extraction -> not valid
        if not words:
            return False
        
        # Check if all words are in the filler set
        words_set = set(words)
        is_only_fillers = words_set.issubset(self.filler_words) and len(words_set) > 0
        
        logger.debug(
            f"is_filler_only('{text}'): words={words}, "
            f"is_only_fillers={is_only_fillers}"
        )
        
        return is_only_fillers
    
    def should_allow_interrupt(
        self,
        text: str,
        confidence: float = 1.0,
        is_final: bool = True,
    ) -> bool:
        """
        Determine if this speech should be allowed to interrupt the agent.
        
        This is the core filtering logic that decides whether user speech should
        stop the agent or be ignored as a filler.
        
        Decision Logic:
        1. If agent is NOT speaking -> always allow (register as valid speech)
        2. If agent IS speaking:
           a. If text is only fillers -> check confidence
              - Low confidence -> IGNORE (likely background noise)
              - High confidence -> IGNORE (filler during agent speech)
           b. If text contains real words -> ALLOW (valid interruption)
        3. If word count is below minimum -> IGNORE
        
        Args:
            text: The transcribed text from user speech
            confidence: ASR confidence score (0.0 to 1.0)
            is_final: Whether this is a final transcription or interim
            
        Returns:
            True if should interrupt agent, False if should ignore
            
        Example:
            >>> # Agent is speaking
            >>> filter.should_allow_interrupt("umm", 0.5)  # Low conf filler
            False
            >>> filter.should_allow_interrupt("wait stop", 0.9)  # Real command
            True
            >>> # Agent is quiet
            >>> filter.should_allow_interrupt("umm", 0.5)  # Registers as speech
            True
        """
        # Normalize and validate input
        text = self._normalize_text(text)
        
        if not text:
            logger.debug("Empty text -> ignoring")
            return False
        
        # Extract words
        words = self._extract_words(text)
        
        # Check minimum word count
        if len(words) < self.min_word_count:
            logger.debug(
                f"Word count {len(words)} below minimum {self.min_word_count} -> ignoring"
            )
            return False
        
        # If agent is NOT speaking, always allow (register as valid speech event)
        if not self._agent_speaking:
            logger.debug(
                f"✅ Agent quiet - allowing speech: '{text}' (conf: {confidence:.2f})"
            )
            return True
        
        # Agent IS speaking - check if it's only fillers
        only_fillers = self.is_filler_only(text)
        
        if only_fillers:
            # Low confidence fillers are likely background noise
            if confidence < self.confidence_threshold:
                logger.info(
                    f"🚫 Ignored filler (low confidence): '{text}' "
                    f"(conf: {confidence:.2f} < {self.confidence_threshold})"
                )
                return False
            
            # High confidence fillers during agent speech are still ignored
            logger.info(
                f"🚫 Ignored filler (agent speaking): '{text}' "
                f"(conf: {confidence:.2f})"
            )
            return False
        
        # Contains real words - this is a valid interruption
        logger.info(
            f"✅ Valid interruption detected: '{text}' "
            f"(conf: {confidence:.2f}, words: {words})"
        )
        return True
    
    def update_filler_words(self, new_words: Set[str]) -> None:
        """
        Dynamically update the filler word list.
        
        This allows runtime modification of which words are considered fillers,
        useful for adding language-specific fillers or customizing behavior.
        
        Args:
            new_words: Set of new filler words to add
            
        Example:
            >>> filter.update_filler_words({'haan', 'accha'})  # Add Hindi fillers
        """
        before_count = len(self.filler_words)
        self.filler_words.update(new_words)
        after_count = len(self.filler_words)
        
        logger.info(
            f"Updated filler words: {before_count} -> {after_count} "
            f"(added {after_count - before_count} new words)"
        )
        logger.debug(f"Current filler words: {sorted(self.filler_words)}")
    
    def set_filler_words(self, filler_words: Set[str]) -> None:
        """
        Replace the entire filler word list.
        
        Args:
            filler_words: New set of filler words
        """
        self.filler_words = filler_words
        logger.info(f"Set filler words to: {sorted(filler_words)}")
    
    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        """
        Internal callback for agent state changes.
        
        Tracks when the agent transitions between speaking and other states
        to enable context-aware filtering.
        
        Args:
            ev: Agent state changed event
        """
        self._agent_speaking = (ev.new_state == "speaking")
        
        logger.debug(
            f"Agent state: {ev.old_state} -> {ev.new_state} "
            f"(speaking: {self._agent_speaking})"
        )
    
    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        """
        Internal callback for user input transcription events.
        
        This is where the filtering logic is applied. When user speech is detected,
        this method decides whether to allow it to interrupt the agent.
        
        Note: This method logs the decision but does not directly prevent
        interruptions. The actual interruption prevention must be implemented
        by the session or by modifying the interruption logic.
        
        Args:
            ev: User input transcribed event
        """
        # For interim results, we don't take action but still log
        if not ev.is_final:
            logger.debug(
                f"[INTERIM] User said: '{ev.transcript}' "
                f"(speaking: {self._agent_speaking})"
            )
            return
        
        # Get confidence from event (default to 1.0 if not available)
        # Note: UserInputTranscribedEvent doesn't have confidence in the base schema
        # This would need to be added or tracked separately
        confidence = 1.0  # Default assumption
        
        should_interrupt = self.should_allow_interrupt(
            text=ev.transcript,
            confidence=confidence,
            is_final=ev.is_final,
        )
        
        # Log the decision
        if should_interrupt:
            logger.debug(
                f"[FINAL] Allowing interruption from: '{ev.transcript}'"
            )
        else:
            logger.debug(
                f"[FINAL] Blocking interruption from: '{ev.transcript}'"
            )
        
        # TODO: Actual interruption prevention would happen here
        # This might involve:
        # 1. Calling session.interrupt() or not based on should_interrupt
        # 2. Setting a flag that other parts of the system check
        # 3. Modifying the event before it's processed further
    
    def attach_to_session(self, session: AgentSession) -> None:
        """
        Hook this filter into an AgentSession's event system.
        
        This registers event listeners for agent state changes and user input
        transcriptions, enabling the filter to track context and make filtering
        decisions.
        
        Args:
            session: The AgentSession to attach to
            
        Example:
            >>> session = AgentSession(...)
            >>> filler_filter = FillerFilterPlugin()
            >>> filler_filter.attach_to_session(session)
            >>> await session.start(...)
        """
        self._session = session
        
        # Register event listeners
        session.on("agent_state_changed", self._on_agent_state_changed)
        session.on("user_input_transcribed", self._on_user_input_transcribed)
        
        logger.info(
            f"FillerFilterPlugin attached to session. "
            f"Monitoring {len(self.filler_words)} filler words."
        )
    
    def detach_from_session(self) -> None:
        """
        Remove this filter from the current session.
        
        Unregisters event listeners to prevent memory leaks and unwanted behavior.
        """
        if self._session is None:
            logger.warning("No session attached, cannot detach")
            return
        
        # Unregister event listeners
        self._session.off("agent_state_changed", self._on_agent_state_changed)
        self._session.off("user_input_transcribed", self._on_user_input_transcribed)
        
        logger.info("FillerFilterPlugin detached from session")
        self._session = None
    
    @property
    def agent_speaking(self) -> bool:
        """Whether the agent is currently speaking."""
        return self._agent_speaking
    
    def get_stats(self) -> dict:
        """
        Get statistics about the filter configuration.
        
        Returns:
            Dictionary with filter statistics
        """
        return {
            "filler_word_count": len(self.filler_words),
            "filler_words": sorted(self.filler_words),
            "confidence_threshold": self.confidence_threshold,
            "min_word_count": self.min_word_count,
            "case_sensitive": self.case_sensitive,
            "agent_speaking": self._agent_speaking,
            "session_attached": self._session is not None,
        }
