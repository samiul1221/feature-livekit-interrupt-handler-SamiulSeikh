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


def _elongation_intensity_for_token(word: str) -> tuple[str, float]:
    """Top-level helper: return (normalized_token, intensity 0.0-1.0) for a single token.

    This matches the logic in the nested helper used by `_get_emotional_context` but
    is lifted to module-level so other methods can reuse the raw numeric score.
    """
    import re
    match = re.search(r'(.)\1{1,}', word)
    if match:
        repetition_count = len(match.group(0))
        base = re.sub(r'(.)\1+', r'\1', word)

        if repetition_count <= 2:
            intensity = 0.2
        elif repetition_count == 3:
            intensity = 0.4
        elif repetition_count == 4:
            intensity = 0.6
        elif repetition_count == 5:
            intensity = 0.8
        else:
            intensity = 1.0
        return base, intensity
    return word, 0.0


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
        room: Optional['rtc.Room'] = None,
    ):
        self.base_stt = base_stt
        self.confidence_scorer = confidence_scorer
        self.agent_state_tracker = agent_state_tracker
        self.filter_interim = filter_interim
        self.filter_final = filter_final
        self.room = room
        
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
    
    def _get_emotional_context(self, text: str) -> str | None:
        """Maps pure filler words to structured context tags with robust real-world handling."""
        cleaned = text.strip().lower().rstrip(".,!?;:")
        words = cleaned.split()
        
        # Analyze elongation intensity with fuzzy thresholds (longer = more intense emotion)
        def get_elongation_intensity(word: str) -> tuple[str, float]:
            """Returns (base_word, intensity_score) using fuzzy logic.
            Intensity score: 0.0 (minimal) to 1.0 (maximum)
            """
            import re
            # Find sequences of repeated characters (2+ repetitions)
            match = re.search(r'(.)\1{1,}', word)
            if match:
                repeated_char = match.group(1)
                repetition_count = len(match.group(0))
                
                # Normalize to base form (reduce to single char for matching)
                # This handles yeahhh -> yeah, ummm -> um, hmm -> hm
                base = re.sub(r'(.)\1+', r'\1', word)
                
                # Fuzzy scoring: gradual intensity curve
                # 2 chars (mm) = 0.2
                # 3 chars (mmm) = 0.4
                # 4 chars (mmmm) = 0.6
                # 5 chars (mmmmm) = 0.8
                # 6+ chars = 1.0
                
                if repetition_count <= 2:
                    intensity = 0.2
                elif repetition_count == 3:
                    intensity = 0.4
                elif repetition_count == 4:
                    intensity = 0.6
                elif repetition_count == 5:
                    intensity = 0.8
                else:
                    intensity = 1.0
                
                return base, intensity
            return word, 0.0
        
        # Single-word fillers
        hesitant_markers = {
            "um", "uh", "er", "uhh", "umm", "ummm", "uhhh", "ehh", "eh",
            "erm", "errm", "ah", "ahh", "ahhh"
        }
        
        thinking_markers = {
            "hmm", "hmmm", "hmmmm", "hm", "mmm", "mmmm"
        }
        
        agreement_markers = {
            "aha", "yeah", "yep", "yup", "right", "ok", "okay", "mhm", "mm-hmm",
            "uh-huh", "sure", "alright", "gotcha", "got it"
        }
        
        disagreement_markers = {
            "nah", "nope", "uh-uh", "mm-mm", "no way"
        }
        
        uncertainty_markers = {
            "i don't know", "i'm not sure", "maybe", "perhaps", "i guess",
            "kinda", "sorta", "sort of", "kind of"
        }
        
        # Multi-word phrases
        thinking_phrases = {
            "let me see", "let me think", "give me a second", "hold on",
            "one moment", "just a moment", "wait a minute", "let's see"
        }
        
        # Check exact phrase match first (for multi-word)
        if cleaned in thinking_phrases or cleaned in uncertainty_markers:
            return "<user_state>thinking</user_state>"
        
        # Single word or very short utterances
        if len(words) == 1:
            word = words[0]
            base_word, intensity = get_elongation_intensity(word)
            
            if base_word in hesitant_markers or word in hesitant_markers:
                # Fuzzy thresholds for hesitation
                if intensity >= 0.7:  # 6+ chars: very hesitant
                    return "<user_state>very_hesitant</user_state>"
                elif intensity >= 0.4:  # 4-5 chars: quite hesitant
                    return "<user_state>quite_hesitant</user_state>"
                else:  # 2-3 chars: mild hesitation
                    return "<user_state>hesitant</user_state>"
            
            elif base_word in thinking_markers or word in thinking_markers:
                # Fuzzy thresholds for thinking (hmm is introspective by nature)
                if intensity >= 0.75:  # 6+ chars: deeply thinking
                    return "<user_state>deeply_thinking</user_state>"
                elif intensity >= 0.5:  # 5+ chars: pondering
                    return "<user_state>pondering</user_state>"
                elif intensity >= 0.2:  # 3+ chars: active thinking
                    return "<user_state>thinking</user_state>"
                else:  # 2 chars: brief acknowledgment/thinking
                    return "<user_state>considering</user_state>"
            
            elif base_word in agreement_markers or word in agreement_markers:
                if intensity >= 0.6:  # 5+ chars: strong agreement
                    return "<user_state>strongly_agrees</user_state>"
                return "<user_state>agrees</user_state>"
            
            elif base_word in disagreement_markers or word in disagreement_markers:
                return "<user_state>disagrees</user_state>"
        
        # Two-word combinations (common real-world patterns)
        if len(words) == 2:
            phrase = " ".join(words)
            if phrase in thinking_phrases or phrase in uncertainty_markers:
                return "<user_state>thinking</user_state>"
            
            # Analyze elongation in compound patterns with fuzzy logic
            word0_base, intensity0 = get_elongation_intensity(words[0])
            word1_base, intensity1 = get_elongation_intensity(words[1])
            
            # "um yeah" / "uh okay" -> still agreement but hesitant
            if (word0_base in hesitant_markers or words[0] in hesitant_markers) and \
               (word1_base in agreement_markers or words[1] in agreement_markers):
                if intensity0 >= 0.5:  # significant hesitation
                    return "<user_state>very_hesitant_agreement</user_state>"
                return "<user_state>hesitant_agreement</user_state>"
            
            if (word0_base in hesitant_markers or words[0] in hesitant_markers) and \
               (word1_base in disagreement_markers or words[1] in disagreement_markers):
                return "<user_state>hesitant_disagreement</user_state>"
        
        # Three-word combinations
        if len(words) <= 4:
            phrase = " ".join(words)
            if phrase in thinking_phrases or phrase in uncertainty_markers:
                return "<user_state>thinking</user_state>"
        
        return None

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
    
    async def _send_suppression_event(self, text: str, reason: str):
        """Send a custom room event when a filler is suppressed."""
        if self.room and self.room.local_participant:
            try:
                import json
                payload = json.dumps({
                    "type": "filler_suppressed",
                    "text": text,
                    "reason": reason
                })
                # Publish data to the room so the frontend can display "Thinking..." or similar
                await self.room.local_participant.publish_data(
                    payload,
                    topic="filler_detection",
                    reliable=True
                )
            except Exception as e:
                logger.warning(f"Failed to send suppression event: {e}")

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
                
                # Check for emotional context
                context_tag = self._get_emotional_context(text)
                
                if should_suppress:
                    should_filter = True
                    filter_reason = result.get('reason', 'full_pipeline_filter')
                    
                    # Smart Override: If we have a high-intensity emotional signal, 
                    # we forward it even if it's a filler so the LLM can react.
                    high_intensity_tags = {
                        "<user_state>very_hesitant</user_state>",
                        "<user_state>quite_hesitant</user_state>",
                        "<user_state>deeply_thinking</user_state>",
                        "<user_state>pondering</user_state>",
                        "<user_state>strongly_agrees</user_state>",
                        "<user_state>hesitant_agreement</user_state>",
                        "<user_state>hesitant_disagreement</user_state>"
                    }
                    
                    if context_tag and context_tag in high_intensity_tags:
                        should_filter = False
                        logger.info(f"⚠️ Suppression overridden by high-intensity context: {context_tag}")
                    
                    if should_filter:
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
                
                # Send visual feedback event
                await self._send_suppression_event(text, filter_reason)
                
                continue  # Don't yield this event
            else:
                # Paralinguistic Analysis: Inject context tags for pure fillers when agent is listening
                if not self.agent_speaking and event.is_final and event.alternatives:
                    context_tag = self._get_emotional_context(text)
                    # compute numeric intensity for the phrase
                    words = text.strip().lower().rstrip(".,!?;:").split()
                    max_intensity = 0.0
                    for w in words:
                        _, score = _elongation_intensity_for_token(w)
                        if score > max_intensity:
                            max_intensity = score

                    if context_tag:
                        original_text = event.alternatives[0].text

                        # Compose an explanatory score tag for the LLM. This gives both the
                        # paralinguistic intensity and the ASR confidence so the LLM can reason
                        # about numeric values.
                        score_tag = (
                            f"<paralinguistic score=\"{max_intensity:.2f}\" "
                            f"asr_conf=\"{confidence:.2f}\">"
                            f"score=filler_elongation(0..1); asr_confidence(0..1)"
                            f"</paralinguistic>"
                        )

                        event.alternatives[0].text = f"{original_text} {context_tag} {score_tag}"
                        logger.info(f"Paralinguistic injection: '{original_text}' -> '{event.alternatives[0].text}'")

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
