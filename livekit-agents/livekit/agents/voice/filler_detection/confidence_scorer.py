"""
Multi-factor confidence scoring for filler detection.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Combine multiple detection signals into a unified confidence score.
    
    Weighted factors:
    - Semantic similarity: 40%
    - String matching: 20%
    - Temporal patterns: 20%
    - ASR confidence: 15%
    - Audio features: 5%
    """
    
    WEIGHTS = {
        'semantic': 0.40,
        'string_match': 0.20,
        'temporal': 0.20,
        'asr_confidence': 0.15,
        'audio': 0.05,
    }
    
    THRESHOLD_DEFINITE = 0.80  # Definitely filler
    THRESHOLD_LIKELY = 0.50    # Likely filler
    
    # Ambiguous fillers that require stricter context checking
    AMBIGUOUS_FILLERS = {'you know', 'like', 'i mean', 'actually', 'basically'}

    def __init__(
        self,
        semantic_detector=None,
        temporal_analyzer=None,
        audio_filter=None,
        filler_words: Optional[set] = None,
    ):
        self.semantic_detector = semantic_detector
        self.temporal_analyzer = temporal_analyzer
        self.audio_filter = audio_filter
        self.filler_words = filler_words or {
            'uh', 'uhh', 'um', 'umm', 'hmm', 'er', 'ah',
            'haan', 'accha'
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for string matching."""
        if not text:
            return text

        # Collapse excessive repeated characters: 'hmmmmmm' -> 'hmm'
        # and 'ummmmmm' -> 'umm'. Keeps repeated form to match filler tokens like 'umm'.
        def _repl(m):
            ch = m.group(1)
            # Keep at most 3 characters of the same char
            return ch * 3

        normalized = re.sub(r"(.)\1{2,}", _repl, text.lower().strip())
        return normalized
    
    def _string_match_score(self, text: str) -> float:
        """Calculate string matching score (0.0-1.0)."""
        normalized = self._normalize_text(text)
        words = set(normalized.split())
        
        # Check if all words are fillers
        if words and words.issubset(self.filler_words):
            return 1.0
        
        # Check if any words are fillers
        filler_count = sum(1 for w in words if w in self.filler_words)
        if not words:
            return 0.0
        
        # Partial score based on ratio
        return filler_count / len(words)
    
    def calculate_confidence(
        self,
        text: str,
        asr_confidence: float = 1.0,
        audio_data=None,
    ) -> Dict[str, float]:
        """
        Calculate overall filler confidence from all available signals.
        
        Args:
            text: Transcribed text
            asr_confidence: ASR confidence score (0.0-1.0)
            audio_data: Optional raw audio data for audio analysis
            
        Returns:
            Dict with scores from each component and final confidence
        """
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        # 1. Semantic similarity (40%)
        if self.semantic_detector is not None:
            semantic_result = self.semantic_detector.calculate_similarity(text)
            semantic_score = semantic_result['filler_similarity']
            scores['semantic'] = semantic_score
            weighted_sum += semantic_score * self.WEIGHTS['semantic']
            total_weight += self.WEIGHTS['semantic']
        else:
            scores['semantic'] = None
        
        # 2. String matching (20%)
        string_score = self._string_match_score(text)
        scores['string_match'] = string_score
        weighted_sum += string_score * self.WEIGHTS['string_match']
        total_weight += self.WEIGHTS['string_match']
        
        # 3. Temporal patterns (20%)
        if self.temporal_analyzer is not None:
            temporal_result = self.temporal_analyzer.analyze(text, asr_confidence)
            temporal_score = temporal_result['filler_score']
            scores['temporal'] = temporal_score
            scores['temporal_details'] = temporal_result
            weighted_sum += temporal_score * self.WEIGHTS['temporal']
            total_weight += self.WEIGHTS['temporal']
        else:
            scores['temporal'] = None
        
        # 4. ASR confidence (15%)
        # Lower ASR confidence increases filler likelihood
        asr_score = 1.0 - asr_confidence
        scores['asr_confidence'] = asr_score
        weighted_sum += asr_score * self.WEIGHTS['asr_confidence']
        total_weight += self.WEIGHTS['asr_confidence']
        
        # 5. Audio features (5%)
        if self.audio_filter is not None and audio_data is not None:
            is_filler_audio, audio_features = self.audio_filter.is_filler_audio(audio_data)
            audio_score = audio_features.get('filler_score', 0.0)
            scores['audio'] = audio_score
            scores['audio_details'] = audio_features
            weighted_sum += audio_score * self.WEIGHTS['audio']
            total_weight += self.WEIGHTS['audio']
        else:
            scores['audio'] = None
        
        # --- Contextual Disambiguation for Ambiguous Fillers ---
        # If the text contains ambiguous fillers (e.g., "you know") but also contains
        # other non-filler words, we should significantly reduce the confidence.
        normalized_text = self._normalize_text(text)
        words = set(normalized_text.split())
        
        # Check if any ambiguous filler is present
        has_ambiguous = any(amb in normalized_text for amb in self.AMBIGUOUS_FILLERS)
        
        if has_ambiguous:
            # Count non-filler words
            # We consider words that are NOT in the strict filler list AND not ambiguous fillers
            non_filler_count = 0
            for w in words:
                if w not in self.filler_words and w not in self.AMBIGUOUS_FILLERS:
                    # Also check if 'w' is part of a multi-word ambiguous filler
                    # This is a simple heuristic; for "you know", 'you' and 'know' might be counted.
                    # Better: check if the phrase is EXACTLY the ambiguous filler.
                    non_filler_count += 1
            
            # If the text is NOT just the ambiguous filler (or other fillers), penalize
            # Example: "you know" -> penalty 0 (it IS the filler)
            # Example: "do you know" -> penalty applied
            
            # Check if the text is EXACTLY one of the ambiguous fillers (or a combination of fillers)
            is_pure_filler = False
            if normalized_text in self.AMBIGUOUS_FILLERS:
                is_pure_filler = True
            elif words.issubset(self.filler_words.union(self.AMBIGUOUS_FILLERS)):
                 # It's all fillers, e.g. "um you know"
                 is_pure_filler = True
            
            if not is_pure_filler:
                # It has other content. Penalize heavily.
                # We multiply the weighted sum by a penalty factor (e.g., 0.2)
                # effectively saying "this is likely NOT a filler".
                penalty_factor = 0.2
                weighted_sum *= penalty_factor
                scores['ambiguity_penalty'] = penalty_factor
                logger.debug(f"Applied ambiguity penalty to '{text}' (factor {penalty_factor})")

        # Calculate final confidence (normalize by actual total weight)
        final_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Determine classification
        if final_confidence > self.THRESHOLD_DEFINITE:
            classification = 'definite_filler'
        elif final_confidence > self.THRESHOLD_LIKELY:
            classification = 'likely_filler'
        else:
            classification = 'genuine_speech'
        
        return {
            'confidence': final_confidence,
            'classification': classification,
            'scores': scores,
            'text': text,
        }
    
    def should_suppress(
        self,
        text: str,
        agent_speaking: bool,
        asr_confidence: float = 1.0,
        audio_data=None,
    ) -> tuple[bool, dict]:
        """
        Determine if speech should be suppressed.
        
        Args:
            text: Transcribed text
            agent_speaking: Whether agent is currently speaking
            asr_confidence: ASR confidence score
            audio_data: Optional audio data
            
        Returns:
            Tuple of (should_suppress, confidence_dict)
        """
        # Never suppress if agent is not speaking
        if not agent_speaking:
            return False, {'reason': 'agent_not_speaking'}
        
        # Calculate confidence
        result = self.calculate_confidence(text, asr_confidence, audio_data)
        
        # Suppression logic based on confidence
        confidence = result['confidence']
        classification = result['classification']
        
        # Definite filler -> always suppress
        if classification == 'definite_filler':
            should_suppress = True
            result['reason'] = 'definite_filler'
        
        # Likely filler -> suppress if agent speaking
        elif classification == 'likely_filler':
            should_suppress = True
            result['reason'] = 'likely_filler_during_speech'
        
        # Genuine speech -> never suppress
        else:
            should_suppress = False
            result['reason'] = 'genuine_speech'
        
        return should_suppress, result
