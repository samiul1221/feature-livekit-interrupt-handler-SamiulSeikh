"""
Unit tests for enhanced filler detection system.
"""

import pytest
import numpy as np
from livekit.agents.voice.filler_detection import (
    SemanticFillerDetector,
    TemporalAnalyzer,
    AudioFeatureFilter,
    ConfidenceScorer,
)


class TestSemanticDetector:
    """Test semantic embedding detection."""
    
    @pytest.fixture
    def detector(self):
        return SemanticFillerDetector()
    
    def test_detect_single_filler(self, detector):
        assert detector.detect("uh") is True
        assert detector.detect("umm") is True
    
    def test_detect_command(self, detector):
        assert detector.detect("stop") is False
        assert detector.detect("wait") is False
    
    def test_similarity_scores(self, detector):
        result = detector.calculate_similarity("ummm")
        assert result['filler_similarity'] > 0.7
        assert result['filler_similarity'] > result['command_similarity']


class TestTemporalAnalyzer:
    """Test temporal pattern analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return TemporalAnalyzer()
    
    def test_repetition_detection(self, analyzer):
        # Add same word multiple times
        for _ in range(3):
            analyzer.add_event("uh")
        
        assert analyzer.detect_repetition("uh") is True
    
    def test_speech_rate_calculation(self, analyzer):
        analyzer.add_event("hello world test")
        rate = analyzer.calculate_speech_rate()
        assert rate >= 0.0
    
    def test_rapid_single_words(self, analyzer):
        # Add multiple single-word events
        for word in ["uh", "um", "hmm"]:
            analyzer.add_event(word)
        
        result = analyzer.analyze("uh")
        assert result['is_rapid_single'] is True


class TestAudioFeatureFilter:
    """Test audio feature extraction."""
    
    @pytest.fixture
    def audio_filter(self):
        return AudioFeatureFilter()
    
    def test_extract_features(self, audio_filter):
        # Create dummy audio (1 second at 16kHz)
        audio_data = np.random.randn(16000).astype(np.float32)
        features = audio_filter.extract_features(audio_data)
        
        if features.get('enabled'):
            assert 'duration' in features
            assert 'mean_energy' in features
    
    def test_filler_detection(self, audio_filter):
        # Short, low-energy audio (filler characteristics)
        short_audio = np.random.randn(4000).astype(np.float32) * 0.05
        is_filler, features = audio_filter.is_filler_audio(short_audio)
        
        if features.get('enabled'):
            assert 'filler_score' in features


class TestConfidenceScorer:
    """Test multi-factor confidence scoring."""
    
    @pytest.fixture
    def scorer(self):
        semantic = SemanticFillerDetector()
        temporal = TemporalAnalyzer()
        return ConfidenceScorer(
            semantic_detector=semantic,
            temporal_analyzer=temporal,
        )
    
    def test_string_match_score(self, scorer):
        score = scorer._string_match_score("uh um")
        assert score > 0.5
    
    def test_calculate_confidence(self, scorer):
        result = scorer.calculate_confidence("uh")
        
        assert 'confidence' in result
        assert 'classification' in result
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_should_suppress_filler(self, scorer):
        should_suppress, result = scorer.should_suppress(
            text="uh",
            agent_speaking=True,
        )
        
        assert should_suppress is True
    
    def test_should_not_suppress_command(self, scorer):
        should_suppress, result = scorer.should_suppress(
            text="stop now",
            agent_speaking=True,
        )
        
        assert should_suppress is False
    
    def test_never_suppress_when_agent_quiet(self, scorer):
        should_suppress, result = scorer.should_suppress(
            text="uh",
            agent_speaking=False,
        )
        
        assert should_suppress is False


@pytest.mark.asyncio
class TestSTTWrapper:
    """Test STT wrapper integration."""
    
    async def test_quick_filler_check(self):
        # This would require mocking the STT and session
        # Placeholder for integration tests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
