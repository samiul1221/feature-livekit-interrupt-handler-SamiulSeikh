"""
Performance benchmark for enhanced filler detection system.
Tests latency and accuracy of all detection components.
"""

import time
import numpy as np
from livekit.agents.voice.filler_detection import (
    SemanticFillerDetector,
    TemporalAnalyzer,
    AudioFeatureFilter,
    ConfidenceScorer,
)


def benchmark_semantic_detector():
    """Benchmark semantic detection latency."""
    detector = SemanticFillerDetector()
    
    test_cases = [
        "uh",
        "umm wait",
        "stop please",
        "you know what I mean",
        "this is a longer sentence with multiple words to test",
    ]
    
    print("\n=== Semantic Detector Benchmark ===")
    
    for text in test_cases:
        start = time.perf_counter()
        for _ in range(100):
            result = detector.calculate_similarity(text)
        end = time.perf_counter()
        
        avg_latency = (end - start) / 100 * 1000
        print(f"  '{text[:30]}...': {avg_latency:.3f}ms per call")
        print(f"    Result: filler={result['is_filler']}, score={result['filler_similarity']:.3f}")


def benchmark_temporal_analyzer():
    """Benchmark temporal analysis latency."""
    analyzer = TemporalAnalyzer()
    
    # Warm up with some events
    for i in range(5):
        analyzer.add_event(f"test {i}")
    
    print("\n=== Temporal Analyzer Benchmark ===")
    
    start = time.perf_counter()
    for _ in range(1000):
        analyzer.analyze("uh")
    end = time.perf_counter()
    
    avg_latency = (end - start) / 1000 * 1000
    print(f"  Analyze: {avg_latency:.3f}ms per call")


def benchmark_audio_features():
    """Benchmark audio feature extraction."""
    audio_filter = AudioFeatureFilter()
    
    if not audio_filter.enabled:
        print("\n=== Audio Features: DISABLED (librosa not available) ===")
        return
    
    print("\n=== Audio Feature Filter Benchmark ===")
    
    # Generate test audio (1 second)
    audio_data = np.random.randn(16000).astype(np.float32)
    
    start = time.perf_counter()
    for _ in range(10):  # Fewer iterations as this is slower
        is_filler, features = audio_filter.is_filler_audio(audio_data)
    end = time.perf_counter()
    
    avg_latency = (end - start) / 10 * 1000
    print(f"  Feature extraction: {avg_latency:.3f}ms per call")


def benchmark_confidence_scorer():
    """Benchmark multi-factor confidence scoring."""
    semantic = SemanticFillerDetector()
    temporal = TemporalAnalyzer()
    audio = AudioFeatureFilter()
    
    scorer = ConfidenceScorer(
        semantic_detector=semantic,
        temporal_analyzer=temporal,
        audio_filter=audio,
    )
    
    print("\n=== Confidence Scorer Benchmark ===")
    
    test_cases = [
        "uh",
        "wait",
        "umm stop",
    ]
    
    for text in test_cases:
        start = time.perf_counter()
        for _ in range(100):
            result = scorer.calculate_confidence(text)
        end = time.perf_counter()
        
        avg_latency = (end - start) / 100 * 1000
        print(f"  '{text}': {avg_latency:.3f}ms per call")
        print(f"    Confidence: {result['confidence']:.3f}, Class: {result['classification']}")


def benchmark_end_to_end():
    """Benchmark complete pipeline."""
    semantic = SemanticFillerDetector()
    temporal = TemporalAnalyzer()
    
    scorer = ConfidenceScorer(
        semantic_detector=semantic,
        temporal_analyzer=temporal,
    )
    
    print("\n=== End-to-End Pipeline Benchmark ===")
    
    test_cases = [
        ("uh", True),  # Filler during speech
        ("stop", True),  # Command during speech
        ("uh", False),  # Filler when quiet
    ]
    
    for text, agent_speaking in test_cases:
        start = time.perf_counter()
        for _ in range(100):
            should_suppress, result = scorer.should_suppress(
                text=text,
                agent_speaking=agent_speaking,
            )
        end = time.perf_counter()
        
        avg_latency = (end - start) / 100 * 1000
        print(f"  '{text}' (agent_speaking={agent_speaking}): {avg_latency:.3f}ms")
        print(f"    Suppress: {should_suppress}, Confidence: {result.get('confidence', 0):.3f}")


def test_accuracy():
    """Test detection accuracy on known cases."""
    semantic = SemanticFillerDetector()
    temporal = TemporalAnalyzer()
    
    scorer = ConfidenceScorer(
        semantic_detector=semantic,
        temporal_analyzer=temporal,
    )
    
    print("\n=== Accuracy Test ===")
    
    # Test cases: (text, expected_is_filler, agent_speaking)
    test_cases = [
        ("uh", True, True),
        ("umm", True, True),
        ("hmm", True, True),
        ("stop", False, True),
        ("wait", False, True),
        ("hold on", False, True),
        ("uh wait", False, True),  # Mixed, should not suppress
        ("you know", True, True),
        ("this is a real sentence", False, True),
        ("uh", False, False),  # Filler when quiet should not suppress
    ]
    
    correct = 0
    total = len(test_cases)
    
    for text, expected_is_filler, agent_speaking in test_cases:
        should_suppress, result = scorer.should_suppress(
            text=text,
            agent_speaking=agent_speaking,
        )
        
        expected_suppress = expected_is_filler and agent_speaking
        is_correct = should_suppress == expected_suppress
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"  {status} '{text}' (speaking={agent_speaking}): "
              f"expected={expected_suppress}, got={should_suppress}, "
              f"conf={result.get('confidence', 0):.3f}")
    
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("ENHANCED FILLER DETECTION - PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    benchmark_semantic_detector()
    benchmark_temporal_analyzer()
    benchmark_audio_features()
    benchmark_confidence_scorer()
    benchmark_end_to_end()
    test_accuracy()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nTarget: < 50ms total latency")
    print("Target: > 95% accuracy")


if __name__ == "__main__":
    main()
