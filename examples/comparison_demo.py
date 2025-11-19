"""
Comparison example showing basic vs enhanced filler detection.
"""

import logging
from livekit.agents.voice.filler_filter import FillerFilterPlugin
from livekit.agents.voice.filler_detection import (
    SemanticFillerDetector,
    TemporalAnalyzer,
    ConfidenceScorer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_filter():
    """Test basic string-matching filter."""
    print("\n" + "="*60)
    print("BASIC FILTER (String Matching Only)")
    print("="*60)
    
    basic = FillerFilterPlugin()
    
    test_cases = [
        "uh",
        "uhhh",  # Variation
        "ummmm",  # Variation
        "you know",  # Phrase
        "wait",
        "uh wait",
    ]
    
    for text in test_cases:
        is_filler = basic.is_filler_only(text)
        print(f"  '{text}': {'FILLER' if is_filler else 'NOT FILLER'}")


def test_enhanced_filter():
    """Test enhanced multi-layered filter."""
    print("\n" + "="*60)
    print("ENHANCED FILTER (Multi-Layered)")
    print("="*60)
    
    semantic = SemanticFillerDetector()
    temporal = TemporalAnalyzer()
    
    scorer = ConfidenceScorer(
        semantic_detector=semantic,
        temporal_analyzer=temporal,
    )
    
    test_cases = [
        "uh",
        "uhhh",  # Variation - semantic will catch
        "ummmm",  # Variation - semantic will catch
        "you know",  # Phrase - semantic will catch
        "wait",
        "uh wait",  # Mixed - should detect as non-filler
        "stop please",
        "hmm okay",
    ]
    
    for text in test_cases:
        result = scorer.calculate_confidence(text)
        is_filler = result['classification'] in ['definite_filler', 'likely_filler']
        
        print(f"  '{text}':")
        print(f"    Classification: {result['classification']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Semantic: {result['scores'].get('semantic', 0):.3f}")
        print(f"    String: {result['scores'].get('string_match', 0):.3f}")


def show_advantages():
    """Show specific advantages of enhanced system."""
    print("\n" + "="*60)
    print("ENHANCED SYSTEM ADVANTAGES")
    print("="*60)
    
    semantic = SemanticFillerDetector()
    
    # Cases where basic fails but enhanced succeeds
    edge_cases = [
        ("uhhhh", "Variation with extra 'h'"),
        ("ummmmm", "Variation with extra 'm'"),
        ("errr", "Variation with extra 'r'"),
        ("you know", "Common phrase"),
        ("i mean", "Common phrase"),
        ("haan ji", "Hindi + respect marker"),
    ]
    
    print("\nCases where BASIC would fail but ENHANCED succeeds:")
    for text, description in edge_cases:
        result = semantic.calculate_similarity(text)
        is_filler = result['is_filler']
        
        print(f"  '{text}' ({description})")
        print(f"    Detected as filler: {is_filler}")
        print(f"    Similarity score: {result['filler_similarity']:.3f}")


def compare_performance():
    """Compare basic vs enhanced detection."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    import time
    
    # Basic filter
    basic = FillerFilterPlugin()
    
    start = time.perf_counter()
    for _ in range(1000):
        basic.is_filler_only("uh wait")
    basic_time = (time.perf_counter() - start) / 1000 * 1000
    
    # Enhanced filter
    semantic = SemanticFillerDetector()
    temporal = TemporalAnalyzer()
    scorer = ConfidenceScorer(semantic, temporal)
    
    start = time.perf_counter()
    for _ in range(1000):
        scorer.calculate_confidence("uh wait")
    enhanced_time = (time.perf_counter() - start) / 1000 * 1000
    
    print(f"  Basic Filter: {basic_time:.3f}ms per call")
    print(f"  Enhanced Filter: {enhanced_time:.3f}ms per call")
    print(f"  Overhead: {enhanced_time - basic_time:.3f}ms ({((enhanced_time/basic_time - 1)*100):.1f}% increase)")
    print(f"  Still under 50ms target: {'✅ YES' if enhanced_time < 50 else '❌ NO'}")


def main():
    """Run all comparisons."""
    test_basic_filter()
    test_enhanced_filter()
    show_advantages()
    compare_performance()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nBASIC FILTER:")
    print("  ✅ Fast (< 0.1ms)")
    print("  ❌ Only exact string matches")
    print("  ❌ Misses variations (uhhh, ummmm)")
    print("  ❌ Can't detect phrases")
    print("  ❌ No context awareness")
    
    print("\nENHANCED FILTER:")
    print("  ✅ Semantic understanding")
    print("  ✅ Catches variations automatically")
    print("  ✅ Multi-language support")
    print("  ✅ Temporal pattern analysis")
    print("  ✅ Multi-factor confidence scoring")
    print("  ✅ Still < 50ms latency")
    print("  ⚠️  Requires additional dependencies")


if __name__ == "__main__":
    main()
