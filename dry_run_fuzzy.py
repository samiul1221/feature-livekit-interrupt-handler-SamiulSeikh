import asyncio
import logging
import sys
import os

# Add the local package to the path to avoid import errors
sys.path.append(os.path.join(os.getcwd(), "livekit-agents"))

# Mock the dependencies to avoid loading heavy libraries like sentence-transformers/numpy/pandas
# which are causing binary incompatibility issues in this environment
import sys
from unittest.mock import MagicMock

# Mock modules that might cause issues
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["livekit"] = MagicMock()
sys.modules["livekit.agents"] = MagicMock()
sys.modules["livekit.agents.stt"] = MagicMock()

# Now import the wrapper - we need to be careful about imports inside the wrapper
# We'll manually load the wrapper class to avoid importing the whole package tree
# which triggers the heavy imports

# Define the class directly for testing logic if import fails, 
# OR try to import just the wrapper if possible.
# Given the traceback, the issue is deep in the dependency tree (pandas/numpy).
# So we will extract the logic we want to test into this script to verify it 
# independently of the environment issues.

print("=== Extracting Logic for Dry Run (bypassing env issues) ===")

def get_elongation_intensity(word: str) -> tuple[str, float]:
    """Top-level helper returning (base_word, intensity) for a single token.

    We keep the same mapping as the nested helper in `_get_emotional_context`.
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

def _get_emotional_context(text: str) -> str | None:
    """Maps pure filler words to structured context tags with robust real-world handling."""
    cleaned = text.strip().lower().rstrip(".,!?;:")
    words = cleaned.split()
    
    # Use helper get_elongation_intensity defined at module scope
    
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

async def run_dry_run():
    print("=== Starting Fuzzy Logic Paralinguistic Pipeline Dry Run ===")
    
    # Test cases covering various scenarios
    test_cases = [
        # Hesitation intensity
        "um",           # mild
        "ummm",         # moderate
        "ummmmmm",      # intense
        "uh",           # mild
        "uhhhhh",       # intense
        
        # Thinking intensity
        "hm",           # considering
        "hmm",          # thinking
        "hmmm",         # pondering
        "hmmmmm",       # deeply thinking
        
        # Agreement intensity
        "yeah",         # agree
        "yeahhh",       # strong agree
        "ok",           # agree
        
        # Combinations
        "um yeah",      # hesitant agreement
        "ummm yeah",    # very hesitant agreement
        "uh nope",      # hesitant disagreement
        
        # Phrases
        "let me see",   # thinking
        "i don't know", # thinking/uncertainty
        
        # Non-fillers (should return None)
        "hello world",
        "what time is it"
    ]
    
    print(f"\n{'INPUT':<20} | {'OUTPUT TAG':<40}")
    print("-" * 65)
    
    for text in test_cases:
        tag = _get_emotional_context(text)
        # compute max intensity across words for debug/visibility
        words = text.strip().lower().rstrip(".,!?;:").split()
        max_intensity = 0.0
        for w in words:
            _, intensity = get_elongation_intensity(w)
            if intensity > max_intensity:
                max_intensity = intensity

        # Simulate an ASR confidence (mock) for debug clarity
        if max_intensity >= 0.8:
            asr_confidence = 0.85
        elif max_intensity >= 0.6:
            asr_confidence = 0.90
        elif max_intensity > 0.0:
            asr_confidence = 0.92
        else:
            asr_confidence = 0.97

        # Print context tag, intensity score (membership-like) and simulated ASR confidence
        score_label_text = (
            f"<paralinguistic score=\"{max_intensity:.2f}\" "
            f"asr_conf=\"{asr_confidence:.2f}\">"
            f"score=filler_elongation(0..1); asr_confidence(0..1)"
            f"</paralinguistic>"
        )

        injected_text = f"{text} {tag or ''} {score_label_text}"
        print(
            f"'{text}'".ljust(20)
            + " | "
            + str(tag)
            + f"   score={max_intensity:.2f}"
            + f"   asr_conf={asr_confidence:.2f}"
            + f"   -> injected: {injected_text}"
        )
        
    print("\n=== Dry Run Complete ===")

if __name__ == "__main__":
    asyncio.run(run_dry_run())
