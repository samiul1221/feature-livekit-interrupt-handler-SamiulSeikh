"""
Semantic embedding-based filler detection using sentence transformers.
"""

from __future__ import annotations

import logging
from typing import Dict, List
import re
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class SemanticFillerDetector:
    """
    Detect fillers using semantic similarity with pre-computed embeddings.
    Uses cosine similarity to compare incoming speech with known fillers and commands.
    """
    
    FILLER_PHRASES = [
        "uh", "uhh", "uhhh",
        "um", "umm", "ummm",
        "hmm", "hmmm", "hmmmm",
        "er", "err", "errr",
        "ah", "ahh", "ahhh",
        "you know", "like", "i mean",
        "haan", "accha", "theek",  # Hindi
        "este", "pues", "eh",  # Spanish
    ]
    
    COMMAND_PHRASES = [
        "wait", "stop", "hold on", "pause",
        "no", "yes", "okay", "listen",
        "excuse me", "sorry", "hello",
        "help", "repeat", "again",
    ]
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        filler_threshold: float = 0.75,
        command_margin: float = 0.1,
    ):
        self.model_name = model_name
        self.filler_threshold = filler_threshold
        self.command_margin = command_margin
        
        self.model = None
        self.filler_embeddings = None
        self.command_embeddings = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        
    def _load_model(self):
        """Load sentence transformer model and pre-compute embeddings."""
        try:
            logger.info(f"Loading semantic model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Pre-compute filler embeddings
            self.filler_embeddings = self.model.encode(
                self.FILLER_PHRASES,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Pre-compute command embeddings
            self.command_embeddings = self.model.encode(
                self.COMMAND_PHRASES,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            logger.info(
                f"Loaded {len(self.FILLER_PHRASES)} filler and "
                f"{len(self.COMMAND_PHRASES)} command embeddings"
            )
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.model = None
    
    def calculate_similarity(self, text: str) -> Dict[str, float]:
        """
        Calculate semantic similarity scores.
        
        Returns:
            Dict with 'filler_similarity', 'command_similarity', 'is_filler'
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return {
                'filler_similarity': 0.0,
                'command_similarity': 0.0,
                'is_filler': False,
                'method': 'fallback'
            }
        
        try:
            # Normalize repetitive characters (e.g., "hmmmmmm" -> "hmmm")
            text = self._normalize_repeated_chars(text)

            # Encode input text
            text_embedding = self.model.encode(
                [text.lower().strip()],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            # Calculate cosine similarity with fillers
            filler_similarities = np.dot(self.filler_embeddings, text_embedding)
            max_filler_sim = float(np.max(filler_similarities))
            
            # Calculate cosine similarity with commands
            command_similarities = np.dot(self.command_embeddings, text_embedding)
            max_command_sim = float(np.max(command_similarities))
            
            # Classify as filler if:
            # 1. Similarity to fillers exceeds threshold
            # 2. Similarity to fillers is higher than commands by margin
            is_filler = (
                max_filler_sim > self.filler_threshold and
                max_filler_sim > max_command_sim + self.command_margin
            )
            
            return {
                'filler_similarity': max_filler_sim,
                'command_similarity': max_command_sim,
                'is_filler': is_filler,
                'method': 'semantic'
            }
            
        except Exception as e:
            logger.error(f"Error in semantic detection: {e}")
            return {
                'filler_similarity': 0.0,
                'command_similarity': 0.0,
                'is_filler': False,
                'method': 'error'
            }
    
    def detect(self, text: str) -> bool:
        """Simple boolean detection."""
        result = self.calculate_similarity(text)
        return result['is_filler']

    def _normalize_repeated_chars(self, text: str, max_repeats: int = 3) -> str:
        """
        Normalize overly long repeated characters inside words.

        Examples:
            - "hmmmmmm" -> "hmm"
            - "ummmmmm" -> "umm"

        This helps the semantic model and string-matching to detect
        elongated filler variants.
        """
        if not text:
            return text

        # Reduce any repeated char to at most max_repeats occurrences.
        # Keep punctuation and whitespace intact.
        def _repl(m):
            ch = m.group(1)
            return ch * max_repeats

        # Replace runs of the same character (3 or more) with max_repeats
        normalized = re.sub(r"(.)\1{2,}", _repl, text)
        return normalized
    
    def add_filler_phrase(self, phrase: str):
        """Add new filler phrase and recompute embeddings."""
        if phrase not in self.FILLER_PHRASES:
            self.FILLER_PHRASES.append(phrase)
            if self.model is not None:
                self.filler_embeddings = self.model.encode(
                    self.FILLER_PHRASES,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
    
    def add_command_phrase(self, phrase: str):
        """Add new command phrase and recompute embeddings."""
        if phrase not in self.COMMAND_PHRASES:
            self.COMMAND_PHRASES.append(phrase)
            if self.model is not None:
                self.command_embeddings = self.model.encode(
                    self.COMMAND_PHRASES,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
