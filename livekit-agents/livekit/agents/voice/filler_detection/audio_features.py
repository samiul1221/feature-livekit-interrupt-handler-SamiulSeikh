"""
Audio feature extraction for filler detection.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Audio features disabled. Install with: pip install librosa")


class AudioFeatureFilter:
    """
    Extract audio features to detect fillers.
    Analyzes pitch, energy, and duration characteristics.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        short_duration_threshold: float = 0.3,  # seconds
        low_energy_threshold: float = 0.1,
        fast_mode: bool = True,
    ):
        self.sample_rate = sample_rate
        self.short_duration_threshold = short_duration_threshold
        self.low_energy_threshold = low_energy_threshold
        self.enabled = LIBROSA_AVAILABLE or fast_mode
        self.fast_mode = fast_mode
    
    def extract_features(self, audio_data: np.ndarray) -> dict:
        """
        Extract audio features from raw audio data.
        
        Args:
            audio_data: numpy array of audio samples
            
        Returns:
            Dict with audio features
        """
        if not self.enabled:
            return {'enabled': False}
        
        try:
            # Calculate duration
            duration = len(audio_data) / self.sample_rate
            
            # Calculate energy (RMS)
            if self.fast_mode or not LIBROSA_AVAILABLE:
                # Fast numpy RMS (approximate)
                mean_energy = float(np.sqrt(np.mean(audio_data ** 2)))
            else:
                rms = librosa.feature.rms(y=audio_data)[0]
                mean_energy = float(np.mean(rms))
            
            # Calculate pitch variation (fallbacks to quick estimate if fast_mode)
            if self.fast_mode or not LIBROSA_AVAILABLE:
                pitch_variation = 0.0
                # quick autocorrelation method (low-cost) for rough pitch estimate
                try:
                    if len(audio_data) > 512:
                        window = audio_data[:4096]
                        corr = np.correlate(window, window, mode='full')
                        corr = corr[len(corr) // 2:]
                        d = np.diff(corr)
                        start = np.where(d > 0)[0]
                        if start.size > 0:
                            # compute a small local peak
                            local = corr[start[0]:start[0] + 500]
                            if local.size > 0:
                                peak = np.argmax(local)
                                # invert peak index to get proxy for frequency
                                pitch_variation = float(1.0 / (peak + 1))
                except Exception:
                    pitch_variation = 0.0
            else:
                pitches, magnitudes = librosa.piptrack(
                    y=audio_data,
                    sr=self.sample_rate,
                    fmin=50,
                    fmax=400
                )
                pitch_variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
            
            # Calculate zero crossing rate (indicates voiced vs unvoiced)
            if self.fast_mode or not LIBROSA_AVAILABLE:
                # simple zero crossing estimate
                mean_zcr = float(np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2.0)
            else:
                zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
                mean_zcr = float(np.mean(zcr))
            
            return {
                'enabled': True,
                'duration': duration,
                'mean_energy': mean_energy,
                'pitch_variation': pitch_variation,
                'zero_crossing_rate': mean_zcr,
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def is_filler_audio(self, audio_data: np.ndarray) -> tuple[bool, dict]:
        """
        Determine if audio has filler characteristics.
        
        Filler characteristics:
        - Very short duration (< 0.3s)
        - Low energy (quiet)
        - Low pitch variation (monotone)
        
        Returns:
            Tuple of (is_filler, features_dict)
        """
        if not self.enabled:
            return False, {'enabled': False}
        
        features = self.extract_features(audio_data)
        
        if not features.get('enabled', False):
            return False, features
        
        # Filler indicators
        is_short = features['duration'] < self.short_duration_threshold
        is_low_energy = features['mean_energy'] < self.low_energy_threshold
        is_monotone = features['pitch_variation'] < 20.0  # Low pitch variation
        
        # Calculate filler score
        filler_score = 0.0
        
        if is_short:
            filler_score += 0.3
        
        if is_low_energy:
            filler_score += 0.3
        
        if is_monotone:
            filler_score += 0.4
        
        is_filler = filler_score > 0.6
        
        features['is_short'] = is_short
        features['is_low_energy'] = is_low_energy
        features['is_monotone'] = is_monotone
        features['filler_score'] = filler_score
        features['is_filler'] = is_filler
        
        return is_filler, features
    
    def analyze_from_bytes(self, audio_bytes: bytes) -> tuple[bool, dict]:
        """
        Analyze audio from raw bytes.
        
        Args:
            audio_bytes: Raw audio bytes (PCM format)
            
        Returns:
            Tuple of (is_filler, features_dict)
        """
        if not self.enabled:
            return False, {'enabled': False}
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return self.is_filler_audio(audio_data)
        except Exception as e:
            logger.error(f"Error analyzing audio bytes: {e}")
            return False, {'enabled': False, 'error': str(e)}
