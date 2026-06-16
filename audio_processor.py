from collections import Counter

import librosa
import numpy as np

from raaga_database import identify_raaga
from swara_database import SWARA_FREQUENCIES, closest_swara

# pyin pitch-tracking bounds, spanning the three octaves of the swara database
# (lower S = 155.56 Hz up through the tara-sthayi swaras).
FREQ_MIN = 140.0   # Hz
FREQ_MAX = 1250.0  # Hz

# Minimum pyin voicing probability for a frame to count; drops low-confidence
# frames that pyin pins to the fmin floor.
VOICED_PROB_MIN = 0.5


class AudioProcessor:
    def extract_features(self, filepath: str) -> dict:
        y, sr = librosa.load(filepath)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        dominant_pitches = self._get_dominant_pitches(pitches, magnitudes)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        return {
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "tempo": float(tempo),
            "sample_rate": int(sr),
            "dominant_pitches": dominant_pitches,
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "mfcc_means": mfccs.mean(axis=1).tolist(),
        }

    def extract_swaras(self, filepath: str) -> dict:
        """Extract the swaras of a WAV, MP3, or M4A file and identify raagas.

        Detects the distinct swaras present across the recording and runs them
        through :func:`identify_raaga`. Returns both the detected swaras and the
        top 3 raaga matches::

            {"swaras": [...], "raagas": [{...}, {...}, {...}]}
        """
        swaras = self.detected_swaras(filepath)
        return {"swaras": swaras, "raagas": identify_raaga(swaras, top_n=3)}

    @staticmethod
    def fundamental_frequency(frequencies: list[float]) -> float | None:
        """Estimate the fundamental as the median of the detected frequencies."""
        if not frequencies:
            return None
        return float(np.median(frequencies))

    def detected_swaras(self, filepath: str, min_fraction: float = 0.03) -> list[str]:
        """Return the distinct swaras present across a phrase, low to high.

        Each voiced frame's fundamental is mapped to its closest swara; a swara
        is kept only if it appears in at least ``min_fraction`` of the voiced
        frames, which filters out transient noise and pitch-glide frames between
        notes. Ordering follows ascending middle-octave frequency.
        """
        frequencies = self.detected_frequencies(filepath)
        if not frequencies:
            return []

        counts = Counter(closest_swara(f) for f in frequencies)
        threshold = len(frequencies) * min_fraction
        swaras = [swara for swara, count in counts.items() if count >= threshold]
        return sorted(swaras, key=lambda swara: SWARA_FREQUENCIES[swara])

    def detected_frequencies(self, filepath: str) -> list[float]:
        """Return the fundamental frequency (Hz) of each voiced time window.

        Uses pYIN, which tracks the fundamental directly instead of taking the
        loudest spectral bin, so overtones no longer masquerade as the pitch.
        Unvoiced frames (NaN) and low-confidence frames (voicing probability
        below ``VOICED_PROB_MIN``) are dropped.
        """
        y, sr = librosa.load(filepath)

        f0, _voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=FREQ_MIN, fmax=FREQ_MAX, sr=sr
        )

        return [
            float(freq)
            for freq, prob in zip(f0, voiced_prob)
            if not np.isnan(freq) and prob >= VOICED_PROB_MIN
        ]

    def _get_dominant_pitches(self, pitches, magnitudes, top_n=10) -> list:
        pitch_list = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_list.append(float(pitch))

        if not pitch_list:
            return []

        pitch_list.sort(reverse=True)
        return pitch_list[:top_n]
