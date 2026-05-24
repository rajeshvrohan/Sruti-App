import librosa
import numpy as np


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
