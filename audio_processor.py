import os
import re
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile

try:
    import imageio_ffmpeg

    # Path to the ffmpeg binary bundled with the pip-installed imageio-ffmpeg
    # package, so decoding does not depend on a system-wide ffmpeg install.
    FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    # Fall back to a system ffmpeg on PATH if imageio-ffmpeg is unavailable.
    FFMPEG_BINARY = "ffmpeg"

from raaga_database import identify_raaga
from swara_database import SWARA_FREQUENCIES, closest_swara

# pyin pitch-tracking bounds, spanning the three octaves of the swara database
# (lower S = 155.56 Hz up through the tara-sthayi swaras).
FREQ_MIN = 140.0   # Hz
FREQ_MAX = 1250.0  # Hz

# Minimum pyin voicing probability for a frame to count; drops low-confidence
# frames that pyin pins to the fmin floor.
VOICED_PROB_MIN = 0.5

# Extensions librosa/soundfile can decode directly. Anything else (MP3, M4A,
# MP4, ...) is first transcoded to WAV with the bundled ffmpeg binary.
NATIVE_EXTENSIONS = {".wav"}

# Matches the "Duration: HH:MM:SS.ss" line ffmpeg prints to stderr when probing
# a file's metadata, used to read a clip's length without decoding its samples.
_FFMPEG_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _transcode_to_wav(filepath: str) -> str:
    """Decode an audio/video file to a temporary mono WAV using ffmpeg.

    Uses the imageio-ffmpeg bundled binary (:data:`FFMPEG_BINARY`), so no
    system-wide ffmpeg install is required. Returns the path to the temp WAV;
    the caller is responsible for deleting it. Propagates ``FileNotFoundError``
    when the binary is missing, and raises ``ValueError`` when ffmpeg cannot
    decode the input (e.g. the file has no audio stream).
    """
    fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                FFMPEG_BINARY, "-y", "-i", filepath,
                "-vn",                       # drop any video stream
                "-acodec", "pcm_s16le",      # decode to standard PCM WAV
                "-ar", "44100", "-ac", "1",  # 44.1 kHz mono
                audio_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        _silent_remove(audio_path)
        stderr = exc.stderr.decode("utf-8", "ignore").strip().splitlines()
        detail = stderr[-1] if stderr else "unknown ffmpeg error"
        raise ValueError(f"Could not extract audio from the file: {detail}")
    except FileNotFoundError:
        _silent_remove(audio_path)
        raise
    return audio_path


def _silent_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class AudioProcessor:
    def get_duration(self, filepath: str) -> float:
        """Return the length of an audio/video file in seconds.

        Reads only container/header metadata — it does not decode or load the
        waveform into memory — so it stays cheap even for very long files. This
        lets callers reject over-long uploads before the costly pitch analysis
        (and before decoding gigabytes of samples) ever runs.

        WAV files are read directly with soundfile; other formats (MP3, M4A,
        MP4, ...) are probed with the same bundled ffmpeg binary used for
        transcoding. Raises ``ValueError`` if the duration cannot be determined,
        and propagates ``FileNotFoundError`` when ffmpeg is required but missing.
        """
        if os.path.splitext(filepath)[1].lower() in NATIVE_EXTENSIONS:
            return float(soundfile.info(filepath).duration)
        return self._probe_duration_with_ffmpeg(filepath)

    @staticmethod
    def _probe_duration_with_ffmpeg(filepath: str) -> float:
        """Read a file's duration (seconds) from ffmpeg's metadata output.

        Invokes ``ffmpeg -i`` with no output target: ffmpeg prints the input's
        metadata (including its ``Duration``) to stderr and exits non-zero, all
        without decoding the media, so this is fast regardless of clip length.
        """
        proc = subprocess.run(
            [FFMPEG_BINARY, "-i", filepath],
            capture_output=True,
        )
        stderr = proc.stderr.decode("utf-8", "ignore")
        match = _FFMPEG_DURATION_RE.search(stderr)
        if not match:
            raise ValueError("Could not determine the audio duration.")
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

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

    def extract_swaras(self, filepath: str, debug: bool = False) -> dict:
        """Extract the swaras of a WAV, MP3, M4A, or MP4 file and identify raagas.

        Detects the distinct swaras present across the recording and runs them
        through :func:`identify_raaga`. Returns both the detected swaras and the
        top 3 raaga matches::

            {"swaras": [...], "raagas": [{...}, {...}, {...}]}

        Non-WAV inputs (MP3, M4A, MP4) are first transcoded to a temporary WAV
        with the bundled imageio-ffmpeg binary before being passed to librosa;
        the temp file is removed once analysis completes.

        When ``debug`` is True, the raw dominant frequency of every analysis
        window (not just the surviving filtered list) is printed to stdout.
        """
        analysis_path = filepath
        temp_audio = None
        if os.path.splitext(filepath)[1].lower() not in NATIVE_EXTENSIONS:
            analysis_path = temp_audio = _transcode_to_wav(filepath)

        try:
            swaras = self.detected_swaras(analysis_path, debug=debug)
        finally:
            if temp_audio:
                _silent_remove(temp_audio)

        if debug:
            print(f"[debug] final filtered swaras: {swaras}", flush=True)

        return {"swaras": swaras, "raagas": identify_raaga(swaras, top_n=3)}

    @staticmethod
    def fundamental_frequency(frequencies: list[float]) -> float | None:
        """Estimate the fundamental as the median of the detected frequencies."""
        if not frequencies:
            return None
        return float(np.median(frequencies))

    def detected_swaras(
        self,
        filepath: str,
        min_segment_frames: int = 5,
        stability_cents: float = 60.0,
        debug: bool = False,
    ) -> list[str]:
        """Return the distinct swaras present across a phrase, low to high.

        The pitch track is segmented into distinct note events: runs of
        consecutive confident frames whose pitch stays stable (within
        ``stability_cents`` of the running segment median). A run counts as a
        real note only if it spans at least ``min_segment_frames`` frames, which
        discards the pitch-glide frames between notes. Each segment's median
        frequency is mapped to its closest swara.

        Unlike a frame-count histogram, this gives every sustained note equal
        weight regardless of how long it is held, so brief melodic notes in a
        phrase are preserved instead of being swamped by long, drone-like low
        notes.
        """
        times, f0, voiced_prob = self._analyze_pitch(filepath, debug=debug)
        segments = self._segment_notes(
            times, f0, voiced_prob, min_segment_frames, stability_cents, debug=debug
        )

        swaras = {closest_swara(median_hz) for _s, _e, _n, median_hz in segments}
        return sorted(swaras, key=lambda swara: SWARA_FREQUENCIES[swara])

    def _segment_notes(
        self,
        times,
        f0,
        voiced_prob,
        min_frames: int = 5,
        stability_cents: float = 60.0,
        debug: bool = False,
    ) -> list[tuple[float, float, int, float]]:
        """Cluster the per-frame pitch track into stable note segments.

        Walks the frames in time order, growing a segment while each confident
        frame stays within ``stability_cents`` of the segment's running median.
        A gap (unvoiced/low-confidence frame) or a pitch jump beyond the
        tolerance closes the current segment and starts a new one. Segments
        shorter than ``min_frames`` are discarded as transients/glides.

        Returns a list of ``(start_time, end_time, frame_count, median_hz)``.
        """
        segments: list[tuple[float, float, int, float]] = []
        run: list[tuple[float, float]] = []  # (time, freq) of the current segment

        def close(segment_run):
            if len(segment_run) >= min_frames:
                freqs = [freq for _t, freq in segment_run]
                segments.append(
                    (segment_run[0][0], segment_run[-1][0], len(segment_run), float(np.median(freqs)))
                )

        for t, freq, prob in zip(times, f0, voiced_prob):
            if np.isnan(freq) or prob < VOICED_PROB_MIN:
                close(run)
                run = []
                continue

            freq = float(freq)
            if run:
                running_median = float(np.median([f for _t, f in run]))
                if abs(1200.0 * np.log2(freq / running_median)) > stability_cents:
                    # Pitch jumped to a new note — close the current segment.
                    close(run)
                    run = [(float(t), freq)]
                    continue
            run.append((float(t), freq))

        close(run)

        if debug:
            print(f"[debug] {len(segments)} stable note segments:", flush=True)
            for start, end, n, median_hz in segments:
                print(
                    f"[debug]   {start:7.3f}-{end:7.3f}s  ({n:3d} frames)  "
                    f"median={median_hz:8.2f} Hz  -> {closest_swara(median_hz)}",
                    flush=True,
                )

        return segments

    def _analyze_pitch(self, filepath: str, debug: bool = False):
        """Run pYIN and return ``(times, f0, voiced_prob)`` arrays.

        pYIN tracks the fundamental directly instead of taking the loudest
        spectral bin, so overtones do not masquerade as the pitch. When
        ``debug`` is True, the raw pYIN result for *every* analysis window is
        printed — including the unvoiced/low-confidence frames later steps drop —
        so the full pitch track is visible.
        """
        y, sr = librosa.load(filepath)

        f0, _voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=FREQ_MIN, fmax=FREQ_MAX, sr=sr
        )
        times = librosa.times_like(f0, sr=sr)

        if debug:
            print(
                f"[debug] raw dominant frequency per time window "
                f"({len(f0)} windows):",
                flush=True,
            )
            for i, (t, freq, prob) in enumerate(zip(times, f0, voiced_prob)):
                if np.isnan(freq):
                    print(
                        f"[debug]   window {i:5d}  t={t:7.3f}s  "
                        f"f0=     NaN Hz  p={float(prob):.2f}  (unvoiced)",
                        flush=True,
                    )
                else:
                    kept = float(prob) >= VOICED_PROB_MIN
                    print(
                        f"[debug]   window {i:5d}  t={t:7.3f}s  "
                        f"f0={float(freq):8.2f} Hz  p={float(prob):.2f}  "
                        f"~{closest_swara(float(freq))}"
                        f"{'' if kept else '  (dropped: low confidence)'}",
                        flush=True,
                    )

        return times, f0, voiced_prob

    def detected_frequencies(self, filepath: str, debug: bool = False) -> list[float]:
        """Return the fundamental frequency (Hz) of each voiced, confident window.

        Unvoiced frames (NaN) and low-confidence frames (voicing probability
        below ``VOICED_PROB_MIN``) are dropped. This is a flat list with no note
        structure; :meth:`detected_swaras` uses :meth:`_segment_notes` instead.
        """
        _times, f0, voiced_prob = self._analyze_pitch(filepath, debug=debug)
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
