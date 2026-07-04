import os

# Must be set before librosa imports numba. On the 512MB Render plan the cold
# numba JIT compile (LLVM) spikes RSS enough to get the worker OOM-killed
# (gunicorn logs "SIGKILL! Perhaps out of memory?"). Windowed analysis keeps
# the interpreted (JIT-off) path fast enough, so disabling JIT trades a little
# CPU for staying inside the memory limit. Do not re-enable on this plan.
os.environ["NUMBA_DISABLE_JIT"] = "1"

import re
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf

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

# Sample rate the analysis runs at. This matches the historical default of
# librosa.load (22050 Hz); the pyin frame rate — and therefore the
# min_segment_frames note-length threshold — depends on it, so keep it fixed
# regardless of the source file's native rate.
ANALYSIS_SAMPLE_RATE = 22050

# pyin's time and memory both grow linearly with signal length, so a full song
# (~6 min) would OOM the instance and blow the request timeout. Instead of
# grinding the whole recording, sample a few short windows spread across it and
# union their swaras — a raaga's swara vocabulary recurs throughout the piece,
# so this captures it at a fraction of the cost and with bounded memory. A clip
# shorter than one window is analysed whole, exactly as before.
ANALYSIS_WINDOW_SECONDS = 12.0
ANALYSIS_MAX_WINDOWS = 5


def _analysis_windows(n_samples: int, sr: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` sample ranges to run pitch detection on.

    Short signals yield a single whole-signal window (preserving the original
    behaviour). Longer signals yield up to :data:`ANALYSIS_MAX_WINDOWS`
    non-overlapping windows spaced evenly from the start to the end of the
    recording, so the sampled swaras cover the whole piece.
    """
    win = int(ANALYSIS_WINDOW_SECONDS * sr)
    if n_samples <= win:
        return [(0, n_samples)]
    count = min(ANALYSIS_MAX_WINDOWS, n_samples // win)
    if count <= 1:
        return [(0, win)]
    stride = (n_samples - win) / (count - 1)
    return [(round(i * stride), round(i * stride) + win) for i in range(count)]

# Reject inputs longer than this before spending CPU transcoding them. An
# overlong clip is what pins the ffmpeg subprocess long enough to trip the
# gunicorn worker timeout on a constrained instance.
MAX_TRANSCODE_DURATION_SECONDS = 30 * 60  # 30 minutes

# Hard subprocess ceilings so a stuck ffmpeg can never block a worker forever.
_PROBE_TIMEOUT_SECONDS = 30
_TRANSCODE_TIMEOUT_SECONDS = 300

# ffmpeg prints "Duration: HH:MM:SS.ss" to stderr while reading the header.
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


def _probe_duration_seconds(filepath: str) -> float | None:
    """Return the input's duration in seconds from ffmpeg's header metadata.

    Runs the bundled ffmpeg with no output file so it only parses the container
    header (cheap, no decode) and prints a ``Duration:`` line to stderr.
    Returns ``None`` when the duration cannot be determined. Propagates
    ``FileNotFoundError`` when the ffmpeg binary is missing.
    """
    try:
        proc = subprocess.run(
            [FFMPEG_BINARY, "-i", filepath],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None
    # ffmpeg exits non-zero when given no output file; the metadata we want is
    # on stderr regardless, so we parse it rather than checking the exit code.
    match = _DURATION_RE.search(proc.stderr.decode("utf-8", "ignore"))
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _transcode_to_wav(filepath: str) -> str:
    """Decode an audio/video file to a temporary mono WAV using ffmpeg.

    Uses the imageio-ffmpeg bundled binary (:data:`FFMPEG_BINARY`), so no
    system-wide ffmpeg install is required. Rejects inputs longer than
    :data:`MAX_TRANSCODE_DURATION_SECONDS` before decoding, and caps the
    transcode with a timeout so a stuck ffmpeg cannot block a gunicorn worker.
    Returns the path to the temp WAV; the caller is responsible for deleting
    it. Propagates ``FileNotFoundError`` when the binary is missing, and raises
    ``ValueError`` when the file is too long, times out, or cannot be decoded
    (e.g. it has no audio stream).
    """
    duration = _probe_duration_seconds(filepath)
    if duration is not None and duration > MAX_TRANSCODE_DURATION_SECONDS:
        limit_min = MAX_TRANSCODE_DURATION_SECONDS // 60
        raise ValueError(
            f"Audio is too long ({duration / 60:.1f} minutes). "
            f"The maximum length is {limit_min} minutes."
        )

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
            timeout=_TRANSCODE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _silent_remove(audio_path)
        raise ValueError(
            "Transcoding timed out; the file is too large or complex to process."
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


def _load_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float32 signal using soundfile.

    soundfile reads via libsndfile, which has no ffmpeg/audioread dependency,
    so this never spawns a subprocess (unlike ``librosa.load``'s audioread
    fallback). Multi-channel audio is downmixed to mono and the signal is
    resampled to :data:`ANALYSIS_SAMPLE_RATE`, matching the default behaviour
    of ``librosa.load`` so downstream pitch analysis is unchanged.
    """
    y, sr = sf.read(filepath, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != ANALYSIS_SAMPLE_RATE:
        # librosa.resample is pure numpy/soxr — no ffmpeg subprocess.
        y = librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SAMPLE_RATE)
        sr = ANALYSIS_SAMPLE_RATE
    # pyin and the librosa feature extractors want a contiguous float32 array.
    return np.ascontiguousarray(y, dtype=np.float32), int(sr)


class AudioProcessor:
    def get_duration(self, filepath: str) -> float:
        """Return the duration of an audio file in seconds.

        Native formats are read directly by librosa. Non-native inputs (MP3,
        M4A, MP4) are first transcoded to a temporary WAV with the bundled
        ffmpeg binary, mirroring :meth:`extract_swaras`. Raises
        ``FileNotFoundError`` if ffmpeg is required but unavailable.
        """
        analysis_path = filepath
        temp_audio = None
        if os.path.splitext(filepath)[1].lower() not in NATIVE_EXTENSIONS:
            analysis_path = temp_audio = _transcode_to_wav(filepath)

        try:
            return float(librosa.get_duration(path=analysis_path))
        finally:
            if temp_audio:
                _silent_remove(temp_audio)

    def _load_audio(self, filepath: str) -> tuple[np.ndarray, int]:
        """Load any supported input as a mono float32 signal via soundfile.

        Native WAV is read directly with soundfile (no ffmpeg). Non-native
        inputs (MP3, M4A, MP4) are transcoded to a temporary WAV with the
        bundled ffmpeg binary first, then read with soundfile. This keeps the
        only ffmpeg call inside the timeout-guarded :func:`_transcode_to_wav`,
        so librosa/audioread never spawns ffmpeg during analysis.
        """
        analysis_path = filepath
        temp_audio = None
        if os.path.splitext(filepath)[1].lower() not in NATIVE_EXTENSIONS:
            analysis_path = temp_audio = _transcode_to_wav(filepath)
        try:
            return _load_wav(analysis_path)
        finally:
            if temp_audio:
                _silent_remove(temp_audio)

    def extract_features(self, filepath: str) -> dict:
        y, sr = self._load_audio(filepath)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        dominant_pitches = self._get_dominant_pitches(pitches, magnitudes)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa 0.11 returns tempo as a shape-(1,) array; unwrap to a scalar.
        tempo = float(np.ravel(tempo)[0])
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        return {
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "tempo": tempo,
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

    def _read_analysis_windows(self, filepath: str):
        """Yield ``(signal, sr, start_seconds)`` for each sampled window.

        Reads only the sampled windows straight from the file with soundfile
        (seeking, never loading the whole recording) and resamples just that
        ~12s chunk to :data:`ANALYSIS_SAMPLE_RATE`. Non-native inputs are
        transcoded to a temporary WAV once first. This bounds peak memory to a
        single window regardless of song length — a full-signal load and
        resample would OOM the 512MB instance on a ~6-minute upload.
        """
        analysis_path = filepath
        temp_audio = None
        if os.path.splitext(filepath)[1].lower() not in NATIVE_EXTENSIONS:
            analysis_path = temp_audio = _transcode_to_wav(filepath)
        try:
            info = sf.info(analysis_path)
            for start, end in _analysis_windows(info.frames, info.samplerate):
                seg, sr = sf.read(
                    analysis_path, start=start, stop=end,
                    dtype="float32", always_2d=False,
                )
                if seg.ndim > 1:
                    seg = seg.mean(axis=1)
                if sr != ANALYSIS_SAMPLE_RATE:
                    seg = librosa.resample(
                        seg, orig_sr=sr, target_sr=ANALYSIS_SAMPLE_RATE
                    )
                yield (
                    np.ascontiguousarray(seg, dtype=np.float32),
                    ANALYSIS_SAMPLE_RATE,
                    start / info.samplerate,
                )
        finally:
            if temp_audio:
                _silent_remove(temp_audio)

    def _analyze_pitch(self, filepath: str, debug: bool = False):
        """Run pYIN and return ``(times, f0, voiced_prob)`` arrays.

        pYIN tracks the fundamental directly instead of taking the loudest
        spectral bin, so overtones do not masquerade as the pitch. To keep time
        and memory bounded on long recordings, pYIN is run only on the sampled
        windows from :meth:`_read_analysis_windows` rather than the whole
        signal; the per-window frame tracks are concatenated (with a NaN
        separator so a note cannot be merged across the gap between two
        windows). When ``debug`` is True, the raw pYIN result for *every*
        analysis frame is printed — including the unvoiced/low-confidence frames
        later steps drop.
        """
        times_parts, f0_parts, prob_parts = [], [], []
        window_count = 0
        analysed_seconds = 0.0
        for seg, sr, start_seconds in self._read_analysis_windows(filepath):
            window_count += 1
            analysed_seconds += len(seg) / sr
            f0_w, _voiced_flag, prob_w = librosa.pyin(
                seg, fmin=FREQ_MIN, fmax=FREQ_MAX, sr=sr
            )
            times_w = librosa.times_like(f0_w, sr=sr) + start_seconds
            if window_count > 1:
                # NaN separator closes any open note at the window boundary.
                times_parts.append(times_w[:1])
                f0_parts.append(np.array([np.nan], dtype=f0_w.dtype))
                prob_parts.append(np.array([0.0], dtype=prob_w.dtype))
            times_parts.append(times_w)
            f0_parts.append(f0_w)
            prob_parts.append(prob_w)

        times = np.concatenate(times_parts)
        f0 = np.concatenate(f0_parts)
        voiced_prob = np.concatenate(prob_parts)

        if debug:
            print(
                f"[debug] analysed {window_count} window(s) spanning the "
                f"recording ({analysed_seconds:.1f}s of audio)",
                flush=True,
            )
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
        below ``VOICED_PROB_MIN``) are dropped. 
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

