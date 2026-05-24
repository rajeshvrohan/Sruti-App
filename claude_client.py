import os
import anthropic


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-6"

    def analyze_carnatic(self, features: dict) -> str:
        prompt = self._build_prompt(features)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def _build_prompt(self, features: dict) -> str:
        return f"""You are an expert in Carnatic classical music. Analyze the following audio features
extracted from a recording and provide insights about the music.

Audio Features:
- Duration: {features.get('duration', 'N/A'):.2f} seconds
- Tempo: {features.get('tempo', 'N/A'):.1f} BPM
- Sample Rate: {features.get('sample_rate', 'N/A')} Hz
- Spectral Centroid Mean: {features.get('spectral_centroid_mean', 'N/A'):.2f} Hz
- Dominant Pitches (Hz): {features.get('dominant_pitches', [])}
- MFCC Means: {features.get('mfcc_means', [])}

Based on these features, please analyze:
1. Possible Raga identification or characteristics
2. Tala (rhythmic cycle) inference from tempo
3. General musical characteristics (mood, style)
4. Any notable observations about the tonal structure

Provide a concise, informative analysis suitable for both students and enthusiasts of Carnatic music."""
