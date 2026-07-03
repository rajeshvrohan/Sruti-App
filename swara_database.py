SHADJA_HZ = 311.13  # Madhya sthayi (middle octave) S = D#4 / Eb4

# Pythagorean shruti ratios relative to S, one octave.
_SWARA_RATIOS: dict[str, tuple[int, int]] = {
    "S":  (1,   1),    # Shadja              — unison
    "R1": (256, 243),  # Shuddha Rishabha    — Pythagorean minor 2nd
    "R2": (9,   8),    # Chatushruti Rishabha — major 2nd
    "G1": (32,  27),   # Shuddha Gandhara    — Pythagorean minor 3rd
    "G2": (81,  64),   # Sadharana Gandhara  — Pythagorean major 3rd
    "M1": (4,   3),    # Shuddha Madhyama    — perfect 4th
    "M2": (729, 512),  # Prati Madhyama      — Pythagorean tritone
    "P":  (3,   2),    # Panchama            — perfect 5th
    "D1": (128, 81),   # Shuddha Dhaivata    — Pythagorean minor 6th
    "D2": (27,  16),   # Chatushruti Dhaivata — Pythagorean major 6th
    "N1": (16,  9),    # Shuddha Nishada     — Pythagorean minor 7th
    "N2": (243, 128),  # Kakali Nishada      — Pythagorean major 7th
}

# Octave multipliers: lower (mandra) 0.5, middle (madhya) 1.0, higher (tara) 2.0.
_OCTAVE_MULTIPLIERS: dict[str, float] = {"lower": 0.5, "middle": 1.0, "higher": 2.0}

SWARA_FREQUENCIES_BY_OCTAVE: dict[str, dict[str, float]] = {
    octave: {
        name: round(SHADJA_HZ * mult * num / den, 2)
        for name, (num, den) in _SWARA_RATIOS.items()
    }
    for octave, mult in _OCTAVE_MULTIPLIERS.items()
}

# Canonical middle-octave mapping (swara name -> Hz), for ordering and display.
SWARA_FREQUENCIES: dict[str, float] = SWARA_FREQUENCIES_BY_OCTAVE["middle"]

# (swara, Hz) pairs across all three octaves, for matching.
_ALL_OCTAVE_FREQUENCIES: list[tuple[str, float]] = [
    (name, freq)
    for octave in SWARA_FREQUENCIES_BY_OCTAVE.values()
    for name, freq in octave.items()
]


def closest_swara(frequency: float) -> str:
    """Swara name (without octave) closest to `frequency`, matched across all three octaves."""
    name, _ = min(_ALL_OCTAVE_FREQUENCIES, key=lambda pair: abs(pair[1] - frequency))
    return name
