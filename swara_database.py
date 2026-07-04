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
    """Swara name (without octave) closest to `frequency`, matched across all three octaves.

    NOTE: assumes the fixed absolute tonic :data:`SHADJA_HZ`. For real
    recordings, which are in the performer's own key, use
    :func:`closest_swara_relative` with an estimated/known tonic instead.
    """
    name, _ = min(_ALL_OCTAVE_FREQUENCIES, key=lambda pair: abs(pair[1] - frequency))
    return name


import math

# Each swara's position in cents above Sa (from its Pythagorean ratio). Carnatic
# swaras are intervals above the tonic, so classification must be relative.
SWARA_CENTS: dict[str, float] = {
    name: 1200.0 * math.log2(num / den) for name, (num, den) in _SWARA_RATIOS.items()
}


def closest_swara_relative(frequency: float, tonic_hz: float) -> str:
    """Swara whose interval above the tonic is closest to ``frequency``'s.

    Works in any key: ``frequency`` is expressed as cents above ``tonic_hz``
    (the performer's Sa), folded into one octave, then matched to the nearest
    swara position. This is the tonic-relative counterpart of
    :func:`closest_swara`.
    """
    cents = (1200.0 * math.log2(frequency / tonic_hz)) % 1200.0
    return min(
        SWARA_CENTS,
        key=lambda s: min(abs(cents - SWARA_CENTS[s]), 1200.0 - abs(cents - SWARA_CENTS[s])),
    )
