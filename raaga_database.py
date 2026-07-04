"""Common Carnatic raagas: each with its arohana (ascent) and avarohana (descent).

Swaras follow the 12-semitone convention (S R1 R2 G1 G2 M1 M2 P D1 D2 N1 N2);
S' is the upper-octave shadja closing a scale.
"""

def _fold_octave(swara: str) -> str:
    """Collapse the upper-octave shadja S' down to S."""
    return "S" if swara == "S'" else swara


def raaga_swaras(raaga: dict) -> set[str]:
    """Return the set of distinct swaras (octave-folded) used by a raaga."""
    return {_fold_octave(s) for s in raaga["arohana"] + raaga["avarohana"]}


def match_raagas(detected_swaras, top_n: int | None = None) -> list[dict]:
    """Strict match: candidate raagas must contain every detected swara.

    Ranked by scale coverage (score 0-1). Returns dicts of `name`, `score`,
    `missing`, best first.
    """
    detected = {_fold_octave(s) for s in detected_swaras}

    matches: list[dict] = []
    for name, raaga in RAAGAS.items():
        scale = raaga_swaras(raaga)
        if detected - scale:
            continue  # input contains a swara this raaga does not use
        score = len(detected & scale) / len(scale)
        matches.append(
            {"name": name, "score": round(score, 3), "missing": sorted(scale - detected)}
        )

    matches.sort(key=lambda m: (-m["score"], m["name"]))
    return matches[:top_n] if top_n is not None else matches


def identify_raaga(detected_swaras, top_n: int = 3) -> list[dict]:
    """Rank raagas by overlap with the detected swaras, tolerating noisy/partial input.

    match_percentage is the Jaccard overlap between the detected swaras and the
    raaga's scale, so a candidate loses ground both for scale swaras that were
    not heard and for detected swaras it cannot explain — a pentatonic raaga
    can no longer outrank its parent scale just because its few swaras were all
    present. Ties favour the fewest unexplained detected swaras. Returns up to
    top_n dicts with `name`, `match_percentage`, `matched`, `missing`,
    `unexpected`, best first.
    """
    detected = {_fold_octave(s) for s in detected_swaras}
    if not detected:
        return []

    results: list[dict] = []
    for name, raaga in RAAGAS.items():
        scale = raaga_swaras(raaga)
        matched = detected & scale
        if not matched:
            continue
        results.append(
            {
                "name": name,
                "match_percentage": round(100 * len(matched) / len(detected | scale), 1),
                "matched": _sorted_swaras(matched),
                "missing": _sorted_swaras(scale - detected),
                "unexpected": _sorted_swaras(detected - scale),
            }
        )

    results.sort(
        key=lambda r: (-r["match_percentage"], len(r["unexpected"]), r["name"])
    )
    return results[:top_n]


_SWARA_ORDER = ["S", "R1", "R2", "G1", "G2", "M1", "M2", "P", "D1", "D2", "N1", "N2"]


def _sorted_swaras(swaras) -> list[str]:
    """Return swaras ordered low to high by their position in the octave."""
    return sorted(swaras, key=lambda s: _SWARA_ORDER.index(s) if s in _SWARA_ORDER else 99)


RAAGAS: dict[str, dict] = {
    "Shankarabharanam": {
        "name": "Shankarabharanam",
        "arohana": ["S", "R2", "G2", "M1", "P", "D2", "N2", "S'"],
        "avarohana": ["S'", "N2", "D2", "P", "M1", "G2", "R2", "S"],
        "description": "29th melakarta; the Carnatic equivalent of the Western "
        "major scale and one of the most fundamental sampurna raagas.",
    },
    "Kalyani": {
        "name": "Kalyani",
        "arohana": ["S", "R2", "G2", "M2", "P", "D2", "N2", "S'"],
        "avarohana": ["S'", "N2", "D2", "P", "M2", "G2", "R2", "S"],
        "description": "65th melakarta (Mechakalyani); a bright, auspicious raaga "
        "with the prati madhyama, equivalent to the Lydian mode.",
    },
    "Bhairavi": {
        "name": "Bhairavi",
        "arohana": ["S", "R2", "G1", "M1", "P", "D2", "N1", "S'"],
        "avarohana": ["S'", "N1", "D1", "P", "M1", "G1", "R2", "S"],
        "description": "Bhashanga janya of Natabhairavi; uses chatushruti dhaivata "
        "(D2) while ascending and shuddha dhaivata (D1) while descending.",
    },
    "Todi": {
        "name": "Todi",
        "arohana": ["S", "R1", "G1", "M1", "P", "D1", "N1", "S'"],
        "avarohana": ["S'", "N1", "D1", "P", "M1", "G1", "R1", "S"],
        "description": "8th melakarta (Hanumatodi); a heavy, deeply emotive raaga "
        "rich in gamaka, suited to elaborate exposition.",
    },
    "Kharaharapriya": {
        "name": "Kharaharapriya",
        "arohana": ["S", "R2", "G1", "M1", "P", "D2", "N1", "S'"],
        "avarohana": ["S'", "N1", "D2", "P", "M1", "G1", "R2", "S"],
        "description": "22nd melakarta; corresponds to the Dorian mode and is the "
        "parent of many popular janya raagas.",
    },
    "Mohanam": {
        "name": "Mohanam",
        "arohana": ["S", "R2", "G2", "P", "D2", "S'"],
        "avarohana": ["S'", "D2", "P", "G2", "R2", "S"],
        "description": "Pentatonic (audava) janya of Harikambhoji with no madhyama "
        "or nishada; cheerful and widely used, akin to the major pentatonic.",
    },
    "Hamsadhwani": {
        "name": "Hamsadhwani",
        "arohana": ["S", "R2", "G2", "P", "N2", "S'"],
        "avarohana": ["S'", "N2", "P", "G2", "R2", "S"],
        "description": "Pentatonic janya of Shankarabharanam dropping madhyama and "
        "dhaivata; bright and festive, popular for concert openers.",
    },
    "Abhogi": {
        "name": "Abhogi",
        "arohana": ["S", "R2", "G1", "M1", "D2", "S'"],
        "avarohana": ["S'", "D2", "M1", "G1", "R2", "S"],
        "description": "Pentatonic janya of Kharaharapriya without panchama or "
        "nishada; serene and contemplative in character.",
    },
    "Mayamalavagowla": {
        "name": "Mayamalavagowla",
        "arohana": ["S", "R1", "G2", "M1", "P", "D1", "N2", "S'"],
        "avarohana": ["S'", "N2", "D1", "P", "M1", "G2", "R1", "S"],
        "description": "15th melakarta; the traditional first raaga taught to "
        "beginners, with symmetric semitone intervals around S and P.",
    },
    "Charukesi": {
        "name": "Charukesi",
        "arohana": ["S", "R2", "G2", "M1", "P", "D1", "N1", "S'"],
        "avarohana": ["S'", "N1", "D1", "P", "M1", "G2", "R2", "S"],
        "description": "26th melakarta; blends a major-scale lower tetrachord with "
        "a minor upper tetrachord, giving a poignant, versatile mood.",
    },
    "Kambhoji": {
        "name": "Kambhoji",
        "arohana": ["S", "R2", "G2", "M1", "P", "D2", "S'"],
        "avarohana": ["S'", "N1", "D2", "P", "M1", "G2", "R2", "S"],
        "description": "Janya of Harikambhoji that omits nishada while ascending; "
        "a majestic, classic raaga rich in gamaka.",
    },
    "Natabhairavi": {
        "name": "Natabhairavi",
        "arohana": ["S", "R2", "G1", "M1", "P", "D1", "N1", "S'"],
        "avarohana": ["S'", "N1", "D1", "P", "M1", "G1", "R2", "S"],
        "description": "20th melakarta; the Carnatic natural minor scale and parent "
        "of Bhairavi, Hindolam and many others.",
    },
    "Hindolam": {
        "name": "Hindolam",
        "arohana": ["S", "G1", "M1", "D1", "N1", "S'"],
        "avarohana": ["S'", "N1", "D1", "M1", "G1", "S"],
        "description": "Pentatonic janya of Natabhairavi without rishabha or "
        "panchama; meditative and soothing (akin to Hindustani Malkauns).",
    },
    "Durga": {
        "name": "Durga",
        "arohana": ["S", "R2", "M1", "P", "D2", "S'"],
        "avarohana": ["S'", "D2", "P", "M1", "R2", "S"],
        "description": "Pentatonic janya of Shankarabharanam without gandhara or "
        "nishada; simple, devotional and pleasant.",
    },
    "Harikambhoji": {
        "name": "Harikambhoji",
        "arohana": ["S", "R2", "G2", "M1", "P", "D2", "N1", "S'"],
        "avarohana": ["S'", "N1", "D2", "P", "M1", "G2", "R2", "S"],
        "description": "28th melakarta; corresponds to the Mixolydian mode and "
        "parents Kambhoji, Mohanam and Mand among others.",
    },
    "Saveri": {
        "name": "Saveri",
        "arohana": ["S", "R1", "M1", "P", "D1", "S'"],
        "avarohana": ["S'", "N2", "D1", "P", "M1", "G2", "R1", "S"],
        "description": "Janya of Mayamalavagowla with a pentatonic ascent and a "
        "fuller descent; a devotional, plaintive raaga.",
    },
    "Sriranjani": {
        "name": "Sriranjani",
        "arohana": ["S", "R2", "G1", "M1", "D2", "N1", "S'"],
        "avarohana": ["S'", "N1", "D2", "M1", "G1", "R2", "S"],
        "description": "Janya of Kharaharapriya that omits panchama entirely; "
        "graceful and well suited to lighter compositions.",
    },
    "Madhyamavati": {
        "name": "Madhyamavati",
        "arohana": ["S", "R2", "M1", "P", "N1", "S'"],
        "avarohana": ["S'", "N1", "P", "M1", "R2", "S"],
        "description": "Pentatonic janya of Kharaharapriya without gandhara or "
        "dhaivata; auspicious and traditionally sung to close a concert.",
    },
    "Arabhi": {
        "name": "Arabhi",
        "arohana": ["S", "R2", "M1", "P", "D2", "S'"],
        "avarohana": ["S'", "N2", "D2", "P", "M1", "G2", "R2", "S"],
        "description": "Janya of Shankarabharanam with a pentatonic ascent and a "
        "sampurna descent; a strong, spirited ghana raaga.",
    },
    "Bilahari": {
        "name": "Bilahari",
        "arohana": ["S", "R2", "G2", "P", "D2", "S'"],
        "avarohana": ["S'", "N2", "D2", "P", "M1", "G2", "R2", "S"],
        "description": "Janya of Shankarabharanam with a pentatonic ascent and a "
        "full descent; bright, lively and festive.",
    },
    "Shanmukhapriya": {
        "name": "Shanmukhapriya",
        "arohana": ["S", "R2", "G1", "M2", "P", "D1", "N1", "S'"],
        "avarohana": ["S'", "N1", "D1", "P", "M2", "G1", "R2", "S"],
        "description": "56th melakarta; an intense raaga with prati madhyama that "
        "conveys pathos and gravity.",
    },
    "Simhendramadhyamam": {
        "name": "Simhendramadhyamam",
        "arohana": ["S", "R2", "G1", "M2", "P", "D1", "N2", "S'"],
        "avarohana": ["S'", "N2", "D1", "P", "M2", "G1", "R2", "S"],
        "description": "57th melakarta; a grand, emotive raaga (the Carnatic "
        "counterpart of the Hungarian minor scale).",
    },
    "Keeravani": {
        "name": "Keeravani",
        "arohana": ["S", "R2", "G1", "M1", "P", "D1", "N2", "S'"],
        "avarohana": ["S'", "N2", "D1", "P", "M1", "G1", "R2", "S"],
        "description": "21st melakarta; the harmonic minor scale, popular across "
        "Carnatic, film and fusion music.",
    },
    "Vachaspati": {
        "name": "Vachaspati",
        "arohana": ["S", "R2", "G2", "M2", "P", "D2", "N1", "S'"],
        "avarohana": ["S'", "N1", "D2", "P", "M2", "G2", "R2", "S"],
        "description": "64th melakarta; equivalent to the Lydian dominant scale, "
        "bright with prati madhyama and kaisiki nishada.",
    },
}
