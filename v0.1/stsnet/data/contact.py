"""
Contact parser for SSLL description strings.

Extracts contact location and type from natural-language Swedish phase
descriptions and normalises them to canonical phonological classes.

Public API
----------
parse_contact(phase: str) -> tuple[str, str] | None
    Returns (location, contact_type) or None if no contact in this phase.

    location:     one of CONTACT_LOCATIONS  (e.g. "chest", "other_hand")
    contact_type: one of CONTACT_TYPES      (e.g. "single", "repeated")

CONTACT_LOCATIONS — 22 classes (including "none")
CONTACT_TYPES     — 4 classes
"""

import re

# ---------------------------------------------------------------------------
# Canonical class sets
# ---------------------------------------------------------------------------

CONTACT_LOCATIONS: list[str] = [
    "none",
    # face / head
    "mouth",       # mun(nen)
    "chin",        # hak(an)
    "nose",        # näs(an)
    "forehead",    # pann(an)
    "cheek",       # kind(en)
    "ear",         # öra/örat
    "top_of_head", # hjässan
    "face",        # ansiktet (non-specific)
    "head",        # huvud (non-specific)
    # neck / throat
    "neck",        # hals / nacke
    # torso
    "chest",       # bröstet (incl. left/right/upper/lower sides)
    "stomach",     # magen
    "hip",         # höften
    # arm
    "shoulder",    # axeln
    "upper_arm",   # överarmen
    "forearm",     # underarmen
    "elbow",       # armveck / armbåge
    "wrist",       # handled
    # other hand / passive hand
    "other_hand",  # varandra, ovanpå flata handen, etc.
    # spatial (near but not touching body part — treated as that location)
    "temple",      # tinningen
    "back",        # ryggen
]

CONTACT_TYPES: list[str] = [
    "none",
    "single",     # unmarked / kort — one contact event
    "repeated",   # upprepade kontakter
    "sustained",  # bibehållen kontakt — moves while maintaining contact
]

# ---------------------------------------------------------------------------
# Location normalisation map
# ---------------------------------------------------------------------------
# Keys are lowercased Swedish expressions; values are CONTACT_LOCATIONS entries.
# Order matters: more specific entries should appear first in the lookup.

_LOCATION_RAW: list[tuple[str, str]] = [
    # ----- face -----
    ("sidan av munnen",          "mouth"),
    ("bredvid munnen",           "mouth"),
    ("munnen",                   "mouth"),
    ("mun",                      "mouth"),

    ("vänstra sidan av hakan",   "chin"),
    ("högra sidan av hakan",     "chin"),
    ("hakans",                   "chin"),
    ("hakan",                    "chin"),
    ("under hakan",              "chin"),
    ("haka",                     "chin"),

    ("sidan av näsan",           "nose"),
    ("bredvid näsan",            "nose"),
    ("näsan",                    "nose"),
    ("näsa",                     "nose"),

    ("sidan av pannan",          "temple"),
    ("vid sidan av pannan",      "temple"),
    ("bredvid pannan",           "forehead"),
    ("pannan",                   "forehead"),
    ("panna",                    "forehead"),

    ("respektive kind",          "cheek"),
    ("kinden",                   "cheek"),
    ("kind",                     "cheek"),

    ("bredvid örat",             "ear"),
    ("örat",                     "ear"),
    ("öra",                      "ear"),

    ("hjässan",                  "top_of_head"),
    ("hjässa",                   "top_of_head"),

    ("tinningen",                "temple"),
    ("tinning",                  "temple"),

    ("ansiktet",                 "face"),
    ("ansikte",                  "face"),

    ("huvud",                    "head"),

    # ----- neck / throat -----
    ("sidan av halsen",          "neck"),
    ("vid sidan av halsen",      "neck"),
    ("bredvid halsen",           "neck"),
    ("halsen",                   "neck"),
    ("hals",                     "neck"),
    ("nacken",                   "neck"),
    ("nacke",                    "neck"),

    # ----- torso -----
    ("nedre delen av högra sidan av bröstet",  "chest"),
    ("övre delen av vänstra sidan av bröstet", "chest"),
    ("övre delen av bröstet",    "chest"),
    ("nedre delen av bröstet",   "chest"),
    ("vänstra sidan av bröstet", "chest"),
    ("den vänstra sidan av bröstet", "chest"),
    ("högra sidan av bröstet",   "chest"),
    ("bröstet",                  "chest"),
    ("bröst",                    "chest"),

    ("höger sidan av magen",     "stomach"),
    ("magen",                    "stomach"),
    ("mage",                     "stomach"),
    ("buken",                    "stomach"),

    ("höftet",                   "hip"),
    ("höften",                   "hip"),
    ("höft",                     "hip"),

    ("ryggen",                   "back"),
    ("rygg",                     "back"),

    # ----- arm -----
    ("respektive axel",          "shoulder"),
    ("axlarna",                  "shoulder"),
    ("högra axeln",              "shoulder"),
    ("vänstra axeln",            "shoulder"),
    ("höger axel",               "shoulder"),
    ("axeln",                    "shoulder"),
    ("axel",                     "shoulder"),

    ("vänstra överarmen",        "upper_arm"),
    ("överarmen",                "upper_arm"),
    ("överarm",                  "upper_arm"),

    ("armvecket",                "elbow"),
    ("armveck",                  "elbow"),
    ("armbågen",                 "elbow"),
    ("armbåge",                  "elbow"),

    ("vänstra underarmen",       "forearm"),
    ("underarmen",               "forearm"),
    ("underarm",                 "forearm"),

    ("handleden",                "wrist"),
    ("handled",                  "wrist"),

    # ----- other hand / passive hand -----
    # "varandra" = each other → hands touch
    ("varandras tummar",         "other_hand"),
    ("varandra",                 "other_hand"),
    # spatial preposition targets that are handshape words → passive hand
    ("flata handen",             "other_hand"),
    ("den flata handen",         "other_hand"),
    ("knutna handen",            "other_hand"),
    ("den knutna handen",        "other_hand"),
    ("sprethanden",              "other_hand"),
    ("den andra sprethanden",    "other_hand"),
    ("tumhanden",                "other_hand"),
    ("pekfingret",               "other_hand"),
    ("det andra pekfingret",     "other_hand"),
    ("krokfingret",              "other_hand"),
    ("lillfingret",              "other_hand"),
    ("dubbelkroken",             "other_hand"),
    ("den andra handen",         "other_hand"),
    ("andra handen",             "other_hand"),
    ("den andra flata handen",   "other_hand"),
    ("andra flata handen",       "other_hand"),
    ("den andra tumhanden",      "other_hand"),
    ("n-handen",                 "other_hand"),
    ("l-handen",                 "other_hand"),
    ("d-handen",                 "other_hand"),
    ("a-handen",                 "other_hand"),
    ("s-handen",                 "other_hand"),
    ("vinkelhanden",             "other_hand"),
    ("tumvinkelhanden",          "other_hand"),
    ("nyphanden",                "other_hand"),
    ("handen",                   "other_hand"),
]

# Build from longest to shortest for greedy matching
_LOCATION_MAP: dict[str, str] = {}
for raw, canonical in sorted(_LOCATION_RAW, key=lambda x: -len(x[0])):
    _LOCATION_MAP[raw.lower()] = canonical


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "kontakt" or "kontakter" (plural form, e.g. "upprepade kontakter med pannan")
_KW = r'kontakt(?:er)?'

# Contact-type modifiers (appear BEFORE "kontakt[er]")
_pat_type = re.compile(
    r'\b(upprepade?|bibehållen|kort|slutligen)\s+' + _KW,
    re.IGNORECASE,
)

# "kontakt[er] med X"
_BOUNDARY = r'(?:[,.]|$|\s+(?:förs|och|upprepas|samtidigt|växelvis|under|ovanpå|bakom|framför|vid|längs|samt|men))'
_pat_med = re.compile(
    r'\b' + _KW + r'\s+med\s+([\w\s]+?)' + _BOUNDARY,
    re.IGNORECASE,
)

# "kontakt[er] PREPOSITION X"
_PREPS = r'(?:ovanpå|bredvid|bakom|under|framför|vid sidan av|längs med|mot)'
_pat_spatial = re.compile(
    r'\b' + _KW + r'\s+(' + _PREPS + r')\s+([\w\s]+?)' + _BOUNDARY,
    re.IGNORECASE,
)

# bare "kontakt[er]" — presence check
_pat_bare = re.compile(r'\b' + _KW + r'\b', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_location(raw: str) -> str:
    """Map a raw location string to a canonical CONTACT_LOCATIONS entry."""
    raw = raw.strip().lower().rstrip('.,')
    # Greedy longest-match
    for key, canonical in sorted(_LOCATION_MAP.items(), key=lambda x: -len(x[0])):
        if raw == key or raw.startswith(key):
            return canonical
    # Fallback heuristics for unseen expressions
    if any(w in raw for w in ('handen', 'fingret', 'fingrar', 'fingertopparna', 'tummen')):
        return "other_hand"
    if any(w in raw for w in ('bröstet', 'bröst')):
        return "chest"
    if any(w in raw for w in ('armen', 'arm')):
        return "upper_arm"
    return "other_hand"   # safest fallback for unknown hand/body parts


_pat_upprepas = re.compile(r'\bupprepas\b', re.IGNORECASE)

def _parse_contact_type(phase: str) -> str:
    """Extract contact type from the phase string."""
    m = _pat_type.search(phase)
    if m:
        word = m.group(1).lower()
        if word.startswith("upprepade"):
            return "repeated"
        if word == "bibehållen":
            return "sustained"
        # kort / slutligen → single
        return "single"
    # "upprepas" anywhere in phrase (trailing repetition marker, e.g. "kontakt med X, upprepas")
    if _pat_upprepas.search(phase):
        return "repeated"
    return "single"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_contact(phase: str) -> tuple[str, str] | None:
    """
    Parse contact information from one description phase.

    Returns (location, contact_type) where both are canonical class strings,
    or None if the phase contains no contact mention.

    location     ∈ CONTACT_LOCATIONS
    contact_type ∈ CONTACT_TYPES
    """
    if not _pat_bare.search(phase):
        return None    # no contact word at all

    contact_type = _parse_contact_type(phase)

    # Try "kontakt med X" first
    m = _pat_med.search(phase)
    if m:
        loc = _normalise_location(m.group(1))
        return loc, contact_type

    # Try "kontakt PREPOSITION X"
    m = _pat_spatial.search(phase)
    if m:
        # Try prep+location first (e.g. "vid sidan av pannan" → temple),
        # fall back to just the location word
        full = (m.group(1) + " " + m.group(2)).strip()
        loc = _normalise_location(full) if full in _LOCATION_MAP else _normalise_location(m.group(2))
        return loc, contact_type

    # "kontakt" present but target unclear — treat as other_hand
    # (most bare "kontakt" references are between the hands)
    return "other_hand", contact_type
