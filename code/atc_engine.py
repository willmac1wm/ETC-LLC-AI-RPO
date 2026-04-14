"""
ATC Engine — shared parsing, state, and facility logic.

Imported by both:
  atc_whisper_server.py  (local STARS appliance)
  atc_api_server.py      (cloud REST/WebSocket API)

No hardware dependencies. No Flask. No socketio.
"""

import re
import json
import time
import glob
import datetime
from pathlib import Path

# ── Resolve project root regardless of where code lives ───────────────────────
_HERE     = Path(__file__).resolve().parent        # code/
BASE_DIR  = _HERE.parent                           # ETC-LLC-AI-RPO/
FACILITY_DIR = BASE_DIR / "data" / "facilities"

# ─────────────────────────────────────────────────────────────────────────────
# DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────

AIRLINE_MAP = {
    'american':'AAL','delta':'DAL','united':'UAL','southwest':'SWA',
    'jetblue':'JBU','alaska':'ASA','spirit':'NKS','frontier':'FFT',
    'fedex':'FDX','federal':'FDX','ups':'UPS','atlas':'GTI',
    'lufthansa':'DLH','british':'BAW','emirates':'UAE',
    'envoy':'ENY','skywest':'SKW','republic':'RPA','endeavor':'EDV',
    'horizon':'QXE','cape air':'KAP','sun country':'SCX','allegiant':'AAY',
    'air force':'AIO','navy':'NVY','army':'AFS',
}

NUM_WORDS = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4',
    'five':'5','six':'6','seven':'7','eight':'8','nine':'9','niner':'9',
}

ICAO_NAMES = {
    'AAL':'American','DAL':'Delta','UAL':'United','SWA':'Southwest',
    'JBU':'JetBlue','ASA':'Alaska','NKS':'Spirit','FFT':'Frontier',
    'FDX':'FedEx','UPS':'UPS','GTI':'Atlas','DLH':'Lufthansa',
    'BAW':'Speedbird','UAE':'Emirates','ENY':'Envoy','SKW':'SkyWest',
    'RPA':'Republic','EDV':'Endeavor','QXE':'Horizon','KAP':'Cape Air',
    'SCX':'Sun Country','AAY':'Allegiant','AIO':'Air Force','NVY':'Navy',
}

# NATO phonetic → letter  ('delta'/'golf' conflict with airline names;
# airline lookup runs first so this is safe)
NATO_ALPHA = {
    'alpha':'A','bravo':'B','charlie':'C','foxtrot':'F','golf':'G',
    'hotel':'H','india':'I','juliet':'J','kilo':'K','lima':'L',
    'mike':'M','oscar':'O','papa':'P','quebec':'Q','romeo':'R',
    'sierra':'S','tango':'T','uniform':'U','victor':'V','whiskey':'W',
    'xray':'X','x-ray':'X','yankee':'Y','zulu':'Z',
}

_ALPHA_NATO = {v: k for k, v in NATO_ALPHA.items()}   # reverse map for readback

# ─────────────────────────────────────────────────────────────────────────────
# FACILITY DATA PACKS
# ─────────────────────────────────────────────────────────────────────────────

_facility: dict = {}


def list_facilities() -> list[str]:
    """Return sorted list of available facility IDs."""
    if not FACILITY_DIR.exists():
        return []
    return [Path(f).stem for f in sorted(glob.glob(str(FACILITY_DIR / "*.json")))]


def load_facility(facility_id: str) -> dict:
    """Load a facility JSON pack. Returns the dict or {} on failure."""
    global _facility
    path = FACILITY_DIR / f"{facility_id}.json"
    if not path.exists():
        print(f"[!] Facility pack not found: {path}")
        return {}
    with open(path) as f:
        _facility = json.load(f)
    print(f"[*] Loaded facility: {_facility.get('facility_name', facility_id)}")
    return _facility


def facility_fixes() -> set:
    return set(_facility.get("fixes", []))


def facility_approach_code(appr_type: str, runway: str) -> str | None:
    """Look up STARS code from loaded facility pack, or None to use computed value."""
    for airport_approaches in _facility.get("approaches", {}).values():
        for appr in airport_approaches:
            if appr["type"] == appr_type and appr["runway"].upper() == runway.upper():
                return appr["stars_code"]
    return None


def get_loaded_facility() -> dict:
    return _facility


# ─────────────────────────────────────────────────────────────────────────────
# PARSER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _w2d(t: str) -> str:
    """Replace spoken number-words with digits."""
    for w, d in NUM_WORDS.items():
        t = re.sub(r"\b" + w + r"\b", d, t, flags=re.IGNORECASE)
    return t


def _resolve_nnumber(text: str) -> tuple[str | None, str]:
    """
    Parse N-number from spoken text.
    'november one two three foxtrot alpha'  →  ('N123FA', '')
    Returns (callsign, remainder) or (None, text).
    """
    lower = text.lower()
    m = re.search(r"\bnovember\b", lower)
    if not m:
        return None, text
    after  = lower[m.end():].strip()
    tokens = after.split()
    chars, consumed = [], 0
    for tok in tokens:
        tok = tok.strip(".,")
        if tok in NUM_WORDS:
            chars.append(NUM_WORDS[tok]); consumed += 1
        elif tok in NATO_ALPHA:
            chars.append(NATO_ALPHA[tok]); consumed += 1
        elif re.match(r'^\d$', tok):
            chars.append(tok); consumed += 1
        else:
            break
    if len(chars) < 2:
        return None, text
    return 'N' + ''.join(chars), ' '.join(tokens[consumed:])


_APPR_TYPES = [
    (r"ils\s+z\b",        "ILSZ"),
    (r"ils\b",            "ILS"),
    (r"rnav\s*\(gnss\)",  "RNAV"),
    (r"rnav\s*\(gps\)",   "RNAV"),
    (r"rnav\b",           "RNAV"),
    (r"lda\b",            "LDA"),
    (r"loc\b",            "LOC"),
    (r"visual\b",         "VIS"),
]

_CMD_PATTERNS = [
    r"heading\s+\d",
    r"turn\s+(left|right)",
    r"(climb|descend|maintain)\s+\d",
    r"flight\s+level\s+\d",
    r"speed\s+\d",
    r"squawk\s+\d{4}",
    r"cleared\s+(ils|rnav|visual|loc|lda)",
    r"(proceed|fly)\s+direct",
    r"contact\s+\w+\s+\d{3}\.",
]

_last_callsign: str | None = None
_last_callsign_time: float = 0.0
_CALLSIGN_TTL = 45.0


def _has_command_tokens(lower_text: str) -> bool:
    return any(re.search(p, lower_text) for p in _CMD_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_atc_command(text: str) -> tuple[str | None, str | None]:
    """
    Parse a plain-English ATC instruction into (callsign, command_string).

    command_string token format
    ───────────────────────────
    H180        heading 180
    TL / TR     turn left / right (precedes H token)
    FH          fly heading (precedes H token)
    A30         altitude 3000 ft  (A + hundreds)
    A350        flight level 350  (A + FL value ≥ 180)
    S250        speed 250 knots
    SQ4521      squawk 4521
    APPR_ILS_27L  approach clearance
    DCT_KEYED   direct-to fix
    FREQ_119.1  frequency change (verbal only, no STARS keystroke)
    """
    global _last_callsign, _last_callsign_time
    lower    = text.lower()
    callsign = None
    rest     = ""

    # 1) Airline callsign
    for name, icao in AIRLINE_MAP.items():
        idx = lower.find(name)
        if idx != -1:
            after = _w2d(text[idx + len(name):].strip())
            m = re.match(r"\s*(\d{1,4})", after)
            if m:
                callsign = icao + m.group(1)
                rest     = after[m.end():].strip()
                break

    # 2) N-number (GA)
    if not callsign:
        callsign, rest = _resolve_nnumber(text)

    # 3) Implicit: reuse last callsign if command tokens are present
    if not callsign:
        age = time.time() - _last_callsign_time
        if _last_callsign and age < _CALLSIGN_TTL and _has_command_tokens(lower):
            callsign = _last_callsign
            rest     = text
            print(f"[~] Implicit callsign: {callsign} ({age:.0f}s ago)")
        else:
            return None, None

    _last_callsign      = callsign
    _last_callsign_time = time.time()

    r      = _w2d(rest.lower())
    tokens = []

    # Heading
    tl = re.search(r"turn\s+left\s+(?:heading\s+)?(\d{1,3})", r)
    tr = re.search(r"turn\s+right\s+(?:heading\s+)?(\d{1,3})", r)
    fh = re.search(r"(?:fly|proceed)\s+(?:direct\s+)?heading\s+(\d{1,3})", r)
    if tl:     tokens += ["TL", "H" + tl.group(1).zfill(3)]
    elif tr:   tokens += ["TR", "H" + tr.group(1).zfill(3)]
    elif fh:   tokens += ["FH", "H" + fh.group(1).zfill(3)]
    else:
        h = re.search(r"heading\s+(\d{1,3})", r)
        if h: tokens.append("H" + h.group(1).zfill(3))

    # Altitude
    afl = re.search(r"(?:flight level|fl)\s+(\d{2,3})", r)
    aft = re.search(r"(?:climb|descend|maintain|altitude)(?:\s+and\s+maintain)?\s+(\d{3,5})", r)
    if afl:   tokens.append("A" + afl.group(1))
    elif aft: tokens.append("A" + str(int(aft.group(1)) // 100))

    # Speed
    spd = re.search(r"(?:(?:reduce|increase|maintain)\s+)?speed\s+(\d{2,3})", r)
    if spd: tokens.append("S" + spd.group(1))

    # Squawk
    sq = re.search(r"squawk\s+(\d{4})", r)
    if sq: tokens.append("SQ" + sq.group(1))

    # Approach clearance
    clr = re.search(r"cleared\s+(.+?)(?:\s+approach)?(?:\s+runway\s+|\s+)(\d{1,2}[LRC]?)\b", r)
    if clr:
        raw_type = clr.group(1).strip()
        appr_rwy  = clr.group(2).upper()
        for pattern, tag in _APPR_TYPES:
            if re.search(pattern, raw_type):
                tokens.append(f"APPR_{tag}_{appr_rwy}")
                break

    # Direct-to fix (use original-case rest for uppercase fix names)
    dtf = re.search(r"(?:proceed\s+direct|fly\s+direct|direct)\s+([A-Za-z]{3,5})\b", rest)
    if dtf:
        fix = dtf.group(1).upper()
        if fix not in {"LEFT","RIGHT","TURN","HEADING","SPEED","CLIMB","DESCEND"}:
            tokens.append("DCT_" + fix)

    # Frequency
    freq = re.search(r"(?:contact|monitor|over to)\s+\w+\s+(?:on\s+)?(\d{3}\.\d{1,3})", r)
    if freq: tokens.append("FREQ_" + freq.group(1))

    if not tokens:
        return callsign, None
    return callsign, callsign + " " + " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# READBACK GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_pilot_readback(command: str | None) -> str | None:
    if not command:
        return None

    def spoken(n):
        w = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        return " ".join(w[int(d)] for d in str(n) if d.isdigit())

    def spoken_freq(f):
        parts = f.split(".")
        return spoken(parts[0]) + " point " + spoken(parts[1])

    parts = command.strip().split()
    cs    = parts[0]

    if cs.startswith('N') and len(cs) > 1 and cs[1].isdigit():
        cs_spoken = ' '.join(
            'november' if c == 'N' else NUM_WORDS.get(c, _ALPHA_NATO.get(c, c).lower())
            for c in cs
        )
    else:
        name      = ICAO_NAMES.get(cs[:3], cs[:3])
        cs_spoken = name + " " + spoken(cs[3:])

    rb = []
    i  = 1
    while i < len(parts):
        t  = parts[i]
        nx = parts[i+1] if i + 1 < len(parts) else ""
        if   t == "TL" and nx.startswith("H"): rb.append("turn left heading "  + spoken(nx[1:])); i += 2
        elif t == "TR" and nx.startswith("H"): rb.append("turn right heading " + spoken(nx[1:])); i += 2
        elif t == "FH" and nx.startswith("H"): rb.append("fly heading "        + spoken(nx[1:])); i += 2
        elif t.startswith("H"):
            rb.append("heading " + spoken(t[1:])); i += 1
        elif t.startswith("A") and not t.startswith("APPR"):
            alt = int(t[1:])
            if alt >= 180: rb.append("climb and maintain flight level " + spoken(alt))
            else:          rb.append("descend and maintain " + spoken(alt * 100))
            i += 1
        elif t.startswith("S") and not t.startswith("SQ"):
            rb.append("reduce speed " + spoken(t[1:]) + " knots"); i += 1
        elif t.startswith("SQ"):
            rb.append("squawk " + spoken(t[2:])); i += 1
        elif t.startswith("APPR_"):
            _, appr_type, rwy = t.split("_", 2)
            rwy_spoken = spoken(rwy.rstrip("LRC"))
            suffix = {"L":" left","R":" right","C":" center"}.get(
                rwy[-1] if rwy[-1].isalpha() else "", "")
            rb.append(f"cleared {appr_type} approach runway {rwy_spoken}{suffix}"); i += 1
        elif t.startswith("DCT_"):
            rb.append("proceed direct " + t[4:]); i += 1
        elif t.startswith("FREQ_"):
            rb.append("contact " + spoken_freq(t[5:])); i += 1
        else:
            i += 1

    return (", ".join(rb) + ", " + cs_spoken) if rb else None


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND MAPPER  (internal tokens → STARS keystrokes)
# ─────────────────────────────────────────────────────────────────────────────

def map_to_stars(command_str: str | None) -> list[str]:
    """
    Convert internal command string into ordered STARS keystroke strings.

    DAL452 TL H180 A30 SQ4521  →  ['DAL452', 'H180', 'D030', 'SQ4521']
    """
    if not command_str:
        return []
    parts      = command_str.strip().split()
    keystrokes = [parts[0]]   # callsign selects the track
    i = 1
    while i < len(parts):
        tok = parts[i]
        if tok in ('TL', 'TR', 'FH'):
            i += 1; continue
        if tok.startswith('H'):
            keystrokes.append(tok)
        elif tok.startswith('A') and not tok.startswith('APPR'):
            alt = int(tok[1:])
            keystrokes.append(f"A{alt:03d}" if alt >= 180 else f"D{alt:03d}")
        elif tok.startswith('S') and not tok.startswith('SQ'):
            keystrokes.append(tok)
        elif tok.startswith('SQ'):
            keystrokes.append(tok)
        elif tok.startswith('APPR_'):
            _, appr_type, rwy = tok.split('_', 2)
            fac_code = facility_approach_code(appr_type, rwy)
            if fac_code:
                keystrokes.append(fac_code)
            else:
                prefix = {'ILS':'CI','ILSZ':'CI','RNAV':'CR',
                          'LOC':'CL','LDA':'CL','VIS':'CV'}.get(appr_type, 'C')
                keystrokes.append(f"{prefix}{rwy}")
        elif tok.startswith('DCT_'):
            keystrokes.append('D' + tok[4:])
        # FREQ_ → verbal only, no STARS keystroke
        i += 1
    return keystrokes


# ─────────────────────────────────────────────────────────────────────────────
# STATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_aircraft_states: dict = {}


def update_aircraft_state(callsign: str, command_str: str) -> None:
    """Update per-aircraft state dict. Does NOT emit via socketio — caller handles that."""
    if not callsign or not command_str:
        return
    state  = _aircraft_states.setdefault(callsign, {})
    tokens = command_str.split()[1:]
    for tok in tokens:
        if tok in ('TL','TR','FH'): continue
        if   tok.startswith('H')                             : state['hdg']  = tok[1:]
        elif tok.startswith('A') and not tok.startswith('APPR'):
            alt = int(tok[1:])
            state['alt'] = f"FL{alt}" if alt >= 180 else str(alt * 100)
        elif tok.startswith('S') and not tok.startswith('SQ'): state['spd']  = tok[1:]
        elif tok.startswith('SQ')                            : state['sq']   = tok[2:]
        elif tok.startswith('APPR_'):
            _, t, r = tok.split('_', 2); state['appr'] = f"{t} {r}"
        elif tok.startswith('DCT_')                          : state['dct']  = tok[4:]
    state['last_cmd'] = command_str
    state['ts']       = datetime.datetime.now().strftime("%H:%M:%S")


def get_aircraft_states() -> dict:
    return dict(_aircraft_states)


def remove_aircraft(callsign: str) -> bool:
    if callsign in _aircraft_states:
        del _aircraft_states[callsign]
        return True
    return False


def get_confidence(callsign: str | None, tokens: list[str]) -> tuple[bool, str]:
    """Require at least one actionable token (not just turn hints or frequency)."""
    actionable = [t for t in tokens
                  if t not in ('TL','TR','FH') and not t.startswith('FREQ_')]
    if not actionable:
        return False, "callsign only — no actionable command"
    return True, f"{len(actionable)} token(s)"


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: full pipeline on a text string
# ─────────────────────────────────────────────────────────────────────────────

def process_text(text: str) -> dict:
    """
    Run the full pipeline on a plain-text ATC instruction.
    Returns a dict ready to send as JSON.
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    cs, command  = parse_atc_command(text)
    readback     = generate_pilot_readback(command)
    tokens       = command.split()[1:] if command else []
    confident, reason = get_confidence(cs, tokens)
    stars_keys   = map_to_stars(command)

    if cs and command:
        update_aircraft_state(cs, command)

    return {
        "ts":               ts,
        "transcript":       text,
        "callsign":         cs,
        "command":          command,
        "readback":         readback,
        "stars_keys":       stars_keys,
        "confident":        confident,
        "confidence_reason": reason,
    }
