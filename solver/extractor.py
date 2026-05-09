import re
from typing import Any, Dict, List, Optional

from formula_parser import is_valid_formula


# =====================================
# BM STOPWORDS — KRITIKAL
# Perkataan BM yang DILARANG parse sebagai simbol kimia
# FIX BUG #6: "Sebatian" → "Se" (Selenium)
# FIX BUG #7: kemolaran 0.4 tersalah jadi pOH
# =====================================
BM_STOPWORDS = {
    # Perkataan BM biasa
    "Sebatian", "sebatian", "Dalam", "dalam", "Ialah", "ialah",
    "Jika", "jika", "Dan", "dan", "Dengan", "dengan", "Atau", "atau",
    "Suatu", "suatu", "Satu", "satu", "Untuk", "untuk", "Antara", "antara",
    "Bagi", "bagi", "Kepada", "kepada", "Daripada", "daripada",
    "Apabila", "apabila", "Semasa", "semasa", "Selepas", "selepas",
    "Sebelum", "sebelum", "Oleh", "oleh", "Pada", "pada",
    "Adalah", "adalah", "Akan", "akan", "Telah", "telah",
    "Yang", "yang", "Ini", "ini", "Itu", "itu",
    "Boleh", "boleh", "Perlu", "perlu", "Mesti", "mesti",
    "Mengandungi", "mengandungi", "Dilarutkan", "dilarutkan",
    "Dipanaskan", "dipanaskan", "Dicampurkan", "dicampurkan",
    "Bertindak", "bertindak", "Terhasil", "terhasil",
    "Meningkat", "meningkat", "Menurun", "menurun",
    "Berkurang", "berkurang", "Bertambah", "bertambah",
    # English words yang boleh tersalah parse
    "Calculate", "Find", "Determine", "What", "Which", "How",
    "Mass", "Volume", "Molarity", "Formula", "Jawapan", "Diberi", "Pengiraan",
    "Mol", "Rate", "Change", "Time", "Ar", "RTP", "STP",
    "Carbon", "Dioxide", "Hydrogen", "Oxygen", "Nitrogen", "Sulfur",
    "Chlorine", "Water", "Acid", "Base", "Salt", "Gas",
    "Hitungkan", "Tentukan", "Berapakah", "Apakah", "Nyatakan", "Terangkan",
    "Jelaskan", "Bandingkan", "Langkah",
    "Jisim", "Isipadu", "Kepekatan", "Kemolaran",
    "Tindak", "Balas", "Larutan", "Unsur", "Atom",
    # Dua huruf yang sering tersalah parse
    "In", "Of", "At", "To", "By", "As", "An", "Is", "Be",
    "No", "Do", "Go", "Up", "On", "Or",
}


# =====================================
# NORMALIZATION
# =====================================
def normalize_text(text: str) -> str:
    t = text.strip()
    t = t.replace("\u00a0", " ")
    t = t.replace("\\text{", "")
    t = t.replace("}", "")
    t = t.replace("$", "")
    t = t.replace("→", "->")
    t = t.replace("−", "-")
    t = t.replace("⁻", "-")
    t = t.replace("₁", "1").replace("₂", "2").replace("₃", "3").replace("₄", "4")
    t = t.replace("₅", "5").replace("₆", "6").replace("₇", "7").replace("₈", "8").replace("₉", "9")
    t = t.replace("₀", "0")
    t = t.replace("cm³", "cm3").replace("dm³", "dm3")
    t = t.replace("cm 3", "cm3").replace("dm 3", "dm3")
    t = t.replace("mℓ", "ml")
    t = t.replace("mol dm^-3", "mol dm3").replace("mol dm-3", "mol dm3").replace("mol dm⁻3", "mol dm3")
    t = t.replace("×10^", "e").replace("x10^", "e")
    t = t.replace("× 10^", "e").replace("x 10^", "e")
    t = t.replace("×10", "e").replace("x10", "e")
    t = re.sub(r"\s+", " ", t)
    return t


# =====================================
# ION CHARGE PARSER
# FIX BUG #3: parse cas ion dari teks soalan
# e.g. "SO4 2-" → charge=-2, "MnO4 -" → charge=-1
# =====================================
def extract_ion_charge(text: str) -> Optional[int]:
    """
    Extract ionic charge from question text.
    FIX BUG #7b: Must only match charge that appears AFTER whitespace
    or at end of string — NOT numbers inside formula like NO3, SO4, Cr2O7.

    Valid:   "SO4 2-"  "MnO4 -"  "NO3 -"  "Cr2O7 2-"  "NH4 +"
    Invalid: "NO3" (the 3 is part of formula, not charge)
    """
    t = normalize_text(text)

    # Pattern: whitespace then number then sign at end of token
    # e.g. "SO4 2-"  "Cr2O7 2-"
    m = re.search(r'\s(\d+)\s*[-−](?!\d)', t)
    if m:
        return -int(m.group(1))

    m = re.search(r'\s(\d+)\s*[+](?!\d)', t)
    if m:
        return +int(m.group(1))

    # Single sign after whitespace: "MnO4 -"  "NO3 -"  "NH4 +"
    if re.search(r'\s[-−]\s*$', t.strip()):
        return -1
    if re.search(r'\s[+]\s*$', t.strip()):
        return +1

    return None


def extract_species_with_charge(text: str, formulas: List[str]) -> Optional[Dict[str, Any]]:
    """
    For oxidation number questions:
    Extract species formula AND ionic charge from question text.
    FIX BUG #3: properly detect charge for ions like SO4²⁻, Cr2O7²⁻, MnO4⁻
    """
    if not formulas:
        return None

    q = normalize_text(text)
    species = formulas[0]

    # Try to find charge explicitly stated in question
    charge = extract_ion_charge(q)

    # If no charge found via general parse, check species string itself
    if charge is None:
        m = re.search(r'([A-Z][A-Za-z0-9()]+)\s*(\d*)\s*([+\-−])', q)
        if m:
            sign_char = m.group(3)
            num_str = m.group(2).strip()
            num = int(num_str) if num_str else 1
            charge = num if sign_char == '+' else -num

    return {"species": species, "charge": charge}


# =====================================
# GENERIC EXTRACTORS
# =====================================
def extract_valid_formulas(text: str) -> List[str]:
    t = normalize_text(text)

    candidates = re.findall(r"\b[A-Z][A-Za-z0-9()·.]*[A-Za-z0-9)]\b", t)

    valid: List[str] = []
    seen = set()

    for cand in candidates:
        # FIX BUG #6: skip BM stopwords
        if cand in BM_STOPWORDS:
            continue
        if cand in seen:
            continue
        # Extra guard: skip single-letter uppercase that are BM/EN words
        if len(cand) == 1 and cand not in {"H", "N", "O", "S", "C", "P", "K", "I", "U", "V", "W", "Y"}:
            continue
        if is_valid_formula(cand):
            valid.append(cand)
            seen.add(cand)

    return valid


def extract_mass_values(text: str) -> List[float]:
    vals: List[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(mg|g)\b", text.lower()):
        value = float(m.group(1))
        unit = m.group(2)
        vals.append(value / 1000.0 if unit == "mg" else value)
    return vals


def extract_volume_cm3_values(text: str) -> List[float]:
    return [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(cm3|ml)\b", text.lower())]


def extract_volume_dm3_values(text: str) -> List[float]:
    return [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*dm3\b", text.lower())]


def extract_mole_values(text: str) -> List[float]:
    vals: List[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*mol\b", text.lower()):
        vals.append(float(m.group(1)))
    return vals


def extract_molarity_values(text: str) -> List[float]:
    """
    FIX BUG #7: only extract molarity values when EXPLICITLY tied to mol dm-3 unit.
    Do NOT extract bare numbers as molarity — prevents 0.4 mol/dm3 being read as pOH=0.4
    """
    vals: List[float] = []
    tl = text.lower()
    # Must be explicitly "X mol dm3" or "X M" (with word boundary)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*mol\s*dm[-−]?3", tl):
        vals.append(float(m.group(1)))
    # Also catch "X mol dm-3" already normalised
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*mol\s*dm3", tl):
        v = float(m.group(1))
        if v not in vals:
            vals.append(v)
    return vals


def extract_ph(text: str) -> Optional[float]:
    m = re.search(r"\bph\s*=?\s*(\d+(?:\.\d+)?)", text.lower())
    return float(m.group(1)) if m else None


def extract_poh(text: str) -> Optional[float]:
    """
    FIX BUG #7: only extract pOH when 'poh' keyword is explicitly present.
    Never infer pOH from molarity values.
    """
    m = re.search(r"\bpoh\s*=?\s*(\d+(?:\.\d+)?)", text.lower())
    return float(m.group(1)) if m else None


def extract_h_plus(text: str) -> Optional[float]:
    tl = normalize_text(text).lower()

    patterns = [
        r"\[\s*h\+\s*\]\s*=?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"hydrogen[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mol\s*dm3",
        r"ion hidrogen[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mol\s*dm3",
        r"kepekatan\s*h\+[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, tl)
        if m:
            return float(m.group(1))

    sci = re.search(r"(\d+(?:\.\d+)?)\s*[×x]?\s*10\^?\s*(-?\d+)", text.lower())
    if sci:
        base = float(sci.group(1))
        power = int(sci.group(2))
        return base * (10 ** power)

    return None


def extract_oh_minus(text: str) -> Optional[float]:
    tl = normalize_text(text).lower()

    patterns = [
        r"\[\s*oh-\s*\]\s*=?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"ion hidroksida[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mol\s*dm3",
        r"kepekatan\s*oh-[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, tl)
        if m:
            return float(m.group(1))

    sci = re.search(r"(\d+(?:\.\d+)?)\s*[×x]?\s*10\^?\s*(-?\d+)", text.lower())
    if sci and ("oh" in text.lower() or "hidroksida" in text.lower()):
        base = float(sci.group(1))
        power = int(sci.group(2))
        return base * (10 ** power)

    return None


def extract_temperatures(text: str) -> List[float]:
    return [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*°?\s*c\b", text.lower())]


def extract_times(text: str) -> List[float]:
    return [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds|min|minute|minutes)\b", text.lower())]


def extract_condition(text: str) -> Optional[str]:
    q = text.lower()
    if "stp" in q:
        return "STP"
    if "rtp" in q or "room" in q or "keadaan bilik" in q:
        return "RTP"
    return None


def extract_equation(text: str) -> Optional[str]:
    """
    Extract clean chemical equation from question text.
    LHS must start with a capital-letter formula token (not BM words).
    Handles equation at start, middle, or end of question.
    """
    t = normalize_text(text)
    arrow_pos = t.find('->')
    if arrow_pos == -1:
        return None

    left  = t[:arrow_pos].strip()
    right = t[arrow_pos + 2:].strip()

    # Trim right at first clause/sentence boundary
    for sep in ['. ', ', jika ', ', apabila ', ', dan ', ', if ',
                ', when ', ', berapakah', ', hitungkan']:
        pos = right.find(sep)
        if pos != -1:
            right = right[:pos]

    # Trim left: find last occurrence of a clause separator,
    # then take everything after it as potential equation LHS
    for sep in ['. ', ', berapakah', ', hitungkan', ', jika ',
                ', apabila ', 'tindak balas ', 'reaction ', 'persamaan ']:
        pos = left.rfind(sep)
        if pos != -1:
            candidate = left[pos + len(sep):].strip()
            if candidate and re.match(r'^[A-Z0-9]', candidate):
                left = candidate
                break

    # Walk left string to find where a valid formula chain begins
    # Valid start: digit (stoich coeff) or uppercase letter
    m = re.search(r'(\d*[A-Z][A-Za-z0-9()]*(?:\s*[+]\s*\d*[A-Z][A-Za-z0-9()]*)*)$', left)
    if m:
        left = m.group(1).strip()

    eq = (left + " -> " + right.strip()).strip()
    eq = re.sub(r'\s+', ' ', eq)

    # Validate: LHS must start with formula char and be reasonable length
    lhs_part = eq.split('->')[0].strip()
    if not re.match(r'^[0-9A-Z]', lhs_part):
        return None
    if len(lhs_part) > 60:
        return None

    return eq if '->' in eq else None


def _find_target_formula(question: str, rhs_formulas: List[str], lhs_formulas: List[str]) -> Optional[str]:
    """
    Find which RHS formula the question is ASKING ABOUT.
    Priority:
    1. Formula mentioned after 'isipadu X' or 'jisim X' pattern
    2. RHS formula explicitly in question (not in LHS)
    3. Last RHS formula as fallback
    """
    q_upper = question.upper()

    for pattern in [
        r'(?:ISIPADU|VOLUME)\s+(?:GAS\s+)?([A-Z][A-Za-z0-9()]+)',
        r'(?:JISIM|MASS)\s+(?:LOGAM\s+|PEPEJAL\s+|AIR\s+)?([A-Z][A-Za-z0-9()]+)',
        r'([A-Z][A-Za-z0-9()]+)\s+(?:YANG\s+TERHASIL|YANG\s+DIHASILKAN|FORMED)',
    ]:
        m = re.search(pattern, q_upper)
        if m:
            candidate = m.group(1)
            if candidate in rhs_formulas:
                return candidate

    for f in rhs_formulas:
        if f in question and f not in lhs_formulas:
            return f

    return rhs_formulas[-1] if rhs_formulas else None


def extract_isotope_data(text: str) -> Optional[Dict[str, List[float]]]:
    normalized = text.replace("–", "-")
    masses = [float(x) for x in re.findall(r"\b[A-Z][a-z]?-(\d+(?:\.\d+)?)", normalized)]
    if not masses:
        return None

    percent_abundances = [float(x) for x in re.findall(r"\((\d+(?:\.\d+)?)%\)", normalized)]
    if len(percent_abundances) == len(masses):
        return {"isotope_masses": masses, "abundances": percent_abundances}

    # plain % without brackets e.g. "20% isotop boron-10"
    plain_pct = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", normalized)]
    if len(plain_pct) == len(masses):
        return {"isotope_masses": masses, "abundances": plain_pct}

    ratio_match = re.search(r"(?:nisbah[^\d]*)(\d+(?:\s*:\s*\d+)+)", normalized.lower())
    if ratio_match:
        abundances = [float(x.strip()) for x in ratio_match.group(1).split(":")]
        if len(abundances) == len(masses):
            return {"isotope_masses": masses, "abundances": abundances}

    plain_numbers = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", normalized)]
    filtered = [x for x in plain_numbers if x not in masses]
    if len(filtered) >= len(masses):
        return {"isotope_masses": masses, "abundances": filtered[:len(masses)]}

    return None


def extract_subatomic_data(text: str) -> Optional[Dict[str, int]]:
    q = text.lower()
    if not any(k in q for k in ["proton", "neutron", "electron", "nukleon", "nucleon"]):
        return None

    A = None
    Z = None

    patterns_A = [
        r"nombor nukleon\s*\(?a\)?\s*=?\s*(\d+)",
        r"nukleon\s+(\d+)",
        r"a\s*=\s*(\d+)",
        r"nucleon number\s*\(?a\)?\s*=\s*(\d+)",
        r"\ba\s*=\s*(\d+)"
    ]
    patterns_Z = [
        r"nombor proton\s*\(?z\)?\s*=?\s*(\d+)",
        r"proton number\s*\(?z\)?\s*=?\s*(\d+)",
        r"proton\s+(\d+)",
        r"\bz\s*=\s*(\d+)"
    ]

    for ptn in patterns_A:
        m = re.search(ptn, q)
        if m:
            A = int(m.group(1))
            break

    for ptn in patterns_Z:
        m = re.search(ptn, q)
        if m:
            Z = int(m.group(1))
            break

    nuclide = re.search(r"\b(\d{1,3})\s*(\d{1,3})\s*[A-Z][a-z]?\b", text)
    if nuclide and A is None and Z is None:
        A = int(nuclide.group(1))
        Z = int(nuclide.group(2))

    if A is not None and Z is not None:
        return {"A": A, "Z": Z}

    return None


def extract_empirical_masses(text: str) -> Optional[Dict[str, float]]:
    """
    FIX BUG #6: For % composition questions, parse percentages directly.
    No longer relies on formula extraction (which would get "Se" from "Sebatian").
    """
    q = normalize_text(text)
    ql = q.lower()

    if not any(k in ql for k in ["empirical", "formula empirik", "empirik", "%", "peratus"]):
        return None

    # Method 1: explicit "El = X g" format
    matches = re.findall(r"([A-Z][a-z]?)\s*=\s*(\d+(?:\.\d+)?)\s*g", q)
    if matches:
        return {el: float(mass) for el, mass in matches}

    # Method 2: "X g El" format
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*g\s*([A-Z][a-z]?)", q)
    if matches:
        return {el: float(mass) for mass, el in matches}

    # Method 3: percentage composition e.g. "40% karbon, 6.7% hidrogen, 53.3% oksigen"
    # FIX: parse element names in BM/EN then map to symbols
    element_name_map = {
        "karbon": "C", "carbon": "C",
        "hidrogen": "H", "hydrogen": "H",
        "oksigen": "O", "oxygen": "O",
        "nitrogen": "N",
        "sulfur": "S", "belerang": "S",
        "klorin": "Cl", "chlorine": "Cl",
        "natrium": "Na", "sodium": "Na",
        "kalium": "K", "potassium": "K",
        "kalsium": "Ca", "calcium": "Ca",
        "besi": "Fe", "iron": "Fe",
        "tembaga": "Cu", "copper": "Cu",
        "zink": "Zn", "zinc": "Zn",
        "fosforus": "P", "phosphorus": "P",
    }

    pct_pattern = re.findall(r"(\d+(?:\.\d+)?)\s*%\s*([a-z]+)", ql)
    if pct_pattern:
        result = {}
        for pct_str, name in pct_pattern:
            symbol = element_name_map.get(name.lower())
            if symbol:
                result[symbol] = float(pct_str)
        if result:
            return result

    # Method 4: percentage with symbol e.g. "75% C, 25% H"
    pct_sym = re.findall(r"(\d+(?:\.\d+)?)\s*%\s*([A-Z][a-z]?)\b", q)
    if pct_sym:
        result = {}
        for pct_str, sym in pct_sym:
            if sym not in BM_STOPWORDS and is_valid_formula(sym):
                result[sym] = float(pct_str)
        if result:
            return result

    return None


def extract_oxidation_target(text: str, formulas: List[str]) -> Optional[Dict[str, Any]]:
    q = text.lower()
    if not any(k in q for k in ["nombor pengoksidaan", "oxidation number", "pengoksidaan"]):
        return None

    # FIX BUG #3: extract charge from question text, not just from species string
    charge = extract_ion_charge(normalize_text(text))

    species = formulas[0] if formulas else None
    if not species:
        m = re.search(r"dalam\s+([A-Z][A-Za-z0-9()+-]+)", text)
        species = m.group(1) if m else None

    if not species:
        return None

    target = None

    m = re.search(r"nombor pengoksidaan\s+([A-Z][a-z]?)", text)
    if m:
        target = m.group(1)

    if not target:
        m = re.search(r"(?:dalam|of)\s+([A-Z][a-z]?)\s*(?:dalam|in)", text)
        if m:
            target = m.group(1)

    if not target:
        parsed_symbols = re.findall(r"[A-Z][a-z]?", species)
        for sym in parsed_symbols:
            if sym not in {"O", "H", "K", "Na", "Li", "Mg", "Ca", "Ba"}:
                target = sym
                break
        if not target and parsed_symbols:
            target = parsed_symbols[0]

    # If charge still None, try to parse from species string itself
    if charge is None:
        m = re.search(r"([+-]\d*|\d*[+-])\s*$", species)
        if m:
            token = m.group(1).strip()
            if token in ["+", "-"]:
                charge = 1 if token == "+" else -1
            elif token.endswith("+"):
                charge = int(token[:-1]) if token[:-1] else 1
            elif token.endswith("-"):
                charge = -(int(token[:-1]) if token[:-1] else 1)
            elif token.startswith("+"):
                charge = int(token[1:]) if token[1:] else 1
            elif token.startswith("-"):
                charge = -(int(token[1:]) if token[1:] else 1)

    return {"species": species, "target_element": target, "charge": charge}


def extract_redox_change(text: str) -> Optional[Dict[str, Any]]:
    q = text.lower()
    if not any(k in q for k in ["oxidation", "reduction", "penurunan", "pengoksidaan"]):
        return None

    pairs = re.findall(r"([A-Z][a-z]?)\s*\^?\s*(\d+)([+-])", text)
    if len(pairs) >= 2 and pairs[0][0] == pairs[1][0]:
        before = float(pairs[0][1]) * (1 if pairs[0][2] == "+" else -1)
        after = float(pairs[1][1]) * (1 if pairs[1][2] == "+" else -1)
        return {"element": pairs[0][0], "before_ox": before, "after_ox": after}

    m = re.search(r"(?:daripada|from)?\s*(-?\d+(?:\.\d+)?)\s*(?:kepada|to)\s*(-?\d+(?:\.\d+)?)", q)
    if m:
        return {"before_ox": float(m.group(1)), "after_ox": float(m.group(2))}

    return None


# =====================================
# THERMOCHEMISTRY DETECTOR
# FIX BUG #4: detect thermochemistry BEFORE volume/mol chain
# =====================================
def is_thermochemistry_question(ql: str) -> bool:
    """
    Returns True if question is about heat/enthalpy calculation.
    FIX BUG #4: prevents routing to gas volume solver for thermo questions.
    """
    thermo_keywords = [
        "entalpi", "enthalpy", "delta h", "δh",
        "haba", "heat", "kalorimetri", "calorimetry",
        "suhu meningkat", "suhu menurun", "suhu naik", "suhu turun",
        "temperature rise", "temperature drop",
        "exothermic", "endothermic", "eksotermik", "endotermik",
        "pemelarutan", "pembakaran", "peneutralan",
        "dissolution", "combustion", "neutralization", "neutralisation",
    ]
    return any(k in ql for k in thermo_keywords)


def is_titration_question(ql: str) -> bool:
    """
    Returns True if question is about acid-base titration.
    FIX BUG #4: prevents routing to gas volume solver for titration questions.
    """
    titration_keywords = [
        "titrat", "pentitratan", "dititrat",
        "neutralis", "meneutralkan", "neutralkan",
        "neutralise", "neutralize", "neutralization", "neutralisation",
        "titration", "titrate",
    ]
    return any(k in ql for k in titration_keywords)


# =====================================
# STRUCTURED EXTRACTOR
# =====================================
def structured_extract(question: str) -> Optional[Dict[str, Any]]:
    q = normalize_text(question)
    ql = q.lower()

    formulas = extract_valid_formulas(q)
    masses = extract_mass_values(q)
    volumes_cm3 = extract_volume_cm3_values(q)
    volumes_dm3 = extract_volume_dm3_values(q)
    moles = extract_mole_values(q)
    molarities = extract_molarity_values(q)
    ph = extract_ph(q)
    poh = extract_poh(q)         # FIX BUG #7: now only extracts if "poh" keyword present
    h_plus = extract_h_plus(q)
    oh_minus = extract_oh_minus(q)
    temperatures = extract_temperatures(q)
    times = extract_times(q)
    condition = extract_condition(q)
    equation = extract_equation(q)

    particle_words = any(k in ql for k in ["particle", "particles", "zarah", "atom", "atoms", "molecule", "molecules"])
    volume_words = any(k in ql for k in ["volume", "isi padu", "gas", "dm3", "cm3"])
    mole_words = any(k in ql for k in ["mol", "mole", "bilangan mol"])

    # =====================================
    # PRIORITY 0 — THERMOCHEMISTRY
    # FIX BUG #4: MUST be checked BEFORE volume/mol chain
    # "50cm3 larutan" was triggering gas solver instead of enthalpy
    # =====================================
    if is_thermochemistry_question(ql):
        # FIX: calc_moles priority order:
        # 1. molarity x volume (larutan: HCl + NaOH)
        # 2. jisim bahan terlarut / Mr (pemelarutan: NaOH dalam air)
        # 3. mol eksplisit dalam soalan
        calc_moles = None
        if molarities and volumes_cm3:
            calc_moles = molarities[0] * (min(volumes_cm3) / 1000.0)
        elif molarities and volumes_dm3:
            calc_moles = molarities[0] * min(volumes_dm3)
        elif masses and formulas and not molarities:
            # pemelarutan: 2g NaOH → mol = 2/40 = 0.05
            try:
                from formula_parser import molar_mass as _mm
                calc_moles = masses[0] / _mm(formulas[0])
            except Exception:
                calc_moles = None
        elif moles:
            calc_moles = moles[0]

        # FIX BUG #9: jisim LARUTAN bukan jisim bahan terlarut
        if volumes_cm3:
            total_mass = sum(volumes_cm3)
        elif masses and not molarities:
            total_mass = masses[0]
        else:
            total_mass = None

        # FIX BUG (Q3): detect "suhu meningkat X°C" = single delta_T
        # e.g. "suhu meningkat 13.5°C" — tiada suhu awal/akhir eksplisit
        delta_t_match = re.search(
            r'suhu\s+(?:meningkat|naik|turun|menurun)\s+(\d+(?:\.\d+)?)',
            ql
        )
        if delta_t_match and len(temperatures) < 2:
            delta_t_val = float(delta_t_match.group(1))
            if 'turun' in ql or 'menurun' in ql:
                delta_t_val = -delta_t_val
            # Use dummy base temp 25°C; solver only needs delta_t
            t_initial = 25.0
            t_final = 25.0 + delta_t_val
            temperatures = [t_initial, t_final]

        if len(temperatures) >= 2:
            if total_mass and calc_moles:
                return {
                    "task": "delta_h_from_calorimetry",
                    "mass_g": total_mass,
                    "temp_initial": temperatures[0],
                    "temp_final": temperatures[1],
                    "moles": calc_moles,
                }
            if total_mass:
                return {
                    "task": "calorimetry",
                    "mass_g": total_mass,
                    "temp_initial": temperatures[0],
                    "temp_final": temperatures[1],
                }

    # =====================================
    # PRIORITY 1 — TITRATION
    # FIX BUG #4: MUST be checked BEFORE volume/mol chain
    # "20cm3 HCl meneutralkan 40cm3 NaOH" was routing to gas solver
    # =====================================
    if is_titration_question(ql):
        if len(molarities) >= 2 and len(volumes_cm3) >= 1:
            data: Dict[str, Any] = {
                "task": "titration_find_volume",
                "known_molarity": molarities[0],
                "known_volume_cm3": volumes_cm3[0],
                "known_formula": formulas[0] if formulas else "asid",
                "unknown_molarity": molarities[1],
                "unknown_formula": formulas[1] if len(formulas) > 1 else "bes",
            }
            if equation:
                data["equation"] = equation
            return data

        if len(molarities) == 1 and len(volumes_cm3) >= 2:
            data = {
                "task": "titration_find_molarity",
                "known_molarity": molarities[0],
                "known_volume_cm3": volumes_cm3[0],
                "known_formula": formulas[0] if formulas else "diketahui",
                "unknown_formula": formulas[1] if len(formulas) > 1 else "tidak diketahui",
                "unknown_volume_cm3": volumes_cm3[1],
            }
            if equation:
                data["equation"] = equation
            return data

        if masses and len(volumes_cm3) >= 1 and len(formulas) >= 2:
            data = {
                "task": "titration_find_molarity",
                "known_mass_g": masses[0],
                "known_formula": formulas[0],
                "unknown_formula": formulas[1],
                "unknown_volume_cm3": volumes_cm3[-1],
            }
            if equation:
                data["equation"] = equation
            return data

    # =====================================
    # PRIORITY 2 — pH / pOH
    # =====================================
    if "poh" in ql:
        if oh_minus:
            return {"task": "poh_from_oh", "oh_minus": oh_minus}
        if moles:
            return {"task": "poh_from_oh", "oh_minus": moles[0]}

    if "ph" in ql and any(k in ql for k in ["oh-", "oh -", "hidroksida", "hydroxide"]):
        if oh_minus:
            return {"task": "ph_from_poh", "oh_conc": oh_minus}
        # FIX Q8: kepekatan OH- given as molarity value, not oh_minus
        # e.g. 'kepekatan OH- ialah 0.001 mol dm-3' → molarities=[0.001]
        if molarities:
            return {"task": "ph_from_poh", "oh_conc": molarities[0]}
        if moles:
            return {"task": "ph_from_poh", "oh_conc": moles[0]}

    if "ph" in ql and any(k in ql for k in ["h+", "hcl", "hno3", "h2so4", "asid", "acid", "kepekatan h"]):
        if h_plus:
            return {"task": "ph_from_h", "h_plus": h_plus}
        if moles:
            return {"task": "ph_from_h", "h_plus": moles[0]}

    # =====================================
    # PRIORITY 3 — SPECIFIC HIGH-CONFIDENCE TASKS
    # =====================================

    # JMR
    if any(k in ql for k in ["jmr", "jisim molekul relatif", "relative molecular mass", "jisim molar"]):
        if formulas:
            return {"task": "jmr", "formula": formulas[0], "formulas": formulas}

    # Subatomic
    sub = extract_subatomic_data(q)
    if sub:
        return {"task": "subatomic", **sub}

    # Relative atomic mass from isotopes
    isotope_data = extract_isotope_data(q)
    if isotope_data and any(k in ql for k in ["ar", "jisim atom relatif", "relative atomic mass", "isotop", "isotope"]):
        return {"task": "ar_from_abundance", **isotope_data}

    # Empirical formula — FIX BUG #6: uses new % parser, not formula extractor
    empirical = extract_empirical_masses(q)
    if empirical:
        return {"task": "empirical_formula", "element_masses": empirical}

    # Oxidation number — FIX BUG #3: charge now correctly parsed
    ox = extract_oxidation_target(q, formulas)
    if ox:
        return {"task": "oxidation_number", **ox}

    # Redox change
    redox_change = extract_redox_change(q)
    if redox_change and any(k in ql for k in ["oxidation", "reduction", "pengoksidaan", "penurunan"]):
        return {"task": "redox_change", **redox_change}

    # Stoichiometry — equation present + mass given
    # FIX Q4/Q5: detect mass-to-VOLUME (isipadu gas) vs mass-to-MASS
    # Always parse given_formula from LHS, target from RHS
    if equation and masses and any(k in ql for k in [
        "stoichiometry", "stoikiometri", "formed", "terbentuk", "hasilkan",
        "produce", "reacts", "reaction", "mendapan", "precipitate", "pemendapan",
        "terhasil", "dihasilkan", "bertindak balas", "bertindak", "sepenuhnya",
        "completely", "dipanaskan", "dibakar", "terurai", "terbentuk"
    ]):
        eq_parts = equation.split('->')
        lhs_formulas = re.findall(r'[A-Z][A-Za-z0-9()]*', eq_parts[0]) if len(eq_parts) >= 1 else []
        rhs_formulas = re.findall(r'[A-Z][A-Za-z0-9()]*', eq_parts[1]) if len(eq_parts) >= 2 else []

        # FIX BUG A: given_formula = first LHS formula (no filter against soalan formulas[])
        # Because soalan formulas[] may not contain all equation formulas
        given_f = lhs_formulas[0] if lhs_formulas else (formulas[0] if formulas else None)

        # FIX BUG B: use smart target detection
        target_f = _find_target_formula(q, rhs_formulas, lhs_formulas)

        if given_f and target_f and given_f != target_f:
            asks_volume = any(k in ql for k in [
                "isipadu", "volume", "dm3", "cm3", "liter", "litre"
            ])
            task = "stoichiometry_mass_to_volume" if asks_volume else "stoichiometry_mass_to_mass"
            return {
                "task": task,
                "equation": equation,
                "given_formula": given_f,
                "given_mass_g": masses[0],
                "target_formula": target_f,
                "condition": condition,
            }

    # Calorimetry only (no moles available)
    if any(k in ql for k in ["haba", "heat", "calorimetry", "kalorimetri", "q =", "diserap", "dibebaskan"]) and len(temperatures) >= 2 and masses:
        return {
            "task": "calorimetry",
            "mass_g": masses[0],
            "temp_initial": temperatures[0],
            "temp_final": temperatures[1],
        }

    # Enthalpy only (Q and moles given directly)
    if any(k in ql for k in ["enthalpy", "entalpi", "delta h", "δh"]) and moles:
        m_q = re.search(r"(\d+(?:\.\d+)?)\s*(j|kj)\b", ql)
        if m_q:
            qv = float(m_q.group(1)) * (1000 if m_q.group(2) == "kj" else 1)
            return {"task": "enthalpy", "Q_joule": qv, "moles": moles[0]}

    # Thermochemistry type (eksotermik/endotermik)
    if any(k in ql for k in ["eksotermik", "endotermik", "exothermic", "endothermic", "jenis tindak balas"]) and len(temperatures) >= 2:
        return {
            "task": "thermochemistry_type",
            "temp_initial": temperatures[0],
            "temp_final": temperatures[1],
        }

    # Rate from two points
    if any(k in ql for k in ["kadar", "rate"]) and len(times) >= 2:
        if len(volumes_cm3) >= 2 and len(times) >= 2:
            return {
                "task": "rate_from_points",
                "time1": times[0], "value1": volumes_cm3[0],
                "time2": times[1], "value2": volumes_cm3[1],
            }

    # Rate average
    if any(k in ql for k in ["kadar", "rate"]) and len(times) >= 1:
        values = volumes_cm3 or volumes_dm3 or masses or moles
        if len(values) >= 2:
            return {"task": "rate_average", "change": values[1] - values[0], "time": times[0]}
        if len(values) >= 1:
            return {"task": "rate_average", "change": values[0], "time": times[0]}

    # pH / pOH (lower priority fallback)
    if poh is not None and ("ph" in ql or "nilai ph" in ql):
        return {"task": "ph_from_poh", "poh": poh}

    if h_plus is not None and ("ph" in ql or "nilai ph" in ql or "[h+]" in ql or "hydrogen ion" in ql or "ion hidrogen" in ql):
        return {"task": "ph_from_h", "h_plus": h_plus}

    if ph is not None and any(k in ql for k in ["kemolaran", "[h+]", "hcl", "hydrogen ion", "ion hidrogen"]):
        return {"task": "h_from_ph", "ph": ph}

    if oh_minus is not None and ("poh" in ql or "nilai poh" in ql):
        return {"task": "poh_from_oh", "oh_minus": oh_minus}

    if poh is not None and any(k in ql for k in ["[oh", "oh-", "ion hidroksida"]):
        return {"task": "oh_from_poh", "poh": poh}

    # Concentration g/dm3
    if any(k in ql for k in ["g dm3", "g dm-3", "kepekatan"]) and masses and (volumes_cm3 or volumes_dm3):
        data: Dict[str, Any] = {"task": "concentration_g_dm3", "mass_g": masses[0]}
        if volumes_cm3:
            data["volume_cm3"] = volumes_cm3[0]
        if volumes_dm3:
            data["volume_dm3"] = volumes_dm3[0]
        return data

    # FIX BUG #10: mass_from_molarity — "jisim yang diperlukan untuk membuat larutan"
    # e.g. "berapa jisim NaOH untuk buat 500cm3 larutan 0.5 mol dm-3?"
    if any(k in ql for k in ["diperlukan", "required", "needed", "membuat larutan",
                               "prepare", "sediakan", "buat larutan"]):
        if molarities and (volumes_cm3 or volumes_dm3) and formulas:
            vol_dm3 = (min(volumes_cm3) / 1000.0) if volumes_cm3 else min(volumes_dm3)
            return {
                "task": "mass_from_molarity",
                "molarity": molarities[0],
                "volume_dm3": vol_dm3,
                "formula": formulas[0],
                "formulas": formulas,
            }

    # Molarity from mass
    if any(k in ql for k in ["kemolaran", "molarity"]) and masses and formulas and (volumes_dm3 or volumes_cm3):
        data: Dict[str, Any] = {
            "task": "molarity_from_mass",
            "mass_g": masses[0],
            "formula": formulas[0],
            "formulas": formulas,
        }
        if volumes_dm3:
            data["volume_dm3"] = volumes_dm3[0]
        if volumes_cm3:
            data["volume_cm3"] = volumes_cm3[0]
        return data

    # Dilution
    if any(k in ql for k in ["dilution", "pencairan", "m1v1", "m2v2"]) and len(molarities) >= 2:
        v_candidates = [float(x) for x in re.findall(
            r"(?:v2|akhir|final volume|isipadu akhir)\s*=?\s*(\d+(?:\.\d+)?)", ql
        )]
        V2 = v_candidates[0] if v_candidates else (
            volumes_cm3[0] if volumes_cm3 else (volumes_dm3[0] if volumes_dm3 else None)
        )
        if V2 is not None:
            return {"task": "dilution", "M1": molarities[0], "M2": molarities[1], "V2": V2}

    # =====================================
    # MOL / GAS CHAIN — LOWEST PRIORITY
    # FIX BUG #4: only reached if NOT thermo/titration question
    # =====================================
    if particle_words and volume_words and (volumes_dm3 or volumes_cm3):
        data: Dict[str, Any] = {"task": "particles_from_volume", "condition": condition}
        if volumes_dm3:
            data["volume_dm3"] = volumes_dm3[0]
        if volumes_cm3:
            data["volume_cm3"] = volumes_cm3[0]
        return data

    if particle_words and masses and formulas:
        return {
            "task": "particles_from_mass",
            "mass_g": masses[0],
            "formula": formulas[0],
            "formulas": formulas,
        }

    if any(k in ql for k in ["mass", "jisim"]) and volume_words and formulas and (volumes_dm3 or volumes_cm3):
        data = {
            "task": "mass_from_volume",
            "formula": formulas[0],
            "formulas": formulas,
            "condition": condition,
        }
        if volumes_dm3:
            data["volume_dm3"] = volumes_dm3[0]
        if volumes_cm3:
            data["volume_cm3"] = volumes_cm3[0]
        return data

    if any(k in ql for k in ["volume", "isi padu", "isipadu"]) and masses and formulas:
        return {
            "task": "volume_from_mass",
            "mass_g": masses[0],
            "formula": formulas[0],
            "formulas": formulas,
            "condition": condition,
        }

    # FIX BUG #1: multi-formula "jumlah mol" question
    # e.g. "5.6g N2 dan 3.2g O2 — jumlah mol?"
    if mole_words and len(formulas) >= 2 and len(masses) >= 2:
        if any(k in ql for k in ["jumlah", "total", "keseluruhan", "combined", "semua", "campuran"]):
            return {
                "task": "moles_multi",
                "formulas": formulas,
                "masses": masses,
            }

    if mole_words and masses and formulas:
        return {
            "task": "moles_from_mass",
            "mass_g": masses[0],
            "formula": formulas[0],
            "formulas": formulas,
        }

    if mole_words and (volumes_dm3 or volumes_cm3):
        data = {"task": "moles_from_volume", "condition": condition}
        if volumes_dm3:
            data["volume_dm3"] = volumes_dm3[0]
        if volumes_cm3:
            data["volume_cm3"] = volumes_cm3[0]
        return data

    if particle_words and moles:
        return {"task": "particles_from_moles", "moles": moles[0]}

    if volume_words and moles:
        return {"task": "volume_from_moles", "moles": moles[0], "condition": condition}

    if any(k in ql for k in ["mass", "jisim"]) and moles and formulas:
        return {
            "task": "mass_from_moles",
            "moles": moles[0],
            "formula": formulas[0],
            "formulas": formulas,
        }

    return None
