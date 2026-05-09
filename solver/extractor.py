import re
from typing import Any, Dict, List, Optional

from formula_parser import is_valid_formula


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
    t = t.replace("×10^", "e")
    t = t.replace("x10^", "e")
    t = t.replace("× 10^", "e")
    t = t.replace("x 10^", "e")
    t = t.replace("×10", "e")
    t = t.replace("x10", "e")
    t = re.sub(r"\s+", " ", t)
    return t


# =====================================
# GENERIC EXTRACTORS
# =====================================
def extract_valid_formulas(text: str) -> List[str]:
    t = normalize_text(text)

    candidates = re.findall(r"\b[A-Z][A-Za-z0-9()·.]*[A-Za-z0-9)]\b", t)

    bad_words = {
        "Calculate", "Find", "Determine", "What", "Which", "How",
        "Mass", "Volume", "Molarity", "Formula", "Jawapan", "Diberi", "Pengiraan",
        "Mol", "Rate", "Change", "Time", "Ar", "RTP", "STP",
        "Carbon", "Dioxide", "Hydrogen", "Oxygen", "Nitrogen", "Sulfur",
        "Chlorine", "Water", "Acid", "Base", "Salt", "Gas",
        "Hitungkan", "Tentukan", "Berapakah", "Apakah", "Nyatakan", "Terangkan",
        "Jelaskan", "Bandingkan", "Jawapan", "Diberi", "Pengiraan", "Langkah",
        "Jisim", "Isipadu", "Kepekatan", "Kemolaran",
        "Dalam", "Dengan", "Untuk", "Antara", "Tindak", "Balas", "Larutan",
    }

    valid: List[str] = []
    seen = set()

    for cand in candidates:
        if cand in bad_words or cand in seen:
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
    vals: List[float] = []
    tl = text.lower()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:mol\s*dm3|m\b)", tl):
        vals.append(float(m.group(1)))
    return vals


def extract_ph(text: str) -> Optional[float]:
    m = re.search(r"\bph\s*=?\s*(\d+(?:\.\d+)?)", text.lower())
    return float(m.group(1)) if m else None


def extract_poh(text: str) -> Optional[float]:
    m = re.search(r"\bpoh\s*=?\s*(\d+(?:\.\d+)?)", text.lower())
    return float(m.group(1)) if m else None


def extract_h_plus(text: str) -> Optional[float]:
    tl = normalize_text(text).lower()

    patterns = [
        r"\[\s*h\+\s*\]\s*=?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        r"hydrogen[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mol\s*dm3",
        r"ion hidrogen[^\d]*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mol\s*dm3",
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
    t = normalize_text(text)
    t = t.replace(".", " ")

    m = re.search(r"([A-Za-z0-9()\[\]+.\s]+(?:->)[A-Za-z0-9()\[\]+.\s]+)", t)
    if m:
        eq = re.sub(r"\s*->\s*", " -> ", m.group(1).strip())
        eq = re.sub(r"\s+", " ", eq).strip()

        stop_words = [
            " calculate ", " find ", " determine ", " when ", " what ", " volume ",
            " mass ", " moles ", " particles ", " of ", " in ", " at ", " used ",
            " formed ", " reacts ", " reaction "
        ]
        eq_lower = f" {eq.lower()} "
        cut_pos = len(eq)
        for w in stop_words:
            pos = eq_lower.find(w)
            if pos != -1:
                cut_pos = min(cut_pos, pos)
        eq = eq[:cut_pos].strip()

        if "->" in eq:
            return eq
    return None


def extract_isotope_data(text: str) -> Optional[Dict[str, List[float]]]:
    normalized = text.replace("–", "-")
    masses = [float(x) for x in re.findall(r"\b[A-Z][a-z]?-(\d+(?:\.\d+)?)", normalized)]
    if not masses:
        return None

    percent_abundances = [float(x) for x in re.findall(r"\((\d+(?:\.\d+)?)%\)", normalized)]
    if len(percent_abundances) == len(masses):
        return {"isotope_masses": masses, "abundances": percent_abundances}

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
    q = normalize_text(text)
    if not any(k in q.lower() for k in ["empirical", "formula empirik"]):
        return None

    matches = re.findall(r"([A-Z][a-z]?)\s*=\s*(\d+(?:\.\d+)?)\s*g", q)
    if matches:
        return {el: float(mass) for el, mass in matches}

    matches = re.findall(r"(\d+(?:\.\d+)?)\s*g\s*([A-Z][a-z]?)", q)
    if matches:
        return {el: float(mass) for mass, el in matches}

    return None


def extract_oxidation_target(text: str, formulas: List[str]) -> Optional[Dict[str, Any]]:
    q = text.lower()
    if not any(k in q for k in ["nombor pengoksidaan", "oxidation number", "pengoksidaan"]):
        return None

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

    charge = None
    m = re.search(r"([+-]\d+|\d+[+-]|[+-])\s*$", species)
    if m:
        token = m.group(1)
        if token in ["+", "-"]:
            charge = 1 if token == "+" else -1
        elif token.endswith("+"):
            charge = int(token[:-1])
        elif token.endswith("-"):
            charge = -int(token[:-1])
        elif token.startswith("+"):
            charge = int(token[1:])
        elif token.startswith("-"):
            charge = -int(token[1:])

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
    poh = extract_poh(q)
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
    # PRIORITY 0: pH / pOH / TITRATION
    # Must be checked BEFORE mol/volume chain
    # =====================================

    # pOH detection
    if "poh" in ql:
        if oh_minus:
            return {"task": "poh_from_oh", "oh_conc": oh_minus[0]}
        if moles:
            return {"task": "poh_from_oh", "oh_conc": moles[0]}

    # pH from OH- (alkali)
    if "ph" in ql and any(k in ql for k in ["oh-", "oh -", "hidroksida", "hydroxide"]):
        if oh_minus:
            return {"task": "ph_from_poh", "oh_conc": oh_minus[0]}
        if moles:
            return {"task": "ph_from_poh", "oh_conc": moles[0]}

    # pH from H+ (acid)  
    if "ph" in ql and any(k in ql for k in ["h+", "hcl", "hno3", "h2so4", "asid", "acid"]):
        if h_plus:
            return {"task": "ph_from_h", "h_conc": h_plus[0]}
        if moles:
            return {"task": "ph_from_h", "h_conc": moles[0]}

    # Titration
    if any(k in ql for k in ["titrat", "pentitratan", "dititrat", "neutralis"]):
        if len(molarities) >= 2 and volumes_cm3:
            return {
                "task": "titration_find_volume",
                "known_molarity": molarities[0],
                "known_volume_cm3": volumes_cm3[0],
                "known_formula": formulas[0] if formulas else "NaOH",
                "unknown_molarity": molarities[1],
                "unknown_formula": formulas[1] if len(formulas) > 1 else "HCl",
            }
        if molarities and volumes_cm3:
            return {
                "task": "titration_find_volume",
                "known_molarity": molarities[0],
                "known_volume_cm3": volumes_cm3[0],
                "known_formula": formulas[0] if formulas else "NaOH",
                "unknown_molarity": molarities[-1],
                "unknown_formula": formulas[-1] if formulas else "HCl",
            }

    # =====================================
    # PRIORITY: SPECIFIC / HIGH-CONFIDENCE TASKS
    # =====================================

    # JMR
    if any(k in ql for k in ["jmr", "jisim molekul relatif", "relative molecular mass"]):
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

    # Empirical formula
    empirical = extract_empirical_masses(q)
    if empirical:
        return {"task": "empirical_formula", "element_masses": empirical}

    # Oxidation number
    ox = extract_oxidation_target(q, formulas)
    if ox:
        return {"task": "oxidation_number", **ox}

    # Redox increase / decrease
    redox_change = extract_redox_change(q)
    if redox_change and any(k in ql for k in ["oxidation", "reduction", "pengoksidaan", "penurunan"]):
        return {"task": "redox_change", **redox_change}

    # Stoichiometry mass-to-mass
    if equation and len(formulas) >= 2 and masses and any(k in ql for k in ["stoichiometry", "stoikiometri", "formed", "terbentuk", "hasilkan", "produce", "reacts", "reaction"]):
        return {
            "task": "stoichiometry_mass_to_mass",
            "equation": equation,
            "given_formula": formulas[0],
            "given_mass_g": masses[0],
            "target_formula": formulas[1],
        }

    # Titration
    if any(k in ql for k in ["titrate", "titration", "pentitratan", "neutralise", "neutralization", "neutralisation"]) and len(formulas) >= 2:
        if len(molarities) >= 2 and volumes_cm3:
            data = {
                "task": "titration_find_volume",
                "known_molarity": molarities[0],
                "known_volume_cm3": volumes_cm3[0],
                "known_formula": formulas[0],
                "unknown_molarity": molarities[1],
                "unknown_formula": formulas[1],
            }
            if equation:
                data["equation"] = equation
            return data

        if masses and volumes_cm3:
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

        if len(molarities) >= 1 and len(volumes_cm3) >= 2:
            data = {
                "task": "titration_find_molarity",
                "known_molarity": molarities[0],
                "known_volume_cm3": volumes_cm3[0],
                "known_formula": formulas[0],
                "unknown_formula": formulas[1],
                "unknown_volume_cm3": volumes_cm3[1],
            }
            if equation:
                data["equation"] = equation
            return data

    # Delta H from calorimetry
    if any(k in ql for k in ["enthalpy", "entalpi", "delta h", "Δh"]) and len(temperatures) >= 2 and masses and moles:
        return {
            "task": "delta_h_from_calorimetry",
            "mass_g": masses[0],
            "temp_initial": temperatures[0],
            "temp_final": temperatures[1],
            "moles": moles[0],
        }

    # Calorimetry only
    if any(k in ql for k in ["haba", "heat", "calorimetry", "kalorimetri", "q =", "diserap", "dibebaskan"]) and len(temperatures) >= 2 and masses:
        return {
            "task": "calorimetry",
            "mass_g": masses[0],
            "temp_initial": temperatures[0],
            "temp_final": temperatures[1],
        }

    # Enthalpy only
    if any(k in ql for k in ["enthalpy", "entalpi", "delta h", "Δh"]) and moles:
        m_q = re.search(r"(\d+(?:\.\d+)?)\s*(j|kj)\b", ql)
        if m_q:
            qv = float(m_q.group(1)) * (1000 if m_q.group(2) == "kj" else 1)
            return {"task": "enthalpy", "Q_joule": qv, "moles": moles[0]}

    # Thermochemistry type
    if any(k in ql for k in ["eksotermik", "endotermik", "exothermic", "endothermic", "jenis tindak balas"]) and len(temperatures) >= 2:
        return {
            "task": "thermochemistry_type",
            "temp_initial": temperatures[0],
            "temp_final": temperatures[1],
        }

    # Rate from two points
    if any(k in ql for k in ["kadar", "rate"]) and len(times) >= 2:
        numeric_pairs = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:min|minute|minutes|s|sec|second|seconds)\D+(\d+(?:\.\d+)?)\s*(?:cm3|ml|dm3|g|mol)",
            ql
        )
        if len(numeric_pairs) >= 2:
            t1, v1 = map(float, numeric_pairs[0])
            t2, v2 = map(float, numeric_pairs[1])
            return {"task": "rate_from_points", "time1": t1, "value1": v1, "time2": t2, "value2": v2}

        if len(volumes_cm3) >= 2 and len(times) >= 2:
            return {
                "task": "rate_from_points",
                "time1": times[0],
                "value1": volumes_cm3[0],
                "time2": times[1],
                "value2": volumes_cm3[1],
            }

    # Rate average
    if any(k in ql for k in ["kadar", "rate"]) and len(times) >= 1:
        values = volumes_cm3 or volumes_dm3 or masses or moles
        if len(values) >= 2:
            return {"task": "rate_average", "change": values[1] - values[0], "time": times[0]}
        if len(values) >= 1:
            return {"task": "rate_average", "change": values[0], "time": times[0]}

    # pH / pOH
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

    # Concentration g dm3
    if any(k in ql for k in ["g dm3", "g dm-3", "kepekatan"]) and masses and (volumes_cm3 or volumes_dm3):
        data: Dict[str, Any] = {"task": "concentration_g_dm3", "mass_g": masses[0]}
        if volumes_cm3:
            data["volume_cm3"] = volumes_cm3[0]
        if volumes_dm3:
            data["volume_dm3"] = volumes_dm3[0]
        return data

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
        v_candidates = [float(x) for x in re.findall(r"(?:v2|akhir|final volume|isipadu akhir)\s*=?\s*(\d+(?:\.\d+)?)", ql)]
        V2 = v_candidates[0] if v_candidates else (
            volumes_cm3[0] if volumes_cm3 else (
                volumes_dm3[0] if volumes_dm3 else None
            )
        )
        if V2 is not None:
            return {"task": "dilution", "M1": molarities[0], "M2": molarities[1], "V2": V2}

    # =====================================
    # MOL / GAS CHAIN
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

    if any(k in ql for k in ["volume", "isi padu"]) and masses and formulas:
        return {
            "task": "volume_from_mass",
            "mass_g": masses[0],
            "formula": formulas[0],
            "formulas": formulas,
            "condition": condition,
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