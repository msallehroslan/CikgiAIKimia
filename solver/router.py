# router.py — FIXED v3.1.0
# Changes:
#   FIX BUG #2: RTP condition correctly passed through
#   FIX BUG #4: thermochemistry + titration checked BEFORE mol/volume chain
#   Cleaner classify() fallback with same priority order as extractor

from extractor import structured_extract, is_thermochemistry_question, is_titration_question


def classify(question: str) -> str:
    """
    Fallback classifier — only used when structured_extract returns None.
    Priority order mirrors extractor.py structured_extract().
    """
    q = question.lower()

    # ── PRIORITY 0: THERMOCHEMISTRY ────────────────────────────────────────
    # FIX BUG #4: must come BEFORE any volume/mol check
    if is_thermochemistry_question(q):
        if any(k in q for k in ["entalpi", "enthalpy", "delta h", "pemelarutan",
                                  "pembakaran", "peneutralan", "dissolution",
                                  "combustion", "neutralization", "neutralisation"]):
            return "delta_h_from_calorimetry"
        return "calorimetry"

    # ── PRIORITY 1: TITRATION ───────────────────────────────────────────────
    # FIX BUG #4: must come BEFORE mol/volume check
    if is_titration_question(q):
        return "titration_find_molarity"

    # ── PRIORITY 2: pH / pOH ───────────────────────────────────────────────
    if "poh" in q and "ph" in q:
        return "ph_from_poh"
    if "poh" in q:
        return "poh_from_oh"
    if "ph" in q and any(k in q for k in ["oh-", "hidroksida", "naoh", "koh"]):
        return "ph_from_poh"
    if "ph" in q and any(k in q for k in ["h+", "hcl", "hno3", "h2so4", "asid", "acid"]):
        return "ph_from_h"
    if "ph" in q:
        return "ph_from_h"

    # ── PRIORITY 3: SPECIFIC TASKS ─────────────────────────────────────────
    if any(k in q for k in ["jmr", "jisim molekul relatif", "jisim molar", "molar mass"]):
        return "jmr"

    if any(k in q for k in ["nombor pengoksidaan", "oxidation number"]):
        return "oxidation_number"

    if any(k in q for k in ["isotop", "isotope", "jisim atom relatif", "relative atomic mass"]):
        return "ar_from_abundance"

    if any(k in q for k in ["proton", "neutron", "electron", "nukleon", "nucleon"]):
        return "subatomic"

    if any(k in q for k in ["formula empirik", "empirical formula", "empirik"]):
        return "empirical_formula"

    # ── PRIORITY 4: STOICHIOMETRY ──────────────────────────────────────────
    if "->" in q or "reacts" in q:
        if any(k in q for k in ["mass", "jisim", "mendapan", "precipitate"]):
            return "stoichiometry_mass_to_mass"

    # ── PRIORITY 5: RATE ───────────────────────────────────────────────────
    if any(k in q for k in ["rate", "kadar"]):
        return "rate_average"

    # ── PRIORITY 6: MOL / GAS CHAIN ───────────────────────────────────────
    # FIX BUG #2: RTP keyword → condition="RTP" handled in extractor
    # Here we just route to correct task
    if any(k in q for k in ["particle", "particles", "zarah", "molecule", "molecules"]):
        if any(k in q for k in ["dm3", "cm3", "volume", "isi padu"]):
            return "particles_from_volume"
        if any(k in q for k in ["mass", "jisim", " g", "gram"]):
            return "particles_from_mass"
        if any(k in q for k in ["mol", "mole"]):
            return "particles_from_moles"

    if any(k in q for k in ["mass", "jisim"]) and any(k in q for k in ["dm3", "cm3", "volume", "isi padu", "gas"]):
        return "mass_from_volume"

    if any(k in q for k in ["volume", "isi padu", "isipadu"]) and any(k in q for k in ["mass", "jisim", " g", "gram"]):
        return "volume_from_mass"

    if any(k in q for k in ["mol", "mole", "bilangan mol"]):
        if any(k in q for k in ["mass", "jisim", " g", "gram"]):
            return "moles_from_mass"
        if any(k in q for k in ["volume", "isi padu", "dm3", "cm3"]):
            return "moles_from_volume"

    if any(k in q for k in ["volume", "isi padu", "dm3", "cm3"]) and any(k in q for k in ["mol", "mole"]):
        return "volume_from_moles"

    return "unknown"


def route(question: str):
    """
    Priority:
    1. structured_extract (HIGH ACCURACY — deterministic Python)
    2. classify() fallback
    """
    data = structured_extract(question)

    if data:
        return data["task"], data

    task = classify(question)
    return task, None
