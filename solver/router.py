# router.py (FULL UPDATED)

from extractor import structured_extract


def classify(question: str) -> str:
    q = question.lower()

    # ===============================
    # pH / pOH FIRST - before any mol/volume check
    # ===============================
    if "poh" in q:
        return "poh_from_oh"
    if "ph" in q and any(k in q for k in ["oh-", "hidroksida", "naoh", "koh"]):
        return "ph_from_poh"
    if "ph" in q and any(k in q for k in ["hcl", "hno3", "h2so4", "asid", "acid"]):
        return "ph_from_h"

    # ===============================
    # TITRATION - before mol check
    # ===============================
    if any(k in q for k in ["titrat", "pentitratan", "dititrat", "neutralis"]):
        return "titration_find_volume"

    if any(k in q for k in ["jmr", "jisim molekul relatif", "jisim molar", "molar mass"]):
        return "jmr"

        # ===============================
    # MOL / GAS (PRIORITY HIGH)
    # ===============================

    # particles chain
    if any(k in q for k in ["particle", "particles", "zarah", "atom", "molecule"]):
        if any(k in q for k in ["dm3", "cm3", "volume", "isi padu", "gas"]):
            return "particles_from_volume"
        if any(k in q for k in ["mass", "jisim", " g", "gram"]):
            return "particles_from_mass"
        if any(k in q for k in ["mol", "mole"]):
            return "particles_from_moles"

    # mass-volume chain
    if any(k in q for k in ["mass", "jisim"]) and any(k in q for k in ["dm3", "cm3", "volume", "isi padu", "gas"]):
        return "mass_from_volume"

    if any(k in q for k in ["volume", "isi padu"]) and any(k in q for k in ["mass", "jisim", " g", "gram"]):
        return "volume_from_mass"

    # mole conversions
    # pH / pOH must come BEFORE mol check
    if "poh" in q:
        return "poh_from_oh"
    if "ph" in q and any(k in q for k in ["oh-", "oh -", "hidroksida", "hydroxide", "alkali", "naoh", "koh"]):
        return "ph_from_poh"
    if "ph" in q and any(k in q for k in ["kepekatan h+", "h+ jika", "h+ apabila", "concentration"]):
        return "h_from_ph"

    # Titration must come BEFORE mol check
    if any(k in q for k in ["titrat", "pentitratan", "neutralis", "dititrat"]):
        return "titration_find_volume"

    if any(k in q for k in ["mol", "mole", "bilangan mol"]):
        if any(k in q for k in ["mass", "jisim", " g", "gram"]):
            return "moles_from_mass"
        if any(k in q for k in ["volume", "isi padu", "dm3", "cm3"]):
            return "moles_from_volume"

    if any(k in q for k in ["volume", "isi padu", "dm3", "cm3"]) and any(k in q for k in ["mol", "mole"]):
        return "volume_from_moles"

    # ===============================
    # STOICHIOMETRY (IMPORTANT FIX)
    # ===============================
    if "->" in q or "reacts" in q or "reaction" in q:
        if any(k in q for k in ["mass", "jisim"]):
            return "stoichiometry_mass_to_mass"

    # ===============================
    # ACID / BASE
    # ===============================
    if "poh" in q and "ph" in q:
        return "ph_from_poh"

    if "poh" in q:
        return "poh_from_oh"

    if "ph" in q:
        return "ph_from_h"

    # TITRATION (FIXED)
    if any(k in q for k in ["titration", "titrate", "pentitratan", "neutralise", "neutralization", "neutralisation"]):
        return "titration_find_volume"

    # ===============================
    # THERMOCHEMISTRY (FIXED CHAIN)
    # ===============================
    if any(k in q for k in ["enthalpy", "entalpi", "delta h", "Δh"]):
        return "delta_h_from_calorimetry"

    if any(k in q for k in ["heat", "haba", "calorimetry", "kalorimetri"]):
        return "calorimetry"

    # ===============================
    # RATE
    # ===============================
    if any(k in q for k in ["rate", "kadar"]):
        return "rate_average"

    # ===============================
    # ATOMIC STRUCTURE
    # ===============================
    if any(k in q for k in ["isotope", "isotop", "jisim atom relatif", " ar "]):
        return "ar_from_abundance"

    if any(k in q for k in ["proton", "electron", "neutron", "nucleon", "nukleon"]):
        return "subatomic"

    # ===============================
    # REDOX (FIXED)
    # ===============================
    if any(k in q for k in ["nombor pengoksidaan", "oxidation number"]):
        return "oxidation_number"

    return "unknown"


def route(question: str):
    """
    Priority:
    1. structured extraction (HIGH ACCURACY)
    2. fallback classification
    """

    data = structured_extract(question)

    if data:
        return data["task"], data

    # fallback classifier
    task = classify(question)

    return task, None