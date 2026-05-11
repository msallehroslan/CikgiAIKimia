"""
solver/dispatcher.py — Cikgu AI Kimia
=======================================
Central dispatch: maps task strings → solver function calls.

This is the MISSING FILE that main.py imports as solve_by_task.
Without this, the system crashes on startup.

All solver functions remain deterministic Python — NO LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("cikgu.dispatcher")

# Import all solver functions
from solver_engine import (
    calculate_molar_mass,
    solve_moles_from_mass,
    solve_moles_from_volume,
    solve_stoichiometry,
    solve_thermochemistry,
    solve_ph,
    solve_titration,
    solve_concentration,
    solve_dilution,
    solve_voltaic_cell,
    solve_molarity_from_dh,
    solve_empirical_formula,
    solve_rate_of_reaction,
    solve_relative_ar,
    solve_mass_from_molarity,
    solve_molar_mass,
)

NA = 6.02e23
VM_RTP = 24.0
VM_STP = 22.4


# ── Missing solvers (implemented here until moved to solver_engine.py) ─────

def _solve_subatomic(data: dict) -> dict:
    """Proton/neutron/electron from A and Z. Deterministic."""
    A = data.get("A")
    Z = data.get("Z")
    if A is None or Z is None:
        return {"error": "Perlu A (nombor nukleon) dan Z (nombor proton)"}
    protons   = Z
    neutrons  = A - Z
    electrons = Z   # neutral atom
    if neutrons < 0:
        return {"error": f"Nombor nukleon ({A}) kurang dari nombor proton ({Z})"}
    lang = data.get("lang", "BM")
    if lang == "BM":
        answer = (
            f"🧮 Jawapan Cikgu AI Kimia\n\n"
            f"Diberi:\n  A = {A}, Z = {Z}\n\n"
            f"Jawapan:\n"
            f"  Proton   = Z = {protons}\n"
            f"  Neutron  = A − Z = {A} − {Z} = {neutrons}\n"
            f"  Elektron = Z = {electrons} (atom neutral)"
        )
    else:
        answer = (
            f"🧮 Cikgu AI Kimia Answer\n\n"
            f"Given:\n  A = {A}, Z = {Z}\n\n"
            f"Answer:\n"
            f"  Protons   = Z = {protons}\n"
            f"  Neutrons  = A − Z = {A} − {Z} = {neutrons}\n"
            f"  Electrons = Z = {electrons} (neutral atom)"
        )
    return {"answer": answer, "protons": protons,
            "neutrons": neutrons, "electrons": electrons, "task": "subatomic"}


def _solve_particles_from_moles(data: dict) -> dict:
    """Number of particles = n × NA."""
    moles = data.get("moles")
    if moles is None:
        return {"error": "Perlu bilangan mol"}
    particles = moles * NA
    lang = data.get("lang", "BM")
    if lang == "BM":
        answer = (
            f"🧮 Jawapan Cikgu AI Kimia\n\n"
            f"Diberi:\n  n = {moles} mol\n\n"
            f"Formula:\n  Bilangan zarah = n × NA\n\n"
            f"Pengiraan:\n  = {moles} × 6.02×10²³\n"
            f"  = {particles:.3e} zarah\n\n"
            f"Jawapan:\n  Bilangan zarah = {particles:.3e}"
        )
    else:
        answer = (
            f"🧮 Cikgu AI Kimia Answer\n\n"
            f"Given:\n  n = {moles} mol\n\n"
            f"Formula:\n  Number of particles = n × NA\n\n"
            f"Calculation:\n  = {moles} × 6.02×10²³\n"
            f"  = {particles:.3e} particles\n\n"
            f"Answer:\n  Number of particles = {particles:.3e}"
        )
    return {"answer": answer, "particles": particles, "task": "particles_from_moles"}


def _solve_particles_from_mass(data: dict) -> dict:
    """Particles = (mass/M) × NA."""
    mass    = data.get("mass_g")
    formula = data.get("formula", "")
    if mass is None or not formula:
        return {"error": "Perlu jisim (g) dan formula kimia"}
    M = calculate_molar_mass(formula)
    if not M:
        return {"error": f"Tidak kenal formula: {formula}"}
    n         = mass / M
    particles = n * NA
    lang = data.get("lang", "BM")
    answer = (
        f"🧮 Jawapan Cikgu AI Kimia\n\n"
        f"Diberi:\n  m = {mass} g, Formula = {formula}, M = {M} g mol⁻¹\n\n"
        f"Pengiraan:\n  n = {mass} ÷ {M} = {round(n,5)} mol\n"
        f"  Bilangan zarah = {round(n,5)} × 6.02×10²³ = {particles:.3e}\n\n"
        f"Jawapan:\n  Bilangan zarah = {particles:.3e}"
    )
    return {"answer": answer, "particles": particles, "n_mol": round(n,5), "task": "particles_from_mass"}


def _solve_particles_from_volume(data: dict) -> dict:
    """Particles from gas volume at RTP/STP."""
    vm  = VM_STP if data.get("condition", "RTP").upper() == "STP" else VM_RTP
    vol_dm3 = data.get("volume_dm3") or (data.get("volume_cm3", 0) / 1000)
    if not vol_dm3:
        return {"error": "Perlu isipadu gas"}
    n         = vol_dm3 / vm
    particles = n * NA
    lang = data.get("lang", "BM")
    answer = (
        f"🧮 Jawapan Cikgu AI Kimia\n\n"
        f"Diberi:\n  V = {vol_dm3} dm³, Vm = {vm} dm³ mol⁻¹\n\n"
        f"Pengiraan:\n  n = {vol_dm3} ÷ {vm} = {round(n,5)} mol\n"
        f"  Bilangan zarah = {round(n,5)} × 6.02×10²³ = {particles:.3e}\n\n"
        f"Jawapan:\n  Bilangan zarah = {particles:.3e}"
    )
    return {"answer": answer, "particles": particles, "n_mol": round(n,5), "task": "particles_from_volume"}


def _solve_oxidation_number(data: dict) -> dict:
    """
    Solve oxidation number for a target element in a species.
    Uses standard rules: O=-2, H=+1, group 1=+1, group 2=+2.
    Sum of oxidation numbers = ion charge (or 0 for neutral).
    """
    from formula_parser import parse_formula

    species        = data.get("species", "")
    target_element = data.get("target_element")
    charge         = data.get("charge") or 0   # ion charge

    if not species or not target_element:
        return {"error": "Perlu spesies dan unsur sasaran"}

    # Known fixed oxidation numbers
    FIXED_OX = {
        "O":  -2, "H": +1, "F": -1,
        "Na": +1, "K": +1, "Li": +1, "Rb": +1, "Cs": +1,
        "Mg": +2, "Ca": +2, "Ba": +2, "Sr": +2,
        "Al": +3,
        "Cl": -1, "Br": -1, "I": -1,   # in binary compounds
    }
    # Exceptions: O in peroxides = -1, H in metal hydrides = -1
    # (SPM level doesn't require these edge cases)

    try:
        composition = parse_formula(species)
    except Exception as e:
        return {"error": f"Tidak dapat parse formula '{species}': {e}"}

    if target_element not in composition:
        return {"error": f"Unsur '{target_element}' tidak dijumpai dalam '{species}'"}

    target_count = composition[target_element]

    # Sum known oxidation contributions
    known_sum = 0
    for elem, count in composition.items():
        if elem == target_element:
            continue
        ox = FIXED_OX.get(elem)
        if ox is not None:
            known_sum += ox * count
        # else: unknown — assume 0 (simplification for SPM level)

    # Solve: known_sum + target_element_ox * count = charge
    # target_element_ox = (charge - known_sum) / count
    ox_number = (charge - known_sum) / target_count

    if ox_number != int(ox_number):
        ox_str = f"{ox_number:.2f}"   # non-integer (fractional oxidation)
    else:
        ox_number = int(ox_number)
        ox_str = f"+{ox_number}" if ox_number >= 0 else str(ox_number)

    lang = data.get("lang", "BM")
    other_contributions = [
        f"{elem}: {FIXED_OX[elem]:+d} × {count} = {FIXED_OX[elem]*count:+d}"
        for elem, count in composition.items()
        if elem != target_element and elem in FIXED_OX
    ]
    contrib_str = "\n  ".join(other_contributions) if other_contributions else "(tiada)"

    if lang == "BM":
        answer = (
            f"🧮 Jawapan Cikgu AI Kimia\n\n"
            f"Diberi:\n  Spesies = {species}, cas ion = {charge:+d}\n\n"
            f"Sumbangan oksidaan lain:\n  {contrib_str}\n\n"
            f"Pengiraan:\n"
            f"  Jumlah cas = 0 (atau {charge:+d} untuk ion)\n"
            f"  {known_sum:+d} + (x × {target_count}) = {charge}\n"
            f"  x = ({charge} − {known_sum}) ÷ {target_count} = {ox_str}\n\n"
            f"Jawapan:\n  Nombor pengoksidaan {target_element} = {ox_str}"
        )
    else:
        answer = (
            f"🧮 Cikgu AI Kimia Answer\n\n"
            f"Given:\n  Species = {species}, ion charge = {charge:+d}\n\n"
            f"Other oxidation contributions:\n  {contrib_str}\n\n"
            f"Calculation:\n"
            f"  Total charge = {charge}\n"
            f"  {known_sum:+d} + (x × {target_count}) = {charge}\n"
            f"  x = ({charge} − {known_sum}) ÷ {target_count} = {ox_str}\n\n"
            f"Answer:\n  Oxidation number of {target_element} = {ox_str}"
        )
    return {"answer": answer, "oxidation_number": ox_str, "task": "oxidation_number"}


def _solve_mass_from_volume(data: dict) -> dict:
    formula     = data.get("formula", "")
    vol_dm3     = data.get("volume_dm3") or (data.get("volume_cm3", 0) / 1000)
    condition   = data.get("condition", "RTP")
    ar_override = data.get("ar_override")
    vm          = VM_STP if condition.upper() == "STP" else VM_RTP
    M           = calculate_molar_mass(formula, ar_override)
    if not M:
        return {"error": f"Tidak kenal formula: {formula}"}
    n    = vol_dm3 / vm
    mass = n * M
    return {"answer": f"Jisim {formula} = {round(mass,3)} g", "mass_g": round(mass,3),
            "n_mol": round(n,5), "task": "mass_from_volume"}


def _solve_volume_from_mass(data: dict) -> dict:
    formula   = data.get("formula", "")
    mass      = data.get("mass_g", 0)
    condition = data.get("condition", "RTP")
    vm        = VM_STP if condition.upper() == "STP" else VM_RTP
    M         = calculate_molar_mass(formula)
    if not M:
        return {"error": f"Tidak kenal formula: {formula}"}
    n   = mass / M
    vol = n * vm
    return {"answer": f"Isipadu {formula} = {round(vol,4)} dm³ = {round(vol*1000,1)} cm³",
            "volume_dm3": round(vol,4), "n_mol": round(n,5), "task": "volume_from_mass"}


def _solve_volume_from_moles(data: dict) -> dict:
    moles     = data.get("moles", 0)
    condition = data.get("condition", "RTP")
    vm        = VM_STP if condition.upper() == "STP" else VM_RTP
    vol       = moles * vm
    return {"answer": f"Isipadu = {round(vol,4)} dm³ = {round(vol*1000,1)} cm³",
            "volume_dm3": round(vol,4), "task": "volume_from_moles"}


def _solve_mass_from_moles(data: dict) -> dict:
    formula = data.get("formula", "")
    moles   = data.get("moles", 0)
    M       = calculate_molar_mass(formula)
    if not M:
        return {"error": f"Tidak kenal formula: {formula}"}
    mass = moles * M
    return {"answer": f"Jisim {formula} = {round(mass,3)} g",
            "mass_g": round(mass,3), "task": "mass_from_moles"}


def _solve_molarity_from_mass(data: dict) -> dict:
    mass    = data.get("mass_g", 0)
    formula = data.get("formula", "")
    vol_dm3 = data.get("volume_dm3") or (data.get("volume_cm3", 0) / 1000)
    M       = calculate_molar_mass(formula)
    if not M or not vol_dm3:
        return {"error": "Perlu formula dan isipadu"}
    n        = mass / M
    molarity = n / vol_dm3
    return {"answer": f"Kemolaran = {round(molarity,4)} mol dm⁻³",
            "molarity": round(molarity,4), "n_mol": round(n,5), "task": "molarity_from_mass"}


def _solve_concentration_g_dm3(data: dict) -> dict:
    mass    = data.get("mass_g", 0)
    vol_dm3 = data.get("volume_dm3") or (data.get("volume_cm3", 0) / 1000)
    if not vol_dm3:
        return {"error": "Perlu isipadu"}
    conc = mass / vol_dm3
    return {"answer": f"Kepekatan = {round(conc,3)} g dm⁻³",
            "conc_g_dm3": round(conc,3), "task": "concentration_g_dm3"}


# ── DISPATCH TABLE ─────────────────────────────────────────────────────────

def _stoich_dispatch(data: dict) -> dict:
    """Universal stoichiometry dispatch — parses equation and routes."""
    eq = data.get("equation", "")
    # Parse given/target from data (extractor already extracted these)
    return solve_stoichiometry(
        given_formula   = data["given_formula"],
        target_formula  = data["target_formula"],
        given_coeff     = int(data.get("given_coeff", 1)),
        target_coeff    = int(data.get("target_coeff", 1)),
        given_mass_g    = data.get("given_mass_g"),
        given_mol       = data.get("given_mol"),
        given_vol_cm3   = data.get("given_volume_cm3"),
        given_vol_dm3   = data.get("given_volume_dm3"),
        given_molarity  = data.get("given_molarity"),
        given_solution_cm3 = data.get("given_solution_cm3"),
        want            = data.get("want", "mass"),
        condition       = data.get("condition", "RTP"),
        ar_override     = data.get("ar_override"),
        lang            = data.get("lang", "BM"),
    )


TASK_DISPATCH: Dict[str, Any] = {
    # ── Mol calculations ──────────────────────────────────────────────
    "moles_from_mass":        lambda d: solve_moles_from_mass(
                                   d["formula"], d["mass_g"], d.get("lang","BM")),
    "moles_from_volume":      lambda d: solve_moles_from_volume(
                                   d.get("volume_cm3", (d.get("volume_dm3",0))*1000),
                                   d.get("condition","RTP"), d.get("lang","BM")),
    "mass_from_moles":        _solve_mass_from_moles,
    "volume_from_moles":      _solve_volume_from_moles,
    "mass_from_volume":       _solve_mass_from_volume,
    "volume_from_mass":       _solve_volume_from_mass,

    # ── Particles ─────────────────────────────────────────────────────
    "particles_from_moles":   _solve_particles_from_moles,
    "particles_from_mass":    _solve_particles_from_mass,
    "particles_from_volume":  _solve_particles_from_volume,

    # ── Concentration ─────────────────────────────────────────────────
    "molarity_from_mass":     _solve_molarity_from_mass,
    "concentration_g_dm3":    _solve_concentration_g_dm3,
    "mass_from_molarity":     lambda d: solve_mass_from_molarity(
                                   d["formula"], d["molarity"],
                                   d.get("volume_cm3", d.get("volume_dm3",0)*1000),
                                   d.get("lang","BM")),
    "dilution":               lambda d: solve_dilution(
                                   d["M1"], d.get("V1", d.get("volume_cm3",0)),
                                   v2_cm3=d.get("V2"), m2=d.get("M2"),
                                   lang=d.get("lang","BM")),

    # ── Stoichiometry ─────────────────────────────────────────────────
    "stoichiometry_mass_to_mass":   _stoich_dispatch,
    "stoichiometry_mass_to_volume": lambda d: _stoich_dispatch({**d, "want":"volume_rtp"}),
    "stoichiometry_volume_to_mass": lambda d: _stoich_dispatch({**d, "want":"mass"}),
    "stoichiometry_volume_to_volume": lambda d: _stoich_dispatch({**d, "want":"volume_rtp"}),

    # ── JMR / Molar mass ──────────────────────────────────────────────
    "jmr":                    lambda d: solve_molar_mass(
                                   d["formula"], d.get("ar_override"), d.get("lang","BM")),

    # ── pH ────────────────────────────────────────────────────────────
    "ph_from_h":              lambda d: solve_ph(d["h_plus"], "H+", d.get("lang","BM")),
    "ph_from_poh":            lambda d: solve_ph(d["oh_conc"], "OH-", d.get("lang","BM")),
    "poh_from_oh":            lambda d: solve_ph(d["oh_minus"], "OH-", d.get("lang","BM")),
    "h_from_ph":              lambda d: {"answer": f"[H+] = 10^-{d['ph']} = "
                                         f"{10**(-d['ph']):.2e} mol dm⁻³",
                                         "h_plus": 10**(-d['ph']),
                                         "task": "h_from_ph"},

    # ── Thermochemistry ───────────────────────────────────────────────
    "calorimetry":            lambda d: solve_thermochemistry(
                                   delta_T      = d["temp_final"] - d["temp_initial"],
                                   volume_cm3_total = d["mass_g"],
                                   lang=d.get("lang","BM")),
    "delta_h_from_calorimetry": lambda d: solve_thermochemistry(
                                   delta_T      = d["temp_final"] - d["temp_initial"],
                                   volume_cm3_total = d["mass_g"],
                                   n_mol_given  = d.get("moles"),
                                   molarity     = d.get("molarity"),
                                   lang=d.get("lang","BM")),
    "molarity_from_delta_h":  lambda d: solve_molarity_from_dh(
                                   d["delta_h_kj_mol"],
                                   d["delta_t"],
                                   d["volume1_cm3"] + d["volume2_cm3"],
                                   lang=d.get("lang","BM")),

    # ── Titration ────────────────────────────────────────────────────
    "titration_find_molarity": lambda d: solve_titration(
                                    d["known_volume_cm3"], d["known_molarity"],
                                    v2_cm3=d.get("unknown_volume_cm3"),
                                    lang=d.get("lang","BM")),
    "titration_find_volume":   lambda d: solve_titration(
                                    d["known_volume_cm3"], d["known_molarity"],
                                    m2=d.get("unknown_molarity"),
                                    lang=d.get("lang","BM")),

    # ── Empirical formula ─────────────────────────────────────────────
    "empirical_formula":       lambda d: solve_empirical_formula(
                                    d["element_masses"], d.get("lang","BM")),

    # ── Atomic structure ──────────────────────────────────────────────
    "subatomic":               _solve_subatomic,

    # ── Relative Ar ───────────────────────────────────────────────────
    "ar_from_abundance":       lambda d: solve_relative_ar(
                                    list(zip(d["isotope_masses"], d["abundances"])),
                                    d.get("lang","BM")),

    # ── Rate of reaction ──────────────────────────────────────────────
    "rate_average":            lambda d: solve_rate_of_reaction(
                                    d["change"], d["time"], lang=d.get("lang","BM")),
    "rate_from_points":        lambda d: solve_rate_of_reaction(
                                    abs(d["value2"]-d["value1"]),
                                    abs(d["time2"]-d["time1"]),
                                    lang=d.get("lang","BM")),

    # ── Voltaic cell ──────────────────────────────────────────────────
    "voltaic_cell":            lambda d: solve_voltaic_cell(
                                    d["e0_cathode"], d["e0_anode"], d.get("lang","BM")),

    # ── Oxidation number ──────────────────────────────────────────────
    "oxidation_number":        _solve_oxidation_number,

    # ── Multi-mol ─────────────────────────────────────────────────────
    "moles_multi":             lambda d: {
                                    "answer": "\n".join(
                                        solve_moles_from_mass(f, m).get("answer","")
                                        for f, m in zip(d["formulas"], d["masses"])
                                    ),
                                    "task": "moles_multi",
                               },
}


# ── Main dispatch function ─────────────────────────────────────────────────

def solve_by_task(task: str, data: dict) -> str:
    """
    Central dispatch: maps task string → solver call → returns answer string.

    This is what main.py calls as solve_fn(task, data).
    Returns the solver's 'answer' string directly.
    Raises RuntimeError if task is unknown or solver returns an error.
    """
    fn = TASK_DISPATCH.get(task)
    if fn is None:
        raise ValueError(f"Unknown task: '{task}'. Add to TASK_DISPATCH in dispatcher.py")

    logger.debug(f"[dispatch] task={task} data_keys={list(data.keys())}")

    try:
        result = fn(data)
    except KeyError as e:
        raise RuntimeError(f"Task '{task}' missing required field: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Solver error task='{task}': {e}") from e

    if isinstance(result, dict):
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("answer") or str(result)

    return str(result)
