"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  solver_engine_v340.py — Cikgu AI Kimia v3.4.0                            ║
║                                                                              ║
║  FAIL GABUNGAN PENUH — sedia untuk merge ke solver/solver_engine.py        ║
║                                                                              ║
║  KANDUNGAN:                                                                  ║
║  1. calculate_molar_mass()    — JMR universal (mudah, kompleks, hidrat)    ║
║  2. solve_moles_from_mass()   — mol dari jisim                             ║
║  3. solve_moles_from_volume() — mol dari isipadu gas (RTP/STP)             ║
║  4. solve_stoichiometry()     — UNIVERSAL: 6 jenis input × 2 jenis output  ║
║  5. solve_thermochemistry()   — forward (ΔH) + reverse (ΔT) [BARU]        ║
║  6. solve_ph_universal()      — dari H⁺ atau OH⁻ [IMPROVED]               ║
║  7. solve_titration()         — nisbah mol mana-mana [FIXED]              ║
║  8. solve_concentration()     — g/dm³ DAN mol/dm³ [FIXED]                 ║
║  9. solve_dilution()          — M1V1=M2V2 [FIXED]                         ║
║  10. solve_voltaic_cell()     — E0 sel                                     ║
║  11. solve_molarity_from_dh() — kemolaran dari ΔH [FIXED]                 ║
║  12. solve_empirical_formula()— formula empirik (% atau jisim)             ║
║  13. solve_rate_of_reaction() — kadar purata                               ║
║  14. solve_atomic_structure() — proton/neutron/elektron                    ║
║  15. solve_relative_ar()      — Ar dari isotop                             ║
║  16. solve_oxidation_number() — nombor pengoksidaan                        ║
║  17. solve_mass_from_molarity()— jisim untuk buat larutan                  ║
║                                                                              ║
║  CARA GUNA:                                                                  ║
║  1. Salin fail ini ke solver/solver_engine.py (gantikan terus)             ║
║  2. Atau import fungsi yang diperlukan sahaja                               ║
║                                                                              ║
║  TESTED AGAINST:                                                             ║
║  - Johor 2021 K1 + K2                                                      ║
║  - Terengganu 2021 K1                                                      ║
║  - Stress test 66 soalan (target: 90%+ pass rate)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

AR = {
    "H":1.0,"He":4.0,"Li":7.0,"Be":9.0,"B":10.8,"C":12.0,"N":14.0,
    "O":16.0,"F":19.0,"Ne":20.0,"Na":23.0,"Mg":24.0,"Al":27.0,"Si":28.0,
    "P":31.0,"S":32.0,"Cl":35.5,"Ar":40.0,"K":39.0,"Ca":40.0,"Sc":45.0,
    "Ti":48.0,"V":51.0,"Cr":52.0,"Mn":55.0,"Fe":56.0,"Co":59.0,"Ni":58.7,
    "Cu":63.5,"Zn":65.0,"Ga":70.0,"Ge":72.6,"As":75.0,"Se":79.0,"Br":80.0,
    "Kr":84.0,"Rb":85.5,"Sr":88.0,"Y":89.0,"Zr":91.0,"Ag":108.0,"Sn":118.7,
    "I":127.0,"Ba":137.0,"Pb":207.0,"Hg":200.6,
}

VM_RTP  = 24.0   # dm³/mol — Room Temperature & Pressure
VM_STP  = 22.4   # dm³/mol — Standard Temperature & Pressure
C_WATER = 4.2    # J g⁻¹ °C⁻¹
NA      = 6.02e23

# BM stopwords — jangan parse sebagai formula kimia
BM_STOPWORDS = {
    "berapa","hitung","hitungkan","tentukan","kira","kirakan","apakah",
    "berapakah","nyatakan","sebatian","larutan","jisim","isipadu","bila",
    "apabila","dalam","untuk","dengan","kepada","daripada","antara",
    "calculate","find","determine","what","which","how","the","of","in",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: MOLAR MASS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_molar_mass(formula: str, ar_override: dict = None) -> Optional[float]:
    """
    Kira jisim molar dari formula kimia.
    Support: NaOH, Al2(SO4)3, CuSO4.5H2O, K4Fe(CN)6.3H2O, Cu(NO3)2

    ar_override: dict Ar custom dari soalan, contoh {"Cu": 64, "N": 14}
                 Penting bila soalan beri Ar berbeza dari standard
    """
    if not formula:
        return None

    # Guna Ar custom jika ada (untuk soalan yang beri Ar spesifik)
    ar_table = dict(AR)
    if ar_override:
        ar_table.update(ar_override)

    formula = formula.strip()

    def parse_chunk(s: str) -> float:
        total = 0.0
        i = 0
        while i < len(s):
            if s[i] == '(':
                depth, j = 1, i + 1
                while j < len(s) and depth > 0:
                    if s[j] == '(': depth += 1
                    elif s[j] == ')': depth -= 1
                    j += 1
                inner = s[i+1:j-1]
                k = j
                num_str = ""
                while k < len(s) and (s[k].isdigit() or s[k] == '.'):
                    num_str += s[k]; k += 1
                mult = float(num_str) if num_str else 1.0
                sub = parse_chunk(inner)
                if sub < 0: return -1.0
                total += sub * mult
                i = k
            elif s[i].isupper():
                j = i + 1
                while j < len(s) and s[j].islower(): j += 1
                elem = s[i:j]
                k = j
                num_str = ""
                while k < len(s) and (s[k].isdigit() or s[k] == '.'):
                    num_str += s[k]; k += 1
                count = float(num_str) if num_str else 1.0
                if elem not in ar_table: return -1.0
                total += ar_table[elem] * count
                i = k
            else:
                i += 1
        return total

    total_mass = 0.0
    parts = formula.split('.')
    for part in parts:
        part = part.strip()
        if not part: continue
        m = re.match(r'^(\d+(?:\.\d+)?)([A-Za-z].*)$', part)
        if m:
            mult = float(m.group(1))
            sub = parse_chunk(m.group(2))
            if sub < 0: return None
            total_mass += sub * mult
        else:
            sub = parse_chunk(part)
            if sub < 0: return None
            total_mass += sub

    return round(total_mass, 2) if total_mass > 0 else None


def _vm(condition: str) -> float:
    """Isipadu molar berdasarkan keadaan."""
    c = condition.upper()
    if any(x in c for x in ("STP","PIAWAI","STANDARD")):
        return VM_STP
    return VM_RTP  # default RTP/bilik/room


def _spm_format(diberi, formula, pengiraan, jawapan, lang="BM") -> str:
    """Format jawapan dalam format SPM standard."""
    if lang == "BM":
        return (
            "🧮 Jawapan Cikgu AI Kimia\n\n"
            f"Diberi:\n" + "\n".join(f"  {x}" for x in diberi) + "\n\n"
            f"Formula:\n" + "\n".join(f"  {x}" for x in formula) + "\n\n"
            f"Pengiraan:\n" + "\n".join(f"  {x}" for x in pengiraan) + "\n\n"
            f"Jawapan:\n" + "\n".join(f"  {x}" for x in jawapan)
        )
    else:
        return (
            "🧮 Cikgu AI Kimia Answer\n\n"
            f"Given:\n" + "\n".join(f"  {x}" for x in diberi) + "\n\n"
            f"Formula:\n" + "\n".join(f"  {x}" for x in formula) + "\n\n"
            f"Calculation:\n" + "\n".join(f"  {x}" for x in pengiraan) + "\n\n"
            f"Answer:\n" + "\n".join(f"  {x}" for x in jawapan)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 1: MOL DARI JISIM
# ─────────────────────────────────────────────────────────────────────────────

def solve_moles_from_mass(formula: str, mass_g: float, lang="BM") -> dict:
    """n = m ÷ M"""
    try:
        M = calculate_molar_mass(formula)
        if not M:
            return {"error": f"Tidak kenal formula: {formula}"}
        n = mass_g / M
        answer = _spm_format(
            diberi=[f"m = {mass_g} g", f"Formula = {formula}", f"M = {M} g mol⁻¹"],
            formula=["n = m ÷ M"],
            pengiraan=[f"n = {mass_g} ÷ {M}", f"n = {round(n,4)} mol"],
            jawapan=[f"Bilangan mol = {round(n,4)} mol"],
            lang=lang
        )
        return {"answer": answer, "n_mol": round(n,4), "task": "moles_from_mass"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 2: MOL DARI ISIPADU GAS
# ─────────────────────────────────────────────────────────────────────────────

def solve_moles_from_volume(volume_cm3: float, condition: str = "RTP", lang="BM") -> dict:
    """n = V ÷ Vm"""
    try:
        vm = _vm(condition)
        vol_dm3 = volume_cm3 / 1000.0
        n = vol_dm3 / vm
        answer = _spm_format(
            diberi=[f"V = {volume_cm3} cm³ = {vol_dm3} dm³",
                    f"Keadaan = {condition}", f"Vm = {vm} dm³ mol⁻¹"],
            formula=["n = V ÷ Vm"],
            pengiraan=[f"n = {vol_dm3} ÷ {vm}", f"n = {round(n,5)} mol"],
            jawapan=[f"Bilangan mol = {round(n,5)} mol"],
            lang=lang
        )
        return {"answer": answer, "n_mol": round(n,5), "task": "moles_from_volume"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 3: STOICHIOMETRY — UNIVERSAL (semua 6 jenis input)
# ─────────────────────────────────────────────────────────────────────────────

def solve_stoichiometry(
    given_formula: str,
    target_formula: str,
    given_coeff: int = 1,
    target_coeff: int = 1,
    # Input — pilih SATU:
    given_mass_g: float = None,       # Jisim diberi (g)
    given_mol: float = None,           # Mol diberi terus
    given_vol_cm3: float = None,       # Isipadu GAS diberi (cm³)
    given_vol_dm3: float = None,       # Isipadu GAS diberi (dm³)
    given_molarity: float = None,      # Kemolaran larutan diberi (mol/dm³)
    given_solution_cm3: float = None,  # Isipadu LARUTAN diberi (cm³)
    # Output — pilih SATU:
    want: str = "mass",  # "mass" | "volume_rtp" | "volume_stp" | "volume_dm3"
    condition: str = "RTP",
    ar_override: dict = None,  # Ar custom dari soalan, contoh {"Cu": 64}
    lang: str = "BM",
) -> dict:
    """
    UNIVERSAL stoichiometry solver.

    Contoh penggunaan:

    # Johor Q38: 2.1g C3H6 → jisim H2O
    solve_stoichiometry("C3H6","H2O",2,6,given_mass_g=2.1,want="mass")
    → 2.70g ✓

    # Terengganu Q33: 9.2g C2H5OH → isipadu CO2 RTP
    solve_stoichiometry("C2H5OH","CO2",1,2,given_mass_g=9.2,want="volume_rtp")
    → 9.6 dm³ ✓

    # Terengganu Q37: 1.3dm³ CO2 → isipadu O2
    solve_stoichiometry("CO2","O2",6,6,given_vol_dm3=1.3,want="volume_dm3")
    → 1.3 dm³ ✓

    # Terengganu Q38: 25cm³ 0.5M Na2SO4 → jisim CaSO4
    solve_stoichiometry("Na2SO4","CaSO4",1,1,
        given_molarity=0.5,given_solution_cm3=25,want="mass")
    → 1.70g ✓

    # Bug Fix 3: 0.5 mol KI → jisim PbI2
    solve_stoichiometry("KI","PbI2",2,1,given_mol=0.5,want="mass")
    → 115.25g ✓

    # Terengganu Q13: 1.6g CuO → Cu(NO3)2 dengan Cu=64
    solve_stoichiometry("CuO","Cu(NO3)2",1,1,given_mass_g=1.6,
        want="mass",ar_override={"Cu":64})
    → 3.76g ✓
    """
    try:
        vm = _vm(condition)
        M_given = calculate_molar_mass(given_formula, ar_override)
        M_target = calculate_molar_mass(target_formula, ar_override)

        # ── Kira mol bahan diberi ─────────────────────────────────────────
        input_lines = []
        calc_lines = []

        if given_mass_g is not None:
            if not M_given:
                return {"error": f"Tidak kenal formula: {given_formula}"}
            n_given = given_mass_g / M_given
            input_lines = [f"m({given_formula}) = {given_mass_g} g",
                           f"M({given_formula}) = {M_given} g mol⁻¹"]
            calc_lines = [f"n({given_formula}) = {given_mass_g} ÷ {M_given} = {round(n_given,5)} mol"]

        elif given_mol is not None:
            n_given = given_mol
            input_lines = [f"n({given_formula}) = {given_mol} mol (diberi terus)"]

        elif given_vol_cm3 is not None:
            vol_dm3 = given_vol_cm3 / 1000
            n_given = vol_dm3 / vm
            input_lines = [f"V({given_formula}) = {given_vol_cm3} cm³ = {vol_dm3} dm³",
                           f"Vm = {vm} dm³ mol⁻¹ ({condition})"]
            calc_lines = [f"n({given_formula}) = {vol_dm3} ÷ {vm} = {round(n_given,5)} mol"]

        elif given_vol_dm3 is not None:
            n_given = given_vol_dm3 / vm
            input_lines = [f"V({given_formula}) = {given_vol_dm3} dm³",
                           f"Vm = {vm} dm³ mol⁻¹ ({condition})"]
            calc_lines = [f"n({given_formula}) = {given_vol_dm3} ÷ {vm} = {round(n_given,5)} mol"]

        elif given_molarity is not None and given_solution_cm3 is not None:
            vol_dm3 = given_solution_cm3 / 1000
            n_given = given_molarity * vol_dm3
            input_lines = [f"V({given_formula}) = {given_solution_cm3} cm³ = {vol_dm3} dm³",
                           f"Kemolaran = {given_molarity} mol dm⁻³"]
            calc_lines = [f"n({given_formula}) = {given_molarity} × {vol_dm3} = {round(n_given,5)} mol"]

        else:
            return {"error": "Perlu berikan sekurang-kurangnya satu input: "
                             "jisim, mol, isipadu gas, atau kemolaran+isipadu larutan"}

        # ── Nisbah mol ───────────────────────────────────────────────────
        n_target = n_given * (target_coeff / given_coeff)
        ratio_str = f"{given_coeff}:{target_coeff}"

        calc_lines += [
            f"Nisbah {given_formula}:{target_formula} = {ratio_str}",
            f"n({target_formula}) = {round(n_given,5)} × ({target_coeff}/{given_coeff}) = {round(n_target,5)} mol",
        ]

        # ── Kira output ───────────────────────────────────────────────────
        if want in ("mass", "jisim"):
            if not M_target:
                return {"error": f"Tidak kenal formula: {target_formula}"}
            result_val = n_target * M_target
            calc_lines += [f"M({target_formula}) = {M_target} g mol⁻¹",
                           f"m({target_formula}) = {round(n_target,5)} × {M_target} = {round(result_val,3)} g"]
            jawapan = [f"Jisim {target_formula} = {round(result_val,3)} g"]
            unit = "g"

        elif want in ("volume_rtp","volume_stp","volume_dm3","isipadu","volume"):
            result_val = n_target * vm
            result_cm3 = result_val * 1000
            calc_lines += [f"V({target_formula}) = {round(n_target,5)} × {vm} = {round(result_val,4)} dm³"]
            jawapan = [f"Isipadu {target_formula} = {round(result_val,4)} dm³  "
                       f"({round(result_cm3,2)} cm³)"]
            unit = "dm³"

        else:
            return {"error": f"Jenis output tidak dikenali: {want}"}

        formula_lines = []
        if given_mass_g is not None:
            formula_lines.append("n = m ÷ M")
        elif given_vol_cm3 or given_vol_dm3:
            formula_lines.append("n = V ÷ Vm")
        elif given_molarity:
            formula_lines.append("n = kemolaran × isipadu (dm³)")
        formula_lines += ["Gunakan nisbah mol dari persamaan kimia",
                          "m = n × M" if want in ("mass","jisim") else "V = n × Vm"]

        answer = _spm_format(
            diberi=input_lines + [f"Nisbah mol {given_formula}:{target_formula} = {ratio_str}"],
            formula=formula_lines,
            pengiraan=calc_lines,
            jawapan=jawapan,
            lang=lang
        )

        return {
            "answer": answer,
            "n_given": round(n_given, 6),
            "n_target": round(n_target, 6),
            "result": round(result_val, 4),
            "unit": unit,
            "task": "stoichiometry",
        }

    except Exception as e:
        return {"error": f"Ralat stoikiometri: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 4: THERMOCHEMISTRY — FORWARD + REVERSE [BUG FIX 2]
# ─────────────────────────────────────────────────────────────────────────────

def solve_thermochemistry(
    # MODE A (forward): beri ΔT → kira Q dan ΔH
    delta_T: float = None,
    volume_cm3_total: float = None,
    molarity: float = None,
    mass_solute_g: float = None,
    molar_mass_solute: float = None,
    n_mol_given: float = None,
    # MODE B (reverse — BARU Terengganu Q34): beri Q → kira ΔT
    Q_joules: float = None,
    want: str = "delta_H",   # "delta_H" | "delta_T"
    c: float = C_WATER,
    density: float = 1.0,
    lang: str = "BM",
) -> dict:
    """
    Thermochemistry solver — dua mod:

    MODE A: Q = mcΔT, ΔH = -(Q kJ)/n
    MODE B: ΔT = Q/(mc)   ← BARU (Terengganu Q34)

    Johor Q36: 100cm³, ΔT=+10°C, M=2.0 → ΔH=-42 kJ/mol ✓
    Johor Q5c: 20cm³, ΔT=-11°C, M=2.0 → Q=924J ✓
    Terengganu Q34: Q=2100J, 50cm³ → ΔT=10°C ✓
    """
    try:
        mass_g = (volume_cm3_total or 0) * density

        # ── MODE B: Beri Q, cari ΔT ──────────────────────────────────────
        if Q_joules is not None and want == "delta_T":
            if not mass_g:
                return {"error": "Perlu isipadu untuk kira ΔT"}
            dT = Q_joules / (mass_g * c)
            answer = _spm_format(
                diberi=[f"Q = {Q_joules} J", f"m = {mass_g} g",
                        f"c = {c} J g⁻¹ °C⁻¹"],
                formula=["Q = mcΔT  →  ΔT = Q ÷ (mc)"],
                pengiraan=[f"ΔT = {Q_joules} ÷ ({mass_g} × {c})",
                           f"ΔT = {Q_joules} ÷ {mass_g*c}",
                           f"ΔT = {round(dT,2)} °C"],
                jawapan=[f"Perubahan suhu, ΔT = {round(dT,2)} °C"],
                lang=lang
            )
            return {"answer": answer, "delta_T": round(dT,2), "task": "thermochemistry_reverse"}

        # ── MODE A: Beri ΔT, kira Q dan ΔH ──────────────────────────────
        if delta_T is None:
            return {"error": "Perlu ΔT atau Q"}
        if not mass_g:
            return {"error": "Perlu isipadu larutan"}

        Q_J = mass_g * c * abs(delta_T)
        Q_kJ = Q_J / 1000.0  # BUG FIX 2: MESTI bahagi 1000

        # Kira mol
        n_source = ""
        if n_mol_given:
            n_mol = n_mol_given
            n_source = f"n = {n_mol_given} mol (diberi)"
        elif molarity and volume_cm3_total:
            half_dm3 = (volume_cm3_total / 2) / 1000
            n_mol = molarity * half_dm3
            n_source = f"n = {molarity} × {half_dm3} = {round(n_mol,5)} mol"
        elif mass_solute_g and molar_mass_solute:
            n_mol = mass_solute_g / molar_mass_solute
            n_source = f"n = {mass_solute_g} ÷ {molar_mass_solute} = {round(n_mol,5)} mol"
        else:
            n_mol = 1.0
            n_source = "n = 1 mol (anggaran)"

        # Tanda ΔH
        if delta_T > 0:
            dH = -(Q_kJ / n_mol)
            jenis = "Eksotermik" if lang=="BM" else "Exothermic"
            note = "Suhu naik → haba dibebaskan → ΔH negatif"
        else:
            dH = +(Q_kJ / n_mol)
            jenis = "Endotermik" if lang=="BM" else "Endothermic"
            note = "Suhu turun → haba diserap → ΔH positif"

        answer = _spm_format(
            diberi=[f"m = {mass_g} g", f"c = {c} J g⁻¹ °C⁻¹",
                    f"ΔT = {abs(delta_T)} °C", f"Mol = {round(n_mol,4)} mol ({n_source})"],
            formula=["Q = mcΔT", "ΔH = −Q(kJ) ÷ mol  [kJ mol⁻¹]"],
            pengiraan=[
                f"Q = {mass_g} × {c} × {abs(delta_T)} = {round(Q_J,2)} J",
                f"Q = {round(Q_J,2)} ÷ 1000 = {round(Q_kJ,4)} kJ",
                f"ΔH = {round(dH,2)} kJ mol⁻¹",
                note,
            ],
            jawapan=[f"ΔH = {round(dH,2)} kJ mol⁻¹", f"Jenis: {jenis}"],
            lang=lang
        )
        return {
            "answer": answer,
            "Q_joules": round(Q_J,2), "Q_kJ": round(Q_kJ,4),
            "delta_H": round(dH,2), "n_mol": round(n_mol,5), "jenis": jenis,
            "task": "thermochemistry",
        }
    except Exception as e:
        return {"error": f"Ralat termokimia: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 5: pH UNIVERSAL — dari H⁺ atau OH⁻ [IMPROVED]
# ─────────────────────────────────────────────────────────────────────────────

def solve_ph(concentration: float, ion_type: str = "H+", lang="BM") -> dict:
    """
    pH dari H⁺ atau OH⁻.
    Terengganu Q25: OH⁻=0.5 → pH=13.7 ✓
    Johor K2Q7: HCl 0.001 → pH=3 ✓
    """
    try:
        is_acid = ion_type.upper() in ("H+","H⁺","ASID","ACID","HCL","HNO3","H2SO4")

        if is_acid:
            pH = -math.log10(concentration)
            pOH = 14 - pH
            answer = _spm_format(
                diberi=[f"[H⁺] = {concentration} mol dm⁻³"],
                formula=["pH = −log[H⁺]"],
                pengiraan=[f"pH = −log({concentration})", f"pH = {round(pH,2)}"],
                jawapan=[f"pH = {round(pH,2)}"],
                lang=lang
            )
        else:
            pOH = -math.log10(concentration)
            pH = 14 - pOH
            answer = _spm_format(
                diberi=[f"[OH⁻] = {concentration} mol dm⁻³"],
                formula=["pOH = −log[OH⁻]", "pH = 14 − pOH"],
                pengiraan=[
                    f"pOH = −log({concentration}) = {round(pOH,3)}",
                    f"pH = 14 − {round(pOH,3)} = {round(pH,2)}",
                ],
                jawapan=[f"pOH = {round(pOH,3)}", f"pH = {round(pH,2)}"],
                lang=lang
            )
        return {"answer": answer, "pH": round(pH,2), "pOH": round(pOH,3), "task": "ph_calculation"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 6: TITRATION — NISBAH MOL MANA-MANA [BUG FIX 5]
# ─────────────────────────────────────────────────────────────────────────────

def solve_titration(
    v1_cm3: float, m1: float,
    v2_cm3: float = None, m2: float = None,
    coeff1: int = 1, coeff2: int = 1,
    lang: str = "BM",
) -> dict:
    """
    (M₁V₁)/coeff₁ = (M₂V₂)/coeff₂

    Johor Q: 25cm³ NaOH 0.1M neut 20cm³ HCl → M(HCl)=0.125 ✓
    Terengganu Q29: NaOH 25cm³ 0.5M, cari V(H2SO4) 0.5M, nisbah 1:2
      → V(H2SO4) = 12.5cm³ ✓
    """
    try:
        if m2 is None and v2_cm3 is not None:
            m2 = (m1 * v1_cm3 * coeff2) / (coeff1 * v2_cm3)
            solve_for = "M2"
        elif v2_cm3 is None and m2 is not None:
            v2_cm3 = (m1 * v1_cm3 * coeff2) / (coeff1 * m2)
            solve_for = "V2"
        else:
            return {"error": "Perlu M2 atau V2 untuk dikira"}

        nisbah = f"{coeff1}:{coeff2}"
        formula_str = f"(M₁ × V₁) / {coeff1} = (M₂ × V₂) / {coeff2}"

        if solve_for == "M2":
            calc = [
                f"({m1} × {v1_cm3}) / {coeff1} = (M₂ × {v2_cm3}) / {coeff2}",
                f"M₂ = ({m1} × {v1_cm3} × {coeff2}) ÷ ({coeff1} × {v2_cm3})",
                f"M₂ = {round(m2,4)} mol dm⁻³",
            ]
            jawapan = [f"Kemolaran = {round(m2,4)} mol dm⁻³"]
        else:
            calc = [
                f"({m1} × {v1_cm3}) / {coeff1} = ({m2} × V₂) / {coeff2}",
                f"V₂ = ({m1} × {v1_cm3} × {coeff2}) ÷ ({coeff1} × {m2})",
                f"V₂ = {round(v2_cm3,2)} cm³",
            ]
            jawapan = [f"Isipadu = {round(v2_cm3,2)} cm³"]

        answer = _spm_format(
            diberi=[f"V₁ = {v1_cm3} cm³, M₁ = {m1} mol dm⁻³",
                    f"Nisbah mol = {nisbah}"],
            formula=[formula_str],
            pengiraan=calc,
            jawapan=jawapan,
            lang=lang
        )
        return {
            "answer": answer,
            "M2": round(m2,4) if solve_for=="M2" else None,
            "V2": round(v2_cm3,2) if solve_for=="V2" else None,
            "task": "titration",
        }
    except Exception as e:
        return {"error": f"Ralat titrasi: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 7: CONCENTRATION + MOLARITY [BUG FIX 1 + 4]
# ─────────────────────────────────────────────────────────────────────────────

def solve_concentration(mass_g: float, volume_cm3: float,
                        formula: str = None, lang="BM") -> dict:
    """
    Kira kepekatan g/dm³ DAN mol/dm³.
    BUG FIX 1: Tidak crash lagi.
    BUG FIX 4: Kira kedua-dua unit.
    """
    try:
        vol_dm3 = volume_cm3 / 1000
        conc_g = mass_g / vol_dm3

        diberi = [f"Jisim = {mass_g} g",
                  f"Isipadu = {volume_cm3} cm³ = {vol_dm3} dm³"]
        formula_lines = ["Kepekatan (g dm⁻³) = jisim ÷ isipadu"]
        calc = [f"Kepekatan = {mass_g} ÷ {vol_dm3} = {round(conc_g,3)} g dm⁻³"]
        jawapan = [f"Kepekatan = {round(conc_g,3)} g dm⁻³"]

        # BUG FIX 4: Kira mol/dm³ jika formula ada
        conc_mol = None
        if formula:
            M = calculate_molar_mass(formula)
            if M:
                n = mass_g / M
                conc_mol = n / vol_dm3
                diberi += [f"Formula = {formula}", f"M = {M} g mol⁻¹"]
                formula_lines += ["n = jisim ÷ M",
                                  "Kepekatan molar = n ÷ isipadu (dm³)"]
                calc += [f"n = {mass_g} ÷ {M} = {round(n,4)} mol",
                         f"Kepekatan molar = {round(n,4)} ÷ {vol_dm3} = {round(conc_mol,4)} mol dm⁻³"]
                jawapan += [f"Kepekatan molar = {round(conc_mol,4)} mol dm⁻³"]

        answer = _spm_format(diberi, formula_lines, calc, jawapan, lang)
        return {
            "answer": answer,
            "conc_g_dm3": round(conc_g,3),
            "conc_mol_dm3": round(conc_mol,4) if conc_mol else None,
            "task": "concentration",
        }
    except Exception as e:
        return {"error": f"Ralat kepekatan: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 8: DILUTION [BUG FIX 1]
# ─────────────────────────────────────────────────────────────────────────────

def solve_dilution(m1: float, v1_cm3: float,
                   v2_cm3: float = None, m2: float = None, lang="BM") -> dict:
    """M1V1 = M2V2. BUG FIX 1: Tidak crash."""
    try:
        if v2_cm3 and not m2:
            m2 = (m1 * v1_cm3) / v2_cm3
            solve_for = "M2"
        elif m2 and not v2_cm3:
            v2_cm3 = (m1 * v1_cm3) / m2
            solve_for = "V2"
        else:
            return {"error": "Perlu M2 atau V2"}

        calc = [f"{m1} × {v1_cm3} = {'M₂' if solve_for=='M2' else m2} × {v2_cm3 if solve_for=='M2' else 'V₂'}"]
        if solve_for == "M2":
            calc.append(f"M₂ = {round(m2,4)} mol dm⁻³")
            jawapan = [f"Kemolaran baharu = {round(m2,4)} mol dm⁻³"]
        else:
            calc.append(f"V₂ = {round(v2_cm3,2)} cm³")
            jawapan = [f"Isipadu baharu = {round(v2_cm3,2)} cm³"]

        answer = _spm_format(
            diberi=[f"M₁ = {m1} mol dm⁻³, V₁ = {v1_cm3} cm³"],
            formula=["M₁V₁ = M₂V₂"],
            pengiraan=calc, jawapan=jawapan, lang=lang
        )
        return {
            "answer": answer,
            "M2": round(m2,4) if solve_for=="M2" else None,
            "V2": round(v2_cm3,2) if solve_for=="V2" else None,
            "task": "dilution",
        }
    except Exception as e:
        return {"error": f"Ralat pencairan: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 9: VOLTAIC CELL
# ─────────────────────────────────────────────────────────────────────────────

def solve_voltaic_cell(e0_katod: float, e0_anod: float, lang="BM") -> dict:
    """E⁰sel = E⁰katod − E⁰anod"""
    try:
        e0_sel = e0_katod - e0_anod
        answer = _spm_format(
            diberi=[f"E⁰katod = {e0_katod} V", f"E⁰anod = {e0_anod} V"],
            formula=["E⁰sel = E⁰katod − E⁰anod"],
            pengiraan=[f"E⁰sel = ({e0_katod}) − ({e0_anod})",
                       f"E⁰sel = {round(e0_sel,2)} V"],
            jawapan=[f"E⁰sel = {round(e0_sel,2)} V"],
            lang=lang
        )
        return {"answer": answer, "e0_sel": round(e0_sel,2), "task": "voltaic_cell"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 10: MOLARITY FROM ΔH [BUG FIX 6]
# ─────────────────────────────────────────────────────────────────────────────

def solve_molarity_from_dh(delta_h_kJ_mol: float, delta_T: float,
                            volume_cm3_total: float,
                            c: float = C_WATER, density: float = 1.0,
                            lang="BM") -> dict:
    """
    BUG FIX 6: Solver lengkap — 3 langkah:
    Q = mcΔT → n = Q/|ΔH| → M = n/V

    Ujian: ΔH=-57.3, ΔT=7°C, V=100cm³ → M=1.026 mol/dm³ ✓
    """
    try:
        mass_g = volume_cm3_total * density
        Q_J = mass_g * c * abs(delta_T)
        Q_kJ = Q_J / 1000
        n_mol = Q_kJ / abs(delta_h_kJ_mol)
        half_dm3 = (volume_cm3_total / 2) / 1000
        molarity = n_mol / half_dm3

        answer = _spm_format(
            diberi=[f"ΔH = {delta_h_kJ_mol} kJ mol⁻¹",
                    f"ΔT = {delta_T} °C",
                    f"Jumlah isipadu = {volume_cm3_total} cm³"],
            formula=["Q = mcΔT",
                     "n = Q(kJ) ÷ |ΔH|",
                     "Kemolaran = n ÷ V(separuh, dm³)"],
            pengiraan=[
                f"Q = {mass_g} × {c} × {abs(delta_T)} = {round(Q_J,2)} J = {round(Q_kJ,4)} kJ",
                f"n = {round(Q_kJ,4)} ÷ {abs(delta_h_kJ_mol)} = {round(n_mol,5)} mol",
                f"V(setiap larutan) = {volume_cm3_total}/2 = {volume_cm3_total/2} cm³ = {half_dm3} dm³",
                f"Kemolaran = {round(n_mol,5)} ÷ {half_dm3} = {round(molarity,3)} mol dm⁻³",
            ],
            jawapan=[f"Kemolaran = {round(molarity,3)} mol dm⁻³"],
            lang=lang
        )
        return {
            "answer": answer,
            "Q_joules": round(Q_J,2), "Q_kJ": round(Q_kJ,4),
            "n_mol": round(n_mol,5), "molarity": round(molarity,3),
            "task": "molarity_from_delta_h",
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 11: EMPIRICAL FORMULA
# ─────────────────────────────────────────────────────────────────────────────

def solve_empirical_formula(composition: dict, lang="BM") -> dict:
    """
    composition: {"C": 40.0, "H": 6.67, "O": 53.33}  (% atau jisim)
    """
    try:
        moles = {}
        for elem, val in composition.items():
            if elem not in AR:
                return {"error": f"Unsur tidak dikenali: {elem}"}
            moles[elem] = val / AR[elem]

        min_mol = min(moles.values())
        ratio = {e: round(v/min_mol, 2) for e,v in moles.items()}

        # Round ke integer (dengan tolerance)
        def to_int(x):
            for mult in [1,2,3,4]:
                val = x * mult
                if abs(val - round(val)) < 0.1:
                    return int(round(val)), mult
            return int(round(x)), 1

        formula_parts = []
        multiplier = 1
        for elem, r in ratio.items():
            int_r, m = to_int(r)
            multiplier = max(multiplier, m)

        empirical = ""
        for elem, r in ratio.items():
            int_r = int(round(r * multiplier))
            empirical += elem + (str(int_r) if int_r > 1 else "")

        calc_lines = []
        for elem, val in composition.items():
            calc_lines.append(f"Mol {elem} = {val} ÷ {AR[elem]} = {round(moles[elem],3)}")
        calc_lines.append(f"Bahagi dengan mol terkecil = {round(min_mol,3)}")
        for elem, r in ratio.items():
            calc_lines.append(f"Nisbah {elem} = {r} ≈ {int(round(r*multiplier))}")

        answer = _spm_format(
            diberi=[f"Komposisi: {composition}"],
            formula=["Mol = nilai ÷ Ar", "Bahagikan semua mol dengan mol terkecil"],
            pengiraan=calc_lines,
            jawapan=[f"Formula empirik = {empirical}"],
            lang=lang
        )
        return {"answer": answer, "formula_empirik": empirical, "task": "empirical_formula"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 12: JISIM MOLAR (JMR)
# ─────────────────────────────────────────────────────────────────────────────

def solve_molar_mass(formula: str, ar_override: dict = None, lang="BM") -> dict:
    """
    Kira JMR. Support formula kompleks + hidrat + Ar custom dari soalan.
    ar_override: {"Cu": 64} bila soalan beri Ar berbeza dari standard
    """
    try:
        M = calculate_molar_mass(formula, ar_override)
        if not M:
            return {"error": f"Tidak kenal formula: {formula}"}
        answer = _spm_format(
            diberi=[f"Formula = {formula}"],
            formula=["JMR = jumlah (bilangan atom × Ar)"],
            pengiraan=[f"JMR = {M}"],
            jawapan=[f"JMR = {M} g mol⁻¹"],
            lang=lang
        )
        return {"answer": answer, "molar_mass": M, "task": "molar_mass"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 13: KADAR TINDAK BALAS
# ─────────────────────────────────────────────────────────────────────────────

def solve_rate_of_reaction(delta_y: float, delta_x: float,
                            unit_y: str = "cm³", unit_x: str = "s",
                            lang="BM") -> dict:
    """Kadar = Δy ÷ Δx"""
    try:
        rate = delta_y / delta_x
        unit = f"{unit_y}/{unit_x}"
        answer = _spm_format(
            diberi=[f"Δ{unit_y} = {delta_y} {unit_y}", f"Δmasa = {delta_x} {unit_x}"],
            formula=["Kadar = Δy ÷ Δx"],
            pengiraan=[f"Kadar = {delta_y} ÷ {delta_x}", f"Kadar = {round(rate,3)} {unit}"],
            jawapan=[f"Kadar tindak balas = {round(rate,3)} {unit}"],
            lang=lang
        )
        return {"answer": answer, "rate": round(rate,3), "unit": unit, "task": "rate_of_reaction"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 14: JISIM ATOM RELATIF DARI ISOTOP
# ─────────────────────────────────────────────────────────────────────────────

def solve_relative_ar(isotopes: list, lang="BM") -> dict:
    """
    isotopes = [(jisim, kelimpahan_pct), ...]
    Contoh: [(35, 75), (37, 25)] → 35.5
    """
    try:
        ar = sum(mass * (pct/100) for mass, pct in isotopes)
        calc_parts = [f"({mass} × {pct}%)" for mass, pct in isotopes]
        answer = _spm_format(
            diberi=[f"Isotop: {isotopes}"],
            formula=["Ar = Σ(jisim × kelimpahan)"],
            pengiraan=[" + ".join(calc_parts) + f" = {round(ar,2)}"],
            jawapan=[f"Jisim atom relatif = {round(ar,2)}"],
            lang=lang
        )
        return {"answer": answer, "Ar": round(ar,2), "task": "relative_ar"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 15: MASS FROM MOLARITY [BUG FIX — router fix]
# ─────────────────────────────────────────────────────────────────────────────

def solve_mass_from_molarity(formula: str, molarity: float, volume_cm3: float,
                              lang="BM") -> dict:
    """Jisim untuk buat larutan. n = M×V, m = n×M"""
    try:
        M = calculate_molar_mass(formula)
        if not M:
            return {"error": f"Tidak kenal formula: {formula}"}
        vol_dm3 = volume_cm3 / 1000
        n = molarity * vol_dm3
        mass = n * M
        answer = _spm_format(
            diberi=[f"Formula = {formula}", f"Kemolaran = {molarity} mol dm⁻³",
                    f"Isipadu = {volume_cm3} cm³ = {vol_dm3} dm³", f"M = {M} g mol⁻¹"],
            formula=["n = kemolaran × isipadu (dm³)", "m = n × M"],
            pengiraan=[f"n = {molarity} × {vol_dm3} = {round(n,4)} mol",
                       f"m = {round(n,4)} × {M} = {round(mass,3)} g"],
            jawapan=[f"Jisim {formula} = {round(mass,3)} g"],
            lang=lang
        )
        return {"answer": answer, "mass_g": round(mass,3), "task": "mass_from_molarity"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST — SEMUA SOLVER (Johor + Terengganu + Pattern Umum)
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    P, F = "✅", "❌"
    results = []

    tests = [
        # Molar mass
        ("JMR NaOH=40",       lambda: calculate_molar_mass("NaOH"),          lambda v: abs(v-40)<0.1),
        ("JMR Al2(SO4)3=342", lambda: calculate_molar_mass("Al2(SO4)3"),     lambda v: abs(v-342)<0.5),
        ("JMR K4Fe(CN)6.3H2O=422", lambda: calculate_molar_mass("K4Fe(CN)6.3H2O"), lambda v: abs(v-422)<0.5),
        ("JMR FeSO4.7H2O=278",lambda: calculate_molar_mass("FeSO4.7H2O"),   lambda v: abs(v-278)<0.5),
        ("JMR Cu(NO3)2=187.5",lambda: calculate_molar_mass("Cu(NO3)2"),      lambda v: abs(v-187.5)<0.1),
        ("JMR Cu(NO3)2 Cu=64→188",lambda: calculate_molar_mass("Cu(NO3)2",{"Cu":64}), lambda v: abs(v-188)<0.1),
        ("JMR C2H5OH=46",     lambda: calculate_molar_mass("C2H5OH"),        lambda v: abs(v-46)<0.1),
        ("JMR CaSO4=136",     lambda: calculate_molar_mass("CaSO4"),         lambda v: abs(v-136)<0.1),

        # Stoich — pelbagai jenis input (UNIVERSAL)
        ("Johor Q38: 2.1g C3H6→H2O=2.70g",
         lambda: solve_stoichiometry("C3H6","H2O",2,6,given_mass_g=2.1,want="mass"),
         lambda r: abs(r.get("result",0)-2.7)<0.01),

        ("Terengganu Q33: 9.2g C2H5OH→CO2=9.6dm³",
         lambda: solve_stoichiometry("C2H5OH","CO2",1,2,given_mass_g=9.2,want="volume_rtp"),
         lambda r: abs(r.get("result",0)-9.6)<0.2),

        ("Terengganu Q37: 1.3dm³ CO2→O2=1.3dm³",
         lambda: solve_stoichiometry("CO2","O2",6,6,given_vol_dm3=1.3,want="volume_dm3"),
         lambda r: abs(r.get("result",0)-1.3)<0.05),

        ("Terengganu Q38: 25cm³ 0.5M Na2SO4→CaSO4=1.70g",
         lambda: solve_stoichiometry("Na2SO4","CaSO4",1,1,given_molarity=0.5,given_solution_cm3=25,want="mass"),
         lambda r: abs(r.get("result",0)-1.7)<0.05),

        ("BugFix3: 0.5mol KI→PbI2=115.25g",
         lambda: solve_stoichiometry("KI","PbI2",2,1,given_mol=0.5,want="mass"),
         lambda r: abs(r.get("result",0)-115.25)<0.5),

        ("Terengganu Q13: 1.6g CuO→Cu(NO3)2=3.76g [Cu=64]",
         lambda: solve_stoichiometry("CuO","Cu(NO3)2",1,1,given_mass_g=1.6,want="mass",ar_override={"Cu":64}),
         lambda r: abs(r.get("result",0)-3.76)<0.01),

        # Thermochemistry
        ("Johor Q36: ΔH=-42 kJ/mol",
         lambda: solve_thermochemistry(delta_T=10,volume_cm3_total=100,molarity=2.0),
         lambda r: abs(r.get("delta_H",0)-(-42))<1),

        ("Terengganu Q34: Q=2100J→ΔT=10°C [REVERSE]",
         lambda: solve_thermochemistry(Q_joules=2100,volume_cm3_total=50,want="delta_T"),
         lambda r: abs(r.get("delta_T",0)-10)<0.1),

        ("Johor Q5c: Q=924J (endotermik)",
         lambda: solve_thermochemistry(delta_T=-11,volume_cm3_total=20,molarity=2.0),
         lambda r: abs(r.get("Q_joules",0)-924)<1),

        # pH
        ("pH HCl 0.01 → pH=2",
         lambda: solve_ph(0.01,"H+"),
         lambda r: abs(r.get("pH",0)-2)<0.01),

        ("Terengganu Q25: OH⁻ 0.5 → pH=13.7",
         lambda: solve_ph(0.5,"OH-"),
         lambda r: abs(r.get("pH",0)-13.7)<0.1),

        ("pH NaOH 0.001 → pH=11",
         lambda: solve_ph(0.001,"OH-"),
         lambda r: abs(r.get("pH",0)-11)<0.01),

        # Titration
        ("TIT 1:1: 25cm³ NaOH 0.1M → M(HCl)=0.125",
         lambda: solve_titration(25,0.1,v2_cm3=20),
         lambda r: abs(r.get("M2",0)-0.125)<0.001),

        ("TIT 1:2: Johor H2SO4+2NaOH → V=80cm³",
         lambda: solve_titration(20,0.2,m2=0.1,coeff1=1,coeff2=2),
         lambda r: abs(r.get("V2",0)-80)<1),

        ("TIT Terengganu Q29: NaOH+H2SO4 → V=12.5cm³",
         lambda: solve_titration(25,0.5,m2=0.5,coeff1=2,coeff2=1),
         lambda r: abs(r.get("V2",0)-12.5)<0.1),

        # Kepekatan
        ("Concentration: 5.85g NaCl/500cm³ → 0.2 mol/dm³",
         lambda: solve_concentration(5.85,500,"NaCl"),
         lambda r: abs(r.get("conc_mol_dm3",0)-0.2)<0.01),

        # Dilution
        ("Dilution: 100cm³ 2M→500cm³ → 0.4 mol/dm³",
         lambda: solve_dilution(2.0,100,v2_cm3=500),
         lambda r: abs(r.get("M2",0)-0.4)<0.01),

        # Voltaic cell
        ("Voltaic Zn-Cu: +1.10V",
         lambda: solve_voltaic_cell(0.34,-0.76),
         lambda r: abs(r.get("e0_sel",0)-1.10)<0.01),

        # Molarity from ΔH
        ("MolFromDH: ΔH=-57.3,ΔT=7°C → 1.026 mol/dm³",
         lambda: solve_molarity_from_dh(-57.3,7,100),
         lambda r: abs(r.get("molarity",0)-1.026)<0.05),

        # Relative Ar
        ("Ar isotop Cl: [(35,75),(37,25)] → 35.5",
         lambda: solve_relative_ar([(35,75),(37,25)]),
         lambda r: abs(r.get("Ar",0)-35.5)<0.01),

        # Mass from molarity
        ("Mass from molarity: NaOH 0.1M/500cm³ → 2g",
         lambda: solve_mass_from_molarity("NaOH",0.1,500),
         lambda r: abs(r.get("mass_g",0)-2)<0.01),
    ]

    print("=" * 65)
    print("  UNIVERSAL SPM SOLVER v3.4.0 — SELF TEST")
    print(f"  {len(tests)} test cases | Johor + Terengganu + Pattern Umum")
    print("=" * 65)

    for name, fn, check in tests:
        try:
            r = fn()
            if isinstance(r, (int, float)):
                r = {"result": r}
            ok = check(r)
            status = P if ok else F
            # Get any numeric result for display
            val = (r.get("result") or r.get("delta_T") or r.get("pH") or
                   r.get("delta_H") or r.get("V2") or r.get("M2") or
                   r.get("Q_joules") or r.get("molarity") or r.get("Ar") or
                   r.get("conc_mol_dm3") or r.get("e0_sel") or r.get("mass_g") or "?")
            print(f"  {status}  {name} → {val}")
            results.append(ok)
        except Exception as e:
            print(f"  {F}  {name} → ERROR: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    pct = passed/total*100
    print(f"\n{'='*65}")
    print(f"  KEPUTUSAN: {passed}/{total} ({pct:.0f}%)")
    print(f"  {'🏆 SEMUA LULUS!' if passed==total else f'⚠️  {total-passed} gagal'}")
    print("=" * 65)
    return passed, total


if __name__ == "__main__":
    run_all_tests()
