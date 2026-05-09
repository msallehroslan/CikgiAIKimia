import math
import re
from typing import Dict, List, Optional, Any

from formula_parser import molar_mass, parse_formula, ATOMIC_MASS
from equation_parser import get_ratio
from units import cm3_to_dm3, dm3_to_cm3, j_to_kj

NA = 6.02e23
VM_ROOM = 24.0
VM_STP = 22.4
C_WATER = 4.2


# =====================================
# OUTPUT FORMATTERS
# =====================================
def fmt_num(x: float, dp: int = 3) -> str:
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.{dp}e}"
    s = f"{x:.{dp}f}"
    return s.rstrip("0").rstrip(".")


def spm_format(diberi: List[str], formula: List[str], pengiraan: List[str], jawapan: List[str]) -> str:
    parts: List[str] = []
    if diberi:
        parts.append("Diberi:\n" + "\n".join(diberi))
    if formula:
        parts.append("Formula:\n" + "\n".join(formula))
    if pengiraan:
        parts.append("Pengiraan:\n" + "\n".join(pengiraan))
    if jawapan:
        parts.append("Jawapan:\n" + "\n".join(jawapan))
    return "\n\n".join(parts)


# =====================================
# HELPERS
# =====================================
def get_vm(condition: Optional[str] = None) -> float:
    if not condition:
        return VM_ROOM
    c = condition.strip().lower()
    if c in {"stp", "standard"}:
        return VM_STP
    return VM_ROOM


def condition_label(condition: Optional[str], vm: Optional[float] = None) -> str:
    if condition:
        c = condition.strip().upper()
        if c == "ROOM":
            return "RTP"
        return c
    if vm == VM_STP:
        return "STP"
    return "RTP"


def _pick_formula(data: Dict[str, Any], key: str = "formula") -> Optional[str]:
    value = data.get(key)
    if value:
        return value
    formulas = data.get("formulas") or []
    return formulas[0] if formulas else None


def _pick_target_formula(data: Dict[str, Any]) -> Optional[str]:
    if data.get("target_formula"):
        return data["target_formula"]
    formulas = data.get("formulas") or []
    return formulas[1] if len(formulas) >= 2 else None


def _charge_from_species(species: str, explicit_charge: Optional[int] = None) -> int:
    if explicit_charge is not None:
        return explicit_charge
    s = species.strip()
    if s.endswith("+"):
        m = re.search(r"(\d*)\+$", s)
        return int(m.group(1)) if m and m.group(1) else 1
    if s.endswith("-"):
        m = re.search(r"(\d*)-$", s)
        return -(int(m.group(1)) if m and m.group(1) else 1)
    return 0


# =====================================
# CHAPTER 2 / 3 CORE SOLVERS
# =====================================
def solve_jmr(formula_str: str) -> str:
    parsed = parse_formula(formula_str.replace(".", "·").split("·")[0])
    M = molar_mass(formula_str)
    terms = []
    for el, count in parsed.items():
        if count > 1:
            terms.append(f"{count}({ATOMIC_MASS[el]})")
        else:
            terms.append(f"{ATOMIC_MASS[el]}")
    return spm_format(
        diberi=[f"Formula = {formula_str}"] + [f"{el} = {ATOMIC_MASS[el]}" for el in parsed],
        formula=["JMR = jumlah (bilangan atom × Ar)"],
        pengiraan=[f"JMR = {' + '.join(terms)}", f"JMR = {fmt_num(M, 2)}"],
        jawapan=[f"JMR = {fmt_num(M, 2)}"],
    )


def solve_moles_from_mass(mass_g: float, formula_str: str) -> str:
    M = molar_mass(formula_str)
    n = mass_g / M
    return spm_format(
        diberi=[f"m = {fmt_num(mass_g, 3)} g", f"Formula = {formula_str}", f"M = {fmt_num(M, 2)} g mol⁻¹"],
        formula=["n = m ÷ M"],
        pengiraan=[f"n = {fmt_num(mass_g, 3)} ÷ {fmt_num(M, 2)}", f"n = {fmt_num(n, 3)} mol"],
        jawapan=[f"Bilangan mol = {fmt_num(n, 3)} mol"],
    )


# =====================================
# FIX BUG #1 — MULTI-FORMULA MOL
# =====================================
def solve_moles_multi(formulas: List[str], masses: List[float]) -> str:
    """
    Handle multi-formula "jumlah mol" questions.
    e.g. "5.6g N2 dan 3.2g O2 — hitungkan jumlah bilangan mol gas"
    """
    if not formulas or not masses or len(formulas) != len(masses):
        raise ValueError("Senarai formula dan jisim tidak sepadan.")

    lines_diberi: List[str] = []
    lines_calc: List[str] = []
    lines_jawapan: List[str] = []
    total_moles = 0.0

    for formula, mass in zip(formulas, masses):
        M = molar_mass(formula)
        n = mass / M
        total_moles += n
        lines_diberi.append(f"{formula}: m = {fmt_num(mass, 3)} g,  M = {fmt_num(M, 2)} g mol⁻¹")
        lines_calc.append(f"n({formula}) = {fmt_num(mass, 3)} ÷ {fmt_num(M, 2)} = {fmt_num(n, 4)} mol")
        lines_jawapan.append(f"n({formula}) = {fmt_num(n, 4)} mol")

    individual = " + ".join(
        fmt_num(mass / molar_mass(f), 4) for f, mass in zip(formulas, masses)
    )
    lines_calc.append(f"Jumlah mol = {individual}")
    lines_calc.append(f"Jumlah mol = {fmt_num(total_moles, 4)} mol")
    lines_jawapan.append(f"Jumlah bilangan mol = {fmt_num(total_moles, 4)} mol")

    return spm_format(
        diberi=lines_diberi,
        formula=["n = m ÷ M  (dikira untuk setiap komponen)"],
        pengiraan=lines_calc,
        jawapan=lines_jawapan,
    )


def solve_moles_from_volume(volume_dm3: float, condition: Optional[str] = None) -> str:
    vm = get_vm(condition)
    n = volume_dm3 / vm
    label = condition_label(condition, vm)
    return spm_format(
        diberi=[f"V = {fmt_num(volume_dm3, 3)} dm³", f"Keadaan = {label}", f"Vm = {vm} dm³ mol⁻¹"],
        formula=["n = V ÷ Vm"],
        pengiraan=[f"n = {fmt_num(volume_dm3, 3)} ÷ {vm}", f"n = {fmt_num(n, 3)} mol"],
        jawapan=[f"Bilangan mol = {fmt_num(n, 3)} mol"],
    )


def solve_particles_from_moles(moles: float) -> str:
    particles = moles * NA
    return spm_format(
        diberi=[f"n = {fmt_num(moles, 3)} mol", f"NA = {NA:.2e} mol⁻¹"],
        formula=["Bilangan zarah = n × NA"],
        pengiraan=[f"Bilangan zarah = {fmt_num(moles, 3)} × {NA:.2e}", f"Bilangan zarah = {particles:.3e}"],
        jawapan=[f"Bilangan zarah = {particles:.3e}"],
    )


def solve_volume_from_moles(moles: float, condition: Optional[str] = None) -> str:
    vm = get_vm(condition)
    volume = moles * vm
    label = condition_label(condition, vm)
    return spm_format(
        diberi=[f"n = {fmt_num(moles, 3)} mol", f"Keadaan = {label}", f"Vm = {vm} dm³ mol⁻¹"],
        formula=["V = n × Vm"],
        pengiraan=[f"V = {fmt_num(moles, 3)} × {vm}", f"V = {fmt_num(volume, 3)} dm³"],
        jawapan=[f"Isipadu gas = {fmt_num(volume, 3)} dm³"],
    )


def solve_mass_from_moles(moles: float, formula_str: str) -> str:
    M = molar_mass(formula_str)
    mass_g = moles * M
    return spm_format(
        diberi=[f"n = {fmt_num(moles, 3)} mol", f"Formula = {formula_str}", f"M = {fmt_num(M, 2)} g mol⁻¹"],
        formula=["m = n × M"],
        pengiraan=[f"m = {fmt_num(moles, 3)} × {fmt_num(M, 2)}", f"m = {fmt_num(mass_g, 3)} g"],
        jawapan=[f"Jisim = {fmt_num(mass_g, 3)} g"],
    )


def solve_particles_from_volume(volume_dm3: float, condition: Optional[str] = None) -> str:
    vm = get_vm(condition)
    n = volume_dm3 / vm
    particles = n * NA
    label = condition_label(condition, vm)
    return spm_format(
        diberi=[f"V = {fmt_num(volume_dm3, 3)} dm³", f"Keadaan = {label}", f"Vm = {vm} dm³ mol⁻¹", f"NA = {NA:.2e} mol⁻¹"],
        formula=["n = V ÷ Vm", "Bilangan zarah = n × NA"],
        pengiraan=[f"n = {fmt_num(volume_dm3, 3)} ÷ {vm}", f"n = {fmt_num(n, 3)} mol", f"Bilangan zarah = {fmt_num(n, 3)} × {NA:.2e}", f"Bilangan zarah = {particles:.3e}"],
        jawapan=[f"Bilangan zarah = {particles:.3e}"],
    )


def solve_particles_from_mass(mass_g: float, formula_str: str) -> str:
    M = molar_mass(formula_str)
    n = mass_g / M
    particles = n * NA
    return spm_format(
        diberi=[f"m = {fmt_num(mass_g, 3)} g", f"Formula = {formula_str}", f"M = {fmt_num(M, 2)} g mol⁻¹", f"NA = {NA:.2e} mol⁻¹"],
        formula=["n = m ÷ M", "Bilangan zarah = n × NA"],
        pengiraan=[f"n = {fmt_num(mass_g, 3)} ÷ {fmt_num(M, 2)}", f"n = {fmt_num(n, 3)} mol", f"Bilangan zarah = {fmt_num(n, 3)} × {NA:.2e}", f"Bilangan zarah = {particles:.3e}"],
        jawapan=[f"Bilangan zarah = {particles:.3e}"],
    )


def solve_mass_from_volume(volume_dm3: float, formula_str: str, condition: Optional[str] = None) -> str:
    vm = get_vm(condition)
    n = volume_dm3 / vm
    M = molar_mass(formula_str)
    mass_g = n * M
    label = condition_label(condition, vm)
    return spm_format(
        diberi=[f"V = {fmt_num(volume_dm3, 3)} dm³", f"Formula = {formula_str}", f"Keadaan = {label}", f"Vm = {vm} dm³ mol⁻¹", f"M = {fmt_num(M, 2)} g mol⁻¹"],
        formula=["n = V ÷ Vm", "m = n × M"],
        pengiraan=[f"n = {fmt_num(volume_dm3, 3)} ÷ {vm}", f"n = {fmt_num(n, 3)} mol", f"m = {fmt_num(n, 3)} × {fmt_num(M, 2)}", f"m = {fmt_num(mass_g, 3)} g"],
        jawapan=[f"Jisim = {fmt_num(mass_g, 3)} g"],
    )


def solve_volume_from_mass_multistep(mass_g: float, formula_str: str, condition: Optional[str] = None) -> str:
    M = molar_mass(formula_str)
    n = mass_g / M
    vm = get_vm(condition)
    volume = n * vm
    label = condition_label(condition, vm)
    return spm_format(
        diberi=[f"m = {fmt_num(mass_g, 3)} g", f"Formula = {formula_str}", f"M = {fmt_num(M, 2)} g mol⁻¹", f"Keadaan = {label}", f"Vm = {vm} dm³ mol⁻¹"],
        formula=["n = m ÷ M", "V = n × Vm"],
        pengiraan=[f"n = {fmt_num(mass_g, 3)} ÷ {fmt_num(M, 2)}", f"n = {fmt_num(n, 3)} mol", f"V = {fmt_num(n, 3)} × {vm}", f"V = {fmt_num(volume, 3)} dm³"],
        jawapan=[f"Isipadu gas = {fmt_num(volume, 3)} dm³"],
    )


def solve_stoichiometry_mass_to_mass(equation: str, given_formula: str, given_mass_g: float, target_formula: str) -> str:
    given_M = molar_mass(given_formula)
    given_n = given_mass_g / given_M
    ratio = get_ratio(equation, given_formula, target_formula)
    target_n = given_n * ratio
    target_M = molar_mass(target_formula)
    target_mass = target_n * target_M
    return spm_format(
        diberi=[f"Persamaan = {equation}", f"Jisim {given_formula} = {fmt_num(given_mass_g, 3)} g"],
        formula=["n = m ÷ M", "Nisbah mol daripada persamaan kimia", "m = n × M"],
        pengiraan=[
            f"M({given_formula}) = {fmt_num(given_M, 2)} g mol⁻¹",
            f"n({given_formula}) = {fmt_num(given_mass_g, 3)} ÷ {fmt_num(given_M, 2)} = {fmt_num(given_n, 3)} mol",
            f"Nisbah {given_formula} : {target_formula} = {fmt_num(ratio, 3)}",
            f"n({target_formula}) = {fmt_num(given_n, 3)} × {fmt_num(ratio, 3)} = {fmt_num(target_n, 3)} mol",
            f"M({target_formula}) = {fmt_num(target_M, 2)} g mol⁻¹",
            f"m({target_formula}) = {fmt_num(target_n, 3)} × {fmt_num(target_M, 2)} = {fmt_num(target_mass, 3)} g",
        ],
        jawapan=[f"Jisim {target_formula} = {fmt_num(target_mass, 3)} g"],
    )


def solve_empirical_formula(element_masses: Dict[str, float]) -> str:
    moles: Dict[str, float] = {}
    for el, mass in element_masses.items():
        if el not in ATOMIC_MASS:
            raise ValueError(f"Unsur '{el}' tiada dalam jadual Ar.")
        moles[el] = mass / ATOMIC_MASS[el]

    smallest = min(moles.values())
    ratios = {el: val / smallest for el, val in moles.items()}
    rounded: Dict[str, int] = {}
    for el, ratio in ratios.items():
        r = round(ratio)
        if abs(ratio - r) > 0.15:
            raise ValueError("Nisbah formula empirik tidak hampir kepada integer mudah.")
        rounded[el] = int(r)

    formula = "".join(el if rounded[el] == 1 else f"{el}{rounded[el]}" for el in element_masses.keys())
    pengiraan = []
    for el, mass in element_masses.items():
        pengiraan.append(f"Mol {el} = {fmt_num(mass, 3)} ÷ {ATOMIC_MASS[el]} = {fmt_num(moles[el], 3)}")
    pengiraan.append(f"Bahagi semua dengan mol terkecil = {fmt_num(smallest, 3)}")
    for el in element_masses.keys():
        pengiraan.append(f"Nisbah {el} = {fmt_num(ratios[el], 3)} ≈ {rounded[el]}")
    return spm_format(
        diberi=[f"Jisim unsur = {element_masses}"],
        formula=["Mol = jisim ÷ Ar", "Bahagikan semua mol dengan mol terkecil"],
        pengiraan=pengiraan,
        jawapan=[f"Formula empirik = {formula}"],
    )


def solve_ar_from_abundance(isotope_masses: List[float], abundances: List[float]) -> str:
    if len(isotope_masses) != len(abundances) or not isotope_masses:
        raise ValueError("Data isotop tidak lengkap.")
    total = sum(m * a for m, a in zip(isotope_masses, abundances))
    total_abundance = sum(abundances)
    ar = total / total_abundance
    terms = [f"({a} × {m})" for m, a in zip(isotope_masses, abundances)]
    return spm_format(
        diberi=[f"Jisim isotop = {isotope_masses}", f"Kelimpahan = {abundances}"],
        formula=["Ar = Σ (Kelimpahan × Jisim isotop) ÷ Jumlah kelimpahan"],
        pengiraan=[f"Ar = [{' + '.join(terms)}] ÷ {fmt_num(total_abundance, 2)}", f"Ar = {fmt_num(total, 2)} ÷ {fmt_num(total_abundance, 2)}", f"Ar = {fmt_num(ar, 2)}"],
        jawapan=[f"Jisim atom relatif, Ar = {fmt_num(ar, 2)}"],
    )


def solve_subatomic(A: int, Z: int) -> str:
    proton = Z
    electron = Z
    neutron = A - Z
    return spm_format(
        diberi=[f"Nombor nukleon, A = {A}", f"Nombor proton, Z = {Z}"],
        formula=["Bilangan proton = nombor proton", "Bilangan elektron = nombor proton (atom neutral)", "Bilangan neutron = nombor nukleon − nombor proton"],
        pengiraan=[f"Proton = {proton}", f"Elektron = {electron}", f"Neutron = {A} − {Z} = {neutron}"],
        jawapan=[f"Bilangan proton = {proton}", f"Bilangan elektron = {electron}", f"Bilangan neutron = {neutron}"],
    )


# =====================================
# BAB 6 ACID / BASE / TITRATION
# =====================================
def solve_concentration_g_dm3(mass_g: float, volume_cm3: Optional[float] = None, volume_dm3: Optional[float] = None) -> str:
    if volume_dm3 is None:
        if volume_cm3 is None:
            raise ValueError("Perlu beri isipadu dalam cm³ atau dm³.")
        volume_dm3 = cm3_to_dm3(volume_cm3)
    conc = mass_g / volume_dm3
    pengiraan: List[str] = []
    diberi = [f"Jisim = {fmt_num(mass_g, 3)} g"]
    if volume_cm3 is not None:
        diberi.append(f"Isipadu = {fmt_num(volume_cm3, 3)} cm³")
        pengiraan.append(f"{fmt_num(volume_cm3, 3)} cm³ = {fmt_num(volume_dm3, 3)} dm³")
    else:
        diberi.append(f"Isipadu = {fmt_num(volume_dm3, 3)} dm³")
    pengiraan += [f"Kepekatan = {fmt_num(mass_g, 3)} ÷ {fmt_num(volume_dm3, 3)}", f"Kepekatan = {fmt_num(conc, 3)} g dm⁻³"]
    return spm_format(
        diberi=diberi,
        formula=["Kepekatan (g dm⁻³) = jisim ÷ isipadu"],
        pengiraan=pengiraan,
        jawapan=[f"Kepekatan = {fmt_num(conc, 3)} g dm⁻³"],
    )


def solve_molarity_from_mass(mass_g: float, formula_str: str, volume_dm3: float) -> str:
    M = molar_mass(formula_str)
    n = mass_g / M
    molarity = n / volume_dm3
    return spm_format(
        diberi=[f"Jisim = {fmt_num(mass_g, 3)} g", f"Formula = {formula_str}", f"Jisim molar = {fmt_num(M, 2)} g mol⁻¹", f"Isipadu = {fmt_num(volume_dm3, 3)} dm³"],
        formula=["n = m ÷ M", "M = n ÷ V"],
        pengiraan=[f"n = {fmt_num(mass_g, 3)} ÷ {fmt_num(M, 2)}", f"n = {fmt_num(n, 3)} mol", f"M = {fmt_num(n, 3)} ÷ {fmt_num(volume_dm3, 3)}", f"M = {fmt_num(molarity, 3)} mol dm⁻³"],
        jawapan=[f"Kemolaran = {fmt_num(molarity, 3)} mol dm⁻³"],
    )


def solve_mass_from_molarity(molarity: float, volume_dm3: float, formula_str: str) -> str:
    """
    FIX BUG #10: "Berapakah jisim X untuk membuat Y dm3 larutan Z mol dm-3?"
    Reverse of molarity_from_mass.
    """
    M = molar_mass(formula_str)
    n = molarity * volume_dm3
    mass_g = n * M
    volume_cm3 = dm3_to_cm3(volume_dm3)
    return spm_format(
        diberi=[
            f"Kemolaran = {fmt_num(molarity, 3)} mol dm\u207b\u00b3",
            f"Isipadu = {fmt_num(volume_cm3, 2)} cm\u00b3 = {fmt_num(volume_dm3, 3)} dm\u00b3",
            f"Formula = {formula_str}",
            f"Jisim molar = {fmt_num(M, 2)} g mol\u207b\u00b9",
        ],
        formula=["n = M \u00d7 V", "m = n \u00d7 Mr"],
        pengiraan=[
            f"n = {fmt_num(molarity, 3)} \u00d7 {fmt_num(volume_dm3, 3)}",
            f"n = {fmt_num(n, 3)} mol",
            f"m = {fmt_num(n, 3)} \u00d7 {fmt_num(M, 2)}",
            f"m = {fmt_num(mass_g, 3)} g",
        ],
        jawapan=[f"Jisim {formula_str} = {fmt_num(mass_g, 3)} g"],
    )


def solve_dilution(M1: float, V2: float, M2: float) -> str:
    V1 = (M2 * V2) / M1
    return spm_format(
        diberi=[f"M₁ = {fmt_num(M1, 3)} mol dm⁻³", f"M₂ = {fmt_num(M2, 3)} mol dm⁻³", f"V₂ = {fmt_num(V2, 3)}"],
        formula=["M₁V₁ = M₂V₂"],
        pengiraan=[f"{fmt_num(M1, 3)}(V₁) = {fmt_num(M2, 3)}({fmt_num(V2, 3)})", f"V₁ = ({fmt_num(M2, 3)} × {fmt_num(V2, 3)}) ÷ {fmt_num(M1, 3)}", f"V₁ = {fmt_num(V1, 3)}"],
        jawapan=[f"Isipadu larutan asal, V₁ = {fmt_num(V1, 3)}"],
    )


def solve_ph_from_h(h_conc: float) -> str:
    ph = -math.log10(h_conc)
    return spm_format(
        diberi=[f"[H⁺] = {fmt_num(h_conc, 3)} mol dm⁻³"],
        formula=["pH = − log [H⁺]"],
        pengiraan=[f"pH = − log ({fmt_num(h_conc, 3)})", f"pH = {fmt_num(ph, 2)}"],
        jawapan=[f"pH = {fmt_num(ph, 2)}"],
    )


def solve_h_from_ph(ph: float) -> str:
    h = 10 ** (-ph)
    return spm_format(
        diberi=[f"pH = {fmt_num(ph, 2)}"],
        formula=["[H⁺] = 10⁻pH"],
        pengiraan=[f"[H⁺] = 10^-{fmt_num(ph, 2)}", f"[H⁺] = {h:.3e} mol dm⁻³"],
        jawapan=[f"[H⁺] = {h:.3e} mol dm⁻³"],
    )


def solve_poh_from_oh(oh_conc: float) -> str:
    poh = -math.log10(oh_conc)
    return spm_format(
        diberi=[f"[OH⁻] = {fmt_num(oh_conc, 3)} mol dm⁻³"],
        formula=["pOH = − log [OH⁻]"],
        pengiraan=[f"pOH = − log ({fmt_num(oh_conc, 3)})", f"pOH = {fmt_num(poh, 2)}"],
        jawapan=[f"pOH = {fmt_num(poh, 2)}"],
    )


def solve_oh_from_poh(poh: float) -> str:
    oh = 10 ** (-poh)
    return spm_format(
        diberi=[f"pOH = {fmt_num(poh, 2)}"],
        formula=["[OH⁻] = 10⁻pOH"],
        pengiraan=[f"[OH⁻] = 10^-{fmt_num(poh, 2)}", f"[OH⁻] = {oh:.3e} mol dm⁻³"],
        jawapan=[f"[OH⁻] = {oh:.3e} mol dm⁻³"],
    )


def solve_ph_from_poh(poh: float) -> str:
    ph = 14 - poh
    return spm_format(
        diberi=[f"pOH = {fmt_num(poh, 2)}"],
        formula=["pH + pOH = 14"],
        pengiraan=[f"pH = 14 − {fmt_num(poh, 2)}", f"pH = {fmt_num(ph, 2)}"],
        jawapan=[f"pH = {fmt_num(ph, 2)}"],
    )


def solve_titration_find_volume(known_molarity: float, known_volume_cm3: float, known_formula: str, unknown_molarity: float, unknown_formula: str, equation: str) -> str:
    known_volume_dm3 = cm3_to_dm3(known_volume_cm3)
    known_moles = known_molarity * known_volume_dm3
    ratio = get_ratio(equation, known_formula, unknown_formula)
    unknown_moles = known_moles * ratio
    unknown_volume_dm3 = unknown_moles / unknown_molarity
    unknown_volume_cm3 = dm3_to_cm3(unknown_volume_dm3)
    return spm_format(
        diberi=[f"Persamaan = {equation}", f"{known_formula}: M = {fmt_num(known_molarity, 3)} mol dm⁻³, V = {fmt_num(known_volume_cm3, 3)} cm³", f"{unknown_formula}: M = {fmt_num(unknown_molarity, 3)} mol dm⁻³"],
        formula=["n = MV", "Nisbah mol daripada persamaan kimia", "V = n ÷ M"],
        pengiraan=[
            f"V({known_formula}) = {fmt_num(known_volume_cm3, 3)} cm³ = {fmt_num(known_volume_dm3, 3)} dm³",
            f"n({known_formula}) = {fmt_num(known_molarity, 3)} × {fmt_num(known_volume_dm3, 3)} = {fmt_num(known_moles, 4)} mol",
            f"Nisbah {known_formula} → {unknown_formula} = {fmt_num(ratio, 3)}",
            f"n({unknown_formula}) = {fmt_num(known_moles, 4)} × {fmt_num(ratio, 3)} = {fmt_num(unknown_moles, 4)} mol",
            f"V({unknown_formula}) = {fmt_num(unknown_moles, 4)} ÷ {fmt_num(unknown_molarity, 3)} = {fmt_num(unknown_volume_dm3, 4)} dm³",
            f"V({unknown_formula}) = {fmt_num(unknown_volume_cm3, 3)} cm³",
        ],
        jawapan=[f"Isipadu {unknown_formula} = {fmt_num(unknown_volume_cm3, 3)} cm³"],
    )


def solve_titration_find_molarity(known_mass_g: Optional[float], known_formula: str, known_molarity: Optional[float], known_volume_cm3: Optional[float], unknown_formula: str, unknown_volume_cm3: float, equation: str) -> str:
    calc_lines: List[str] = []
    if known_mass_g is not None:
        known_moles = known_mass_g / molar_mass(known_formula)
        given_lines = [f"Jisim {known_formula} = {fmt_num(known_mass_g, 3)} g"]
        calc_lines += [f"M({known_formula}) = {fmt_num(molar_mass(known_formula), 2)} g mol⁻¹", f"n({known_formula}) = {fmt_num(known_mass_g, 3)} ÷ {fmt_num(molar_mass(known_formula), 2)} = {fmt_num(known_moles, 4)} mol"]
    else:
        if known_molarity is None or known_volume_cm3 is None:
            raise ValueError("Maklumat larutan diketahui tidak lengkap.")
        known_volume_dm3 = cm3_to_dm3(known_volume_cm3)
        known_moles = known_molarity * known_volume_dm3
        given_lines = [f"{known_formula}: M = {fmt_num(known_molarity, 3)} mol dm⁻³, V = {fmt_num(known_volume_cm3, 3)} cm³"]
        calc_lines += [f"V({known_formula}) = {fmt_num(known_volume_cm3, 3)} cm³ = {fmt_num(known_volume_dm3, 3)} dm³", f"n({known_formula}) = {fmt_num(known_molarity, 3)} × {fmt_num(known_volume_dm3, 3)} = {fmt_num(known_moles, 4)} mol"]
    ratio = get_ratio(equation, known_formula, unknown_formula)
    unknown_moles = known_moles * ratio
    unknown_volume_dm3 = cm3_to_dm3(unknown_volume_cm3)
    unknown_molarity_calc = unknown_moles / unknown_volume_dm3
    return spm_format(
        diberi=[f"Persamaan = {equation}"] + given_lines + [f"{unknown_formula}: V = {fmt_num(unknown_volume_cm3, 3)} cm³"],
        formula=["n = m ÷ M atau n = MV", "Nisbah mol daripada persamaan kimia", "M = n ÷ V"],
        pengiraan=calc_lines + [
            f"Nisbah {known_formula} → {unknown_formula} = {fmt_num(ratio, 3)}",
            f"n({unknown_formula}) = {fmt_num(known_moles, 4)} × {fmt_num(ratio, 3)} = {fmt_num(unknown_moles, 4)} mol",
            f"V({unknown_formula}) = {fmt_num(unknown_volume_cm3, 3)} cm³ = {fmt_num(unknown_volume_dm3, 3)} dm³",
            f"M({unknown_formula}) = {fmt_num(unknown_moles, 4)} ÷ {fmt_num(unknown_volume_dm3, 3)} = {fmt_num(unknown_molarity_calc, 3)} mol dm⁻³",
        ],
        jawapan=[f"Kemolaran {unknown_formula} = {fmt_num(unknown_molarity_calc, 3)} mol dm⁻³"],
    )


# =====================================
# BAB 7 RATE
# =====================================
def solve_rate_average(change: float, time: float, quantity_unit: str = "cm³", time_unit: str = "min") -> str:
    rate = change / time
    unit = f"{quantity_unit} {time_unit}⁻¹"
    return spm_format(
        diberi=[f"Perubahan kuantiti = {fmt_num(change, 3)} {quantity_unit}", f"Masa = {fmt_num(time, 3)} {time_unit}"],
        formula=["Kadar = perubahan kuantiti ÷ masa"],
        pengiraan=[f"Kadar = {fmt_num(change, 3)} ÷ {fmt_num(time, 3)}", f"Kadar = {fmt_num(rate, 3)} {unit}"],
        jawapan=[f"Kadar tindak balas = {fmt_num(rate, 3)} {unit}"],
    )


def solve_rate_from_points(time1: float, value1: float, time2: float, value2: float, quantity_unit: str = "cm³", time_unit: str = "min") -> str:
    change = value2 - value1
    delta_t = time2 - time1
    rate = change / delta_t
    unit = f"{quantity_unit} {time_unit}⁻¹"
    return spm_format(
        diberi=[f"Nilai pada masa {fmt_num(time1, 3)} {time_unit} = {fmt_num(value1, 3)} {quantity_unit}", f"Nilai pada masa {fmt_num(time2, 3)} {time_unit} = {fmt_num(value2, 3)} {quantity_unit}"],
        formula=["Kadar = Δy ÷ Δx"],
        pengiraan=[f"Δy = {fmt_num(value2, 3)} − {fmt_num(value1, 3)} = {fmt_num(change, 3)} {quantity_unit}", f"Δx = {fmt_num(time2, 3)} − {fmt_num(time1, 3)} = {fmt_num(delta_t, 3)} {time_unit}", f"Kadar = {fmt_num(change, 3)} ÷ {fmt_num(delta_t, 3)} = {fmt_num(rate, 3)} {unit}"],
        jawapan=[f"Kadar tindak balas = {fmt_num(rate, 3)} {unit}"],
    )


# =====================================
# BAB TERMOKIMIA
# =====================================
def solve_calorimetry(mass_g: float, temp_initial: float, temp_final: float) -> str:
    delta_t = temp_final - temp_initial
    Q = mass_g * C_WATER * delta_t
    return spm_format(
        diberi=[f"m = {fmt_num(mass_g, 3)} g", f"c = {C_WATER} J g⁻¹ °C⁻¹", f"Suhu awal = {fmt_num(temp_initial, 2)}°C", f"Suhu akhir = {fmt_num(temp_final, 2)}°C"],
        formula=["Q = mcΔT"],
        pengiraan=[f"ΔT = {fmt_num(temp_final, 2)} − {fmt_num(temp_initial, 2)} = {fmt_num(delta_t, 2)}°C", f"Q = {fmt_num(mass_g, 3)} × {C_WATER} × {fmt_num(delta_t, 2)}", f"Q = {fmt_num(Q, 2)} J"],
        jawapan=[f"Haba, Q = {fmt_num(Q, 2)} J"],
    )


def solve_enthalpy(Q_joule: float, moles: float) -> str:
    dH_j = -(Q_joule / moles)
    dH_kj = j_to_kj(dH_j)
    return spm_format(
        diberi=[f"Q = {fmt_num(Q_joule, 2)} J", f"Mol = {fmt_num(moles, 3)} mol"],
        formula=["ΔH = − Q ÷ mol"],
        pengiraan=[f"ΔH = − ({fmt_num(Q_joule, 2)} ÷ {fmt_num(moles, 3)})", f"ΔH = {fmt_num(dH_j, 2)} J mol⁻¹", f"ΔH = {fmt_num(dH_kj, 2)} kJ mol⁻¹"],
        jawapan=[f"Perubahan entalpi, ΔH = {fmt_num(dH_kj, 2)} kJ mol⁻¹"],
    )


def solve_thermochemistry_type(temp_initial: float, temp_final: float) -> str:
    delta_t = temp_final - temp_initial
    if delta_t > 0:
        jenis = "Eksotermik"
        penjelasan = "Suhu meningkat, maka haba dibebaskan dan ΔH bernilai negatif."
    elif delta_t < 0:
        jenis = "Endotermik"
        penjelasan = "Suhu menurun, maka haba diserap dan ΔH bernilai positif."
    else:
        jenis = "Tiada perubahan jelas"
        penjelasan = "Suhu tidak berubah, jadi jenis tindak balas tidak dapat dipastikan dengan yakin."
    return spm_format(
        diberi=[f"Suhu awal = {fmt_num(temp_initial, 2)}°C", f"Suhu akhir = {fmt_num(temp_final, 2)}°C"],
        formula=["Bandingkan suhu awal dan suhu akhir"],
        pengiraan=[f"ΔT = {fmt_num(temp_final, 2)} − {fmt_num(temp_initial, 2)} = {fmt_num(delta_t, 2)}°C", penjelasan],
        jawapan=[f"Jenis tindak balas = {jenis}"],
    )


# =====================================
# FIX BUG #5 — DELTA H ENDOTHERMIC
# =====================================
def solve_delta_h_from_calorimetry(mass_g: float, temp_initial: float, temp_final: float, moles: float) -> str:
    """
    FIX BUG #5: Correctly handle ENDOTHERMIC reactions (suhu menurun).

    EKSOTERMIK: temp_final > temp_initial
      → ΔT positif → Q positif → ΔH = -(+Q/mol) = NEGATIF ✓

    ENDOTERMIK: temp_final < temp_initial
      → ΔT negatif → Q negatif → ΔH = -(-Q/mol) = POSITIF ✓
    """
    delta_t = temp_final - temp_initial
    Q = mass_g * C_WATER * delta_t
    dH_j = -(Q / moles)
    dH_kj = j_to_kj(dH_j)

    if delta_t > 0:
        jenis = "Eksotermik"
        tanda_note = "Suhu meningkat → haba dibebaskan → ΔH negatif"
    elif delta_t < 0:
        jenis = "Endotermik"
        tanda_note = "Suhu menurun → haba diserap → ΔH positif"
    else:
        jenis = "Neutral"
        tanda_note = "Tiada perubahan suhu"

    return spm_format(
        diberi=[
            f"m = {fmt_num(mass_g, 3)} g",
            f"c = {C_WATER} J g⁻¹ °C⁻¹",
            f"Suhu awal = {fmt_num(temp_initial, 2)}°C",
            f"Suhu akhir = {fmt_num(temp_final, 2)}°C",
            f"Mol = {fmt_num(moles, 3)} mol",
        ],
        formula=["Q = mcΔT", "ΔH = −Q ÷ mol"],
        pengiraan=[
            f"ΔT = {fmt_num(temp_final, 2)} − {fmt_num(temp_initial, 2)} = {fmt_num(delta_t, 2)}°C",
            f"Q = {fmt_num(mass_g, 3)} × {C_WATER} × ({fmt_num(delta_t, 2)}) = {fmt_num(Q, 2)} J",
            f"ΔH = −({fmt_num(Q, 2)} ÷ {fmt_num(moles, 3)}) = {fmt_num(dH_j, 2)} J mol⁻¹",
            f"ΔH = {fmt_num(dH_kj, 2)} kJ mol⁻¹",
            tanda_note,
        ],
        jawapan=[
            f"Perubahan entalpi, ΔH = {fmt_num(dH_kj, 2)} kJ mol⁻¹",
            f"Jenis tindak balas: {jenis}",
        ],
    )


# =====================================
# BAB REDOX
# =====================================
def solve_oxidation_number(species: str, target_element: str, charge: Optional[int] = None) -> str:
    parsed = parse_formula(species)
    total_charge = _charge_from_species(species, charge)
    if target_element not in parsed:
        raise ValueError(f"Unsur sasaran '{target_element}' tiada dalam {species}.")

    known_sum = 0.0
    known_lines: List[str] = []
    for el, count in parsed.items():
        if el == target_element:
            continue
        if el == "O":
            ox = -2
        elif el == "H":
            ox = +1
        elif el in {"Li", "Na", "K"}:
            ox = +1
        elif el in {"Mg", "Ca", "Ba"}:
            ox = +2
        else:
            raise ValueError(f"Belum dapat tentukan nombor pengoksidaan unsur sokongan '{el}' secara automatik.")
        known_sum += ox * count
        known_lines.append(f"{el} = {ox}")

    target_count = parsed[target_element]
    x = (total_charge - known_sum) / target_count
    lhs_terms = []
    for el, count in parsed.items():
        if el == target_element:
            lhs_terms.append("x" if count == 1 else f"{count}x")
        else:
            if el == "O":
                ox = -2
            elif el == "H":
                ox = +1
            elif el in {"Li", "Na", "K"}:
                ox = +1
            elif el in {"Mg", "Ca", "Ba"}:
                ox = +2
            else:
                ox = 0
            lhs_terms.append(f"{count}({ox:+g})" if count != 1 else f"({ox:+g})")

    return spm_format(
        diberi=[f"Spesies = {species}", f"Cas ion = {total_charge}"] + known_lines,
        formula=["Jumlah nombor pengoksidaan = cas spesies"],
        pengiraan=[
            f"Biarkan nombor pengoksidaan {target_element} = x",
            f"{' + '.join(lhs_terms)} = {total_charge}",
            f"{target_count}x + ({fmt_num(known_sum, 2)}) = {total_charge}" if target_count != 1 else f"x + ({fmt_num(known_sum, 2)}) = {total_charge}",
            f"x = {fmt_num(x, 2)}",
        ],
        jawapan=[f"Nombor pengoksidaan {target_element} = {fmt_num(x, 2)}"],
    )


def solve_redox_change(before: float, after: float, element: str = "unsur") -> str:
    if after > before:
        jenis = "Pengoksidaan"
        sebab = "nombor pengoksidaan meningkat"
    elif after < before:
        jenis = "Penurunan"
        sebab = "nombor pengoksidaan menurun"
    else:
        jenis = "Tiada perubahan redoks"
        sebab = "nombor pengoksidaan tidak berubah"
    return spm_format(
        diberi=[f"Nombor pengoksidaan awal = {fmt_num(before, 2)}", f"Nombor pengoksidaan akhir = {fmt_num(after, 2)}"],
        formula=["Bandingkan nombor pengoksidaan sebelum dan selepas"],
        pengiraan=[f"Perubahan = {fmt_num(after, 2)} − {fmt_num(before, 2)} = {fmt_num(after - before, 2)}", sebab],
        jawapan=[f"{element} mengalami {jenis.lower()}" if jenis != "Tiada perubahan redoks" else jenis],
    )


# =====================================
# DISPATCHER
# =====================================
def solve_by_task(task: str, data: Dict[str, Any]) -> str:
    formula = _pick_formula(data)
    target_formula = _pick_target_formula(data)

    if task == "jmr":
        if not formula:
            raise ValueError("Formula diperlukan.")
        return solve_jmr(formula)

    # FIX BUG #1 — multi-formula mol
    if task == "moles_multi":
        return solve_moles_multi(data["formulas"], data["masses"])

    if task == "moles_from_mass":
        if not formula:
            raise ValueError("Formula diperlukan.")
        return solve_moles_from_mass(data["mass_g"], formula)

    if task == "moles_from_volume":
        volume_dm3 = data.get("volume_dm3")
        if volume_dm3 is None and data.get("volume_cm3") is not None:
            volume_dm3 = cm3_to_dm3(data["volume_cm3"])
        if volume_dm3 is None:
            raise ValueError("Isipadu gas diperlukan.")
        return solve_moles_from_volume(volume_dm3, data.get("condition"))

    if task == "particles_from_moles":
        return solve_particles_from_moles(data["moles"])

    if task == "volume_from_moles":
        return solve_volume_from_moles(data["moles"], data.get("condition"))

    if task == "mass_from_moles":
        if not formula:
            raise ValueError("Formula diperlukan.")
        return solve_mass_from_moles(data["moles"], formula)

    if task == "particles_from_volume":
        volume_dm3 = data.get("volume_dm3")
        if volume_dm3 is None and data.get("volume_cm3") is not None:
            volume_dm3 = cm3_to_dm3(data["volume_cm3"])
        if volume_dm3 is None:
            raise ValueError("Isipadu gas diperlukan.")
        return solve_particles_from_volume(volume_dm3, data.get("condition"))

    if task == "particles_from_mass":
        if not formula:
            raise ValueError("Formula diperlukan.")
        return solve_particles_from_mass(data["mass_g"], formula)

    if task == "mass_from_volume":
        volume_dm3 = data.get("volume_dm3")
        if volume_dm3 is None and data.get("volume_cm3") is not None:
            volume_dm3 = cm3_to_dm3(data["volume_cm3"])
        if volume_dm3 is None:
            raise ValueError("Isipadu gas diperlukan.")
        if not formula:
            raise ValueError("Formula diperlukan.")
        return solve_mass_from_volume(volume_dm3, formula, data.get("condition"))

    if task in {"volume_from_mass", "moles_from_mass_then_volume"}:
        if not formula:
            raise ValueError("Formula diperlukan.")
        return solve_volume_from_mass_multistep(data["mass_g"], formula, data.get("condition"))

    if task == "stoichiometry_mass_to_mass":
        equation = data["equation"]
        if "->" not in equation:
            raise ValueError("Persamaan kimia tidak lengkap.")

        reactants, products = equation.split("->", 1)
        reactant_list = re.findall(r"[A-Z][A-Za-z0-9()]*", reactants)
        product_list = re.findall(r"[A-Z][A-Za-z0-9()]*", products)

        given_formula = data.get("given_formula")
        target_formula = data.get("target_formula")

        if not given_formula and reactant_list:
            given_formula = reactant_list[0]
        if not target_formula and product_list:
            target_formula = product_list[0]

        if not given_formula or not target_formula:
            raise ValueError("Tidak dapat tentukan bahan diberi atau bahan sasaran.")

        return solve_stoichiometry_mass_to_mass(
            equation, given_formula, data["given_mass_g"], target_formula,
        )

    if task == "empirical_formula":
        return solve_empirical_formula(data["element_masses"])

    if task == "ar_from_abundance":
        return solve_ar_from_abundance(data["isotope_masses"], data["abundances"])

    if task == "subatomic":
        return solve_subatomic(int(data["A"]), int(data["Z"]))

    if task == "concentration_g_dm3":
        return solve_concentration_g_dm3(
            data["mass_g"], data.get("volume_cm3"), data.get("volume_dm3"),
        )

    if task == "molarity_from_mass":
        volume_dm3 = data.get("volume_dm3")
        if volume_dm3 is None and data.get("volume_cm3") is not None:
            volume_dm3 = cm3_to_dm3(data["volume_cm3"])
        if volume_dm3 is None:
            raise ValueError("Isipadu larutan diperlukan.")
        if not formula:
            raise ValueError("Formula diperlukan.")
        return solve_molarity_from_mass(data["mass_g"], formula, volume_dm3)

    if task == "mass_from_molarity":
        return solve_mass_from_molarity(
            data["molarity"], data["volume_dm3"], data["formula"]
        )

    if task == "dilution":
        return solve_dilution(data["M1"], data["V2"], data["M2"])

    if task == "ph_from_h":
        h_plus = data.get("h_plus") or data.get("h_conc")
        if h_plus is None:
            raise ValueError("Nilai [H⁺] tidak ditemui.")
        return solve_ph_from_h(h_plus)

    if task == "h_from_ph":
        return solve_h_from_ph(data["ph"])

    if task == "poh_from_oh":
        oh_minus = data.get("oh_minus") or data.get("oh_conc")
        if oh_minus is None:
            raise ValueError("Nilai [OH⁻] tidak ditemui.")
        return solve_poh_from_oh(oh_minus)

    if task == "oh_from_poh":
        return solve_oh_from_poh(data["poh"])

    if task == "ph_from_poh":
        # Support direct pOH value OR [OH-] concentration (two-step)
        poh = data.get("poh")
        if poh is None:
            oh_conc = data.get("oh_conc") or data.get("oh_minus")
            if oh_conc is not None:
                import math as _math
                poh = -_math.log10(oh_conc)
        if poh is None:
            raise ValueError("Nilai pOH atau [OH-] diperlukan.")
        return solve_ph_from_poh(poh)

    if task == "titration_find_volume":
        equation = data.get("equation")
        if not equation:
            equation = f"{data['known_formula']} + {data['unknown_formula']} -> salt + H2O"
        return solve_titration_find_volume(
            data["known_molarity"], data["known_volume_cm3"],
            data["known_formula"], data["unknown_molarity"],
            data["unknown_formula"], equation,
        )

    if task == "titration_find_molarity":
        equation = data.get("equation")
        if not equation:
            equation = f"{data['known_formula']} + {data['unknown_formula']} -> salt + H2O"
        return solve_titration_find_molarity(
            data.get("known_mass_g"), data["known_formula"],
            data.get("known_molarity"), data.get("known_volume_cm3"),
            data["unknown_formula"], data["unknown_volume_cm3"], equation,
        )

    if task == "rate_average":
        return solve_rate_average(
            data["change"], data["time"],
            data.get("quantity_unit", "cm³"), data.get("time_unit", "min"),
        )

    if task == "rate_from_points":
        return solve_rate_from_points(
            data["time1"], data["value1"], data["time2"], data["value2"],
            data.get("quantity_unit", "cm³"), data.get("time_unit", "min"),
        )

    if task == "calorimetry":
        return solve_calorimetry(
            data["mass_g"], data["temp_initial"], data["temp_final"],
        )

    if task == "enthalpy":
        return solve_enthalpy(data["Q_joule"], data["moles"])

    if task == "delta_h_from_calorimetry":
        # Support both raw and pre-computed temperature keys
        t_initial = data.get("temp_initial_raw") or data.get("temp_initial")
        t_final   = data.get("temp_final_raw")   or data.get("temp_final")
        return solve_delta_h_from_calorimetry(
            data["mass_g"], t_initial, t_final, data["moles"],
        )

    if task == "thermochemistry_type":
        return solve_thermochemistry_type(
            data["temp_initial"], data["temp_final"],
        )

    if task == "oxidation_number":
        species = data.get("species")
        if not species:
            formulas = data.get("formulas") or []
            if formulas:
                species = max(formulas, key=len)
        if not species:
            raise ValueError("Spesies redoks tidak ditemui.")
        return solve_oxidation_number(
            species, data["target_element"], data.get("charge"),
        )

    if task == "redox_change":
        return solve_redox_change(
            data["before_ox"], data["after_ox"], data.get("element", "Bahan"),
        )

    raise ValueError(f"Task '{task}' belum disokong.")
