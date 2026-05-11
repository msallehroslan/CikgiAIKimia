import re
from typing import Dict, List, Tuple

# ==============================
# ATOMIC MASS (Ar)
# Based on SPM-style values used across the project
# ==============================
ATOMIC_MASS = {
    "H": 1.0,
    "He": 4.0,
    "Li": 7.0,
    "Be": 9.0,
    "B": 11.0,
    "C": 12.0,
    "N": 14.0,
    "O": 16.0,
    "F": 19.0,
    "Ne": 20.0,
    "Na": 23.0,
    "Mg": 24.0,
    "Al": 27.0,
    "Si": 28.0,
    "P": 31.0,
    "S": 32.0,
    "Cl": 35.5,
    "Ar": 40.0,
    "K": 39.0,
    "Ca": 40.0,
    "Sc": 45.0,
    "Ti": 48.0,
    "Cr": 52.0,
    "Mn": 55.0,
    "Fe": 56.0,
    "Co": 59.0,
    "Ni": 59.0,
    "Cu": 63.5,
    "Zn": 65.0,
    "Ga": 70.0,
    "Ge": 73.0,
    "As": 75.0,
    "Se": 79.0,
    "Br": 80.0,
    "Kr": 84.0,
    "Ag": 108.0,
    "Sn": 119.0,
    "I": 127.0,
    "Ba": 137.0,
    "Pt": 195.0,
    "Au": 197.0,
    "Hg": 201.0,
    "Pb": 207.0,
}

_TOKEN_PATTERN = re.compile(r"([A-Z][a-z]?|\d+|[()\[\]])")
_ELEMENT_PATTERN = re.compile(r"^[A-Z][a-z]?$")
_LEADING_COEFF_PATTERN = re.compile(r"^\s*(\d+)\s*([A-Z(\[])" )
_STATE_PATTERN = re.compile(r"\((aq|s|l|g)\)\s*$", re.IGNORECASE)
_CHARGE_PATTERN = re.compile(r"(?:\^?[+-]\d*|\^?\d*[+-])$")


def normalize_formula(formula: str) -> str:
    """Clean formula text while preserving chemical meaning."""
    f = formula.strip()
    if not f:
        raise ValueError("Formula kosong.")

    # Common Unicode replacements
    f = f.replace("·", ".")
    f = f.replace("•", ".")
    f = f.replace("∙", ".")
    f = f.replace("−", "-")
    f = f.replace("[", "(").replace("]", ")")
    f = f.replace(" ", "")

    # Remove state symbol, e.g. HCl(aq)
    f = _STATE_PATTERN.sub("", f)

    # Remove ionic charge at the end, e.g. SO4^2-, NH4+, Fe3+
    f = _CHARGE_PATTERN.sub("", f)

    # Remove leading stoichiometric coefficient if present, e.g. 2H2O -> H2O
    m = _LEADING_COEFF_PATTERN.match(f)
    if m:
        f = f[m.end(1):].strip()

    if not f:
        raise ValueError("Formula tidak sah selepas normalisasi.")
    return f


def _merge_counts(target: Dict[str, int], source: Dict[str, int], multiplier: int = 1) -> None:
    for el, count in source.items():
        target[el] = target.get(el, 0) + count * multiplier


def _parse_simple_formula(formula: str) -> Dict[str, int]:
    tokens = _TOKEN_PATTERN.findall(formula)
    if not tokens:
        raise ValueError(f"Formula tidak sah: {formula}")

    stack: List[Dict[str, int]] = [{}]
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "(":
            stack.append({})
            i += 1
            continue

        if tok == ")":
            if len(stack) == 1:
                raise ValueError(f"Kurungan tidak sepadan dalam formula: {formula}")
            group = stack.pop()
            i += 1
            mult = 1
            if i < len(tokens) and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            _merge_counts(stack[-1], group, mult)
            continue

        if _ELEMENT_PATTERN.match(tok):
            if tok not in ATOMIC_MASS:
                raise ValueError(f"Unsur '{tok}' tiada dalam jadual Ar.")
            i += 1
            count = 1
            if i < len(tokens) and tokens[i].isdigit():
                count = int(tokens[i])
                i += 1
            stack[-1][tok] = stack[-1].get(tok, 0) + count
            continue

        if tok.isdigit():
            raise ValueError(f"Nombor '{tok}' berada pada kedudukan tidak sah dalam formula: {formula}")

        raise ValueError(f"Token tidak dijangka '{tok}' dalam formula: {formula}")

    if len(stack) != 1:
        raise ValueError(f"Kurungan tidak lengkap dalam formula: {formula}")

    return stack[0]


def _parse_segment_with_multiplier(segment: str) -> Tuple[int, str]:
    """Split 5H2O into (5, H2O)."""
    m = re.match(r"^(\d+)([A-Z(].*)$", segment)
    if m:
        return int(m.group(1)), m.group(2)
    return 1, segment


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Convert chemical formula into element count dictionary.

    Supports:
    - NaOH
    - Ca(OH)2
    - Al2(SO4)3
    - K4[Fe(CN)6]
    - CuSO4·5H2O / CuSO4.5H2O
    - Fe3+ / SO4^2- / HCl(aq)  -> charge/state removed before parsing
    """
    normalized = normalize_formula(formula)
    parts = [p for p in normalized.split(".") if p]
    if not parts:
        raise ValueError(f"Formula tidak sah: {formula}")

    total: Dict[str, int] = {}
    for part in parts:
        mult, subformula = _parse_segment_with_multiplier(part)
        parsed = _parse_simple_formula(subformula)
        _merge_counts(total, parsed, mult)
    return total


def formula_to_string(parsed: Dict[str, int]) -> str:
    """Simple canonical output, mainly for debugging."""
    return "".join(f"{el}{'' if cnt == 1 else cnt}" for el, cnt in sorted(parsed.items()))


def is_valid_formula(candidate: str) -> bool:
    """Reject normal words while accepting real formulas."""
    if not candidate or not re.search(r"[A-Z]", candidate):
        return False
    try:
        parse_formula(candidate)
        return True
    except Exception:
        return False


def molar_mass(formula: str) -> float:
    parsed = parse_formula(formula)
    return sum(ATOMIC_MASS[el] * count for el, count in parsed.items())
