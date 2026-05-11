import re
from typing import Dict, Tuple

from formula_parser import normalize_formula, is_valid_formula

_ARROW_PATTERN = re.compile(r"<=>|⇌|↔|->|→|=")
_STATE_PATTERN = re.compile(r"\((aq|s|l|g)\)\s*$", re.IGNORECASE)


def _normalize_equation(equation: str) -> str:
    eq = equation.strip()
    eq = eq.replace("⇌", "->").replace("↔", "->").replace("→", "->")
    return eq


def _split_species_token(token: str) -> Tuple[int, str]:
    token = token.strip()
    token = _STATE_PATTERN.sub("", token)
    m = re.match(r"^(\d+)\s*([A-Za-z0-9().\[\]+\-^]+)$", token)
    if m:
        coeff = int(m.group(1))
        raw_formula = m.group(2)
    else:
        coeff = 1
        raw_formula = token

    formula = normalize_formula(raw_formula)
    if not is_valid_formula(formula):
        raise ValueError(f"Formula tidak sah dalam persamaan: {raw_formula}")
    return coeff, formula


def parse_equation_side(side: str) -> Dict[str, int]:
    parts = [p.strip() for p in side.split("+") if p.strip()]
    if not parts:
        raise ValueError("Salah satu bahagian persamaan kosong.")

    result: Dict[str, int] = {}
    for part in parts:
        coeff, formula = _split_species_token(part)
        result[formula] = result.get(formula, 0) + coeff
    return result


def parse_equation(equation: str):
    eq = _normalize_equation(equation)
    if "->" not in eq:
        # also support equations using plain '=' if user typed it
        eq = _ARROW_PATTERN.sub("->", eq, count=1)
    if "->" not in eq:
        raise ValueError("Persamaan mesti mengandungi anak panah seperti '->' atau '→'.")

    left, right = [x.strip() for x in eq.split("->", 1)]
    if not left or not right:
        raise ValueError("Persamaan tidak lengkap.")

    return {
        "reactants": parse_equation_side(left),
        "products": parse_equation_side(right),
    }


def get_ratio(equation: str, from_formula: str, to_formula: str) -> float:
    parsed = parse_equation(equation)
    all_species: Dict[str, int] = {}
    all_species.update(parsed["reactants"])
    all_species.update(parsed["products"])

    from_norm = normalize_formula(from_formula)
    to_norm = normalize_formula(to_formula)

    if from_norm not in all_species:
        raise ValueError(f"{from_formula} tiada dalam persamaan.")
    if to_norm not in all_species:
        raise ValueError(f"{to_formula} tiada dalam persamaan.")

    return all_species[to_norm] / all_species[from_norm]
