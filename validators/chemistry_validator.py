"""
validators/chemistry_validator.py — Cikgu AI Kimia  [v4.0 — FULL REWRITE]
==========================================================================
REPLACES: existing validators/chemistry_validator.py

WHAT WAS WRONG IN THE OLD VERSION (6 bugs fixed):
  BUG 1 — OCR errors H20, NaC1, Fe203 passed FormulaValidator as OK
  BUG 2 — EquationValidator completely missing
  BUG 3 — Negative mass/volume only WARNING, should be CRITICAL
  BUG 4 — AnswerValidator never triggered (solver returns str, validator needs dict)
  BUG 5 — validate_extraction only checked 3 fields, missing equation/species
  BUG 6 — ValidationReport missing user_message() for bot reply

MODULE STRUCTURE:
  OCRCorrector        <- corrects 40+ known OCR formula errors
  FormulaValidator    <- validates formula strings (uses OCRCorrector)
  EquationValidator   <- validates equation strings [NEW]
  UnitValidator       <- validates numeric values (negative -> CRITICAL)
  AnswerValidator     <- validates solver output (str OR dict) [FIXED]
  ChemistryValidator  <- integration wrapper used by main.py

INTEGRATION FLOW (main.py):
  data = cv.correct_ocr(data)               <- NEW step
  pre  = cv.validate_extraction(task, data)
  if pre.has_critical:
      return pre.user_message(lang)         <- NEW method
  answer_str = solve_by_task(task, data)
  post = cv.validate_answer(task, answer_str)  <- now accepts str
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# AR table -- import from formula_parser (shared with solver), fallback inline
try:
    from formula_parser import ATOMIC_MASS as AR, parse_formula as _parse_formula
    _HAS_FORMULA_PARSER = True
except ImportError:
    _HAS_FORMULA_PARSER = False
    AR: Dict[str, float] = {
        "H":1.0,"He":4.0,"Li":7.0,"Be":9.0,"B":11.0,"C":12.0,"N":14.0,
        "O":16.0,"F":19.0,"Ne":20.0,"Na":23.0,"Mg":24.0,"Al":27.0,"Si":28.0,
        "P":31.0,"S":32.0,"Cl":35.5,"Ar":40.0,"K":39.0,"Ca":40.0,
        "Ti":48.0,"V":51.0,"Cr":52.0,"Mn":55.0,"Fe":56.0,"Co":59.0,
        "Ni":58.7,"Cu":63.5,"Zn":65.0,"As":75.0,"Se":79.0,"Br":80.0,
        "Kr":84.0,"Ag":108.0,"Sn":118.7,"I":127.0,"Ba":137.0,
        "Hg":200.6,"Pb":207.0,"Au":197.0,"Pt":195.0,
    }

VALID_ELEMENTS = set(AR.keys())


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationIssue:
    severity: str           # "critical" | "warning" | "info"
    code:     str           # machine-readable (logging/metrics)
    message:  str           # human-readable
    field:    Optional[str] = None


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str,
            field: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(severity, code, message, field))

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    @property
    def criticals(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        c = len(self.criticals)
        w = len(self.warnings)
        if c == 0 and w == 0:
            return "OK"
        parts = []
        if c: parts.append(f"{c} critical")
        if w: parts.append(f"{w} warning")
        return ", ".join(parts)

    def user_message(self, lang: str = "BM") -> str:
        """
        FIX BUG 6: Return student-facing HTML message for critical failures.
        Called by main.py when has_critical is True.
        """
        if not self.has_critical:
            return ""
        criticals = self.criticals[:3]
        if lang == "BM":
            lines = ["⚠️ <b>Cikgu AI tidak dapat mengesahkan soalan ini:</b>\n"]
            for i in criticals:
                lines.append(f"• {i.message}")
            lines.append("\n<i>Sila semak semula soalan atau taip dalam teks.</i>")
        else:
            lines = ["⚠️ <b>Cikgu AI could not validate this question:</b>\n"]
            for i in criticals:
                lines.append(f"• {i.message}")
            lines.append("\n<i>Please check the question or type it as text.</i>")
        return "\n".join(lines)


# =============================================================================
# OCR CORRECTOR  (FIX BUG 1)
# =============================================================================

class OCRCorrector:
    """
    Auto-corrects known OCR errors in chemical formula strings.

    DETECTED PATTERNS:
      - digit 0 (zero) vs letter O     : H20->H2O, Fe203->Fe2O3
      - digit 1 (one)  vs letter l     : NaC1->NaCl, CaC12->CaCl2
      - digit 1 (one)  vs Al           : A1Cl3->AlCl3
      - all-caps symbols               : HCL->HCl, NAOH->NaOH

    ANOMALOUS COUNT DETECTION:
      parse_formula('H20') = {H:20} -- H subscript 20 is impossible
      parse_formula('Fe203') = {Fe:203} -- impossible
      These are caught by detect_anomalous_count().
    """

    _EXACT: Dict[str, str] = {
        # zero vs O
        "H20":"H2O",    "Na0H":"NaOH",  "KMn04":"KMnO4",  "K2Cr207":"K2Cr2O7",
        "Fe203":"Fe2O3","A1203":"Al2O3","Cu0":"CuO",       "Mg0":"MgO",
        "Zn0":"ZnO",    "Ca0":"CaO",    "N02":"NO2",       "S02":"SO2",
        "C02":"CO2",    "H2S04":"H2SO4","HN03":"HNO3",     "Ca(0H)2":"Ca(OH)2",
        "Ba(0H)2":"Ba(OH)2","NH40H":"NH4OH","Na2C03":"Na2CO3","CaC03":"CaCO3",
        "Na2S03":"Na2SO3","Na2S203":"Na2S2O3","K2S04":"K2SO4","Na2S04":"Na2SO4",
        "CuS04":"CuSO4","ZnS04":"ZnSO4","FeS04":"FeSO4",
        "Fe2(S04)3":"Fe2(SO4)3","Al2(S04)3":"Al2(SO4)3",
        "Cu(N03)2":"Cu(NO3)2","Fe(N03)3":"Fe(NO3)3","Ca(N03)2":"Ca(NO3)2",
        "K4Fe(CN)6":"K4Fe(CN)6",  # already correct
        # digit 1 vs l
        "NaC1":"NaCl",  "KC1":"KCl",   "CaC12":"CaCl2",  "MgC12":"MgCl2",
        "A1C13":"AlCl3","A1Cl3":"AlCl3","FeC13":"FeCl3",  "ZnC12":"ZnCl2",
        "CuC12":"CuCl2","BaC12":"BaCl2","A1":"Al",
        # all-caps
        "HCL":"HCl",   "NACL":"NaCl",  "NAOH":"NaOH",    "NAHCO3":"NaHCO3",
        "MGSO4":"MgSO4","CASO4":"CaSO4",
    }

    _PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r'([A-Za-z])0([A-Za-z0-9])'), r'\g<1>O\g<2>'),   # 0->O between alphanums
        (re.compile(r'([A-Za-z])0$'),              r'\g<1>O'),         # 0->O at end
        (re.compile(r'\bA[1I](?=[A-Z(])'),         'Al'),              # A1/AI -> Al before formula
        (re.compile(r'\bA[1I]\b'),                  'Al'),              # standalone A1->Al
        (re.compile(r'(?<=[A-Z])C1(?=\d|$|\))'),   'Cl'),              # C1->Cl (NaC1->NaCl)
    ]

    # Max plausible subscript per element in SPM formulas
    # N=10, H=22 to accommodate K4Fe(CN)6 (C:6,N:6) and C12H22O11 (H:22)
    _MAX_SUBSCRIPT: Dict[str, int] = {
        "H":22,"C":12,"N":10,"O":12,"S":4,"Cl":6,"F":6,"Br":3,"I":3,
        "Na":3,"K":4,"Ca":3,"Mg":3,"Al":4,"Fe":4,"Cu":4,"Zn":4,"Mn":2,
        "Cr":2,"Pb":2,"Ag":1,"Ba":1,
    }
    # Known valid complex/organic formulas that have high-count ligands
    _COMPLEX_WHITELIST = {
        "K4Fe(CN)6","K3Fe(CN)6","K2Fe(CN)6","Fe(CN)6",
        "C12H22O11","C6H12O6","C2H5OH","CH3COOH",
    }

    def correct(self, formula: str) -> Tuple[str, bool, str]:
        """Returns (corrected, was_changed, note)."""
        if not formula or not isinstance(formula, str):
            return formula, False, ""
        original = formula
        # 1. Exact lookup
        if formula in self._EXACT:
            corrected = self._EXACT[formula]
            if corrected != formula:
                return corrected, True, f"exact_table:{formula}->{corrected}"
        # 2. Regex patterns
        corrected = formula
        for pattern, replacement in self._PATTERNS:
            corrected = pattern.sub(replacement, corrected)
        if corrected != original:
            return corrected, True, f"regex:{original}->{corrected}"
        return formula, False, ""

    def detect_anomalous_count(self, formula: str) -> Optional[str]:
        """Detect impossible element counts caused by OCR digit errors."""
        if not _HAS_FORMULA_PARSER:
            return None
        # Skip whitelisted complex/organic formulas
        if hasattr(self, "_COMPLEX_WHITELIST") and formula in self._COMPLEX_WHITELIST:
            return None
        try:
            composition = _parse_formula(formula)
        except Exception:
            return None
        for elem, count in composition.items():
            max_ok = self._MAX_SUBSCRIPT.get(elem, 10)
            if count > max_ok:
                return (
                    f"'{formula}' has {elem}:{count} — count {count} is impossible. "
                    f"Likely OCR error (e.g. '0' read as 'O', so '{elem}{count}' "
                    f"should be '{elem}2O' or similar)."
                )
        return None


# =============================================================================
# FORMULA VALIDATOR
# =============================================================================

class FormulaValidator:
    """
    Validates chemical formula strings.
    Uses OCRCorrector to detect and fix common OCR errors.
    """

    _ELEMENT_RE = re.compile(r'([A-Z][a-z]?)')
    _corrector  = OCRCorrector()

    def validate(self, formula: str, field_name: str = "formula") -> ValidationReport:
        report = ValidationReport()

        if not formula or not isinstance(formula, str):
            report.add("critical", "FORMULA_EMPTY", "Formula is empty or missing.", field_name)
            return report

        # Try OCR correction
        corrected, was_changed, note = self._corrector.correct(formula)
        if was_changed:
            report.add("warning", "FORMULA_OCR_CORRECTED",
                       f"'{formula}' looks like OCR error, corrected to '{corrected}'. ({note})",
                       field_name)
            formula = corrected

        # Anomalous subscript count check (H20 -> H:20)
        anomaly = self._corrector.detect_anomalous_count(formula)
        if anomaly:
            report.add("critical", "FORMULA_ANOMALOUS_COUNT", anomaly, field_name)
            return report

        # Validate each segment (CuSO4.5H2O has two segments)
        for seg in formula.split('.'):
            seg = re.sub(r'^\d+', '', seg).strip()
            if seg:
                self._check_segment(seg, report, field_name)

        return report

    def _check_segment(self, seg: str, report: ValidationReport, field_name: str) -> None:
        if not seg[0].isupper():
            report.add("critical", "FORMULA_BAD_START",
                       f"'{seg}' doesn't start with uppercase — likely OCR corruption.", field_name)
            return
        # Balanced parentheses
        depth = 0
        for ch in seg:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            if depth < 0:
                report.add("critical", "FORMULA_UNBALANCED_PARENS",
                           f"Unbalanced parentheses in '{seg}'.", field_name)
                return
        if depth != 0:
            report.add("critical", "FORMULA_UNBALANCED_PARENS",
                       f"Unbalanced parentheses in '{seg}'.", field_name)
            return
        # Valid element symbols
        clean = re.sub(r'[0-9()\[\].]', '', seg)
        for m in self._ELEMENT_RE.finditer(clean):
            elem = m.group(1)
            if elem not in VALID_ELEMENTS:
                report.add("warning", "FORMULA_UNKNOWN_ELEMENT",
                           f"Element '{elem}' in '{seg}' not in SPM AR table. "
                           f"Possible OCR error ('l'->1, 'O'->0).", field_name)

    def correct(self, formula: str) -> Tuple[str, bool]:
        corrected, changed, _ = self._corrector.correct(formula)
        return corrected, changed


# =============================================================================
# EQUATION VALIDATOR  (FIX BUG 2 -- NEW)
# =============================================================================

class EquationValidator:
    """
    Validates chemical equation strings.
    NEW in v4.0 -- the old validator had no equation validation at all.

    Checks: arrow present, non-empty sides, balanced brackets,
    valid element symbols in each species, placeholder word detection.
    """

    _formula_v = FormulaValidator()

    _PLACEHOLDER_WORDS = {
        "salt","gas","precipitate","water","acid","base","product",
        "mendapan","garam","asid","bes","larutan","hasil",
    }

    def validate(self, equation: str, field_name: str = "equation") -> ValidationReport:
        report = ValidationReport()

        if not equation or not isinstance(equation, str):
            report.add("critical", "EQUATION_EMPTY", "Equation is empty.", field_name)
            return report

        # Normalise arrow variants
        eq = (equation
              .replace("→", "->").replace("⇌", "->")
              .replace("⟶", "->").replace("=>", "->"))

        if "->" not in eq:
            report.add("critical", "EQUATION_NO_ARROW",
                       f"Equation '{equation[:50]}' has no arrow (→ or ->). "
                       f"Check for OCR corruption.", field_name)
            return report

        lhs, rhs = [s.strip() for s in eq.split("->", 1)]

        if not lhs:
            report.add("critical", "EQUATION_EMPTY_LHS",
                       "No reactants before arrow.", field_name)
        if not rhs:
            report.add("critical", "EQUATION_EMPTY_RHS",
                       "No products after arrow.", field_name)
        if report.has_critical:
            return report

        for label, side_str in (("reactants", lhs), ("products", rhs)):
            self._check_side(side_str, label, report, field_name)

        return report

    def _check_side(self, side: str, label: str,
                    report: ValidationReport, field_name: str) -> None:
        if side.count("(") != side.count(")"):
            report.add("critical", "EQUATION_UNBALANCED_PARENS",
                       f"Unbalanced parentheses in {label}: '{side[:40]}'.", field_name)

        for token in [t.strip() for t in side.split("+") if t.strip()]:
            token_clean = re.sub(r"^\d+\s*", "", token).strip()
            token_clean = re.sub(r"\((aq|s|l|g)\)$", "", token_clean,
                                  flags=re.IGNORECASE).strip()
            if not token_clean:
                continue
            if token_clean.lower() in self._PLACEHOLDER_WORDS:
                report.add("warning", "EQUATION_PLACEHOLDER_WORD",
                           f"'{token_clean}' in {label} is a placeholder, not a formula. "
                           f"Stoich ratio will default to 1:1.", field_name)
                continue
            sub = self._formula_v.validate(token_clean, f"{field_name}.{label}")
            report.extend(sub)


# =============================================================================
# UNIT / VALUE VALIDATOR  (FIX BUG 3 -- negative values -> CRITICAL)
# =============================================================================

class UnitValidator:
    """
    Validates extracted numeric values for physical plausibility.

    FIX BUG 3: negative mass, volume, moles, concentration -> CRITICAL
    (old version returned WARNING for all out-of-range values).
    """

    def validate_extraction_data(self, data: Dict[str, Any]) -> ValidationReport:
        report = ValidationReport()
        if not data:
            return report

        # Mass (negative = CRITICAL)
        for key in ("mass_g", "given_mass_g", "jisim_g"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val < 0:
                    report.add("critical", "NEGATIVE_MASS",
                               f"Mass {key}={val}g cannot be negative.", key)
                elif val == 0:
                    report.add("warning", "ZERO_MASS",
                               f"Mass {key}=0. Check OCR.", key)
                elif val > 10_000:
                    report.add("warning", "LARGE_MASS",
                               f"Mass {key}={val}g is very large for SPM.", key)

        # Volume (negative = CRITICAL)
        for key in ("volume_cm3","V1","V2","known_volume_cm3",
                    "unknown_volume_cm3","given_volume_cm3"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val < 0:
                    report.add("critical", "NEGATIVE_VOLUME",
                               f"Volume {key}={val}cm³ cannot be negative.", key)
                elif val > 10_000:
                    report.add("warning", "LARGE_VOLUME",
                               f"Volume {key}={val}cm³ is very large.", key)

        for key in ("volume_dm3",):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val < 0:
                    report.add("critical", "NEGATIVE_VOLUME_DM3",
                               f"Volume {key}={val}dm³ cannot be negative.", key)
                elif val > 10:
                    report.add("warning", "LARGE_VOLUME_DM3",
                               f"Volume {key}={val}dm³ is unusually large.", key)

        # Molarity (negative = CRITICAL)
        for key in ("molarity","M1","M2","known_molarity","unknown_molarity","concentration"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val < 0:
                    report.add("critical", "NEGATIVE_MOLARITY",
                               f"Molarity {key}={val} cannot be negative.", key)
                elif val > 20:
                    report.add("warning", "HIGH_MOLARITY",
                               f"Molarity {key}={val} > 20 mol/dm³ (above conc H2SO4). "
                               f"Check for OCR digit error.", key)

        # Moles (negative = CRITICAL)
        val = data.get("moles")
        if val is not None and isinstance(val, (int, float)):
            if val < 0:
                report.add("critical", "NEGATIVE_MOLES",
                           f"Moles={val} cannot be negative.", "moles")
            elif val > 500:
                report.add("warning", "LARGE_MOLES",
                           f"Moles={val} is unusually large.", "moles")

        # pH / pOH (impossible values = CRITICAL)
        for key in ("ph","pH"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val < 0 or val > 14:
                    report.add("critical", "INVALID_PH",
                               f"pH={val} outside [0,14] — impossible for aqueous solution.", key)

        for key in ("poh","pOH"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val < 0 or val > 14:
                    report.add("critical", "INVALID_POH",
                               f"pOH={val} outside [0,14].", key)

        # [H+] and [OH-] must be positive
        for key in ("h_plus","h_conc","H_concentration"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val <= 0:
                    report.add("critical", "NON_POSITIVE_H_CONC",
                               f"[H⁺]={val} must be positive.", key)
                elif val > 20:
                    report.add("warning", "HIGH_H_CONC",
                               f"[H⁺]={val} mol/dm³ is extremely high.", key)

        for key in ("oh_minus","oh_conc","OH_concentration"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val <= 0:
                    report.add("critical", "NON_POSITIVE_OH_CONC",
                               f"[OH⁻]={val} must be positive.", key)

        # Thermochemistry
        val = data.get("delta_H")
        if val is not None and isinstance(val, (int, float)):
            if abs(val) > 10_000:
                report.add("warning", "EXTREME_DELTA_H",
                           f"ΔH={val} kJ/mol is very large. Check J vs kJ.", "delta_H")

        val = data.get("Q_joules")
        if val is not None and isinstance(val, (int, float)):
            if val < 0:
                report.add("warning", "NEGATIVE_Q",
                           f"Q={val}J is negative. Check ΔT sign.", "Q_joules")

        # Specific heat
        for key in ("c_specific_heat","specific_heat","c"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if val <= 0:
                    report.add("critical", "NON_POSITIVE_SPECIFIC_HEAT",
                               f"Specific heat {key}={val} must be positive.", key)
                elif val > 100:
                    report.add("warning", "HIGH_SPECIFIC_HEAT",
                               f"Specific heat {key}={val} J/g°C seems very high (water=4.2).", key)

        # Electrode potentials
        for key in ("e0_cathode","e0_anode"):
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)):
                if abs(val) > 10:
                    report.add("warning", "EXTREME_ELECTRODE_POTENTIAL",
                               f"E°{key}={val}V is outside typical range ±5V.", key)

        return report


# =============================================================================
# ANSWER VALIDATOR  (FIX BUG 4 -- accepts str OR dict)
# =============================================================================

class AnswerValidator:
    """
    Sanity-checks solver output before returning to student.

    FIX BUG 4: solver_engine.solve_by_task() returns str, not dict.
    Old validator called result.get() which crashed on strings.
    This version handles BOTH str and dict.

    For str output: extracts numbers with regex patterns that match
    the solver's own formatted output (e.g. "pH = 2.00", "Jisim H2O = 36 g").
    """

    _EXTRACT = [
        (re.compile(r'pH\s*=\s*([+-]?\d+(?:\.\d+)?)',     re.IGNORECASE), "pH"),
        (re.compile(r'pOH\s*=\s*([+-]?\d+(?:\.\d+)?)',    re.IGNORECASE), "pOH"),
        (re.compile(r'\bn\s*=\s*([+-]?\d+(?:\.\d+)?)\s*mol', re.IGNORECASE), "n_mol"),
        (re.compile(r'(?:Jisim|Mass)\s+\w+\s*=\s*([+-]?\d+(?:\.\d+)?)\s*g', re.IGNORECASE), "mass_g"),
        (re.compile(r'ΔH\s*=\s*([+-]?\d+(?:\.\d+)?)',     re.IGNORECASE), "delta_H"),
        (re.compile(r'E.cel[l]?\s*=\s*([+-]?\d+(?:\.\d+)?)', re.IGNORECASE), "e0_sel"),
        (re.compile(r'Kemolaran\s+\w+\s*=\s*([+-]?\d+(?:\.\d+)?)', re.IGNORECASE), "molarity"),
        (re.compile(r'(?:Isipadu|Volume)\s+\w+\s*=\s*([+-]?\d+(?:\.\d+)?)\s*cm', re.IGNORECASE), "volume_cm3"),
    ]

    def validate_answer(self, task: str, result: Any) -> ValidationReport:
        report = ValidationReport()
        if result is None:
            return report

        values: Dict[str, float] = {}

        if isinstance(result, dict):
            if "error" in result:
                return report
            values = {k: v for k, v in result.items()
                      if isinstance(v, (int, float))}
        elif isinstance(result, str):
            for pattern, key in self._EXTRACT:
                m = pattern.search(result)
                if m:
                    try:
                        values[key] = float(m.group(1))
                    except (ValueError, IndexError):
                        pass

        # pH / pOH must be [0, 14]
        for key in ("pH", "pOH"):
            val = values.get(key)
            if val is not None and not (0 <= val <= 14):
                report.add("critical", f"INVALID_{key}_OUTPUT",
                           f"Solver returned {key}={val:.4f} — must be 0–14. "
                           f"Check [H⁺]/[OH⁻] input.", key)

        # Moles must be positive
        val = values.get("n_mol")
        if val is not None:
            if val < 0:
                report.add("critical", "NEGATIVE_MOLES_OUTPUT",
                           f"Solver returned moles={val:.4f} — impossible.", "n_mol")
            elif val > 500:
                report.add("warning", "LARGE_MOLES_OUTPUT",
                           f"Solver returned {val:.1f} mol — unusually large for SPM.", "n_mol")

        # Mass must be positive
        val = values.get("mass_g")
        if val is not None and val < 0:
            report.add("critical", "NEGATIVE_MASS_OUTPUT",
                       f"Solver returned mass={val:.4f}g — impossible.", "mass_g")

        # ΔH sanity
        val = values.get("delta_H")
        if val is not None and abs(val) > 10_000:
            report.add("warning", "EXTREME_DELTA_H_OUTPUT",
                       f"ΔH={val:.1f} kJ/mol — check J vs kJ confusion. "
                       f"Typical SPM range: 10–1000 kJ/mol.", "delta_H")

        # EMF sanity
        val = values.get("e0_sel")
        if val is not None and abs(val) > 10:
            report.add("warning", "EXTREME_EMF_OUTPUT",
                       f"E°cell={val:.2f}V unusually large.", "e0_sel")

        # Molarity must be positive
        val = values.get("molarity")
        if val is not None and val < 0:
            report.add("critical", "NEGATIVE_MOLARITY_OUTPUT",
                       f"Molarity={val:.4f} cannot be negative.", "molarity")

        # Volume must be positive
        val = values.get("volume_cm3")
        if val is not None and val < 0:
            report.add("critical", "NEGATIVE_VOLUME_OUTPUT",
                       f"Volume={val:.4f}cm³ cannot be negative.", "volume_cm3")

        return report


# =============================================================================
# MAIN INTEGRATION WRAPPER  (FIX BUGS 5 & 6)
# =============================================================================

class ChemistryValidator:
    """
    Combined validator used in main.py pipeline.

    UPDATED main.py USAGE:
        cv = ChemistryValidator()

        # Step 0 (NEW): OCR-correct formulas before validation
        data = cv.correct_ocr(data)

        # Step 1: Pre-solve
        pre = cv.validate_extraction(task, data)
        if pre.has_critical:
            return pre.user_message(lang)      # NEW: bot-ready HTML

        answer_str = solve_by_task(task, data)

        # Step 2: Post-solve (NOW ACCEPTS str)
        post = cv.validate_answer(task, answer_str)
        if post.has_critical:
            logger.error(f'solver sanity fail: {post.summary()}')
    """

    def __init__(self) -> None:
        self._corrector  = OCRCorrector()
        self._formula_v  = FormulaValidator()
        self._equation_v = EquationValidator()
        self._unit_v     = UnitValidator()
        self._answer_v   = AnswerValidator()

    def correct_ocr(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-correct OCR errors in all formula fields.
        Returns a NEW dict. Safe to call even if data has no formula fields.
        """
        if not data:
            return data
        corrected = dict(data)
        for key in ("formula","given_formula","target_formula",
                    "species","oxidant","reductant"):
            val = corrected.get(key)
            if val and isinstance(val, str):
                fixed, changed, _ = self._corrector.correct(val)
                if changed:
                    corrected[key] = fixed
        # formulas list
        flist = corrected.get("formulas")
        if flist and isinstance(flist, list):
            corrected["formulas"] = [
                self._corrector.correct(f)[0] if isinstance(f, str) else f
                for f in flist
            ]
        return corrected

    def validate_extraction(self, task: str, data: Dict[str, Any]) -> ValidationReport:
        """
        Validate extracted data BEFORE calling solver.
        FIX BUG 5: now checks equation, species, oxidant fields.
        """
        report = ValidationReport()
        if not data:
            return report

        # Formula fields (FIX BUG 5: added species, oxidant, reductant)
        for key in ("formula","given_formula","target_formula",
                    "species","oxidant","reductant"):
            val = data.get(key)
            if val and isinstance(val, str):
                report.extend(self._formula_v.validate(val, key))

        # Equation field (FIX BUG 5: was never checked before)
        equation = data.get("equation")
        if equation and isinstance(equation, str):
            report.extend(self._equation_v.validate(equation, "equation"))

        # Numeric fields
        report.extend(self._unit_v.validate_extraction_data(data))

        # Task-specific rules
        report.extend(self._task_specific(task, data))

        return report

    def _task_specific(self, task: str, data: Dict[str, Any]) -> ValidationReport:
        report = ValidationReport()

        # Stoichiometry: coefficients must not be zero
        if "stoichiometry" in task:
            for key in ("given_coeff","target_coeff"):
                val = data.get(key)
                if val is not None and val == 0:
                    report.add("critical","ZERO_STOICH_COEFF",
                               f"Coefficient {key}=0 — equation parse error.", key)
            gf = data.get("given_formula")
            tf = data.get("target_formula")
            if gf and tf and gf == tf:
                report.add("warning","SAME_GIVEN_TARGET",
                           f"given_formula==target_formula=='{gf}'. Check equation.", "given_formula")

        # Empirical formula: percentages should sum to ~100
        if task == "empirical_formula":
            masses = data.get("element_masses", {})
            if masses and isinstance(masses, dict):
                total = sum(v for v in masses.values() if isinstance(v, (int, float)))
                if 50 < total < 200 and not (95 <= total <= 105):
                    report.add("warning","EMPIRICAL_PERCENT_SUM",
                               f"Element percentages sum={total:.1f}% (expected ~100%). "
                               f"Check for OCR digit corruption.", "element_masses")

        # Titration: volumes must be positive
        if "titration" in task:
            for key in ("known_volume_cm3","unknown_volume_cm3"):
                val = data.get(key)
                if val is not None and val <= 0:
                    report.add("critical","NON_POSITIVE_TITRATION_VOLUME",
                               f"Titration volume {key}={val} must be positive.", key)

        # Calorimetry: initial != final temperature
        if task in ("calorimetry","delta_h_from_calorimetry"):
            t_i = data.get("temp_initial")
            t_f = data.get("temp_final")
            if t_i is not None and t_f is not None and t_i == t_f:
                report.add("warning","ZERO_DELTA_T",
                           f"temp_initial==temp_final=={t_i}°C. ΔT=0. Check OCR values.",
                           "temp_initial")

        # Voltaic cell: cathode must be more positive for spontaneous reaction
        if task == "voltaic_cell":
            e_cat = data.get("e0_cathode")
            e_an  = data.get("e0_anode")
            if e_cat is not None and e_an is not None and e_cat <= e_an:
                report.add("warning","NON_SPONTANEOUS_CELL",
                           f"E°cathode({e_cat}V) <= E°anode({e_an}V) — cell is non-spontaneous. "
                           f"Check electrode assignment.", "e0_cathode")

        # Dilution: M2 must be less than M1
        if task == "dilution":
            m1 = data.get("M1")
            m2 = data.get("M2")
            if m1 is not None and m2 is not None and m2 > m1:
                report.add("warning","DILUTION_CONCENTRATION_INCREASE",
                           f"M2={m2} > M1={m1} — dilution should decrease concentration. "
                           f"Check M1/M2 values.", "M2")

        return report

    def validate_answer(self, task: str, result: Any) -> ValidationReport:
        """
        FIX BUG 4: accepts str (from solver_engine) as well as dict.
        """
        return self._answer_v.validate_answer(task, result)
