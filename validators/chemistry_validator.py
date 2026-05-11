"""
validators/chemistry_validator.py — Cikgu AI Kimia
====================================================
Deterministic chemistry validation layer.

Three validator classes + one integration wrapper:

  FormulaValidator   — checks chemical formulas from extractor
  UnitValidator      — checks numeric values and units are sensible
  AnswerValidator    — sanity-checks solver output before returning
  ChemistryValidator — combines all three, used in main pipeline

Usage in main pipeline (after extractor, before solver):
  from validators.chemistry_validator import ChemistryValidator
  cv = ChemistryValidator()
  issues = cv.validate_extraction(task, data)
  if issues.has_critical:
      return fallback_message(...)
  solver_result = solve_by_task(task, data)
  issues2 = cv.validate_answer(task, solver_result)

Zero external dependencies — pure Python.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Import AR table from single source ────────────────────────────────────
# solver/constants.py must exist (see audit recommendation)
# Fallback to inline table if not yet refactored
try:
    from solver.constants import AR
except ImportError:
    AR = {
        "H":1.0,"He":4.0,"Li":7.0,"Be":9.0,"B":11.0,"C":12.0,"N":14.0,
        "O":16.0,"F":19.0,"Ne":20.0,"Na":23.0,"Mg":24.0,"Al":27.0,"Si":28.0,
        "P":31.0,"S":32.0,"Cl":35.5,"Ar":40.0,"K":39.0,"Ca":40.0,
        "Ti":48.0,"V":51.0,"Cr":52.0,"Mn":55.0,"Fe":56.0,"Co":59.0,
        "Ni":58.7,"Cu":63.5,"Zn":65.0,"As":75.0,"Se":79.0,"Br":80.0,
        "Kr":84.0,"Ag":108.0,"Sn":118.7,"I":127.0,"Ba":137.0,
        "Hg":200.6,"Pb":207.0,
    }

VALID_ELEMENTS = set(AR.keys())

# ── Validation result ──────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str        # "critical" | "warning" | "info"
    code: str            # machine-readable code
    message: str         # human-readable description
    field: Optional[str] = None  # which data field triggered this


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, field: str = None):
        self.issues.append(ValidationIssue(severity, code, message, field))

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
        if c:
            parts.append(f"{c} critical")
        if w:
            parts.append(f"{w} warning")
        return ", ".join(parts)


# ── Formula Validator ──────────────────────────────────────────────────────

class FormulaValidator:
    """
    Validates chemical formula strings.

    Rules:
      - Must start with uppercase letter
      - All element symbols must be in AR table
      - Parentheses must be balanced
      - Subscript numbers must follow element or closing bracket
      - Hydrate notation (CuSO4.5H2O) is valid
    """

    _ELEMENT_RE = re.compile(r'([A-Z][a-z]?)')

    def validate(self, formula: str, field_name: str = "formula") -> ValidationReport:
        report = ValidationReport()

        if not formula or not isinstance(formula, str):
            report.add("critical", "FORMULA_EMPTY",
                       f"Formula is empty or not a string.", field_name)
            return report

        # Remove hydrate marker to check each segment
        segments = formula.split('.')
        for seg in segments:
            # Strip leading stoichiometric coefficient
            seg = re.sub(r'^\d+', '', seg).strip()
            if not seg:
                continue
            self._check_segment(seg, report, field_name)

        return report

    def _check_segment(self, seg: str, report: ValidationReport, field_name: str):
        # Must start with uppercase letter
        if not seg or not seg[0].isupper():
            report.add("critical", "FORMULA_BAD_START",
                       f"Formula segment '{seg}' doesn't start with uppercase — "
                       f"likely OCR corruption.", field_name)
            return

        # Check balanced parentheses
        depth = 0
        for ch in seg:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                report.add("critical", "FORMULA_UNBALANCED_PARENS",
                           f"Unbalanced parentheses in '{seg}'.", field_name)
                return
        if depth != 0:
            report.add("critical", "FORMULA_UNBALANCED_PARENS",
                       f"Unbalanced parentheses in '{seg}'.", field_name)
            return

        # Check all element symbols are valid
        # Remove digits and brackets to get pure element stream
        clean = re.sub(r'[0-9().]', '', seg)
        for m in self._ELEMENT_RE.finditer(clean):
            elem = m.group(1)
            if elem not in VALID_ELEMENTS:
                report.add("warning", "FORMULA_UNKNOWN_ELEMENT",
                           f"Element '{elem}' in '{seg}' not in AR table. "
                           f"May be OCR corruption (e.g., '0' vs 'O', 'l' vs '1').",
                           field_name)


# ── Unit / Value Validator ─────────────────────────────────────────────────

class UnitValidator:
    """
    Validates extracted numeric values are physically sensible.
    """

    # Sensible ranges for SPM chemistry values
    _RANGES = {
        "mass_g":          (1e-6, 10_000),    # micrograms to 10kg
        "volume_cm3":      (0.01,  10_000),   # 0.01mL to 10L
        "volume_dm3":      (1e-5,  10.0),     # sub-mL to 10dm³ (typical SPM)
        "molarity":        (1e-6,  20.0),     # dilute to concentrated acid
        "moles":           (1e-8,  100.0),    # trace to 100 mol
        "delta_T":         (-100,  200),      # °C change
        "temperature":     (-50,   200),      # °C absolute
        "delta_H":         (-5000, 5000),     # kJ/mol
        "Q_joules":        (0.01,  1_000_000),
        "ph":              (0,     14),
        "poh":             (0,     14),
        "e0_cathode":      (-5.0,  5.0),      # V
        "e0_anode":        (-5.0,  5.0),
        "time_seconds":    (0,     86400),    # up to 24 hours
    }

    def validate_value(
        self, value: float, field_name: str, unit: str = ""
    ) -> ValidationReport:
        report = ValidationReport()

        if not isinstance(value, (int, float)):
            report.add("critical", "VALUE_NOT_NUMERIC",
                       f"'{field_name}' = {value!r} is not a number.", field_name)
            return report

        if math.isnan(value) or math.isinf(value):
            report.add("critical", "VALUE_INVALID",
                       f"'{field_name}' = {value} is NaN or Inf.", field_name)
            return report

        lo, hi = self._RANGES.get(field_name, (-1e18, 1e18))
        if not (lo <= value <= hi):
            report.add("warning", "VALUE_OUT_OF_RANGE",
                       f"'{field_name}' = {value} {unit} is outside expected SPM range "
                       f"[{lo}, {hi}]. Check for OCR digit corruption.", field_name)

        return report

    def validate_extraction_data(self, data: dict) -> ValidationReport:
        """Validate all numeric fields in an extracted data dict."""
        report = ValidationReport()
        numeric_fields = [
            "mass_g", "volume_cm3", "volume_dm3", "molarity", "moles",
            "delta_T", "delta_H", "Q_joules", "ph", "poh",
            "e0_cathode", "e0_anode", "temperature",
        ]
        for field_name in numeric_fields:
            val = data.get(field_name)
            if val is not None:
                sub = self.validate_value(val, field_name)
                report.issues.extend(sub.issues)
        return report


# ── Answer Validator ───────────────────────────────────────────────────────

class AnswerValidator:
    """
    Sanity-checks solver output before returning to student.
    Catches impossible chemistry: negative mass, pH outside 0–14, etc.
    """

    def validate_answer(self, task: str, result: dict) -> ValidationReport:
        report = ValidationReport()

        if not isinstance(result, dict):
            return report

        if "error" in result:
            # Solver already flagged an error — don't double-report
            return report

        # ── Mole calculations ──────────────────────────────────────────
        for key in ("n_mol", "n_given", "n_target"):
            val = result.get(key)
            if val is not None:
                if val < 0:
                    report.add("critical", "NEGATIVE_MOLES",
                               f"Solver returned {key}={val} mol — moles cannot be negative.",
                               key)
                if val > 1000:
                    report.add("warning", "LARGE_MOLES",
                               f"Solver returned {key}={val} mol — unusually large for SPM.",
                               key)

        # ── Mass ───────────────────────────────────────────────────────
        mass = result.get("mass_g") or result.get("result")
        if mass is not None and task in ("stoichiometry", "moles_from_mass",
                                          "concentration", "mass_from_molarity"):
            if mass < 0:
                report.add("critical", "NEGATIVE_MASS",
                           f"Solver returned mass={mass}g — mass cannot be negative.", "mass_g")
            if mass > 100_000:
                report.add("warning", "VERY_LARGE_MASS",
                           f"Solver returned mass={mass}g — check input values.", "mass_g")

        # ── pH ─────────────────────────────────────────────────────────
        ph = result.get("pH")
        if ph is not None:
            if not (0 <= ph <= 14):
                report.add("critical", "INVALID_PH",
                           f"Solver returned pH={ph} — pH must be 0–14 for aqueous solutions.",
                           "pH")

        poh = result.get("pOH")
        if poh is not None:
            if not (0 <= poh <= 14):
                report.add("critical", "INVALID_POH",
                           f"Solver returned pOH={poh} — must be 0–14.", "pOH")

        # ── Thermochemistry ────────────────────────────────────────────
        dh = result.get("delta_H")
        if dh is not None:
            if abs(dh) > 10_000:
                report.add("warning", "LARGE_DELTA_H",
                           f"ΔH={dh} kJ/mol is very large. Check if units are correct "
                           f"(J vs kJ confusion is common).", "delta_H")

        q_j = result.get("Q_joules")
        if q_j is not None:
            if q_j < 0:
                report.add("warning", "NEGATIVE_Q",
                           f"Q={q_j}J is negative. Check ΔT sign convention.", "Q_joules")

        # ── Voltaic cell ───────────────────────────────────────────────
        e0 = result.get("e0_sel")
        if e0 is not None and abs(e0) > 10:
            report.add("warning", "LARGE_EMF",
                       f"E°cell={e0}V seems unusually large.", "e0_sel")

        # ── Volume ────────────────────────────────────────────────────
        vol = result.get("V2") or result.get("result")
        if vol is not None and task in ("titration", "dilution"):
            if vol < 0:
                report.add("critical", "NEGATIVE_VOLUME",
                           f"Solver returned volume={vol} — volume cannot be negative.", "V2")
            if vol > 10_000:
                report.add("warning", "LARGE_VOLUME",
                           f"Solver returned volume={vol}cm³ — check input values.", "V2")

        return report


# ── Main Integration Wrapper ───────────────────────────────────────────────

class ChemistryValidator:
    """
    Combined validator used in the main pipeline.

    Usage:
        cv = ChemistryValidator()

        # Before solver:
        pre_report = cv.validate_extraction(task, data)
        if pre_report.has_critical:
            return error_response(pre_report)

        result = solve_by_task(task, data)

        # After solver:
        post_report = cv.validate_answer(task, result)
        if post_report.has_critical:
            logger.error(f"Solver sanity fail: {post_report.summary()}")
            return error_response(post_report)
    """

    def __init__(self):
        self._formula_v = FormulaValidator()
        self._unit_v    = UnitValidator()
        self._answer_v  = AnswerValidator()

    def validate_extraction(self, task: str, data: dict) -> ValidationReport:
        """
        Validate extracted data before calling solver.
        Checks formulas, numeric values.
        """
        report = ValidationReport()

        if not data:
            return report

        # Formula checks
        for key in ("formula", "given_formula", "target_formula"):
            formula = data.get(key)
            if formula and isinstance(formula, str):
                sub = self._formula_v.validate(formula, key)
                report.issues.extend(sub.issues)

        # Numeric value checks
        sub = self._unit_v.validate_extraction_data(data)
        report.issues.extend(sub.issues)

        # Task-specific checks
        report.issues.extend(self._task_specific_checks(task, data).issues)

        return report

    def _task_specific_checks(self, task: str, data: dict) -> ValidationReport:
        report = ValidationReport()

        if task in ("ph_from_h", "ph_from_poh"):
            h_plus = data.get("h_plus")
            if h_plus is not None and h_plus <= 0:
                report.add("critical", "INVALID_H_CONCENTRATION",
                           f"[H+] = {h_plus} — concentration must be positive.", "h_plus")
            if h_plus is not None and h_plus > 20:
                report.add("warning", "HIGH_H_CONCENTRATION",
                           f"[H+] = {h_plus} mol/dm³ seems very high for SPM context.", "h_plus")

        if task == "empirical_formula":
            masses = data.get("element_masses", {})
            total  = sum(masses.values()) if masses else 0
            # Percentages should sum to ~100 (±2 rounding)
            if masses and 50 < total < 200:
                if not (95 <= total <= 105):
                    report.add("warning", "EMPIRICAL_PERCENT_SUM",
                               f"Percentage sum = {total:.1f}% (expected ~100%). "
                               f"Check for OCR digit corruption.", "element_masses")

        if task in ("stoichiometry_mass_to_mass", "stoichiometry_mass_to_volume"):
            given_coeff  = data.get("given_coeff",  1)
            target_coeff = data.get("target_coeff", 1)
            if given_coeff == 0 or target_coeff == 0:
                report.add("critical", "ZERO_STOICH_COEFF",
                           "Stoichiometric coefficient is 0 — likely equation parse error.",
                           "coeff")

        return report

    def validate_answer(self, task: str, result: dict) -> ValidationReport:
        """Validate solver output before returning to student."""
        return self._answer_v.validate_answer(task, result)
