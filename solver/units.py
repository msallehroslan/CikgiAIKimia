from typing import Optional


def cm3_to_dm3(v_cm3: float) -> float:
    return v_cm3 / 1000.0


def dm3_to_cm3(v_dm3: float) -> float:
    return v_dm3 * 1000.0


def mg_to_g(m_mg: float) -> float:
    return m_mg / 1000.0


def g_to_mg(m_g: float) -> float:
    return m_g * 1000.0


def kg_to_g(m_kg: float) -> float:
    return m_kg * 1000.0


def g_to_kg(m_g: float) -> float:
    return m_g / 1000.0


def j_to_kj(joule: float) -> float:
    return joule / 1000.0


def kj_to_j(kj: float) -> float:
    return kj * 1000.0


def celsius_delta(t_initial: float, t_final: float) -> float:
    return t_final - t_initial


def ensure_dm3(volume_dm3: Optional[float] = None, volume_cm3: Optional[float] = None) -> float:
    if volume_dm3 is not None:
        return volume_dm3
    if volume_cm3 is not None:
        return cm3_to_dm3(volume_cm3)
    raise ValueError("Perlu beri isipadu dalam dm³ atau cm³.")
