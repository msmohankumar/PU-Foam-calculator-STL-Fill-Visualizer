from typing import Dict

def compute_requirements(
    V_mold_ml: float,
    expansion_multiple: float,
    safety_margin_pct: float,
    ratio_A_to_B: float,
    sg_A: float,
    sg_B: float,
) -> Dict[str, float]:
    s = max(safety_margin_pct, 0.0) / 100.0
    E = max(expansion_multiple, 1e-6)
    r = max(ratio_A_to_B, 1e-6)

    V_liquid = (V_mold_ml / E) * (1.0 + s)

    denom = (r / sg_A) + (1.0 / sg_B)
    m_B = V_liquid / denom
    m_A = r * m_B

    V_A = m_A / sg_A
    V_B = m_B / sg_B

    m_total = m_A + m_B
    V_total = V_A + V_B

    return {
        "V_liquid_ml": V_liquid,
        "m_A_g": m_A,
        "m_B_g": m_B,
        "V_A_ml": V_A,
        "V_B_ml": V_B,
        "m_total_g": m_total,
        "V_total_ml": V_total,
    }
