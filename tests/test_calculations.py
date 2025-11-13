import math
from foamcalc.calculations import compute_requirements

def test_example_alignment():
    # Example: 1000 mL mold, 4.5× expansion, 2:1 A:B, SG_A=1.05, SG_B=1.13
    out = compute_requirements(
        V_mold_ml=1000.0,
        expansion_multiple=4.5,
        safety_margin_pct=0.0,
        ratio_A_to_B=2.0,
        sg_A=1.05,
        sg_B=1.13,
    )
    # Liquid ~222 mL
    assert math.isclose(out["V_liquid_ml"], 222.2222, rel_tol=1e-3)
    # Expect masses near 156 g (A) and 78 g (B) given the weight ratio and SGs
    assert math.isclose(out["m_A_g"], 156.0, rel_tol=0.03)
    assert math.isclose(out["m_B_g"], 78.0, rel_tol=0.03)
