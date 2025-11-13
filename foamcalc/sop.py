import streamlit as st

def default_sop_text() -> str:
    return (
        "Purpose & scope: determine foam thickening time and density at 25.0 ± 0.3 °C. "
        "Equipment: industrial mixer (Type EWTM), Testo 925 thermometer, mixer attachments, foam box with plastic bag, polyol, isocyanate (MDI), c‑pentane, paper cups (250 mL), PPE, scales, distilled water flask, 500 mL plastic container, Metabo band saw. "
        "Process: pre‑mix polyol + c‑pentane to target mass, temper both streams to 25.0 ± 0.3 °C, mix with MDI for 5 s, start timer, pour, probe at ~35 s until fibers appear to capture thickening time, then after 20–25 min cut 5 cm cubes and measure density by displacement."
    )

def sop_ui(default_text: str):
    st.markdown("##### SOP text")
    st.text_area("Reference", value=default_text, height=180)

    st.markdown("##### Thickening time check (25.0 ± 0.3 °C)")
    t_meas = st.number_input("Observed thickening time (s)", min_value=0.0, value=45.0, step=0.5)
    t_nom = st.number_input("Target (s)", min_value=0.0, value=45.0, step=0.5)
    tol = st.number_input("Tolerance (± s)", min_value=0.0, value=4.0, step=0.5)
    if abs(t_meas - t_nom) <= tol:
        st.success("Thickening time within tolerance.")
    else:
        st.warning("Thickening time outside tolerance; re‑temper or re‑test.")

    st.markdown("##### Density check by displacement")
    mass_g = st.number_input("Cube mass m (g)", min_value=0.0, value=12.0, step=0.1)
    disp_ml = st.number_input("Displaced water volume V (mL)", min_value=0.0, value=500.0, step=1.0)
    density_kg_m3 = (mass_g / max(1e-9, disp_ml)) * 1000.0
    st.write(f"Foam density (kg/m³): {density_kg_m3:.1f}")

    d_nom = st.number_input("Target density (kg/m³)", min_value=0.0, value=23.6, step=0.1)
    d_tol = st.number_input("Tolerance (± kg/m³)", min_value=0.0, value=1.0, step=0.1)
    if abs(density_kg_m3 - d_nom) <= d_tol:
        st.success("Density within tolerance.")
    else:
        st.warning("Density out of tolerance; adjust process and re‑evaluate.")
