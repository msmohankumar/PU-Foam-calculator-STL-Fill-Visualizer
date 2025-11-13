import os
from textwrap import dedent

import streamlit as st

def _get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", None)
        except Exception:
            api_key = None
    if not api_key:
        return None
    try:
        from groq import Groq  # lazy import
        return Groq(api_key=api_key)
    except Exception:
        return None

def groq_available() -> bool:
    try:
        return _get_groq_client() is not None
    except Exception:
        return False

def render_groq_panel(
    V_mold_ml: float,
    expansion_multiple: float,
    safety_margin_pct: float,
    ratio_A_to_B: float,
    sg_A: float,
    sg_B: float,
    summary_defaults: dict,
):
    with st.expander("AI Assistant (Groq)"):
        st.write("Generate a concise explanation with formulas and ratios.")
        notes = st.text_area("Notes (ambient temp, test shots, mold material):", height=80)
        model = st.text_input("Groq model", value="llama-3.3-70b-versatile")
        if st.button("Explain my plan"):
            client = _get_groq_client()
            if not client:
                st.error("Groq client not available. Set GROQ_API_KEY and install groq.")
                return
            msg = dedent(f"""
            Provide a concise explanation for sizing a polyurethane foam pour and mixing, including formulas.
            Use these inputs:
            - Mold volume (mL): {V_mold_ml}
            - Expansion (×): {expansion_multiple}
            - Safety margin (%): {safety_margin_pct}
            - Ratio A:B (by weight): {ratio_A_to_B}
            - SG_A (g/mL): {sg_A}
            - SG_B (g/mL): {sg_B}
            Notes: {notes or "N/A"}

            Include:
            - Liquid needed formula and a worked numeric example from inputs.
            - Exact mass split enforcing the weight ratio with SG conversion.
            - Practical timing tips (mixing window, pour handling).
            - One-paragraph rationale for verifying thickening time and density checks.
            """)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in moldmaking and casting."},
                        {"role": "user", "content": msg},
                    ],
                    temperature=0.2,
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"Groq error: {e}")
