from pathlib import Path
import time
import os
import glob
import base64
import mimetypes
import hashlib
import tempfile
from datetime import datetime

# CRITICAL FIX for Streamlit Cloud: Set environment variable to force software rendering BEFORE pyvista is imported
os.environ['PYVISTA_OFF_SCREEN'] = 'True'

from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fpdf import FPDF
from fpdf.errors import FPDFUnicodeEncodingException

# 3D Visualization
import pyvista as pv
from stpyvista import stpyvista

# Load environment variables from a .env file for local development
load_dotenv()

# --- 1. Set Page Config as the first Streamlit command ---
st.set_page_config(page_title="PU Foam + STL Fill Visualizer", page_icon="🧪", layout="wide")

# --- 2. Initialize Session State for Report Data ---
# This is crucial for collecting data across tabs for the final PDF report.
if "report_data" not in st.session_state:
    st.session_state.report_data = {
        "mesh": None,
        "mesh_properties": None,
        "mesh_snapshot_path": None,
        "ai_planner_output": "Not generated yet.",
        "sop_html_path": None,
        "sop_batch_results": None,
        "qc_results": None,
        "ai_explainer_output": "Not generated yet.",
    }

# ------------------------------------------------------------------
# DETAILED PDF REPORT GENERATION (with Unicode Fix)
# ------------------------------------------------------------------

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'PU Foam Calculator & Visualizer Report', 0, 0, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        # Sanitize title to prevent errors
        safe_title = title.encode('latin-1', 'replace').decode('latin-1')
        self.cell(0, 10, safe_title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, content):
        self.set_font('Arial', '', 11)
        # Sanitize body content to prevent errors
        safe_content = str(content).encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 5, safe_content)
        self.ln()

    def add_html_content(self, html_path, resources_dir):
        if not html_path or not html_path.is_file():
            self.chapter_body("SOP HTML file not found.")
            return

        soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'img']):
            text = element.get_text(strip=True)
            # --- UNICODE FIX: Sanitize all text from the HTML file ---
            safe_text = text.encode('latin-1', 'replace').decode('latin-1')

            if element.name in ['h1', 'h2', 'h3']:
                self.set_font('Arial', 'B', 16 - (int(element.name[1]) * 2))
                self.multi_cell(0, 5, safe_text)
                self.ln(2)
            elif element.name in ['p', 'li']:
                self.set_font('Arial', '', 11)
                prefix = "- " if element.name == 'li' else ""
                self.multi_cell(0, 5, prefix + safe_text)
                self.ln(2)
            elif element.name == 'img' and element.get('src'):
                src = Path(element.get('src')).name
                img_path = resources_dir / src
                if img_path.is_file():
                    try:
                        self.image(str(img_path), w=150)
                        self.ln(2)
                    except Exception:
                        self.set_font('Arial', 'I', 9)
                        self.cell(0, 5, f"[Could not embed image: {src}]")
                        self.ln(2)

def generate_detailed_pdf_report():
    """Generates a comprehensive PDF report from session state data."""
    pdf = PDF()
    pdf.add_page()

    # Section 1: 3D Model Visualization
    pdf.chapter_title("1. 3D Model Visualization")
    if st.session_state.report_data.get("mesh_snapshot_path"):
        props = st.session_state.report_data["mesh_properties"]
        body = (
            f"A 3D model was loaded and analyzed.\n\n"
            f"- Bounding Box (X, Y, Z): {props['bounds']}\n"
            f"- Calculated Volume: {props['volume']:.2f} cubic units\n"
            f"- Number of Points: {props['n_points']}\n"
        )
        pdf.chapter_body(body)
        pdf.image(st.session_state.report_data["mesh_snapshot_path"], w=120)
        pdf.ln(5)
    else:
        pdf.chapter_body("No 3D model was uploaded for this session.")

    # Section 2: AI Component Planner
    pdf.chapter_title("2. AI Component Planner")
    pdf.chapter_body(st.session_state.report_data["ai_planner_output"])

    # Section 3: SOP Viewer
    pdf.chapter_title("3. SOP: Determination of Foam Density and Thickening Time")
    sop_path = st.session_state.report_data.get("sop_html_path")
    if sop_path:
        resources_dir = sop_path.with_name(sop_path.stem + "_files")
        pdf.add_html_content(sop_path, resources_dir)
    else:
        pdf.chapter_body("SOP file was not available or not viewed in this session.")

    # Section 4: SOP Batch Calculation
    pdf.chapter_title("4. SOP Batch Calculation")
    batch_results = st.session_state.report_data.get("sop_batch_results")
    if batch_results:
        body = (
            "The following batch was calculated based on SOP ratios.\n\n"
            f"- Input Polyol: {batch_results['polyol_g']:.1f} g\n"
            f"- Required c-Pentane: {batch_results['c_pentane_g']:.1f} g\n"
            f"- Resulting Polyol Mixture: {batch_results['polyol_mixture_g']:.1f} g\n"
            f"- Required MDI (Isocyanate): {batch_results['mdi_g']:.1f} g\n\n"
            f"**Total Final Mixture Weight: {batch_results['total_mixture_g']:.1f} g**"
        )
        pdf.chapter_body(body)
    else:
        pdf.chapter_body("No SOP batch calculation was performed.")

    # Section 5: Quality Check Results
    pdf.chapter_title("5. SOP Quality Check")
    qc_results = st.session_state.report_data.get("qc_results")
    if qc_results:
        body = (
            "Test 1:\n"
            f"  - Thickening Time: {qc_results['t1_time']} s\n"
            f"  - Cube Mass: {qc_results['t1_mass_g']} g\n"
            f"  - Displaced Water: {qc_results['t1_disp_ml']} mL\n"
            f"  - Calculated Density: {qc_results['t1_density']:.2f} kg/m³\n\n"
            "Test 2:\n"
            f"  - Thickening Time: {qc_results['t2_time']} s\n"
            f"  - Cube Mass: {qc_results['t2_mass_g']} g\n"
            f"  - Displaced Water: {qc_results['t2_disp_ml']} mL\n"
            f"  - Calculated Density: {qc_results['t2_density']:.2f} kg/m³\n\n"
            "**Average Results:**\n"
            f"  - Average Thickening Time: {qc_results['avg_time']:.1f} s ({'PASS' if abs(qc_results['avg_time'] - 45.0) <= 4.0 else 'FAIL'})\n"
            f"  - Average Foam Density: {qc_results['avg_density']:.2f} kg/m³ ({'PASS' if abs(qc_results['avg_density'] - 23.6) <= 1.0 else 'FAIL'})"
        )
        pdf.chapter_body(body)
    else:
        pdf.chapter_body("No Quality Check was performed.")

    # Section 6: AI Explanation
    pdf.chapter_title("6. AI Explanation of Foam Process")
    pdf.chapter_body(st.session_state.report_data["ai_explainer_output"])

    # Section 7: Summary and Observations
    pdf.chapter_title("7. Summary and Observations")
    summary_body = (
        f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "User: (Not tracked in this version)\n\n"
        "This report consolidates all data generated and viewed during the application session. "
        "Review all sections for a complete overview of the foam production plan and quality analysis."
    )
    pdf.chapter_body(summary_body)

    # Convert the FPDF bytearray to an immutable bytes object for Streamlit
    return bytes(pdf.output())


# ------------------------------------------------------------------
# UTILITY AND HELPER FUNCTIONS
# ------------------------------------------------------------------
def compute_requirements(V_mold_ml, expansion_multiple, safety_margin_pct, ratio_A_to_B, sg_A, sg_B):
    s = max(float(safety_margin_pct), 0.0) / 100.0
    E = max(float(expansion_multiple), 1e-6)
    r = max(float(ratio_A_to_B), 1e-6)
    V_liquid = (float(V_mold_ml) / E) * (1.0 + s)
    denom = (r / float(sg_A)) + (1.0 / float(sg_B))
    m_B = V_liquid / denom
    m_A = r * m_B
    return {"V_liquid_ml": V_liquid, "m_A_g": m_A, "m_B_g": m_B}

def calculate_sop_components(polyol_g):
    c_pentane_g = polyol_g * (42.6 / 300.0)
    polyol_mixture_g = polyol_g + c_pentane_g
    mdi_g = polyol_mixture_g * (152.0 / 114.2)
    total_mixture_g = polyol_mixture_g + mdi_g
    return {
        "polyol_g": polyol_g, "c_pentane_g": c_pentane_g, "polyol_mixture_g": polyol_mixture_g,
        "mdi_g": mdi_g, "total_mixture_g": total_mixture_g,
    }

def format_number(x, digits=1):
    return f"{float(x):.{digits}f}"

@st.cache_resource(show_spinner="Loading STL...")
def _load_mesh_from_bytes(data: bytes) -> pv.DataSet:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        mesh = pv.read(tmp_path)
    finally:
        os.remove(tmp_path)
    return mesh

def _is_empty(ds) -> bool:
    try:
        attr = getattr(ds, "is_empty", None)
        if callable(attr): return bool(attr())
        if isinstance(attr, bool): return attr
    except Exception: pass
    return getattr(ds, "n_points", 0) == 0

def render_scene(mesh, frac, axis, foam_color, mold_color, mold_opacity, auto_rotate):
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    t = max(0.0, min(1.0, float(frac)))
    if axis == "Z":
        origin, normal = (0, 0, zmin + t * (zmax - zmin)), (0, 0, 1)
    elif axis == "Y":
        origin, normal = (0, ymin + t * (ymax - ymin), 0), (0, 1, 0)
    else: # X
        origin, normal = (xmin + t * (xmax - xmin), 0, 0), (1, 0, 0)
    foam_part = mesh.clip(normal=normal, origin=origin, invert=True)
    plotter = pv.Plotter(window_size=[900, 600], off_screen=True)
    plotter.background_color = "white"
    plotter.add_mesh(mesh, color=mold_color, opacity=mold_opacity, lighting=True, smooth_shading=True)
    if not _is_empty(foam_part):
        plotter.add_mesh(foam_part, color=foam_color, opacity=0.95, lighting=True, smooth_shading=True)
    plotter.view_isometric()
    if auto_rotate:
        try: plotter.camera.azimuth(10)
        except Exception: pass
    stpyvista(plotter, key="pv_single")

def render_sop_with_inlined_assets(sop_html_path: Path) -> str:
    try:
        html = sop_html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Failed to read SOP HTML file at {sop_html_path}: {e}")
        return ""
    soup = BeautifulSoup(html, "html.parser")
    resources_dir = sop_html_path.with_name(sop_html_path.stem + "_files")
    if not resources_dir.is_dir():
        st.warning(f"Companion resources folder not found. Expected at: {resources_dir}")
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src or src.startswith("data:"): continue
        clean_src = Path(src).name
        img_path = (resources_dir / clean_src).resolve()
        if not img_path.is_file():
            st.warning(f"Could not find image: '{clean_src}' at expected path {img_path}")
            continue
        try:
            data = img_path.read_bytes()
            mime, _ = mimetypes.guess_type(str(img_path))
            mime = mime or "image/png"
            b64_data = base64.b64encode(data).decode("ascii")
            img["src"] = f"data:{mime};base64,{b64_data}"
        except Exception as e:
            st.error(f"Failed to read or encode image {img_path}: {e}")
    return str(soup)

# ------------------------------------------------------------------
# APP LAYOUT
# ------------------------------------------------------------------
st.title("PU Foam + STL Fill Visualizer 🧪")
st.markdown("Key SOP Ratios: **Polyol : c-Pentane ≈ 7.04 : 1**  |  **Polyol+c-Pentane : MDI ≈ 1 : 1.33**")

with st.sidebar:
    st.header("General Foam Calculator")
    st.caption("Based on mold volume and expansion.")
    V_mold_ml = st.number_input("Mold Volume (mL)", min_value=0.0, value=1000.0, step=10.0, key="sidebar_volume")
    expansion_multiple = st.number_input("Expansion Multiple (×)", 0.1, value=4.5, step=0.1)
    safety_margin_pct = st.number_input("Safety Margin (%)", 0.0, value=0.0, step=1.0)
    ratio_A_to_B = st.number_input("Mix Ratio A:B (by weight)", 0.01, value=2.0, step=0.1)
    sg_A = st.number_input("Specific Gravity A (g/mL)", 0.1, value=1.05, step=0.01)
    sg_B = st.number_input("Specific Gravity B (g/mL)", 0.1, value=1.13, step=0.01)

    calc_results = compute_requirements(V_mold_ml, expansion_multiple, safety_margin_pct, ratio_A_to_B, sg_A, sg_B)
    st.write("Liquid Needed (mL):", format_number(calc_results["V_liquid_ml"]))
    st.write("Part A Mass (g):", format_number(calc_results["m_A_g"]))
    st.write("Part B Mass (g):", format_number(calc_results["m_B_g"]))
    
    st.markdown("---")
    st.header("Download Full Report")
    st.caption("Generates a detailed PDF of the entire session.")
    
    if st.button("Generate & Download PDF Report"):
        with st.spinner("Generating Detailed PDF... This may take a moment."):
            pdf_bytes = generate_detailed_pdf_report()
            st.session_state.pdf_report = pdf_bytes # Store in session state

    # Use a separate download button that appears after generation
    if "pdf_report" in st.session_state:
        st.download_button(
            label="Click to Download PDF",
            data=st.session_state.pdf_report,
            file_name="PU_Foam_Full_Report.pdf",
            mime="application/pdf",
        )

tab_vis, tab_planner, tab_sop_viewer, tab_sop_calc, tab_sop_qc, tab_ai = st.tabs([
    "3D STL + Fill", "AI Component Planner", "SOP Viewer", "SOP Batch Calculator", "SOP + QC", "AI Explainer"
])

with tab_vis:
    st.subheader("Upload STL and Animate Fill")
    colL, colR = st.columns([2, 1])
    with colR:
        uploaded_file = st.file_uploader("Upload STL", type=["stl"])
        foam_color = st.color_picker("Foam Color", "#FF9900")
        mold_color = st.color_picker("Mold Color", "#AAAAAA")
        mold_opacity = st.slider("Mold Opacity", 0.1, 1.0, 0.3, 0.05)
        fill_axis = st.selectbox("Fill Axis", ["Z", "Y", "X"], 0)
        anim_fps = st.slider("Animation FPS", 1, 30, 12, 1)
        anim_duration_s = st.slider("Duration (s)", 1, 30, 8, 1)
        anim_auto_rotate = st.checkbox("Auto-Rotate", False)
        is_playing = st.toggle("Play", False)
        manual_fill_frac = st.slider("Fill Fraction", 0.0, 1.0, 0.0, 0.01, disabled=is_playing)
    with colL:
        if uploaded_file:
            mesh = _load_mesh_from_bytes(uploaded_file.getvalue())
            if st.session_state.report_data.get("mesh") is None:
                st.session_state.report_data["mesh"] = mesh
                st.session_state.report_data["mesh_properties"] = {
                    "bounds": mesh.bounds, "volume": mesh.volume, "n_points": mesh.n_points
                }
                plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
                plotter.add_mesh(mesh, color=mold_color)
                plotter.view_xy()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    plotter.screenshot(tmp_file.name)
                    st.session_state.report_data["mesh_snapshot_path"] = tmp_file.name
            
            if "anim_t0" not in st.session_state: st.session_state.anim_t0 = None
            if is_playing and st.session_state.anim_t0 is None: st.session_state.anim_t0 = time.time()
            if not is_playing:
                st.session_state.anim_t0 = None
                current_frac = manual_fill_frac
            else:
                elapsed = time.time() - st.session_state.anim_t0
                current_frac = min(1.0, elapsed / anim_duration_s)
            render_scene(mesh, current_frac, fill_axis, foam_color, mold_color, mold_opacity, anim_auto_rotate)
            if is_playing and current_frac < 1.0:
                time.sleep(1.0 / max(1, anim_fps))
                st.rerun()
        else:
            st.info("Upload an STL file to visualize the foam fill.")

with tab_planner:
    st.header("🤖 AI Component Production Planner")
    st.caption("Enter component dimensions OR a direct volume to generate a production plan.")
    with st.form("planner_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Component Volume Input")
            st.caption("Use dimensions OR enter a precise volume below.")
            length_mm = st.number_input("Length (mm)", min_value=0.0, value=100.0, step=10.0)
            width_mm = st.number_input("Width (mm)", min_value=0.0, value=100.0, step=10.0)
            thickness_mm = st.number_input("Thickness (mm)", min_value=0.0, value=10.0, step=1.0)
            st.markdown("---")
            direct_volume_cm3 = st.number_input("Or Enter Volume Directly (cm³)", min_value=0.0, value=V_mold_ml, step=10.0, key="planner_direct_volume")
        with col2:
            st.subheader("Process Parameters")
            planner_safety_margin = st.number_input("Safety Margin (%)", 0.0, 100.0, 15.0, 1.0)
            target_fill_time_s = st.number_input("Target Fill Time (s)", 1.0, 300.0, 10.0, 1.0)
        submitted = st.form_submit_button("Generate Production Plan")
    if submitted:
        if direct_volume_cm3 > 0 and direct_volume_cm3 != V_mold_ml:
            comp_vol_cm3 = direct_volume_cm3
            volume_source_text = f"from directly entered volume of {comp_vol_cm3:.1f} cm³"
        else:
            comp_vol_cm3 = (length_mm / 10.0) * (width_mm / 10.0) * (thickness_mm / 10.0) if (length_mm * width_mm * thickness_mm > 0) else V_mold_ml
            volume_source_text = f"from dimensions ({length_mm}x{width_mm}x{thickness_mm} mm)"
        total_vol_cm3 = comp_vol_cm3 * (1 + planner_safety_margin / 100.0)
        target_density_g_cm3 = 0.0236
        total_foam_mass_g = total_vol_cm3 * target_density_g_cm3
        ratio_mdi_to_polyol_mix = 152.0 / 114.2
        mass_polyol_mix = total_foam_mass_g / (1 + ratio_mdi_to_polyol_mix)
        mass_mdi = total_foam_mass_g - mass_polyol_mix
        mass_polyol = mass_polyol_mix * (300.0 / (300.0 + 42.6))
        mass_cpentane = mass_polyol_mix - mass_polyol
        sg_polyol=1.05; sg_cpentane=0.75; sg_mdi=1.13
        total_liquid_vol_ml = (mass_polyol/sg_polyol) + (mass_cpentane/sg_cpentane) + (mass_mdi/sg_mdi)
        injection_rate_ml_s = total_liquid_vol_ml / target_fill_time_s
        st.subheader("Generated Production Plan")
        with st.spinner("AI is generating your step-by-step plan..."):
            api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
            if not api_key:
                st.error("Groq API Key not found.")
            else:
                try:
                    from groq import Groq
                    client = Groq(api_key=api_key)
                    prompt = f"""
                    As a Foam Process Engineer, create a step-by-step production plan.
                    **Input Data:**
                    - Component Volume Source: {volume_source_text}
                    - Total Foam Volume Needed: {total_vol_cm3:.1f} cm³ (including {planner_safety_margin}% safety margin)
                    - Target Density: 23.6 ± 1.0 kg/m³
                    - Required Injection Rate: {injection_rate_ml_s:.2f} mL/second (for a {target_fill_time_s}s fill)
                    **Required Material Weights (from SOP Ratios):**
                    - **Polyol:** {mass_polyol:.1f} g
                    - **c-Pentane:** {mass_cpentane:.1f} g
                    - **MDI (Isocyanate):** {mass_mdi:.1f} g
                    **Your Task:**
                    1. Create a summary table. 2. Write a clear list for material prep. 3. Write a guide for mixing, injection, and curing. 4. Add a final reminder for QC checks.
                    """
                    resp = client.chat.completions.create(model="qwen/qwen3-32b", messages=[{"role": "user", "content": prompt}])
                    ai_output = resp.choices[0].message.content
                    st.session_state.report_data["ai_planner_output"] = ai_output
                    st.markdown(ai_output)
                except Exception as e:
                    st.error(f"An error occurred with the Groq API: {e}")

with tab_sop_viewer:
    st.header("📘 Standard Operating Procedure Viewer")
    sop_file_path = Path("Determination of foam density and thickening time.htm")
    st.session_state.report_data["sop_html_path"] = sop_file_path
    if sop_file_path.is_file():
        inlined_html = render_sop_with_inlined_assets(sop_file_path)
        st.components.v1.html(inlined_html, height=1200, scrolling=True)
    else:
        st.info("SOP file not found. Place it and its `_files` folder in the app directory.")

with tab_sop_calc:
    st.header("SOP-Based Component Calculator")
    st.caption("Calculates component weights based on the specific ratios from your SOP.")
    with st.expander("About These Ratios"):
        st.markdown("""
        These calculations are based on the weight ratios derived directly from your Standard Operating Procedure:
        - **Step 1 (Polyol Mixture):** 300 g of Polyol is mixed with 42.6 g of c-Pentane, a ratio of **7.04 : 1**.
        - **Step 2 (Final Foam):** 114.2 g of the Polyol/c-Pentane mixture is combined with 152 g of MDI, a ratio of **1 : 1.33**.
        This calculator scales a batch based on your desired amount of Polyol.
        """)
    polyol_input_g = st.number_input("Enter Total Polyol Weight (g)", min_value=0.0, value=300.0, step=10.0, key="sop_polyol_input")
    if polyol_input_g > 0:
        sop_results = calculate_sop_components(polyol_input_g)
        st.session_state.report_data["sop_batch_results"] = sop_results
        st.subheader("Required Component Weights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Polyol (g)", f"{sop_results['polyol_g']:.1f}")
        col1.metric("c-Pentane (g)", f"{sop_results['c_pentane_g']:.1f}")
        col2.metric("Polyol Mixture (g)", f"{sop_results['polyol_mixture_g']:.1f}")
        col2.metric("MDI (Isocyanate) (g)", f"{sop_results['mdi_g']:.1f}")
        col3.metric("Total Final Mixture (g)", f"{sop_results['total_mixture_g']:.1f}")

with tab_sop_qc:
    st.subheader("SOP Quality Control Checks")
    with st.expander("How This Calculator Works and What Each Value Means", expanded=True):
        st.markdown("""
        This tab helps you follow the quality control steps from your SOP. Here’s a breakdown of each measurement and why it's important:
        #### 1. The Two-Test Requirement
        - **What:** Your SOP requires performing each mixing test twice.
        - **Why:** Averaging two results minimizes random errors and provides a more statistically reliable measure of the batch's true quality.
        #### 2. Thickening Time (s)
        - **What:** The time it takes for the liquid to start gelling (when fibers appear). Also known as "gel time."
        - **Why it's critical:** It defines your **"working time"** for pouring.
        #### 3. Foam Density (kg/m³)
        - **What:** The mass per unit of volume, calculated here by the water displacement method.
        - **Why it's critical:** Density controls the foam's strength, insulation value, and material cost.
        """)
    st.info("The mixing test must be performed twice and the average value calculated from the two results.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Test 1")
        st.markdown("###### Thickening Time")
        t1_time = st.number_input("Time (s)", key="t1_time", value=45.0, step=0.5)
        st.markdown("###### Density Measurement")
        t1_mass_g = st.number_input("1. Cube Mass (g)", key="t1_mass", value=12.0, step=0.1)
        t1_disp_ml = st.number_input("2. Displaced Water (g)", key="t1_disp", value=500.0, step=0.1, help="Weight of water displaced by the submerged cube (g = mL).")
        t1_density = (t1_mass_g / max(1e-9, t1_disp_ml)) * 1000.0
        st.write(f"**Test 1 Calculated Density:** `{t1_density:.2f} kg/m³`")
    with col2:
        st.markdown("#### Test 2")
        st.markdown("###### Thickening Time")
        t2_time = st.number_input("Time (s)", key="t2_time", value=45.0, step=0.5)
        st.markdown("###### Density Measurement")
        t2_mass_g = st.number_input("1. Cube Mass (g)", key="t2_mass", value=12.0, step=0.1)
        t2_disp_ml = st.number_input("2. Displaced Water (g)", key="t2_disp", value=500.0, step=0.1, help="Weight of water displaced by the submerged cube (g = mL).")
        t2_density = (t2_mass_g / max(1e-9, t2_disp_ml)) * 1000.0
        st.write(f"**Test 2 Calculated Density:** `{t2_density:.2f} kg/m³`")
    st.markdown("---")
    st.subheader("Average Results and Evaluation")
    avg_time = (t1_time + t2_time) / 2
    avg_density = (t1_density + t2_density) / 2
    st.session_state.report_data["qc_results"] = {
        "t1_time": t1_time, "t1_mass_g": t1_mass_g, "t1_disp_ml": t1_disp_ml, "t1_density": t1_density,
        "t2_time": t2_time, "t2_mass_g": t2_mass_g, "t2_disp_ml": t2_disp_ml, "t2_density": t2_density,
        "avg_time": avg_time, "avg_density": avg_density
    }
    res1, res2 = st.columns(2)
    with res1:
        st.metric("Average Thickening Time (s)", f"{avg_time:.1f}")
        if abs(avg_time - 45.0) <= 4.0: st.success("✅ Within tolerance (45 ± 4 sec)")
        else: st.warning("❌ Outside tolerance (45 ± 4 sec)")
    with res2:
        st.metric("Average Foam Density (kg/m³)", f"{avg_density:.2f}")
        if abs(avg_density - 23.6) <= 1.0: st.success("✅ Within tolerance (23.6 ± 1.0 kg/m³)")
        else: st.warning("❌ Outside tolerance (23.6 ± 1.0 kg/m³)")
    st.markdown("---")
    st.subheader("4. Final Evaluation Steps")
    st.markdown("""
    1.  **Record the results** in the protocol.
    2.  **Print the results** and handle them to the test requestor.
    3.  In case of any deviations, **inform the person responsible**.
    """)

with tab_ai:
    st.subheader("AI-Powered Explainer (Groq)")
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.info("To enable the AI explainer, set your `GROQ_API_KEY` in a `.env` file or Streamlit secrets.")
    else:
        model_selection = st.text_input("Groq Model Name", "qwen/qwen3-32b", help="Supported models: mixtral-8x7b-32768, etc.")
        if "last_ai_inputs" not in st.session_state: st.session_state.last_ai_inputs = {}
        if "last_explanation" not in st.session_state: st.session_state.last_explanation = "Change inputs in the sidebar to automatically generate an explanation."
        current_inputs = {
            "V_mold_ml": V_mold_ml, "expansion_multiple": expansion_multiple, "safety_margin_pct": safety_margin_pct,
            "ratio_A_to_B": ratio_A_to_B, "sg_A": sg_A, "sg_B": sg_B,
        }
        if current_inputs != st.session_state.last_ai_inputs:
            with st.spinner("Inputs changed, generating new explanation from Groq..."):
                try:
                    from groq import Groq
                    client = Groq(api_key=api_key)
                    prompt = f"""
                    Act as a materials engineer. Create a detailed explanation for a polyurethane foam pour using the following data.
                    **Input Data:**
                    - Mold Volume: {V_mold_ml} mL
                    - Foam Expansion Ratio: {expansion_multiple}x
                    - Safety Margin: {safety_margin_pct}%
                    - Mix Ratio (Part A : Part B by weight): {ratio_A_to_B}:1
                    - Specific Gravity of Part A: {sg_A}
                    - Specific Gravity of Part B: {sg_B}
                    **Required Output Structure:**
                    1. Core Concepts & Formulas. 2. Step-by-Step Worked Example. 3. Practical Tips & Visual Cues.
                    """
                    resp = client.chat.completions.create(model=model_selection, messages=[{"role": "user", "content": prompt}])
                    explanation = resp.choices[0].message.content
                    st.session_state.last_explanation = explanation
                    st.session_state.report_data["ai_explainer_output"] = explanation # Save for report
                    st.session_state.last_ai_inputs = current_inputs
                except Exception as e:
                    st.session_state.last_explanation = f"An error occurred with the Groq API: {e}"
        with st.container():
            st.markdown(st.session_state.last_explanation)
        if st.button("Force Refresh Explanation"):
            st.session_state.last_ai_inputs = {}
            st.rerun()
