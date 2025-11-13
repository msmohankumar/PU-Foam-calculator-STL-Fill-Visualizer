# PU Foam Calculator & STL Fill Visualizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pu-foam-calculator-stl-fill-visualizer-by4q8wr5ynlejdyn56wq7g.streamlit.app/)

A comprehensive web application designed for polyurethane (PU) foam process engineers and technicians. This tool combines 3D visualization, AI-powered planning, and SOP-based calculators to streamline the entire foam production workflow—from initial estimation to quality control.

The app is built with Streamlit and PyVista and leverages the Groq API for real-time AI assistance.

## ✨ Key Features

-   **3D STL Fill Visualizer:** Upload any STL file to see a 3D model of a mold and animate the foam filling process. You can customize colors, opacity, and the fill axis (X, Y, or Z) for a clear visual analysis.
-   **AI Production Planner:** Uses a large language model (via Groq) to automatically generate a detailed, step-by-step production plan. Input component dimensions and process parameters to get a complete guide for material preparation, mixing, and injection.
-   **Integrated SOP Viewer:** Displays a Standard Operating Procedure (SOP) directly within the app. It intelligently parses the source HTML and embeds all associated images, ensuring the full document is always accessible and self-contained.
-   **Multiple Foam Calculators:**
    -   **General Purpose Calculator:** For quick, on-the-fly estimations based on mold volume and foam expansion ratios.
    -   **SOP-Based Calculator:** For precise batch calculations using the specific mix ratios derived from your company's official SOP.
    -   **Quality Control (QC) Calculator:** A dedicated tab to help technicians perform and validate QC tests, calculating average thickening time and foam density against predefined SOP tolerances.
-   **AI-Powered Explainer:** An interactive tab that uses AI to dynamically explain the foam calculations. It provides formulas, step-by-step worked examples, and practical tips based on your inputs in the sidebar.

## 🚀 Deployment

This application is deployed on Streamlit Community Cloud. Due to its use of `PyVista` for 3D graphics, it requires a specific configuration to run in a headless Linux environment.

### 1. `requirements.txt`

The Python dependencies are listed in this file. Note that versions for `pyvista` and `Pillow` are kept flexible to ensure compatibility with the latest Python versions on Streamlit Cloud.

streamlit
pyvista
stpyvista
python-dotenv
beautifulsoup4
groq
Pillow

### 2. `packages.txt`

`PyVista` requires system-level OpenGL libraries and a virtual display to function. This file instructs Streamlit to install these using `apt-get` before installing the Python packages.


libgl1-mesa-glx
xvfb

### 3. Environment Variable for Off-Screen Rendering

To prevent `PyVista` from crashing in a headless environment, the following environment variable **must** be set at the very top of `app.py`, before any other imports.


import os
os.environ['PYVISTA_OFF_SCREEN'] = 'True'

### 4. Streamlit Secrets

The AI features require a Groq API key. To deploy the app, add your key to your Streamlit Community Cloud secrets:
1.  Go to your app's settings > "Secrets".
2.  Add a new secret with the following format:
    ```
    GROQ_API_KEY = "your_api_key_here"
    ```

## 💻 How to Run Locally

1.  **Clone the repository:**
    ```
    git clone https://github.com/msmohankumar/PU-Foam-calculator-STL-Fill-Visualizer.git
    cd PU-Foam-calculator-STL-Fill-Visualizer
    ```

2.  **Install Python packages:**
    ```
    pip install -r requirements.txt
    ```

3.  **(Optional) Install system dependencies:**
    On a Debian/Ubuntu system, you may need to install the same packages required for deployment:
    ```
    sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx xvfb
    ```

4.  **Create a `.env` file:**
    Create a file named `.env` in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY="your_api_key_here"
    ```

5.  **Run the Streamlit app:**
    ```
    streamlit run app.py
    ```

## 📁 File Structure

The application requires a specific file structure for the "SOP Viewer" to function correctly. When you save a web page as "HTML, Complete", it creates a `.htm` file and a companion `_files` folder. Both must be present in the repository.


/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── packages.txt # System-level dependencies
├── "Determination of foam density and thickening time.htm" # The SOP HTML file
│
└── /"Determination of foam density and thickening time_files"/ # FOLDER with all images for the SOP
├── image002.gif
├── image004.jpg
└── ... (and all other asset files)

**Note:** If the `_files` folder is missing from the repository, the images in the SOP Viewer will not load.
