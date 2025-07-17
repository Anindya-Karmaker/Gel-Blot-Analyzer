# Gel Blot Analyzer (Previously Imaging-Assistant)

<p align="center">
  <img src="https://github.com/user-attachments/assets/639edac5-089e-4219-a3f5-6d7a06ad1932" width="800" alt="Gel Blot Analyzer main interface showing an annotated gel image">
</p>

**Gel Blot Analyzer** is a comprehensive desktop application designed to streamline the annotation, analysis, and presentation of gel electrophoresis images. Built for researchers, students, and scientists, this tool simplifies the often tedious process of preparing gel images (SDS-PAGE, Western Blots, DNA gels, etc.) for lab notebooks, presentations, and publications.

From precise labeling and molecular weight prediction to powerful densitometric analysis, Gel Blot Analyzer combines essential imaging tools into a single, intuitive workflow.

---

## Key Features

| Feature                  | Description                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Broad Format Support** | Works with most common image formats (TIFF, PNG, BMP, JPG) from any imaging device.                                                       |
| **Image Transformation** | Integrated toolkit for cropping, rotation, alignment, and perspective (skew) correction.                                                  |
| **Advanced Adjustments** | Fine-tune image levels (black/white points) and gamma to visualize faint bands without altering raw data for analysis.                      |
| **Effortless Annotation**| Place standard (L/R/Top) or fully custom markers. Draw lines, arrows, and shapes with custom fonts, colors, and sizes.                     |
| **Densitometry Suite**   | Perform quantitative analysis by defining lanes with rectangular or quadrilateral selections to handle skewed gels.                       |
| **Peak Analysis**        | Features multiple background subtraction methods (Rolling Ball, Valley-to-Valley) and tunable peak detection for accurate band quantification. |
| **MW Prediction**        | Predict the molecular weight of unknown bands using a standard curve generated from your markers with multiple regression models.         |
| **Presets & Workflow**   | Save your favorite molecular weight marker layouts as presets. Undo/Redo support for a non-destructive workflow.                          |
| **Professional Output**  | Export your final annotated image with a transparent background, perfect for posters and publications.                                    |
| **Robust & Stable**      | Includes a built-in exception logger to help diagnose any issues and ensure a stable user experience.                                       |


## Gallery

<p align="center">
  <em>Automatic lane and band detection for densitometry.</em><br>
  <img src="https://github.com/user-attachments/assets/392e9592-639f-4517-b35a-41f947034d6e" width="700" alt="Auto Lane Detection">
</p>
<p align="center">
  <em>Predicting the molecular weight of an unknown band using a standard curve.</em><br>
  <img src="https://github.com/user-attachments/assets/e5add862-6615-4db6-b62b-aaf5859c5917" width="700" alt="Molecular Weight Prediction">
</p>
<p align="center">
  <em>Densitometric analysis plot for quantifying protein bands.</em><br>
  <img src="https://github.com/user-attachments/assets/5157eead-bf64-4302-ba37-c979716825f6" width="700" alt="Densitometry Plot">
</p>

---

## Installation and Usage

### Option 1: Download the Pre-built Application (Recommended)

The easiest way to get started is to download the latest pre-built application for your operating system.

1.  Go to the **[Releases](https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer/releases)** page of this repository.
2.  Download the appropriate file:
    *   **For macOS:** Download the `.dmg` file or the `.app.zip` file.
    *   **For Windows:** Download the `.zip` file for the application.
3.  Unzip the file and run the application. No installation is needed.

### Option 2: Run from Source (For Developers)

If you are a developer and want to run the script directly, you can use Conda for a stable environment.

#### Using Anaconda (Recommended for Stability)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer.git
    cd Gel-Blot-Analyzer
    ```

2.  **Create a New Conda Environment:**
    This command uses the provided file to create a new environment named `gel_analyzer_env` with all the correct library versions.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the Environment:**
    ```bash
    conda activate gel_analyzer_env
    ```

4.  **Run the Program:**
    ```bash
    python Gel_blot_analyzer_v1.0.py
    ```

#### Using `pip`

You can also use `pip` with the provided `requirements.txt` file, although dependency conflicts can be more common.

1.  **Clone the Repository and navigate into it.**

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the program:**
    ```bash
    python Gel_blot_analyzer_v1.0.py
    ```

---

## Building the Application from Source

If you wish to compile the application yourself, you will need `pyinstaller`.

1.  **Activate your Python environment** (either Conda or venv) where you have installed the dependencies.

2.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

3.  **Run the Build Command:**
    This project includes pre-configured `.spec` files for both macOS and Windows. Using these is the most reliable way to build.

    *   **On macOS:**
        ```bash
        pyinstaller build_macos.spec
        ```

    *   **On Windows:**
        ```bash
        pyinstaller build_windows.spec
        ```

4.  The final, self-contained application will be located in the `dist` folder.

---

## Citation

If you use Gel Blot Analyzer in your research, presentations, or publications, please cite it. Your citation helps other researchers and students discover this tool.

#### APA Style (7th Edition)
Karmaker, A. (2024). _Gel Blot Analyzer_ (Version 1.0) [Computer software]. https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer

#### MLA Style (9th Edition)
Karmaker, Anindya. _Gel Blot Analyzer_, Version 1.0, 2024. GitHub, https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer.

#### BibTeX Format
```bibtex
@misc{Karmaker_GelBlotAnalyzer_2024,
  author       = {Karmaker, Anindya},
  title        = {{Gel Blot Analyzer}},
  year         = {2024},
  howpublished = {\url{https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer}},
  note         = {Version 1.0}
}
