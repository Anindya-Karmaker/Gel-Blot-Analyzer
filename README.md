<div align="center">

<img src="https://github.com/user-attachments/assets/b7f1dbdb-4325-4208-b749-cae42556b3f3" width="800" alt="Gel Blot Analyzer main interface showing an annotated gel image">

<h1>Gel Blot Analyzer</h1>

<p>A comprehensive desktop application for annotation, analysis, and presentation of gel electrophoresis images.</p>

<!-- Badges -->
![GitHub Downloads](https://img.shields.io/github/downloads/Anindya-Karmaker/Gel-Blot-Analyzer/total)
![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAnindya-Karmaker%2FGel-Blot-Analyzer&count_bg=%2379C83D&title_bg=%23555555&title=Visits&edge_flat=true)
![License](https://img.shields.io/github/license/Anindya-Karmaker/Gel-Blot-Analyzer?style=flat-square&color=orange)

<br>

<!-- Download Button -->
[<img src="https://img.shields.io/badge/⬇%20Download%20Latest%20Release-2ea44f?style=for-the-badge&logo=github&logoColor=white" alt="Download Latest Release">](https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer/releases/latest)

</div>

---

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
| **Glycan/Oligomer Analysis**| Visualize and predict different protein bands based on glycosylation sites and dimer, trimer, tetramer or other oligomer configuration.         |
| **Presets & Workflow**   | Save your favorite molecular weight marker layouts as presets. Undo/Redo support for a non-destructive workflow.                          |
| **Professional Output**  | Export your final annotated image with a transparent background, perfect for posters and publications.                                    |
| **Robust & Stable**      | Includes a built-in exception logger to help diagnose any issues and ensure a stable user experience.                                       |

---

## Gallery

<p align="center">
  <em>Automatic lane and band detection for densitometry.</em><br>
  <img src="https://github.com/user-attachments/assets/1f579bce-262b-4790-b39c-db85adfdb633" width="700" alt="Auto Lane Detection">
</p>
<p align="center">
  <em>Predicting the molecular weight of an unknown band using a standard curve.</em><br>
  <img src="https://github.com/user-attachments/assets/21752f41-c3ef-436d-89ea-2470e03d44cd" width="700" alt="Molecular Weight Prediction">
</p>
<p align="center">
  <em>Densitometric analysis plot for quantifying protein bands.</em><br>
  <img src="https://github.com/user-attachments/assets/dc3a4d5c-3ec3-4373-9167-cc2223a5f167" width="700" alt="Densitometry Plot">
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

> **Note for macOS:** Since the application is not signed, it may not run directly. Copy the `.app` file to the Applications Folder, open Terminal, and run:
> ```bash
> xattr -cr "/Applications/Gel Blot Analyzer.app"
> ```
>
> **Note for Windows:** Despite being signed, Windows may show a warning since it is not from a widely-known publisher. Click **More Info** → **Run Anyway**.

### Option 2: Run from Source (For Developers)

If you are a developer and want to run the script directly, you can use Conda for a stable environment.

#### Using Anaconda (Recommended for Stability)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer.git
    cd Gel-Blot-Analyzer
    ```

2.  **Create a New Conda Environment:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the Environment:**
    ```bash
    conda activate gel_analyzer_env
    ```

4.  **Run the Program:**
    ```bash
    python Gel_blot_analyzer.py
    ```

    > If you encounter errors due to missing libraries or C-code header mismatches (common on Windows), run:
    > ```bash
    > pip install --force-reinstall --no-cache-dir numpy scikit-image
    > ```

#### Using `pip`

1.  **Clone the Repository and navigate into it.**

2.  **Create and activate a virtual environment:**
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
    python Gel_blot_analyzer.py
    ```

---

## Building the Application from Source

If you wish to compile the application yourself, you will need `pyinstaller`.

1.  **Activate your Python environment** (either Conda or venv).

2.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

3.  **Run the Build Command** using the pre-configured `.spec` files:

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
Karmaker, A. (2026). _Gel Blot Analyzer_ (Version 7.3) [Computer software]. https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer

#### MLA Style (9th Edition)
Karmaker, Anindya. _Gel Blot Analyzer_, Version 7.3, 2026. GitHub, https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer.

#### BibTeX Format
```bibtex
@misc{Karmaker_GelBlotAnalyzer_2026,
  author       = {Karmaker, Anindya},
  title        = {{Gel Blot Analyzer}},
  year         = {2026},
  howpublished = {\url{https://github.com/Anindya-Karmaker/Gel-Blot-Analyzer}},
  note         = {Version 7.3}
}
```
