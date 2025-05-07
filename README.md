# Imaging-Assistant
This software simplifies the process of editing SDS-PAGE, Western Blot (WB), and other gel images for scientific research and publications, making it effortless to add important details like labels, headings, molecular weight markers, and more. With an intuitive interface, you can customize your images precisely the way you want. Save your work with all annotations intact, so you can revisit and refine your images anytime. Plus, take advantage of powerful densitometric analysis tools built right in-giving you accurate, reliable quantification with just a few clicks. Whether you're preparing figures for publication or analyzing experimental results, this tool streamlines your workflow and helps you present your data clearly and professionally.

Features:
* Works with most image format (TIFF,PNG,BMP,JPG) from any device or even camera
* Fully integrated Image Cropping, Alignment and Skew Correction toolkit builtin
* Adjust Low/High Image Contrast and Gamma (For visualizing proteins with faint bands)
* Easily place left/right or top/bottom markers fast with auto detection and marker placement.
* Save your images with transparent labels for those beautiful posters that you will make with this tool!
* Save your Gel Marker Preset as Custom: Includes left/right/top/bottom marker values
* Powerful Undo/Redo process operation
* Option to add custom markers with custom font type, size and color on the image with default top/bottom/left/right arrows builtin and can add more using Webdings font
* Inbuilt Snapping tool to help set markers at the center
* Option to load image from clipboard and save to clipboard for faster operation
* Powerful Keyboard Shortcuts and Tooltips for faster and easier accesibility
* Can perform detailed densitometric analysis including band analysis and band quantification with custom quadrilateral or rectangular selection window
* Different modes and types of background subtraction and peak detection are included
* Powerful Exception Logger builtin to diagnose program crashes or other issues.
  
Example output from the software:

<img width="902" alt="image" src="https://github.com/user-attachments/assets/639edac5-089e-4219-a3f5-6d7a06ad1932" />
<img width="1112" alt="image" src="https://github.com/user-attachments/assets/392e9592-639f-4517-b35a-41f947034d6e" />

<img width="900" alt="image" src="https://github.com/user-attachments/assets/e5add862-6615-4db6-b62b-aaf5859c5917" />
Powerful Densitometric Analyis for predicting protein quantity and purity
<img width="904" alt="image" src="https://github.com/user-attachments/assets/5157eead-bf64-4302-ba37-c979716825f6" />
Fully transparent background with alpha-channel for different applications (Making posters)
<img width="541" alt="image" src="https://github.com/user-attachments/assets/bb55025e-43ba-496f-a22c-477bf3a59973" />




To run using Anaconda without compatiblity issues:

1.Create a new Conda environment:
  Open a terminal or Anaconda Prompt and navigate to the directory where the environment.yml file is located. Then run the following command:
  
  >conda env create -f environment.yml

  This will create a new Conda environment named imaging_assistant_env with all the required libraries.

2.Activate the environment:
  After the environment is created, activate it using the following command:  
  
  >conda activate imaging_assistant_env

3.Run the program:
  Once the environment is activated, you can run your Python program using:
  >cd [Directory, e.g: C:\Users\X\Downloads\Imaging-Assistant-main\Imaging-Assistant-main\]
  >python Imaging_assistant_V6.py

To run the program in Python or Other Distribution(can have compatibility issues with python >3.9.13):

  1. Run the command:
     
     pip install -r C:\Imaging-Assistant-main_3\Imaging-Assistant-main\requirements.txt

     python C:\Imaging-Assistant-main_3\Imaging-Assistant-main\Imaging_assistant_V3.py

To compile the code into an application:
  1. You need to install pyinstaller: pip install pyinstaller
  2. Extract the files into your folder of choice and modify this command(FOR SOME REASON IN SOME ANACONDA VERSION OPENPYXL IS NOT PACKAGED INTO THE FINAL SOFTWARE):

    pyinstaller --noconfirm --onefile --windowed --hidden-import openpyxl.cell._writer --icon=icon.ico --name=Imaging_assistant_v5 Imaging_assistant_V5.py

  3. Modify the directory. Here the directory is the same as Anaconda_prompt (Change as needed)
    
  4. Run the command to get an executable so you can run the software on any PC without needing to install Python or Ananconda Distribution on other computer. (OR USE THE RELEASED VERSION)

If you get issues during compiling exe, execute these commands:
1. pip uninstall pyinstaller
2. pip install --upgrade pyinstaller
3. pyinstaller --noconfirm --onefile --windowed --hidden-import openpyxl.cell._writer --icon=icon.ico --name=Imaging_assistant_V6 Imaging_assistant_V6.py


Would really appreciate if you cite this tool so that other people (Students) can use it without having to spend 4 months on developing something like this:

APA Style (7th Edition):
Karmaker, A. (2024). _Imaging Assistant V6_ [Computer software]. https://github.com/Anindya-Karmaker/Imaging-Assistant

MLA Style (9th Edition):
Karmaker, Anindya. _Imaging Assistant V6_, 2024. GitHub, https://github.com/Anindya-Karmaker/Imaging-Assistant.

Chicago Style (Notes and Bibliography - Bibliography Entry):
Karmaker, Anindya. _Imaging Assistant V6_. 2024. https://github.com/Anindya-Karmaker/Imaging-Assistant.

Chicago Style (Author-Date - Reference List Entry):
Karmaker, Anindya. 2024. _Imaging Assistant V6_. https://github.com/Anindya-Karmaker/Imaging-Assistant.

BIB Format:
@misc{Karmaker_ImagingAssistantV6_2024_misc,
  author = {Karmaker, Anindya},
  title = {{Imaging Assistant V6}},
  year = {2024}, % Adjust year if a specific release date for V6 is known
  howpublished = {\url{https://github.com/Anindya-Karmaker/Imaging-Assistant}},
  note = {Version 6.0. Software} 
}
