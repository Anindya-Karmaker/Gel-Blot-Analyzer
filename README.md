# Imaging-Assistant
This software makes it easy to label and analyze SDS-PAGE, WB and other GEL images (DNA, RNA etc.) easily with powerful densitometric analysis capabilities. It can save the image with all of the options for easy editing and has copy/paste functionality. NOTHING LIKE THIS EXISTS FOR FREE OTHERWISE I WON'T BE SPENDING MY PRECIOUS PHD WEEKENDS ON WORKING ON IT! 
Disclaimer: I am not a professional coder and used ChatGPT, Claude, Gemini and DeepSeek to write most of the supporting code based on my algorithms and ideas. Please download the required libraries using requirements.txt. Used python version 3.9.13 on a M1 MacBook Pro to run and test the code. 
Or directly download the EXE file from releases without needing to install Python or other libraries. 
If someone wants to help me to get me a signing certificate for the application to make it more trustworthy, I would really appreciate it. 

Features:
* Works with most image format (TIFF,PNG,BMP,JPG) from any device or even camera
* Fully integrated Image Cropping, Alignment and Skew Correction toolkit builtin
* Adjust Low/High Image Contrast and Gamma (For visualizing proteins with faint bands)
* Easily place left/right or top/bottom markers fast
* Save your images with transparent labels for those beautiful posters that you will make with this tool!
* Save your Gel Marker Preset as Custom: Includes left/right/top/bottom marker values
* Powerful Undo/Redo process operation
* Option to add custom markers with custom font type, size and color on the image with default top/bottom/left/right arrows builtin and can add more using Webdings font
* Inbuilt Snapping tool to help set markers at the center
* Option to load image from clipboard and save to clipboard for faster operation
* Added SVG export option so the image generated can be edited in Microsoft Word or Other Applications (Work In Progress!)
* Powerful Keyboard Shortcuts and Tooltips for faster and easier accesibility
* Can perform detailed densitometric analysis including band analysis and band quantification with custom quadrilateral or rectangular selection window
* Different modes and types of background subtraction and peak detection are included
* Powerful Exception Logger builtin to diagnose program crashes or other issues.
  
Example output from the software:
<img width="868" alt="image" src="https://github.com/user-attachments/assets/32008260-d4e8-4728-89ad-9d5fc5323d25" />
<img width="866" alt="image" src="https://github.com/user-attachments/assets/1b06ae38-83a7-4aee-884a-1ca59bef396d" />
<img width="1101" alt="image" src="https://github.com/user-attachments/assets/549b2422-0697-447e-b724-5c84ffd606db" />
<img width="867" alt="image" src="https://github.com/user-attachments/assets/f4faa88e-aebc-45d2-a082-645f0ae35182" />

Known Bugs:
* SVG OUPUT IS STILL BUGGY!

To run using Anaconda without compatiblity issues:

1.Create a new Conda environment:
  >Open a terminal or Anaconda Prompt and navigate to the directory where the environment.yml file is located. Then run the following command:
  >conda env create -f environment.yml

  This will create a new Conda environment named imaging_assistant_env with all the required libraries.

2.Activate the environment:
  After the environment is created, activate it using the following command:  
  >conda activate imaging_assistant_env

3.Run the program:
  Once the environment is activated, you can run your Python program using:
  >python Imaging_assistant_V5.py

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
     
Please see the Instruction Video for more details on how to use the software.

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
