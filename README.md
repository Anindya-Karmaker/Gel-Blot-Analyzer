# Imaging-Assistant
This software makes it easy to modify SDS-PAGE, WB and other GEL images easily and add notations like label heading, molecular weight marker and so on. It can save the image with all of the options for easy editing and has copy/paste functionality. I am not a professional coder and used ChatGPT, Claude, Gemini and DeepSeek to write most of the supporting code based on my algorithms and ideas. Please download the required libraries using requirements.txt. Used python version 3.9.13 on a M1 MacBook Pro to run and test the code.

Features:
* Fully integrated Image Cropping, Alignment and Skew Correction toolkit builtin
* Adjust Low/High Image Contrast and Gamma (For visualizing proteins with faint bands)
* Option to add custom markers with custom font type, size and color
* Inbuilt Snapping tool to help set markers 
* Option to load image from clipboard and save to clipboard for faster operation
* Added SVG export option so the image generated can be edited in Microsoft Word or Other Applications (BUGGY!)
* Added Keyboard Shortcuts and Tooltips for major options
* Added Undo/Redo buttons to revert back changes
* Can perform detailed densitometric analysis including band analysis and band quantification
  
Example output from the software:
<img width="969" alt="image" src="https://github.com/user-attachments/assets/c5b6b417-a93f-4d1c-b9fe-e16d422f901a" />
<img width="971" alt="image" src="https://github.com/user-attachments/assets/3943b1ba-d8a9-4e35-9dc8-55cb5cb6a407" />
<img width="969" alt="image" src="https://github.com/user-attachments/assets/0a9489d0-f7f0-4e1c-911f-2ff6fd179067" />
<img width="970" alt="image" src="https://github.com/user-attachments/assets/8f72be1a-e2f3-4557-a204-616d860d4430" />
<img width="970" alt="image" src="https://github.com/user-attachments/assets/69e1f709-d547-4b82-9831-617b4839f9d6" />
<img width="970" alt="image" src="https://github.com/user-attachments/assets/80e22811-fac4-450b-8a79-f57f30b2614b" />

Known Bugs:
* Make sure before adding left/right markers that they have values and not []
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
