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
<img width="968" alt="image" src="https://github.com/user-attachments/assets/a60eb4dc-2b75-4dc8-8cd9-e4101ab6dfd4" />

Have option to predict the molecular weight of the protein/bp of DNA Gel once the ladder markers (left/right) are placed (Accurarcy can vary anywhere from 5 to 10% depending on the quality of the gel) and perform Densitometric analysis:
<img width="971" alt="image" src="https://github.com/user-attachments/assets/6da4bb1c-54d9-4666-be36-80b0b6416d53" />
<img width="967" alt="image" src="https://github.com/user-attachments/assets/e7e65dc4-f129-44a6-ade8-8451d8a91bc4" />
<img width="997" alt="image" src="https://github.com/user-attachments/assets/afc3522b-201f-4695-b8fc-c4b9dc4aafe6" />





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
