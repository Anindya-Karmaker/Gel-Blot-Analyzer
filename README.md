# Imaging-Assistant
This software makes it easy to modify SDS-PAGE, WB and other GEL images easily and add notations like label heading, molecular weight marker and so on. It can save the image with all of the options for easy editing and has copy/paste functionality for PRO users. I am not a professional coder and used ChatGPT to write most of the supporting code based on my algorithms and ideas. Please download the required libraries using requirements.txt. Used python version 3.9.13 on a M1 MacBook Pro to run and test the code.

Example output from the software:
<img width="969" alt="image" src="https://github.com/user-attachments/assets/2898ee86-79ef-4c3e-ba7e-af93de056e37" />


Have option to predict the molecular weight of the protein/bp of DNA Gel once the ladder markers (left/right) are placed (Accurarcy can vary anywhere from 5 to 10% depending on the quality of the gel):
<img width="972" alt="image" src="https://github.com/user-attachments/assets/e9b5124c-7340-42d3-8c7a-f299a30a7f23" />


Other Features:
* Fully integrated Image Cropping and Alignment tool
* Adjust Low/High Image Contrast and Gamma (For visualizing proteins with faint bands)
* Option to add custom markers with custom font type, size and color
* Inbuilt Snapping tool to help set markers in straight lines
* Option to load image from clipboard and save to clipboard for faster operation
* Added Keyboard Shortcuts and Tooltips for major options

To compile the code into an application:
  1. You need to install pyinstaller: pip install pyinstaller
  2. Extract the files into your folder of choice and modify this command:
    pyinstaller --noconfirm --windowed --distpath C:\Imaging-Assistant-main_3\Imaging-Assistant-main\Application\dist --workpath C:\Imaging-Assistant-main_3\Imaging-Assistant-main\Application --icon=C:\Imaging-Assistant-main_3\Imaging-Assistant-main\icon.ico --name=MyApp C:\Imaging-Assistant-main_3\Imaging-Assistant-main\Imaging_assistant_V3.py
    
    Here the directory is: C:\Imaging-Assistant-main_3\Imaging-Assistant-main\ (Change as needed)
  3. Run the command

To run the program in python:
  1. Run the command: pip install -r C:\Imaging-Assistant-main_3\Imaging-Assistant-main\requirements.txt
  2. python C:\Imaging-Assistant-main_3\Imaging-Assistant-main\Imaging_assistant_V3.py



Please see the Instruction Video for more details on how to use the software.
