from PIL import ImageGrab  # Import Pillow's ImageGrab for clipboard access
import sys
from io import BytesIO
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QTextEdit, QHBoxLayout, QCheckBox, QGroupBox, QGridLayout, QWidget, QFileDialog, QSlider, QComboBox, QColorDialog, QMessageBox, QLineEdit, QFontComboBox, QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QKeySequence, QClipboard, QPen, QTransform
from PyQt5.QtCore import Qt
import json
import os
import numpy as np
from PIL import Image, ImageQt


class CombinedSDSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMAGING ASSISTANT BY AK V2.00")
        self.resize(700, 950) # Change for windows/macos viewing
        self.image_path = None
        self.image = None
        self.image_master= None
        self.image_before_padding = None
        self.image_contrasted=None
        self.image_padded=False
        self.left_markers = []
        self.right_markers = []
        self.top_markers = []
        self.current_marker_index = 0
        self.current_top_label_index = 0
        self.font_rotation=-45
        self.image_width=0
        self.image_height=0
        self.new_image_width=0
        self.new_image_height=0
        # Initialize self.marker_values to None initially
        self.marker_values_dict = {
            "Precision Plus All Blue/Unstained": [250, 150, 100, 75, 50, 37, 25, 20, 15, 10],
            "1 kB Plus": [15000, 10000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 850, 650, 500, 400, 300, 200, 100],
        }
        self.top_label=["MWM" , "S1", "S2", "S3" , "S4", "S5" , "S6", "S7", "S8", "S9", "MWM"]
        self.top_label_dict={"Precision Plus All Blue/Unstained": ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"]}
        
        
        self.custom_marker_name = "Custom"
        # Load the config file if exists
        
        self.marker_mode = None
        self.left_marker_shift = 0   # Additional shift for marker text
        self.right_marker_shift = 0   # Additional shift for marker text
        self.right_marker_shift_added=0
        self.top_marker_shift= 10
        self.top_padding = 0
        self.font_color = QColor(0, 0, 0)  # Default to black
        self.font_family = "Arial"  # Default font family
        self.font_size = 12  # Default font size
        self.image_array_backup= None
    
        # Connect UI elements to update the font parameters
        
        self.init_ui()
        

    def init_ui(self):
        layout = QVBoxLayout()
        extra_layout = QHBoxLayout()
    
        # Image display
        self.live_view_label = QLabel("Load an SDS-PAGE image to preview")
        self.live_view_label.setAlignment(Qt.AlignCenter)
        self.live_view_label.setStyleSheet("border: 1px solid black;")
        self.live_view_label.setFixedSize(600,450) # Change for windows/macos viewing
        self.live_view_label.mousePressEvent = self.add_band
        extra_layout.addWidget(self.live_view_label)
        
        
    
        # Load, save, and crop buttons
        buttons_layout = QVBoxLayout()
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        buttons_layout.addWidget(load_button)
        
        paste_button = QPushButton('Paste Image')
        paste_button.clicked.connect(self.paste_image)  # Connect the button to the paste_image method
        buttons_layout.addWidget(paste_button)
    
        
        reset_button = QPushButton("Reset Image")  # Add Reset Image button
        reset_button.clicked.connect(self.reset_image)  # Connect the reset functionality
        buttons_layout.addWidget(reset_button)
        
        copy_button = QPushButton('Copy Image to Clipboard')
        copy_button.clicked.connect(self.copy_to_clipboard)
        buttons_layout.addWidget(copy_button)
        
        save_button = QPushButton("Save Processed Image")
        save_button.clicked.connect(self.save_image)
        buttons_layout.addWidget(save_button)
    
        extra_layout.addLayout(buttons_layout)
        
        # Font options group
        font_options_group = QGroupBox("Font and Image Options")
        font_options_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        font_options_layout = QGridLayout()  # Use a grid layout for compact arrangement
        
        # Font type selection (QFontComboBox)
        font_type_label = QLabel("Font:")
        self.font_combo_box = QFontComboBox()
        self.font_combo_box.setEditable(False)
        self.font_combo_box.setCurrentFont(QFont("Arial"))
        font_options_layout.addWidget(font_type_label, 0, 0)  # Row 0, Column 0
        font_options_layout.addWidget(self.font_combo_box, 0, 1, 1, 2)  # Row 0, Column 1-2
        
        # Font size selection (QSpinBox)
        font_size_label = QLabel("Size:")
        self.font_size_spinner = QSpinBox()
        self.font_size_spinner.setRange(8, 72)  # Set a reasonable font size range
        self.font_size_spinner.setValue(12)  # Default font size
        font_options_layout.addWidget(font_size_label, 1, 0)  # Row 1, Column 0
        font_options_layout.addWidget(self.font_size_spinner, 1, 1)  # Row 1, Column 1
        
        # Font color selection (QPushButton to open QColorDialog)
        self.font_color_button = QPushButton("Color")
        self.font_color_button.clicked.connect(self.select_font_color)
        font_options_layout.addWidget(self.font_color_button, 1, 2)  # Row 1, Column 2
        
        # Font rotation input (QSpinBox)
        font_rotation_label = QLabel("Rotation:")
        self.font_rotation_input = QSpinBox()
        self.font_rotation_input.setRange(-180, 180)  # Set a reasonable font rotation range
        self.font_rotation_input.setValue(-45)  # Default rotation
        self.font_rotation_input.valueChanged.connect(self.update_font)
        
        font_options_layout.addWidget(font_rotation_label, 2, 0)  # Row 2, Column 0
        font_options_layout.addWidget(self.font_rotation_input, 2, 1, 1, 2)  # Row 2, Column 1-2
        
        # Contrast slider
        contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)  # Range for contrast adjustment (0%-200%)
        self.contrast_slider.setValue(100)  # Default value (100% = no change)
        self.contrast_slider.valueChanged.connect(self.update_image_contrast)
        font_options_layout.addWidget(contrast_label, 3, 0)
        font_options_layout.addWidget(self.contrast_slider, 3, 1, 1, 2)
        
        # gamma slider
        gamma_label = QLabel("Gamma:")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(0, 200)  # Range for gamma adjustment (0%-200%)
        self.gamma_slider.setValue(100)  # Default value (100% = no change)
        self.gamma_slider.valueChanged.connect(self.update_image_gamma)
        font_options_layout.addWidget(gamma_label, 4, 0)
        font_options_layout.addWidget(self.gamma_slider, 4, 1, 1, 2)
        
        # Apply button gamma and contrast button
        apply_button = QPushButton("Apply Gamma and Contrast Settings")
        apply_button.clicked.connect(self.save_contrast_options)
        font_options_layout.addWidget(apply_button, 5, 0, 1, 3)  # Span all columns
        
        # Reset gamma and contrast button
        reset_button = QPushButton("Reset Gamma and Contrast Settings")
        reset_button.clicked.connect(self.reset_gamma_contrast)
        font_options_layout.addWidget(reset_button, 6, 0, 1, 3)  # Span all columns
        
        # Connect signals for dynamic updates
        self.font_combo_box.currentFontChanged.connect(self.update_font)
        self.font_size_spinner.valueChanged.connect(self.update_font)
        
        # Set the layout for the font options group box
        font_options_group.setLayout(font_options_layout)
        buttons_layout.addWidget(font_options_group)
        
        layout.addLayout(extra_layout)
        self.setLayout(layout)
        
        marker_options_layout = QHBoxLayout()

        # Left/Right Marker Options
        left_right_marker_group = QGroupBox("Left/Right Marker Options")
        left_right_marker_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        left_right_marker_layout = QVBoxLayout()
        
        # Combo box for marker presets or custom option
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(self.marker_values_dict.keys())
        self.combo_box.addItem("Custom")
        self.combo_box.currentTextChanged.connect(self.on_combobox_changed)
        
        # Textbox to allow modification of marker values (shown when "Custom" is selected)
        self.marker_values_textbox = QLineEdit(self)
        self.marker_values_textbox.setPlaceholderText("Enter custom values as comma-separated list")
        self.marker_values_textbox.setEnabled(False)  # Disable by default
        
        # Rename input for custom option
        self.rename_input = QLineEdit(self)
        self.rename_input.setPlaceholderText("Enter new name for Custom")
        self.rename_input.setEnabled(False)
        
        # Save button for the current configuration
        self.save_button = QPushButton("Save Config", self)
        self.save_button.clicked.connect(self.save_config)
        
        # Add widgets to the Left/Right Marker Options layout
        left_right_marker_layout.addWidget(self.combo_box)
        left_right_marker_layout.addWidget(self.marker_values_textbox)
        left_right_marker_layout.addWidget(self.rename_input)
        left_right_marker_layout.addWidget(self.save_button)
        
        # Set layout for the Left/Right Marker Options group
        left_right_marker_group.setLayout(left_right_marker_layout)
        
        # Top Marker Options
        top_marker_group = QGroupBox("Top Marker Options")
        top_marker_group.setStyleSheet("QGroupBox { font-weight: bold;}")
        
        # Vertical layout for top marker group
        top_marker_layout = QVBoxLayout()
        
        
        # Text input for Top Marker Labels
        self.top_marker_input = QTextEdit(self)
        self.top_label = ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"]
        self.top_marker_input.setText(", ".join(self.top_label))  # Populate with initial values
        self.top_marker_input.setMinimumHeight(40)
        
        # Button to update Top Marker Labels
        self.update_top_labels_button = QPushButton("Update Labels")
        self.update_top_labels_button.clicked.connect(self.update_top_labels)
        
        # Add widgets to the top marker layout
        top_marker_layout.addWidget(self.top_marker_input)
        top_marker_layout.addWidget(self.update_top_labels_button)
        
        # Set the layout for the Top Marker Group
        top_marker_group.setLayout(top_marker_layout)
        
        # Add the group box to the main layout
        layout.addWidget(top_marker_group)
        
        # Add both groups to the horizontal layout
        marker_options_layout.addWidget(left_right_marker_group)
        marker_options_layout.addWidget(top_marker_group)
        
        # Add the horizontal layout to the main layout
        layout.addLayout(marker_options_layout)
        
        main_layout = QHBoxLayout()

        # Cropping parameters group
        cropping_params_group = QGroupBox("Cropping Parameters")
        cropping_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        # Layout for the cropping group box
        cropping_layout = QVBoxLayout()
        
        self.orientation_label = QLabel("Rotation Angle")
        self.orientation_slider = QSlider(Qt.Horizontal)
        self.orientation_slider.setRange(-180, 180)
        self.orientation_slider.setValue(0)
        self.orientation_slider.valueChanged.connect(self.update_live_view)
        
        # Checkbox for showing guides
        self.show_guides_checkbox = QCheckBox("Show Guides", self)
        self.show_guides_checkbox.setChecked(False)
        self.show_guides_checkbox.stateChanged.connect(self.update_live_view)
        
        # Align button to trigger image alignment
        self.align_button = QPushButton("Align Image", self)
        self.align_button.clicked.connect(self.align_image)
        
        # Add these UI components to the cropping layout
        cropping_layout.addWidget(self.orientation_label)
        cropping_layout.addWidget(self.orientation_slider)
        cropping_layout.addWidget(self.show_guides_checkbox)
        cropping_layout.addWidget(self.align_button)
        
        # X sliders in one row
        crop_x_layout = QHBoxLayout()
        crop_x_start_label = QLabel("X Start (%)")
        self.crop_x_start_slider = QSlider(Qt.Horizontal)
        self.crop_x_start_slider.setRange(0, 100)
        self.crop_x_start_slider.setValue(0)
        self.crop_x_start_slider.valueChanged.connect(self.update_live_view)
        crop_x_layout.addWidget(crop_x_start_label)
        crop_x_layout.addWidget(self.crop_x_start_slider)
        
        crop_x_end_label = QLabel("X End (%)")
        self.crop_x_end_slider = QSlider(Qt.Horizontal)
        self.crop_x_end_slider.setRange(0, 100)
        self.crop_x_end_slider.setValue(100)
        self.crop_x_end_slider.valueChanged.connect(self.update_live_view)
        crop_x_layout.addWidget(crop_x_end_label)
        crop_x_layout.addWidget(self.crop_x_end_slider)
        
        cropping_layout.addLayout(crop_x_layout)
        
        # Y sliders in another row
        crop_y_layout = QHBoxLayout()
        crop_y_start_label = QLabel("Y Start (%)")
        self.crop_y_start_slider = QSlider(Qt.Horizontal)
        self.crop_y_start_slider.setRange(0, 100)
        self.crop_y_start_slider.setValue(0)
        self.crop_y_start_slider.valueChanged.connect(self.update_live_view)
        crop_y_layout.addWidget(crop_y_start_label)
        crop_y_layout.addWidget(self.crop_y_start_slider)
        
        crop_y_end_label = QLabel("Y End (%)")
        self.crop_y_end_slider = QSlider(Qt.Horizontal)
        self.crop_y_end_slider.setRange(0, 100)
        self.crop_y_end_slider.setValue(100)
        self.crop_y_end_slider.valueChanged.connect(self.update_live_view)
        crop_y_layout.addWidget(crop_y_end_label)
        crop_y_layout.addWidget(self.crop_y_end_slider)
        
        cropping_layout.addLayout(crop_y_layout)
        
        crop_button = QPushButton("Update Crop")
        crop_button.clicked.connect(self.update_crop)
        cropping_layout.addWidget(crop_button)
        
        
        
        cropping_params_group.setLayout(cropping_layout)
        
        # Padding parameters section (Text inputs for padding)
        padding_params_group = QGroupBox("Adding White Space")
        padding_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        # Layout for the padding group box (All inputs on the same line)
        padding_layout = QVBoxLayout()
        
        # Left space input and label
        left_padding_label = QLabel("Left Space (px):")
        self.left_padding_input = QLineEdit()
        self.left_padding_input.setText("0")  # Default value
        
        # Right space input and label
        right_padding_label = QLabel("Right Space (px):")
        self.right_padding_input = QLineEdit()
        self.right_padding_input.setText("0")  # Default value
        
        # Top space input and label
        top_padding_label = QLabel("Top Space (px):")
        self.top_padding_input = QLineEdit()
        self.top_padding_input.setText("50")  # Default value
        
        # Finalize image button
        self.finalize_button = QPushButton("Add White Space")
        self.finalize_button.clicked.connect(self.finalize_image)
        
        # Reset button to remove padding
        self.reset_padding_button = QPushButton("Remove White Space")
        self.reset_padding_button.clicked.connect(self.remove_padding)
        
        # Add all the elements (inputs and button) to the padding layout
        padding_layout.addWidget(left_padding_label)
        padding_layout.addWidget(self.left_padding_input)
        padding_layout.addWidget(right_padding_label)
        padding_layout.addWidget(self.right_padding_input)
        padding_layout.addWidget(top_padding_label)
        padding_layout.addWidget(self.top_padding_input)
        padding_layout.addWidget(self.finalize_button)
        padding_layout.addWidget(self.reset_padding_button)
        
        padding_params_group.setLayout(padding_layout)
        
        # Add the cropping and padding group boxes to the main horizontal layout
        main_layout.addWidget(cropping_params_group)
        main_layout.addWidget(padding_params_group)
        
        # Set the main layout for the parent widget
        layout.addLayout(main_layout)

        # Create a horizontal layout to hold the marker and font options side by side
        main_layout = QHBoxLayout()
        
        # Marker padding sliders - Group box for marker distance adjustment
        padding_params_group = QGroupBox("Marker Placement and Distance")
        padding_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        # Grid layout for the marker group box
        padding_layout = QGridLayout()
        
        # Left marker: Button, slider, reset, and duplicate in the same row
        left_marker_button = QPushButton("Left Markers")
        left_marker_button.clicked.connect(self.enable_left_marker_mode)
        self.left_padding_slider = QSlider(Qt.Horizontal)
        self.left_padding_slider.setRange(-1000, 1000)
        self.left_padding_slider.setValue(0)
        self.left_padding_slider.valueChanged.connect(self.update_left_padding)
        
        reset_left_button = QPushButton("Reset")
        reset_left_button.clicked.connect(lambda: self.reset_marker('left'))
        duplicate_left_button = QPushButton("Duplicate Left")
        duplicate_left_button.clicked.connect(lambda: self.duplicate_marker('left'))
        
        # Add left marker widgets to the grid layout
        padding_layout.addWidget(left_marker_button, 0, 0)
        padding_layout.addWidget(reset_left_button, 0, 1)
        padding_layout.addWidget(duplicate_left_button, 0, 2)
        padding_layout.addWidget(self.left_padding_slider, 0, 3)
        
        # Right marker: Button, slider, reset, and duplicate in the same row
        right_marker_button = QPushButton("Right Markers")
        right_marker_button.clicked.connect(self.enable_right_marker_mode)
        self.right_padding_slider = QSlider(Qt.Horizontal)
        self.right_padding_slider.setRange(-1000, 1000)
        self.right_padding_slider.setValue(0)
        self.right_padding_slider.valueChanged.connect(self.update_right_padding)
        
        reset_right_button = QPushButton("Reset")
        reset_right_button.clicked.connect(lambda: self.reset_marker('right'))
        duplicate_right_button = QPushButton("Duplicate Right")
        duplicate_right_button.clicked.connect(lambda: self.duplicate_marker('right'))
        
        # Add right marker widgets to the grid layout
        padding_layout.addWidget(right_marker_button, 1, 0)
        padding_layout.addWidget(reset_right_button, 1, 1)
        padding_layout.addWidget(duplicate_right_button, 1, 2)
        padding_layout.addWidget(self.right_padding_slider, 1, 3)
        
        # Top marker: Button, slider, and reset in the same row
        top_marker_button = QPushButton("Top Markers")
        top_marker_button.clicked.connect(self.enable_top_marker_mode)
        self.top_padding_slider = QSlider(Qt.Horizontal)
        self.top_padding_slider.setRange(-1000, 1000)
        self.top_padding_slider.setValue(self.top_padding)
        self.top_padding_slider.valueChanged.connect(self.update_top_padding)
        
        reset_top_button = QPushButton("Reset")
        reset_top_button.clicked.connect(lambda: self.reset_marker('top'))
        
        # Add top marker widgets to the grid layout
        padding_layout.addWidget(top_marker_button, 2, 0)
        padding_layout.addWidget(reset_top_button, 2, 1)
        padding_layout.addWidget(self.top_padding_slider, 2, 2, 1, 2)  # Slider spans 2 columns for better alignment
        
        # Set the layout for the marker group box
        padding_params_group.setLayout(padding_layout)
        
                
        # Add the font options group box to the main layout
        main_layout.addWidget(padding_params_group)

        
        # Add the layout to the main layout of the parent widget
        layout.addLayout(main_layout) 

    

        self.setLayout(layout)
        self.load_config() 
    
    def update_top_labels(self):
        # Retrieve the multiline text from QTextEdit
        self.marker_values=[int(num) if num.strip().isdigit() else num.strip() for num in self.marker_values_textbox.text().strip("[]").split(",")]
        input_text = self.top_marker_input.toPlainText()
        
        # Split the text into a list by commas and strip whitespace
        self.top_label= [label.strip() for label in input_text.split(",") if label.strip()]
        try:
            
            # Ensure that the top_markers list only updates the top_label values serially
            for i in range(min(len(self.top_markers), len(self.top_label))):
                self.top_markers[i] = (self.top_markers[i][0], self.top_label[i])
            
            # If self.top_label has more entries than current top_markers, add them
            # if len(self.top_label) > len(self.top_markers):
            #     additional_markers = [(self.top_markers[-1][0] + 50 * (i + 1), label) 
            #                           for i, label in enumerate(self.top_label[len(self.top_markers):])]
            #     self.top_markers.extend(additional_markers)
            
            # If self.top_label has fewer entries, truncate the list
            if len(self.top_label) < len(self.top_markers):
                self.top_markers = self.top_markers[:len(self.top_label)]
        except:
            pass
        try:
            for i in range(min(len(self.left_markers), len(self.marker_values))):
                self.left_markers[i] = (self.left_markers[i][0], self.marker_values[i])
            if len(self.marker_values) < len(self.left_markers):
                self.left_markers = self.left_markers[:len(self.marker_values)]
                
            for i in range(min(len(self.right_markers), len(self.marker_values))):
                self.right_markers[i] = (self.right_markers[i][0], self.marker_values[i])
            if len(self.marker_values) < len(self.right_markers):
                self.right_markers = self.right_markers[:len(self.marker_values)]
                
        except:
            pass
        print("Updated Top Markers:", self.top_markers)  # Debugging
        print("Updated Left Markers:", self.left_markers)  # Debugging
        print("Updated Left Markers:", self.right_markers)  # Debugging
        
        # Trigger a refresh of the live view
        self.update_live_view()
        
    def reset_marker(self, marker_type):
        if marker_type == 'left':
            self.left_markers.clear()  # Clear left markers
            # self.left_padding_slider.setValue(0)
            self.current_marker_index = 0           
        elif marker_type == 'right':
            self.right_markers.clear()  # Clear right markers
            self.current_marker_index = 0
            # self.right_padding_slider.setValue(0)
        elif marker_type == 'top':
            self.top_markers.clear()  # Clear top markers
            self.current_top_label_index = 0
            # self.top_padding_slider.setValue(0)
    
        # Call update live view after resetting markers
        self.update_live_view()
        
    def duplicate_marker(self, marker_type):
        if marker_type == 'left' and self.right_markers:
            self.left_markers = self.right_markers.copy()  # Copy right markers to left
        elif marker_type == 'right' and self.left_markers:
            self.right_markers = self.left_markers.copy()  # Copy left markers to right

            # self.right_padding = self.image_width*0.9
    
        # Call update live view after duplicating markers
        self.update_live_view()
        
    def on_combobox_changed(self, text):
        """Handle the ComboBox change event to update marker values."""
        if text == "Custom":
            # Enable the textbox for custom values when "Custom" is selected
            self.marker_values_textbox.setEnabled(True)
            self.rename_input.setEnabled(True)
            
            # self.marker_values_textbox.setText()
        else:
            # Update marker values based on the preset option
            self.marker_values = self.marker_values_dict.get(text, [])
            self.marker_values_textbox.setEnabled(False)  # Disable the textbox
            self.rename_input.setEnabled(False)
            self.top_label = self.top_label_dict.get(text, [])
            print("TOP LABEL:",self.top_label)
            self.top_marker_input.setText(", ".join(self.top_label))
            try:
                
                # Ensure that the top_markers list only updates the top_label values serially
                for i in range(min(len(self.top_markers), len(self.top_label))):
                    self.top_markers[i] = (self.top_markers[i][0], self.top_label[i])
                
                # # If self.top_label has more entries than current top_markers, add them
                # if len(self.top_label) > len(self.top_markers):
                #     additional_markers = [(self.top_markers[-1][0] + 50 * (i + 1), label) 
                #                           for i, label in enumerate(self.top_label[len(self.top_markers):])]
                #     self.top_markers.extend(additional_markers)
                
                # If self.top_label has fewer entries, truncate the list
                if len(self.top_label) < len(self.top_markers):
                    self.top_markers = self.top_markers[:len(self.top_label)]
                    
                
                
            except:
                pass
            try:
                self.marker_values_textbox.setText(str(self.marker_values_dict[self.combo_box.currentText()]))
            except:
                pass
    
    # Functions for updating contrast and gamma
    
    def reset_gamma_contrast(self):
        self.contrast_slider.setValue(100)  # Reset contrast to default
        self.gamma_slider.setValue(100)  # Reset gamma to default
        self.image = self.image_master.copy()  # Restore the original image
        self.image_contrasted=self.image.copy()
        self.update_live_view()

    
    def update_image_contrast(self):
        if self.image:
            contrast_factor = self.contrast_slider.value() / 100.0
            gamma_factor = self.gamma_slider.value() / 100.0
            self.image = self.apply_contrast_gamma(self.image_contrasted, contrast=contrast_factor, gamma=gamma_factor)
            self.update_live_view()
    
    def update_image_gamma(self):
        if self.image:
            contrast_factor = self.contrast_slider.value() / 100.0
            gamma_factor = self.gamma_slider.value() / 100.0
            self.image = self.apply_contrast_gamma(self.image_contrasted, contrast=contrast_factor, gamma=gamma_factor)            
            self.update_live_view()
    
    def apply_contrast_gamma(self, image, contrast, gamma):
        # Convert QImage to PIL Image
        pil_image = ImageQt.fromqimage(image)
    
        # Convert PIL Image to numpy array
        img_array = np.asarray(pil_image, dtype=np.float32)
    
        # Apply contrast and gamma adjustments
        img_array = (img_array - 127.5) * contrast + 127.5  # Contrast adjustment
        # img_array = img_array * gamma  # gamma adjustment
        img_array = np.power(img_array / 255.0, gamma) * 255.0  # Gamma adjustment
    
        # Clip values to valid range [0, 255]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
        # Convert back to PIL Image
        adjusted_image = Image.fromarray(img_array)
    
        # Convert back to QImage
        return ImageQt.toqimage(adjusted_image)
    

    def save_contrast_options(self):
        self.image_contrasted=self.image.copy()
        self.image_before_padding=self.image_contrasted.copy()


    def save_config(self):
        """Rename the 'Custom' marker values option and save the configuration."""
        new_name = self.rename_input.text().strip()
        
        # Ensure that the correct dictionary (self.marker_values_dict) is being used
        if self.rename_input.text() != "Enter new name for Custom" and self.rename_input.text() != "":  # Correct condition
            # Save marker values list
            self.marker_values_dict[new_name] = [int(num) if num.strip().isdigit() else num.strip() for num in self.marker_values_textbox.text().strip("[]").split(",")]
            print(self.marker_values_dict)
            
            # Save top_label list under the new_name key
            self.top_label_dict[new_name] = [int(num) if num.strip().isdigit() else num.strip() for num in self.top_marker_input.toPlainText().strip("[]").split(",")]

            print("Top labels:", self.top_label_dict)
            
            try:
                # Save both the marker values and top label (under new_name) to the config file
                with open("Imaging_assistant_config.txt", "w") as f:
                    config = {
                        "marker_values": self.marker_values_dict,
                        "top_label": self.top_label_dict  # Save top_label_dict as a dictionary with new_name as key
                    }
                    json.dump(config, f)  # Save both marker_values and top_label_dict
                print("Config saved successfully.")
            except Exception as e:
                print(f"Error saving config: {e}")
        else:
            print("COULD NOT SAVE")
        
        self.load_config()  # Reload the configuration after saving
    
    
    def load_config(self):
        """Load the configuration from the file."""
        try:
            with open("Imaging_assistant_config.txt", "r") as f:
                config = json.load(f)
                print("Loaded config:", config)
                
                # Load marker values and top label from the file
                self.marker_values_dict = config.get("marker_values", {})
                
                # Retrieve top_label list from the dictionary using the new_name key
                new_name = self.rename_input.text().strip()  # Assuming `new_name` is defined here; otherwise, set it manually
                self.top_label_dict = config.get("top_label", {
                    "Precision Plus All Blue/Unstained": ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"],
                    "1 kB Plus": ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"],
                })  # Default if not found
                self.top_label = self.top_label_dict.get(new_name, ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"])  # Default if not found
                print("Top labels:", self.top_label)
                
        except FileNotFoundError:
            print("Config file not found, using default options.")
            # Set default marker values and top_label if config is not found
            self.marker_values_dict = {
                "Precision Plus All Blue/Unstained": [250, 150, 100, 75, 50, 37, 25, 20, 15, 10],
                "1 kB Plus": [15000, 10000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 850, 650, 500, 400, 300, 200, 100],
            }
            self.top_label_dict = {
                "Precision Plus All Blue/Unstained": ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"],
                "1 kB Plus": ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"],
            }  # Default top labels
            self.top_label = self.top_label_dict.get("Precision Plus All Blue/Unstained", ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"])
        except Exception as e:
            print(f"Error loading config: {e}")
            # Fallback to default values if there is an error
            self.marker_values_dict = {
                "Precision Plus All Blue/Unstained": [250, 150, 100, 75, 50, 37, 25, 20, 15, 10],
                "1 kB Plus": [15000, 10000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 850, 650, 500, 400, 300, 200, 100],
            }
            self.top_label_dict = {
                "Precision Plus All Blue/Unstained": ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"],
                "1 kB Plus": ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"],
            }
            self.top_label = self.top_label_dict.get("Precision Plus All Blue/Unstained", ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"])
    
        # Update combo box options with loaded marker values
        self.combo_box.clear()
        self.combo_box.addItems(self.marker_values_dict.keys())
        self.combo_box.addItem("Custom")
        print("LOADED TOP LABELS:", self.top_label)
    
    def paste_image(self):
        """Handle pasting image from clipboard."""
        self.load_image_from_clipboard()
        self.update_live_view()
    
    def load_image_from_clipboard(self):
        self.reset_image()
        """Load an image from the clipboard into self.image."""
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        # Check if the clipboard contains an image
        if mime_data.hasImage():
            image = clipboard.image()  # Get image from clipboard
            print("Image pasted from clipboard.")
            self.image = image  # Store the image in self.image
            self.original_image = self.image.copy()
            self.image_contrasted= self.image.copy()
            self.image_master=self.image.copy()
            # self.save_right_shift_marker(self.image)
        else:
            print("No image found in clipboard.")
            
        
    def update_font(self):
        """Update the font settings based on UI inputs"""
        # Update font family from the combo box
        self.font_family = self.font_combo_box.currentFont().family()
        
        # Update font size from the spin box
        self.font_size = self.font_size_spinner.value()
        
        self.font_rotation = int(self.font_rotation_input.value())
        
    
        # Once font settings are updated, update the live view immediately
        self.update_live_view()
    
    def select_font_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.font_color = color  # Store the selected color   
            self.update_font()
            
    def load_image(self):
        self.reset_image()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.tif)", options=options
        )
        if file_path:
            self.image_path = file_path
            self.original_image = QImage(self.image_path)  # Keep the original image
            self.image = self.original_image.copy()        # Start with a copy of the original
            self.image_master= self.original_image.copy()  
            self.image_contrasted= self.original_image.copy()  
            self.update_live_view()
    
            # Determine associated config file
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if base_name.endswith("_original"):
                config_name = base_name.replace("_original", "_config.txt")
            else:
                config_name = base_name + "_config.txt"
            config_path = os.path.join(os.path.dirname(file_path), config_name)
    
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as config_file:
                        config_data = json.load(config_file)
                    self.apply_config(config_data)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load config file: {e}")
    
    def apply_config(self, config_data):
        self.left_padding_input.setText(config_data["adding_white_space"]["left"])
        self.right_padding_input.setText(config_data["adding_white_space"]["right"])
        self.top_padding_input.setText(config_data["adding_white_space"]["top"])
    
        self.crop_x_start_slider.setValue(config_data["cropping_parameters"]["x_start_percent"])
        self.crop_x_end_slider.setValue(config_data["cropping_parameters"]["x_end_percent"])
        self.crop_y_start_slider.setValue(config_data["cropping_parameters"]["y_start_percent"])
        self.crop_y_end_slider.setValue(config_data["cropping_parameters"]["y_end_percent"])
    
        self.left_markers = config_data["marker_positions"]["left"]
        self.right_markers = config_data["marker_positions"]["right"]
        self.top_markers = config_data["marker_positions"]["top"]
    
        self.top_label = config_data["marker_labels"]["top"]
        self.top_marker_input.setText(", ".join(self.top_label))
    
        self.font_family = config_data["font_options"]["font_family"]
        self.font_size = config_data["font_options"]["font_size"]
        self.font_rotation = config_data["font_options"]["font_rotation"]
        self.font_color = QColor(config_data["font_options"]["font_color"])
        
        self.top_padding_slider.setValue(config_data["marker_padding"]["top"])
        self.left_padding_slider.setValue(config_data["marker_padding"]["left"])
        self.right_padding_slider.setValue(config_data["marker_padding"]["right"])
        
        #DO NOT KNOW WHY THIS WORKS BUT DIRECT VARIABLE ASSIGNING DOES NOT WORK
        
        font_size_new=self.font_size
        font_rotation_new=self.font_rotation
        
        self.font_combo_box.setCurrentFont(QFont(self.font_family))
        self.font_size_spinner.setValue(font_size_new)
        self.font_rotation_input.setValue(font_rotation_new)

        self.update_live_view()
        
    def get_current_config(self):
        return {
            "adding_white_space": {
                "left": self.left_padding_input.text(),
                "right": self.right_padding_input.text(),
                "top": self.top_padding_input.text(),
            },
            "cropping_parameters": {
                "x_start_percent": self.crop_x_start_slider.value(),
                "x_end_percent": self.crop_x_end_slider.value(),
                "y_start_percent": self.crop_y_start_slider.value(),
                "y_end_percent": self.crop_y_end_slider.value(),
            },
            "marker_positions": {
                "left": self.left_markers,
                "right": self.right_markers,
                "top": self.top_markers,
            },
            "marker_labels": {
                "top": self.top_label,
                "left": [marker[1] for marker in self.left_markers],
                "right": [marker[1] for marker in self.right_markers]
            },
            "marker_padding": {
                "top": self.top_padding_slider.value(),
                "left": self.left_padding_slider.value(),
                "right": self.right_padding_slider.value(),
            },
            "font_options": {
                "font_family": self.font_family,
                "font_size": self.font_size,
                "font_rotation": self.font_rotation,
                "font_color": self.font_color.name(),
            },
        }   
    
    def add_band(self, event):
        # Ensure there's an image loaded and marker mode is active
        if not self.image or not self.marker_mode:
            return
    
        # Get the cursor position from the event
        pos = event.pos()
        cursor_x, cursor_y = pos.x(), pos.y()
    
        # Get the dimensions of the displayed image
        displayed_width = self.live_view_label.width()
        displayed_height = self.live_view_label.height()
    
        # Get the actual dimensions of the loaded image
        image_width = self.image.width()
        image_height = self.image.height()
    
        # Calculate the scaling factor (assuming uniform scaling to maintain aspect ratio)
        scale = min(displayed_width / image_width, displayed_height / image_height)
    
        # Calculate offsets if the image is centered in the live_view_label
        offset_x = (displayed_width - image_width * scale) / 2
        offset_y = (displayed_height - image_height * scale) / 2
    
        # Transform cursor coordinates to the image coordinate space
        image_x = (cursor_x - offset_x) / scale
        image_y = (cursor_y - offset_y) / scale
    
        # Validate that the transformed coordinates are within image bounds
        if not (0 <= image_y <= image_height):
            return  # Ignore clicks outside the image bounds
    
        # Add the band marker based on the active marker mode
        if self.marker_mode == "left" and self.current_marker_index < len(self.marker_values):
            self.left_markers.append((image_y, self.marker_values[self.current_marker_index]))
            self.current_marker_index += 1
        elif self.marker_mode == "right" and self.current_marker_index < len(self.marker_values):
            self.right_markers.append((image_y, self.marker_values[self.current_marker_index]))
            self.current_marker_index += 1
        elif self.marker_mode == "top" and self.current_top_label_index < len(self.top_label):
            self.top_markers.append((image_x, self.top_label[self.current_top_label_index]))
            self.current_top_label_index += 1
    
        # Update the live view with the new markers
        self.update_live_view()
        
    def save_right_shift_marker(self, image):
        self.right_marker_shift=image.width()*0.9
        # print("RIGHT MARKER SHIFT: ", self.right_marker_shift)
        
    def enable_left_marker_mode(self):
        self.marker_mode = "left"
        self.current_marker_index = 0

    def enable_right_marker_mode(self):
        self.marker_mode = "right"
        self.current_marker_index = 0
    
    def enable_top_marker_mode(self):
        self.marker_mode = "top"
        self.current_top_label_index
        
    def remove_padding(self):
        self.image= self.image_before_padding.copy()
        self.image_padded=False
        self.update_live_view()
        
    def finalize_image(self):
        # Get the padding values from the text inputs
        try:
            padding_left = int(self.left_padding_input.text())
            padding_right = int(self.right_padding_input.text())
            padding_top = int(self.top_padding_input.text())
        except ValueError:
            # Handle invalid input (non-integer value)
            print("Please enter valid integers for padding.")
            return
    
        # Ensure self.image_before_padding is initialized
        if self.image_before_padding is None:
            self.image_before_padding = self.image_contrasted.copy()
    
        # Reset to the original image if padding inputs have changed
        if self.image_padded:
            self.image = self.image_before_padding.copy()
            self.image_padded = False
    
        # Calculate the new dimensions with padding
        new_width = self.image.width() + padding_left + padding_right
        new_height = self.image.height() + padding_top
    
        # Create a new blank image with white background
        padded_image = QImage(new_width, new_height, QImage.Format_RGB888)
        padded_image.fill(QColor(255, 255, 255))  # Fill with white
    
        # Copy the original image onto the padded image
        for y in range(self.image.height()):
            for x in range(self.image.width()):
                color = self.image.pixel(x, y)
                padded_image.setPixel(padding_left + x, padding_top + y, color)
    
        self.image = padded_image
        self.image_padded = True
        self.image_contrasted = self.image
    
        # Adjust marker shifts to account for padding
        self.left_marker_shift += padding_left
        # self.right_marker_shift += padding_right
    
        # Update slider values to match the new shifts
        self.left_padding_slider.setValue(self.left_marker_shift)
        # self.right_padding_slider.setValue(self.right_marker_shift_added + self.right_marker_shift)
    
        self.update_live_view()
    
    def update_left_padding(self):
        # Update left padding when slider value changes
        self.left_marker_shift = self.left_padding_slider.value()
        self.update_live_view()

    def update_right_padding(self):
        # Update right padding when slider value changes
        self.right_marker_shift_added = self.right_padding_slider.value()
        self.update_live_view()
        
    def update_top_padding(self):
        # Update top padding when slider value changes
        self.top_marker_shift = self.top_padding_slider.value()
        self.update_live_view()

    def update_live_view(self):
        if not self.image:
            return
    
        # Adjust slider maximum ranges based on the current image width
        self.left_padding_slider.setRange(-self.image.width(), self.image.width())
        self.right_padding_slider.setRange(-self.image.width(), self.image.width())
    
        # Define a higher resolution for processing (e.g., 2x or 3x label size)
        render_scale = 3  # Scale factor for rendering resolution
        render_width = self.live_view_label.width() * render_scale
        render_height = self.live_view_label.height() * render_scale
    
        # Get the crop percentage values from sliders
        x_start_percent = self.crop_x_start_slider.value() / 100
        x_end_percent = self.crop_x_end_slider.value() / 100
        y_start_percent = self.crop_y_start_slider.value() / 100
        y_end_percent = self.crop_y_end_slider.value() / 100
    
        # Calculate the crop boundaries based on the percentages
        x_start = int(self.image.width() * x_start_percent)
        x_end = int(self.image.width() * x_end_percent)
        y_start = int(self.image.height() * y_start_percent)
        y_end = int(self.image.height() * y_end_percent)
    
        # Ensure the cropping boundaries are valid
        if x_start >= x_end or y_start >= y_end:
            QMessageBox.warning(self, "Warning", "Invalid cropping values.")
            self.crop_x_start_slider.setValue(0)
            self.crop_x_end_slider.setValue(100)
            self.crop_y_start_slider.setValue(0)
            self.crop_y_end_slider.setValue(100)
            return
    
        # Crop the image based on the defined boundaries
        cropped_image = self.image.copy(x_start, y_start, x_end - x_start, y_end - y_start)
    
        # Get the orientation value from the slider
        orientation = self.orientation_slider.value()  # Orientation slider value
    
        # Apply the rotation to the cropped image
        rotated_image = cropped_image.transformed(QTransform().rotate(orientation))
    
        # Scale the rotated image to the rendering resolution
        scaled_image = rotated_image.scaled(
            render_width,
            render_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    
        # Render on a high-resolution canvas
        canvas = QImage(render_width, render_height, QImage.Format_RGB888)
        canvas.fill(QColor(255, 255, 255))  # Fill with white background
        self.save_right_shift_marker(scaled_image)
        self.render_image_on_canvas(canvas, scaled_image, x_start, y_start, render_scale)
        
        # Scale the high-resolution canvas down to the label's size for display
        pixmap = QPixmap.fromImage(canvas).scaled(
            self.live_view_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.live_view_label.setPixmap(pixmap)
        

    def render_image_on_canvas(self, canvas, scaled_image, x_start, y_start, render_scale, draw_guides=True):
        painter = QPainter(canvas)
        x_offset = (canvas.width() - scaled_image.width()) // 2
        y_offset = (canvas.height() - scaled_image.height()) // 2
        painter.drawImage(x_offset, y_offset, scaled_image)
    
        # Get the selected font settings
        font = QFont(self.font_combo_box.currentFont().family(), self.font_size_spinner.value() * render_scale)
        font_color = self.font_color if hasattr(self, 'font_color') else QColor(0, 0, 0)  # Default to black if no color selected
    
        painter.setFont(font)
        painter.setPen(font_color)  # Set the font color
    
        # Measure text height for vertical alignment
        font_metrics = painter.fontMetrics()
        text_height = font_metrics.height()
        line_padding = 5 * render_scale  # Space between text and line
    
        # Draw the left markers (aligned right)
        for y_pos, marker_value in self.left_markers:
            y_pos_cropped = (y_pos - y_start) * (scaled_image.height() / self.image.height())
            if 0 <= y_pos_cropped <= scaled_image.height():
                text = f"{marker_value} ⎯ "  ##CHANGE HERE IF YOU WANT TO REMOVE THE "-"
                text_width = font_metrics.horizontalAdvance(text)  # Get text width
                painter.drawText(
                    x_offset + self.left_marker_shift - text_width,
                    y_offset + y_pos_cropped + text_height / 4,  # Adjust for proper text placement
                    text,
                )
    
        # Draw the right markers (aligned left)
        for y_pos, marker_value in self.right_markers:
            y_pos_cropped = (y_pos - y_start) * (scaled_image.height() / self.image.height())
            if 0 <= y_pos_cropped <= scaled_image.height():
                text = f" ⎯ {marker_value}" ##CHANGE HERE IF YOU WANT TO REMOVE THE "-"
                painter.drawText(
                    x_offset + self.right_marker_shift + self.right_marker_shift_added + line_padding,
                    y_offset + y_pos_cropped + text_height / 4,  # Adjust for proper text placement
                    text,
                )
    
        # Draw the top markers (if needed)
        for x_pos, top_label in self.top_markers:
            x_pos_cropped = (x_pos - x_start) * (scaled_image.width() / self.image.width())
            if 0 <= x_pos_cropped <= scaled_image.width():
                painter.save()
                label_x = x_offset + x_pos_cropped
                label_y = y_offset + self.top_marker_shift * render_scale
                painter.translate(label_x, label_y)
                painter.rotate(self.font_rotation)
                painter.drawText(0, 0, f"{top_label}")
                painter.restore()
    
        # Draw guide lines
        if draw_guides and self.show_guides_checkbox.isChecked():
            pen = QPen(Qt.red, 2 * render_scale)
            painter.setPen(pen)
            center_x = canvas.width() // 2
            center_y = canvas.height() // 2
            painter.drawLine(center_x, 0, center_x, canvas.height())  # Vertical line
            painter.drawLine(0, center_y, canvas.width(), center_y)  # Horizontal line
    
        painter.end()

     
    def crop_image(self):
        """Function to crop the current image."""
        if not self.image:
            return None
    
        # Get crop percentage from sliders
        x_start_percent = self.crop_x_start_slider.value() / 100
        x_end_percent = self.crop_x_end_slider.value() / 100
        y_start_percent = self.crop_y_start_slider.value() / 100
        y_end_percent = self.crop_y_end_slider.value() / 100
    
        # Calculate crop boundaries
        x_start = int(self.image.width() * x_start_percent)
        x_end = int(self.image.width() * x_end_percent)
        y_start = int(self.image.height() * y_start_percent)
        y_end = int(self.image.height() * y_end_percent)
    
        # Ensure cropping is valid
        if x_start >= x_end or y_start >= y_end:
            QMessageBox.warning(self, "Warning", "Invalid cropping values.")
            return None
    
        # Crop the image
        cropped_image = self.image.copy(x_start, y_start, x_end - x_start, y_end - y_start)
        return cropped_image
    
    # Modify align_image and update_crop to preserve settings
    def align_image(self):
        """Align the image based on the orientation slider and keep high-resolution updates."""
        if not self.image:
            return
    
        self.draw_guides = False
        self.show_guides_checkbox.setChecked(False)
    
        # Get the orientation value from the slider
        angle = self.orientation_slider.value()
    
        # Perform rotation
        transform = QTransform()
        transform.translate(self.image.width() / 2, self.image.height() / 2)  # Center rotation
        transform.rotate(angle)
        transform.translate(-self.image.width() / 2, -self.image.height() / 2)
    
        rotated_image = self.image.transformed(transform, Qt.SmoothTransformation)
    
        # Create a white canvas large enough to fit the rotated image
        canvas_width = rotated_image.width()
        canvas_height = rotated_image.height()
        high_res_canvas = QImage(canvas_width, canvas_height, QImage.Format_RGB888)
        high_res_canvas.fill(QColor(255, 255, 255))  # Fill with white background
    
        # Render the high-resolution canvas using `render_image_on_canvas`, without guides
        self.render_image_on_canvas(
            high_res_canvas,
            scaled_image=rotated_image,
            x_start=0,
            y_start=0,
            render_scale=1,  # Adjust scale if needed
            draw_guides=False  # Do not draw guides in this case
        )
    
        # Update the main high-resolution image
        self.image = high_res_canvas
        self.image_before_padding = self.image
        self.image_contrasted=self.image.copy()
    
        # Create a low-resolution preview for display in `live_view_label`
        preview = high_res_canvas.scaled(
            self.live_view_label.width(),
            self.live_view_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.live_view_label.setPixmap(QPixmap.fromImage(preview))
    
        # Reset the orientation slider
        self.orientation_slider.setValue(0)
    
    # Modify update_crop to include alignment and configuration preservation
    def update_crop(self):
        """Update the image based on current crop sliders. First align, then crop the image."""
        # Save current configuration
        config = self.get_current_config()
    
        # Align the image first (rotate it)
        self.align_image()
    
        # Now apply cropping
        cropped_image = self.crop_image()
    
        if cropped_image:
            self.image = cropped_image
            self.image_before_padding = self.image
            self.image_contrasted=self.image.copy()
            self.update_live_view()
    
        # Reapply the saved configuration
        self.apply_config(config)
    
        # Reset sliders
        self.crop_x_start_slider.setValue(0)
        self.crop_x_end_slider.setValue(100)
        self.crop_y_start_slider.setValue(0)
        self.crop_y_end_slider.setValue(100)
        self.image_contrasted = self.image
        
    def save_image(self):
        if not self.image:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
    
        options = QFileDialog.Options()
        base_save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)", options=options
        )
        if base_save_path:
            # Check if the base name already contains "_original"
            if "_original" not in os.path.splitext(base_save_path)[0]:
                original_save_path = os.path.splitext(base_save_path)[0] + "_original.png"
                modified_save_path = os.path.splitext(base_save_path)[0] + "_modified.png"
                config_save_path = os.path.splitext(base_save_path)[0] + "_config.txt"
            else:
                original_save_path = base_save_path
                modified_save_path = os.path.splitext(base_save_path)[0].replace("_original", "") + "_modified.png"
                config_save_path = os.path.splitext(base_save_path)[0].replace("_original", "") + "_config.txt"
    
                # Check if "_modified" and "_config" already exist, and overwrite them if so
                if os.path.exists(modified_save_path):
                    os.remove(modified_save_path)
                if os.path.exists(config_save_path):
                    os.remove(config_save_path)
    
            # Save original image (cropped and aligned)
            self.original_image = self.image.copy()  # Save the current (cropped and aligned) image as original
            if not self.original_image.save(original_save_path):
                QMessageBox.warning(self, "Error", f"Failed to save original image to {original_save_path}.")
    
            # Save modified image
            render_scale = 3
            high_res_canvas_width = self.live_view_label.width() * render_scale
            high_res_canvas_height = self.live_view_label.height() * render_scale
            high_res_canvas = QImage(
                high_res_canvas_width, high_res_canvas_height, QImage.Format_RGB888
            )
            high_res_canvas.fill(QColor(255, 255, 255))  # White background
    
            # Define cropping boundaries
            x_start_percent = self.crop_x_start_slider.value() / 100
            y_start_percent = self.crop_y_start_slider.value() / 100
            x_start = int(self.image.width() * x_start_percent)
            y_start = int(self.image.height() * y_start_percent)
    
            # Create a scaled version of the image
            scaled_image = self.image.scaled(
                high_res_canvas_width,
                high_res_canvas_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
    
            # Render the high-resolution canvas without guides for saving
            self.render_image_on_canvas(
                high_res_canvas, scaled_image, x_start, y_start, render_scale, draw_guides=False
            )
    
            if not high_res_canvas.save(modified_save_path):
                QMessageBox.warning(self, "Error", f"Failed to save modified image to {modified_save_path}.")
    
            # Save configuration to a .txt file
            config_data = self.get_current_config()
            try:
                with open(config_save_path, "w") as config_file:
                    json.dump(config_data, config_file, indent=4)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save config file: {e}")
    
            QMessageBox.information(self, "Saved", f"Files saved successfully.")

    
    
    def copy_to_clipboard(self):
        """Copy the image from live view label to the clipboard."""
        if not self.image:
            print("No image to copy.")
            return
    
        # Define a high-resolution canvas for clipboard copy
        render_scale = 3
        high_res_canvas_width = self.live_view_label.width() * render_scale
        high_res_canvas_height = self.live_view_label.height() * render_scale
        high_res_canvas = QImage(
            high_res_canvas_width, high_res_canvas_height, QImage.Format_RGB888
        )
        high_res_canvas.fill(QColor(255, 255, 255))  # White background
    
        # Define cropping boundaries
        x_start_percent = self.crop_x_start_slider.value() / 100
        y_start_percent = self.crop_y_start_slider.value() / 100
        x_start = int(self.image.width() * x_start_percent)
        y_start = int(self.image.height() * y_start_percent)
    
        # Create a scaled version of the image
        scaled_image = self.image.scaled(
            high_res_canvas_width,
            high_res_canvas_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    
        # Render the high-resolution canvas without guides for clipboard
        self.render_image_on_canvas(
            high_res_canvas, scaled_image, x_start, y_start, render_scale, draw_guides=False
        )
        
        # Copy the high-resolution image to the clipboard
        clipboard = QApplication.clipboard()
        clipboard.setImage(high_res_canvas)  # Copy the rendered image
        print("Image copied to clipboard.")
    
    def reset_image(self):
        # Reset the image to original
        self.image_array_backup= None
        if self.image != None:
            self.image = self.image_master.copy()
            self.image_before_padding = self.image.copy()
            self.reset_gamma_contrast()
        self.left_markers.clear()  # Clear left markers
        self.right_markers.clear()  # Clear right markers
        self.top_markers.clear()
        self.update_live_view()  # Update the live view
        self.crop_x_start_slider.setValue(0)
        self.crop_x_end_slider.setValue(100)
        self.crop_y_start_slider.setValue(0)
        self.crop_y_end_slider.setValue(100)
        self.orientation_slider.setValue(0)
        # self.top_padding_slider.setValue(0)
        # self.left_padding_slider.setValue(0)
        # self.right_padding_slider.setValue(0)
        self.marker_mode = None
        self.current_marker_index = 0
        self.current_top_label_index = 0

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CombinedSDSApp()
    window.show()
    sys.exit(app.exec_())
