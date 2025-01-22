
from PIL import ImageGrab  # Import Pillow's ImageGrab for clipboard access
import sys
from io import BytesIO
from PyQt5.QtWidgets import (
    QDesktopWidget, QScrollArea, QSizePolicy, QMainWindow, QApplication, QTabWidget, QLabel, QPushButton, QVBoxLayout, QTextEdit, QHBoxLayout, QCheckBox, QGroupBox, QGridLayout, QWidget, QFileDialog, QSlider, QComboBox, QColorDialog, QMessageBox, QLineEdit, QFontComboBox, QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QKeySequence, QClipboard, QPen, QTransform
from PyQt5.QtCore import Qt, QBuffer
import json
import os
import numpy as np
import matplotlib.pyplot as plt



class LiveViewLabel(QLabel):
    def __init__(self, font_type, font_size, marker_color, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # Enable mouse tracking
        self.preview_marker_enabled = False
        self.preview_marker_text = ""
        self.preview_marker_position = None
        self.marker_font_type = font_type
        self.marker_font_size = font_size
        self.marker_color = marker_color
        self.setFocusPolicy(Qt.StrongFocus)

    def mouseMoveEvent(self, event):
        if self.preview_marker_enabled:
            self.preview_marker_position = event.pos()
            self.update()  # Trigger repaint to show the preview
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.preview_marker_enabled:
            # Place the marker text permanently at the clicked position
            parent = self.parent()
            parent.place_custom_marker(event, self.preview_marker_text)
            # self.preview_marker_enabled = False  # Disable the preview after placing
            self.update()  # Clear the preview
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.preview_marker_enabled and self.preview_marker_position:
            painter = QPainter(self)
            painter.setOpacity(0.5)  # Semi-transparent preview
            font = QFont(self.marker_font_type)
            font.setPointSize(self.marker_font_size)
            painter.setFont(font)
            painter.setPen(self.marker_color)
            text_width = painter.fontMetrics().horizontalAdvance(self.preview_marker_text)
            text_height = painter.fontMetrics().height()
            # Draw the text at the cursor's position
            x, y = self.preview_marker_position.x(), self.preview_marker_position.y()
            painter.drawText(int(x - text_width/2), int(y + text_height/4), self.preview_marker_text)
            painter.end()
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.preview_marker_enabled:
                self.preview_marker_enabled = False  # Turn off the preview
                self.update()  # Clear the overlay
        super().keyPressEvent(event)

class CombinedSDSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.screen = QDesktopWidget().screenGeometry()
        self.screen_width, self.screen_height = self.screen.width(), self.screen.height()
        window_width = int(self.screen_width * 0.5)  # 60% of screen width
        window_height = int(self.screen_height * 0.75)  # 95% of screen height
        self.setWindowTitle("IMAGING ASSISTANT V3")
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        # self.resize(window_width, window_height)
        # self.setFixedSize(window_width, window_height)
        # self.setFixedWidth(window_width)
        # self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.setFixedSize(QSizePolicy.Fixed,QSizePolicy.Fixed)
        # self.resize(700, 950) # Change for windows/macos viewing
        self.image_path = None
        self.image = None
        self.image_master= None
        self.image_before_padding = None
        self.image_contrasted=None
        self.image_before_contrast=None
        self.contrast_applied=False
        self.image_padded=False
        self.left_markers = []
        self.right_markers = []
        self.top_markers = []
        self.custom_markers=[]
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
        self.right_marker_shift = 0   # Additional shift for marker tex
        self.top_marker_shift=0
        self.right_marker_shift_added=0
        self.top_marker_shift_added= 0
        self.top_padding = 0
        self.font_color = QColor(0, 0, 0)  # Default to black
        self.custom_marker_color = QColor(0, 0, 0)  # Default to black
        self.font_family = "Arial"  # Default font family
        self.font_size = 12  # Default font size
        self.image_array_backup= None
        self.run_predict_MW=False
        

        # Main container widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Upper section (Preview and buttons)
        upper_layout = QHBoxLayout()

        # Preview window
        if (int(self.screen_width * 0.3))<540:
            label_width = 540
            label_height = 409
        else:
            label_width = int(self.screen_width * 0.3)  # 30% of screen width
            label_height = int(self.screen_height * 0.35)  # 40% of screen height
    
        self.live_view_label = LiveViewLabel(
            font_type=QFont("Arial"),
            font_size=int(24),
            marker_color=QColor(0,0,0),
            parent=self,
        )
        # Image display
        self.live_view_label.setStyleSheet("border: 1px solid black;")
        # self.live_view_label.setCursor(Qt.CrossCursor)
        self.live_view_label.setFixedSize(label_width, label_height)
        # self.live_view_label.mousePressEvent = self.add_band()
        # self.live_view_label.mousePressEvent = self.add_band
        
        
       

        # Buttons for image loading and saving
        # Load, save, and crop buttons
        buttons_layout = QVBoxLayout()
        load_button = QPushButton("Load Image")
        load_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand width
        load_button.clicked.connect(self.load_image)
        buttons_layout.addWidget(load_button)
        
        paste_button = QPushButton('Paste Image')
        paste_button.clicked.connect(self.paste_image)  # Connect the button to the paste_image method
        paste_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand width
        buttons_layout.addWidget(paste_button)
    
        
        reset_button = QPushButton("Reset Image")  # Add Reset Image button
        reset_button.clicked.connect(self.reset_image)  # Connect the reset functionality
        reset_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand width
        buttons_layout.addWidget(reset_button)
        
        
        
        copy_button = QPushButton('Copy Image to Clipboard')
        copy_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand width
        copy_button.clicked.connect(self.copy_to_clipboard)
        buttons_layout.addWidget(copy_button)
        
        save_button = QPushButton("Save Processed Image")
        save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand width
        save_button.clicked.connect(self.save_image)
        buttons_layout.addWidget(save_button)
        

        predict_button = QPushButton("Predict Molecular Weight")
        predict_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand width
        predict_button.setEnabled(False)  # Initially disabled
        predict_button.clicked.connect(self.predict_molecular_weight)
        buttons_layout.addWidget(predict_button)
        self.predict_button = predict_button
        
        clear_predict_button = QPushButton("Clear Prediction Marker")
        clear_predict_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expand width
        clear_predict_button.setEnabled(True)  # Initially disabled
        clear_predict_button.clicked.connect(self.clear_predict_molecular_weight)
        buttons_layout.addWidget(clear_predict_button)
        buttons_layout.addStretch()
        
        
        upper_layout.addWidget(self.live_view_label)
        upper_layout.addLayout(buttons_layout)
        layout.addLayout(upper_layout)

        # Lower section (Tabbed interface)
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.font_and_image_tab(), "Font and Image Parameters")
        self.tab_widget.addTab(self.create_cropping_tab(), "Crop and Align Parameters")
        self.tab_widget.addTab(self.create_white_space_tab(), "White Space Parameters")
        self.tab_widget.addTab(self.create_markers_tab(), "Marker Parameters")
        
        layout.addWidget(self.tab_widget)
        self.load_config()
        
    
    def font_and_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Group Box for Font Options
        font_options_group = QGroupBox("Font Options")
        font_options_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        font_options_layout = QGridLayout()  # Use a grid layout for compact arrangement
    
        # Font type selection (QFontComboBox)
        font_type_label = QLabel("Font Type:")
        self.font_combo_box = QFontComboBox()
        self.font_combo_box.setEditable(False)
        self.font_combo_box.setCurrentFont(QFont("Arial"))
        font_options_layout.addWidget(font_type_label, 0, 0)  # Row 0, Column 0
        font_options_layout.addWidget(self.font_combo_box, 0, 1, 1, 2)  # Row 0, Column 1-2
    
        # Font size selection (QSpinBox)
        font_size_label = QLabel("Font Size:")
        self.font_size_spinner = QSpinBox()
        self.font_size_spinner.setRange(8, 72)  # Set a reasonable font size range
        self.font_size_spinner.setValue(12)  # Default font size
        font_options_layout.addWidget(font_size_label, 1, 0)  # Row 1, Column 0
        font_options_layout.addWidget(self.font_size_spinner, 1, 1)  # Row 1, Column 1
    
        # Font color selection (QPushButton to open QColorDialog)
        self.font_color_button = QPushButton("Font Color")
        self.font_color_button.clicked.connect(self.select_font_color)
        font_options_layout.addWidget(self.font_color_button, 1, 2)  # Row 1, Column 2
    
        # Font rotation input (QSpinBox)
        font_rotation_label = QLabel("Font Rotation (Top/Bottom Label):")
        self.font_rotation_input = QSpinBox()
        self.font_rotation_input.setRange(-180, 180)  # Set a reasonable font rotation range
        self.font_rotation_input.setValue(-45)  # Default rotation
        self.font_rotation_input.valueChanged.connect(self.update_font)
        font_options_layout.addWidget(font_rotation_label, 2, 0)  # Row 2, Column 0
        font_options_layout.addWidget(self.font_rotation_input, 2, 1, 1, 2)  # Row 2, Column 1-2
    
        # Set the layout for the font options group box
        font_options_group.setLayout(font_options_layout)
        layout.addWidget(font_options_group)
    
        # Group Box for Contrast and Gamma Adjustments
        contrast_gamma_group = QGroupBox("Contrast and Gamma Adjustments")
        contrast_gamma_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        contrast_gamma_layout = QGridLayout()  # Use a grid layout for a clean look
    
        # Bright Region Contrast Slider
        high_contrast_label = QLabel("Bright Region Contrast:")
        self.high_slider = QSlider(Qt.Horizontal)
        self.high_slider.setRange(0, 100)  # Range for contrast adjustment (0%-200%)
        self.high_slider.setValue(100)  # Default value (100% = no change)
        self.high_slider.valueChanged.connect(self.update_image_contrast)
        contrast_gamma_layout.addWidget(high_contrast_label, 0, 0)
        contrast_gamma_layout.addWidget(self.high_slider, 0, 1, 1, 2)
    
        # Dark Region Contrast Slider
        low_contrast_label = QLabel("Dark Region Contrast:")
        self.low_slider = QSlider(Qt.Horizontal)
        self.low_slider.setRange(0, 100)  # Range for contrast adjustment (0%-200%)
        self.low_slider.setValue(0)  # Default value (100% = no change)
        self.low_slider.valueChanged.connect(self.update_image_contrast)
        contrast_gamma_layout.addWidget(low_contrast_label, 1, 0)
        contrast_gamma_layout.addWidget(self.low_slider, 1, 1, 1, 2)
    
        # Gamma Adjustment Slider
        gamma_label = QLabel("Gamma Adjustment:")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(0, 200)  # Range for gamma adjustment (0%-200%)
        self.gamma_slider.setValue(100)  # Default value (100% = no change)
        self.gamma_slider.valueChanged.connect(self.update_image_gamma)
        contrast_gamma_layout.addWidget(gamma_label, 2, 0)
        contrast_gamma_layout.addWidget(self.gamma_slider, 2, 1, 1, 2)
        
        #Invert the image
        invert_button = QPushButton("Invert Image")
        invert_button.clicked.connect(self.invert_image)
        contrast_gamma_layout.addWidget(invert_button, 3, 0, 1, 3)
    
        # Reset Button
        reset_button = QPushButton("Reset Contrast and Gamma")
        reset_button.clicked.connect(self.reset_gamma_contrast)
        contrast_gamma_layout.addWidget(reset_button, 4, 0, 1, 3)  # Span all columns
        
        # Connect signals for dynamic updates
        self.font_combo_box.currentFontChanged.connect(self.update_font)
        self.font_size_spinner.valueChanged.connect(self.update_font)
    
        # Set the layout for the contrast and gamma group box
        contrast_gamma_group.setLayout(contrast_gamma_layout)
        layout.addWidget(contrast_gamma_group)
    
        # Add stretch at the end for proper spacing
        layout.addStretch()
    
        return tab
    
    def create_cropping_tab(self):
        """Create the Cropping tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
    
        # Group Box for Alignment Options
        alignment_params_group = QGroupBox("Alignment Options")
        alignment_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        alignment_layout = QVBoxLayout()
    
        # Rotation Angle and Guide Lines
        rotation_layout = QHBoxLayout()
        self.orientation_label = QLabel("Rotation Angle (Degrees)")
        self.orientation_slider = QSlider(Qt.Horizontal)
        self.orientation_slider.setRange(-180, 180)
        self.orientation_slider.setValue(0)
        self.orientation_slider.valueChanged.connect(self.update_live_view)
    
        self.show_guides_checkbox = QCheckBox("Show Guide Lines", self)
        self.show_guides_checkbox.setChecked(False)
        self.show_guides_checkbox.stateChanged.connect(self.update_live_view)
    
        rotation_layout.addWidget(self.orientation_label)
        rotation_layout.addWidget(self.orientation_slider)
        rotation_layout.addWidget(self.show_guides_checkbox)
    
        # Align Button
        self.align_button = QPushButton("Align Image")
        self.align_button.clicked.connect(self.align_image)
    
        alignment_layout.addLayout(rotation_layout)
        alignment_layout.addWidget(self.align_button)
        alignment_params_group.setLayout(alignment_layout)
    
        # Group Box for Cropping Options
        cropping_params_group = QGroupBox("Cropping Options")
        cropping_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        cropping_layout = QVBoxLayout()
    
        # Cropping Sliders
        crop_slider_layout = QGridLayout()
    
        crop_x_start_label = QLabel("Crop Left (%)")
        self.crop_x_start_slider = QSlider(Qt.Horizontal)
        self.crop_x_start_slider.setRange(0, 100)
        self.crop_x_start_slider.setValue(0)
        self.crop_x_start_slider.valueChanged.connect(self.update_live_view)
        crop_slider_layout.addWidget(crop_x_start_label, 0, 0)
        crop_slider_layout.addWidget(self.crop_x_start_slider, 0, 1)
    
        crop_x_end_label = QLabel("Crop Right (%)")
        self.crop_x_end_slider = QSlider(Qt.Horizontal)
        self.crop_x_end_slider.setRange(0, 100)
        self.crop_x_end_slider.setValue(100)
        self.crop_x_end_slider.valueChanged.connect(self.update_live_view)
        crop_slider_layout.addWidget(crop_x_end_label, 0, 2)
        crop_slider_layout.addWidget(self.crop_x_end_slider, 0, 3)
    
        crop_y_start_label = QLabel("Crop Top (%)")
        self.crop_y_start_slider = QSlider(Qt.Horizontal)
        self.crop_y_start_slider.setRange(0, 100)
        self.crop_y_start_slider.setValue(0)
        self.crop_y_start_slider.valueChanged.connect(self.update_live_view)
        crop_slider_layout.addWidget(crop_y_start_label, 1, 0)
        crop_slider_layout.addWidget(self.crop_y_start_slider, 1, 1)
    
        crop_y_end_label = QLabel("Crop Bottom (%)")
        self.crop_y_end_slider = QSlider(Qt.Horizontal)
        self.crop_y_end_slider.setRange(0, 100)
        self.crop_y_end_slider.setValue(100)
        self.crop_y_end_slider.valueChanged.connect(self.update_live_view)
        crop_slider_layout.addWidget(crop_y_end_label, 1, 2)
        crop_slider_layout.addWidget(self.crop_y_end_slider, 1, 3)
    
        cropping_layout.addLayout(crop_slider_layout)
    
        # Crop Update Button
        crop_button = QPushButton("Update Crop")
        crop_button.clicked.connect(self.update_crop)
        cropping_layout.addWidget(crop_button)
    
        cropping_params_group.setLayout(cropping_layout)
    
        # Add both group boxes to the main layout
        layout.addWidget(alignment_params_group)
        layout.addWidget(cropping_params_group)
        layout.addStretch()
    
        return tab
    
    def create_white_space_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
    
        # Group Box for Padding Parameters
        padding_params_group = QGroupBox("Add White Space to Image for Placing Markers")
        padding_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        padding_layout = QGridLayout()  # Use a grid layout for better alignment
    
        # Left space input and label
        left_padding_label = QLabel("Left Padding (px):")
        self.left_padding_input = QLineEdit()
        self.left_padding_input.setText("100")  # Default value
        self.left_padding_input.setPlaceholderText("Enter padding for the left side")
        self.left_padding_input.setToolTip("Enter the number of pixels to add to the left of the image.")
        padding_layout.addWidget(left_padding_label, 0, 0)
        padding_layout.addWidget(self.left_padding_input, 0, 1)
    
        # Right space input and label
        right_padding_label = QLabel("Right Padding (px):")
        self.right_padding_input = QLineEdit()
        self.right_padding_input.setText("100")  # Default value
        self.right_padding_input.setPlaceholderText("Enter padding for the right side")
        self.right_padding_input.setToolTip("Enter the number of pixels to add to the right of the image.")
        padding_layout.addWidget(right_padding_label, 0, 2)
        padding_layout.addWidget(self.right_padding_input, 0, 3)
    
        # Top space input and label
        top_padding_label = QLabel("Top Padding (px):")
        self.top_padding_input = QLineEdit()
        self.top_padding_input.setText("100")  # Default value
        self.top_padding_input.setPlaceholderText("Enter padding for the top")
        self.top_padding_input.setToolTip("Enter the number of pixels to add to the top of the image.")
        padding_layout.addWidget(top_padding_label, 1, 0)
        padding_layout.addWidget(self.top_padding_input, 1, 1)
    
        # Bottom space input and label
        bottom_padding_label = QLabel("Bottom Padding (px):")
        self.bottom_padding_input = QLineEdit()
        self.bottom_padding_input.setText("50")  # Default value
        self.bottom_padding_input.setPlaceholderText("Enter padding for the bottom")
        self.bottom_padding_input.setToolTip("Enter the number of pixels to add to the bottom of the image.")
        padding_layout.addWidget(bottom_padding_label, 1, 2)
        padding_layout.addWidget(self.bottom_padding_input, 1, 3)
    
        # Buttons for Finalize and Reset
        button_layout = QHBoxLayout()
        self.finalize_button = QPushButton("Add Padding")
        self.finalize_button.clicked.connect(self.finalize_image)
        self.finalize_button.setToolTip("Click to apply the specified padding to the image.")
        button_layout.addWidget(self.finalize_button)
    
        self.reset_padding_button = QPushButton("Remove Padding")
        self.reset_padding_button.clicked.connect(self.remove_padding)
        self.reset_padding_button.setToolTip("Click to remove all added padding and revert the image.")
        button_layout.addWidget(self.reset_padding_button)
    
        # Add padding layout and buttons to the group box
        padding_params_group.setLayout(padding_layout)
    
        # Add group box and buttons to the main layout
        layout.addWidget(padding_params_group)
        layout.addLayout(button_layout)
    
        # Add stretch for spacing
        layout.addStretch()
    
        return tab
        
        
    def create_markers_tab(self):
        """Create the Markers tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
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
        top_marker_group = QGroupBox("Top/Bottom Marker Options")
        top_marker_group.setStyleSheet("QGroupBox { font-weight: bold;}")
        
        # Vertical layout for top marker group
        top_marker_layout = QVBoxLayout()
        
        
        # Text input for Top Marker Labels
        self.top_marker_input = QTextEdit(self)
        self.top_label = ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"]
        self.top_marker_input.setText(", ".join(self.top_label))  # Populate with initial values
        self.top_marker_input.setMinimumHeight(40)
        
        # Button to update Top Marker Labels
        self.update_top_labels_button = QPushButton("Update All Labels")
        self.update_top_labels_button.clicked.connect(self.update_top_labels)
        
        # Add widgets to the top marker layout
        top_marker_layout.addWidget(self.top_marker_input)
        top_marker_layout.addWidget(self.update_top_labels_button)
        
        # Set the layout for the Top Marker Group
        top_marker_group.setLayout(top_marker_layout)
        
        # Add the group box to the main layout
        
        
        # Add both groups to the horizontal layout
        marker_options_layout.addWidget(left_right_marker_group)
        marker_options_layout.addWidget(top_marker_group)
        
        # Add the horizontal layout to the main layout
        layout.addLayout(marker_options_layout)

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
        
        remove_left_button = QPushButton("Remove Last")
        remove_left_button.clicked.connect(lambda: self.reset_marker('left','remove'))
        
        reset_left_button = QPushButton("Reset")
        reset_left_button.clicked.connect(lambda: self.reset_marker('left','reset'))
        
        duplicate_left_button = QPushButton("Duplicate Right")
        duplicate_left_button.clicked.connect(lambda: self.duplicate_marker('left'))
        
        # Add left marker widgets to the grid layout
        padding_layout.addWidget(left_marker_button, 0, 0)
        padding_layout.addWidget(remove_left_button, 0, 1)
        padding_layout.addWidget(reset_left_button, 0, 2)
        padding_layout.addWidget(duplicate_left_button, 0, 3)
        padding_layout.addWidget(self.left_padding_slider, 0, 4,1,2)
        
        # Right marker: Button, slider, reset, and duplicate in the same row
        right_marker_button = QPushButton("Right Markers")
        right_marker_button.clicked.connect(self.enable_right_marker_mode)
        self.right_padding_slider = QSlider(Qt.Horizontal)
        self.right_padding_slider.setRange(-1000, 1000)
        self.right_padding_slider.setValue(0)
        self.right_padding_slider.valueChanged.connect(self.update_right_padding)
        
        remove_right_button = QPushButton("Remove Last")
        remove_right_button.clicked.connect(lambda: self.reset_marker('right','remove'))
        
        reset_right_button = QPushButton("Reset")
        reset_right_button.clicked.connect(lambda: self.reset_marker('right','reset'))
        
        duplicate_right_button = QPushButton("Duplicate Left")
        duplicate_right_button.clicked.connect(lambda: self.duplicate_marker('right'))
        
        # Add right marker widgets to the grid layout
        padding_layout.addWidget(right_marker_button, 1, 0)
        padding_layout.addWidget(remove_right_button, 1, 1)
        padding_layout.addWidget(reset_right_button, 1, 2)
        padding_layout.addWidget(duplicate_right_button, 1, 3)
        padding_layout.addWidget(self.right_padding_slider, 1, 4,1,2)
        
        # Top marker: Button, slider, and reset in the same row
        top_marker_button = QPushButton("Top Markers")
        top_marker_button.clicked.connect(self.enable_top_marker_mode)
        self.top_padding_slider = QSlider(Qt.Horizontal)
        self.top_padding_slider.setRange(-1000, 1000)
        self.top_padding_slider.setValue(0)
        self.top_padding_slider.valueChanged.connect(self.update_top_padding)
        
        remove_top_button = QPushButton("Remove Last")
        remove_top_button.clicked.connect(lambda: self.reset_marker('top','remove'))
        
        reset_top_button = QPushButton("Reset")
        reset_top_button.clicked.connect(lambda: self.reset_marker('top','reset'))
        
        # Add top marker widgets to the grid layout
        padding_layout.addWidget(top_marker_button, 2, 0)
        padding_layout.addWidget(remove_top_button, 2, 1)
        padding_layout.addWidget(reset_top_button, 2, 2)
        padding_layout.addWidget(self.top_padding_slider, 2, 3, 1, 3)  # Slider spans 2 columns for better alignment
        
        for i in range(6):  # Assuming 6 columns in the grid
            padding_layout.setColumnStretch(i, 1)
        
        # Add button and QLineEdit for the custom marker
        self.custom_marker_button = QPushButton("Custom Marker", self)
        self.custom_marker_button.clicked.connect(self.enable_custom_marker_mode)
        
        self.custom_marker_button_left_arrow = QPushButton("←", self)
        
        self.custom_marker_button_right_arrow = QPushButton("→", self)
        
        self.custom_marker_button_top_arrow = QPushButton("↑", self)
        
        self.custom_marker_button_bottom_arrow = QPushButton("↓", self)
        
        
        
        self.custom_marker_text_entry = QLineEdit(self)        
        self.custom_marker_text_entry.setPlaceholderText("Enter custom marker text")
        # self.custom_marker_text_entry.textChanged.connect(self.enable_custom_marker_mode)
        
        self.remove_custom_marker_button = QPushButton("Remove Last", self)
        self.remove_custom_marker_button.clicked.connect(self.remove_custom_marker_mode)
        
        # Add color selection button for custom markers
        self.custom_marker_color_button = QPushButton("Custom Marker Color")
        self.custom_marker_color_button.clicked.connect(self.select_custom_marker_color)
        
        marker_buttons_layout = QHBoxLayout()
        
        # Add the arrow buttons with fixed sizes to the marker buttons layout
        self.custom_marker_button_left_arrow.setFixedSize(30, 30)

        marker_buttons_layout.addWidget(self.custom_marker_button_left_arrow)
        
        self.custom_marker_button_right_arrow.setFixedSize(30, 30)
        marker_buttons_layout.addWidget(self.custom_marker_button_right_arrow)
        
        self.custom_marker_button_top_arrow.setFixedSize(30, 30)
        marker_buttons_layout.addWidget(self.custom_marker_button_top_arrow)
        
        self.custom_marker_button_bottom_arrow.setFixedSize(30, 30)
        marker_buttons_layout.addWidget(self.custom_marker_button_bottom_arrow)
        
        
        #Assign functions to the buttons
        self.custom_marker_button_left_arrow.clicked.connect(lambda: self.arrow_marker("←"))
        self.custom_marker_button_right_arrow.clicked.connect(lambda: self.arrow_marker("→"))
        self.custom_marker_button_top_arrow.clicked.connect(lambda: self.arrow_marker("↑"))
        self.custom_marker_button_bottom_arrow.clicked.connect(lambda: self.arrow_marker("↓"))

        
        # Create a QWidget to hold the QHBoxLayout
        marker_buttons_widget = QWidget()
        marker_buttons_widget.setLayout(marker_buttons_layout)
        
        # Add the custom marker button
        padding_layout.addWidget(self.custom_marker_button, 3, 0)
        
        # Add the text entry for the custom marker
        padding_layout.addWidget(self.custom_marker_text_entry, 3, 1,1,2)
        
        # Add the marker buttons widget to the layout
        padding_layout.addWidget(marker_buttons_widget, 3, 3)  
        
        # Add the remove button
        padding_layout.addWidget(self.remove_custom_marker_button, 3, 4)
        
        # Add the color button
        padding_layout.addWidget(self.custom_marker_color_button, 3, 5)
        
        self.custom_font_type_label = QLabel("Custom Marker Font:", self)
        self.custom_font_type_dropdown = QFontComboBox()
        self.custom_font_type_dropdown.setCurrentFont(QFont("Arial"))
        self.custom_font_type_dropdown.currentFontChanged.connect(self.update_marker_text_font)
        # self.font_type_dropdown.currentIndexChanged.connect(self.update_font_type)
        
        # Font size selector
        self.custom_font_size_label = QLabel("Custom Marker Size:", self)
        self.custom_font_size_spinbox = QSpinBox(self)
        self.custom_font_size_spinbox.setRange(8, 72)  # Allow font sizes from 8 to 72
        self.custom_font_size_spinbox.setValue(12)  # Default font size
        # self.font_size_spinbox.valueChanged.connect(self.update_font_size)
        
        # Grid checkbox
        self.show_grid_checkbox = QCheckBox("Show Snap Grid", self)
        self.show_grid_checkbox.setChecked(False)  # Default: Grid is off
        self.show_grid_checkbox.stateChanged.connect(self.update_live_view)
        
        # Grid size input (optional
        self.grid_size_input = QSpinBox(self)
        self.grid_size_input.setRange(5, 100)  # Grid cell size in pixels
        self.grid_size_input.setValue(20)  # Default grid size
        self.grid_size_input.setPrefix("Grid Size (px): ")
        self.grid_size_input.valueChanged.connect(self.update_live_view)
        
        # Add font type and size widgets to the layout
        padding_layout.addWidget(self.custom_font_type_label, 4, 0,)
        padding_layout.addWidget(self.custom_font_type_dropdown, 4, 1,1,1)  # Span 2 columns
        
        padding_layout.addWidget(self.custom_font_size_label, 4, 2)
        padding_layout.addWidget(self.custom_font_size_spinbox, 4, 3,1,1)
        padding_layout.addWidget(self.show_grid_checkbox,4,4)
        padding_layout.addWidget(self.grid_size_input,4,5)

        
        
        # Set the layout for the marker group box
        padding_params_group.setLayout(padding_layout)
        
                
        # Add the font options group box to the main layout
        layout.addWidget(padding_params_group)
        
        layout.addStretch()
        return tab

    def invert_image(self):
        if self.image:
            inverted_image = self.image.copy()
            inverted_image.invertPixels()
            self.image = inverted_image
            self.update_live_view()
        self.image_before_contrast=self.image.copy()
        self.image_before_padding=self.image.copy()
        self.image_contrasted=self.image.copy()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.live_view_label.setCursor(Qt.ArrowCursor)
        super().keyPressEvent(event)
    
    def update_marker_text_font(self, font: QFont):
        """
        Updates the font of self.custom_marker_text_entry based on the selected font
        from self.custom_font_type_dropdown.
    
        :param font: QFont object representing the selected font from the combobox.
        """
        # Get the name of the selected font
        selected_font_name = font.family()
        
        # Set the font of the QLineEdit
        self.custom_marker_text_entry.setFont(QFont(selected_font_name))
    
    def arrow_marker(self, text: str):
        self.custom_marker_text_entry.clear()
        self.custom_marker_text_entry.setText(text)
    
    def enable_custom_marker_mode(self):
        """Enable the custom marker mode and set the mouse event."""
        self.live_view_label.setCursor(Qt.ArrowCursor)
        custom_text = self.custom_marker_text_entry.text().strip()
    
        self.live_view_label.preview_marker_enabled = True
        self.live_view_label.preview_marker_text = custom_text
        self.live_view_label.marker_font_type=self.custom_font_type_dropdown.currentText()
        self.live_view_label.marker_font_size=self.custom_font_size_spinbox.value()
        self.live_view_label.marker_color=self.custom_marker_color
        
        self.live_view_label.setFocus()
        self.live_view_label.update()
        
        self.marker_mode = "custom"  # Indicate custom marker mode
        self.live_view_label.mousePressEvent = lambda event: self.place_custom_marker(event, custom_text)
        
    def remove_custom_marker_mode(self):
        if hasattr(self, "custom_markers") and isinstance(self.custom_markers, list) and self.custom_markers:
            self.custom_markers.pop()  # Remove the last entry from the list           
        self.update_live_view()  # Update the display
        
    
    def place_custom_marker(self, event, custom_text):
        """Place a custom marker at the cursor location."""
        # Get cursor position from the event
        pos = event.pos()
        cursor_x, cursor_y = pos.x(), pos.y()
    
        # Dimensions of the displayed image
        displayed_width = self.live_view_label.width()
        displayed_height = self.live_view_label.height()
    
        # Dimensions of the actual image
        image_width = self.image.width()
        image_height = self.image.height()
    
        # Calculate scaling factors
        scale = min(displayed_width / image_width, displayed_height / image_height)
    
        # Calculate offsets
        x_offset = (displayed_width - image_width * scale) / 2
        y_offset = (displayed_height - image_height * scale) / 2
    
        # Transform cursor position to image space
        image_x = (cursor_x - x_offset) / scale
        image_y = (cursor_y - y_offset) / scale
        
        # Adjust for snapping to grid
        if self.show_grid_checkbox.isChecked():
            grid_size = self.grid_size_input.value()
            cursor_x = round(cursor_x / grid_size) * grid_size
            cursor_y = round(cursor_y / grid_size) * grid_size
    

    
        # Store the custom marker's position and text
        self.custom_markers = getattr(self, "custom_markers", [])
        self.custom_markers.append((cursor_x, cursor_y, custom_text, self.custom_marker_color, self.custom_font_type_dropdown.currentText(), self.custom_font_size_spinbox.value()))
        # print("CUSTOM_MARKER: ",self.custom_markers)
        # Update the live view to render the custom marker
        self.update_live_view()
        
    def select_custom_marker_color(self):
        """Open a color picker dialog to select the color for custom markers."""
        color = QColorDialog.getColor(self.custom_marker_color, self, "Select Custom Marker Color")
        if color.isValid():
            self.custom_marker_color = color  # Update the custom marker color
    
    def update_top_labels(self):
        # Retrieve the multiline text from QTextEdit
        self.marker_values=[int(num) if num.strip().isdigit() else num.strip() for num in self.marker_values_textbox.text().strip("[]").split(",")]
        input_text = self.top_marker_input.toPlainText()
        
        # Split the text into a list by commas and strip whitespace
        self.top_label= [label.strip() for label in input_text.split(",") if label.strip()]
        try:
            
            # Ensure that the top_markers list only updates the top_label values serially
            if len(self.top_label) < len(self.top_markers):
                self.top_markers = self.top_markers[:len(self.top_label)]
            for i in range(0, len(self.top_markers)):
                self.top_markers[i] = (self.top_markers[i][0], self.top_label[i])
            
            # If self.top_label has more entries than current top_markers, add them
            # if len(self.top_label) > len(self.top_markers):
            #     additional_markers = [(self.top_markers[-1][0] + 50 * (i + 1), label) 
            #                           for i, label in enumerate(self.top_label[len(self.top_markers):])]
            #     self.top_markers.extend(additional_markers)
            
            # If self.top_label has fewer entries, truncate the list
            
        except:
            pass
        try:
            #min(len(self.left_markers)
            for i in range(0, len(self.left_markers)):
                self.left_markers[i] = (self.left_markers[i][0], self.marker_values[i])
            if len(self.marker_values) < len(self.left_markers):
                self.left_markers = self.left_markers[:len(self.marker_values)]
                #min(len(self.right_markers)
            for i in range(0, len(self.right_markers)):
                self.right_markers[i] = (self.right_markers[i][0], self.marker_values[i])
            if len(self.marker_values) < len(self.right_markers):
                self.right_markers = self.right_markers[:len(self.marker_values)]
                
        except:
            pass
        
        # Trigger a refresh of the live view
        self.update_live_view()
        
    def reset_marker(self, marker_type, param):
        if marker_type == 'left':
            if param == 'remove' and len(self.left_markers)!=0:
                self.left_markers.pop()  
            elif param == 'reset':
                self.left_markers.clear()
                self.current_marker_index = 0  

             
        elif marker_type == 'right' and len(self.right_markers)!=0:
            if param == 'remove':
                self.right_markers.pop()  
            elif param == 'reset':
                self.right_markers.clear()
                self.current_marker_index = 0

        elif marker_type == 'top' and len(self.top_markers)!=0:
            if param == 'remove':
                self.top_markers.pop()  
            elif param == 'reset':
                self.top_markers.clear()
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
        
    def on_combobox_changed(self):
        
        text=self.combo_box.currentText()
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
            self.top_label = [str(item) if not isinstance(item, str) else item for item in self.top_label]
            self.top_marker_input.setText(", ".join(self.top_label))
            try:
                
                # Ensure that the top_markers list only updates the top_label values serially
                for i in range(0, len(self.top_markers)):
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
        if self.image_before_contrast==None:
            self.image_before_contrast=self.image_master.copy()
        self.image_contrasted = self.image_before_contrast.copy()  # Update the contrasted image
        self.image_before_padding = self.image_before_contrast.copy()  # Ensure padding resets use the correct base
        self.high_slider.setValue(100)  # Reset contrast to default
        self.low_slider.setValue(0)  # Reset contrast to default
        self.gamma_slider.setValue(100)  # Reset gamma to default
        self.update_live_view()

    
    def update_image_contrast(self):
        if self.contrast_applied==False:
            self.image_before_contrast=self.image.copy()
            self.contrast_applied=True
        
        if self.image:
            high_contrast_factor = self.high_slider.value() / 100.0
            low_contrast_factor = self.low_slider.value() / 100.0
            gamma_factor = self.gamma_slider.value() / 100.0
            self.image = self.apply_contrast_gamma(self.image_contrasted, high_contrast_factor, low_contrast_factor, gamma=gamma_factor)  
            self.update_live_view()
    
    def update_image_gamma(self):
        if self.contrast_applied==False:
            self.image_before_contrast=self.image.copy()
            self.contrast_applied=True
            
        if self.image:
            high_contrast_factor = self.high_slider.value() / 100.0
            low_contrast_factor = self.low_slider.value() / 100.0
            gamma_factor = self.gamma_slider.value() / 100.0
            self.image = self.apply_contrast_gamma(self.image_contrasted, high_contrast_factor, low_contrast_factor, gamma=gamma_factor)            
            self.update_live_view()
    
    def apply_contrast_gamma(self, qimage, high, low, gamma):
        """
        Applies contrast and gamma adjustments to a QImage.
        Converts the QImage to RGBA format, performs the adjustments, and returns the modified QImage.
        """
        #Ensure the image is in the correct format (RGBA8888)
        if qimage.format() != QImage.Format_RGBA8888:
            qimage = qimage.convertToFormat(QImage.Format_RGBA8888)

        # Convert the QImage to a NumPy array
        width = qimage.width()
        height = qimage.height()

        # Get the raw byte data from QImage
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)

        # Create a NumPy array from the pointer (for RGBA format)
        img_array = np.array(ptr).reshape(height, width, 4).astype(np.float32)

        # Normalize the image for contrast and gamma adjustments
        img_array = img_array / 255.0

        # Apply contrast adjustment
        img_array = np.clip((img_array - low) / (high - low), 0, 1)

        # Apply gamma correction
        img_array = np.power(img_array, gamma)

        # Scale back to [0, 255] range
        img_array = (img_array * 255).astype(np.uint8)

        # Convert back to QImage
        qimage = QImage(img_array.data, width, height, img_array.strides[0], QImage.Format_RGBA8888)

        return qimage
    

    def save_contrast_options(self):
        if self.image:
            self.image_contrasted = self.image.copy()  # Save the current image as the contrasted image
            self.image_before_padding = self.image.copy()  # Ensure the pre-padding state is also updated
        else:
            QMessageBox.warning(self, "Error", "No image is loaded to save contrast options.")


    def save_config(self):
        """Rename the 'Custom' marker values option and save the configuration."""
        new_name = self.rename_input.text().strip()
        
        # Ensure that the correct dictionary (self.marker_values_dict) is being used
        if self.rename_input.text() != "Enter new name for Custom" and self.rename_input.text() != "":  # Correct condition
            # Save marker values list
            self.marker_values_dict[new_name] = [int(num) if num.strip().isdigit() else num.strip() for num in self.marker_values_textbox.text().strip("[]").split(",")]
            
            # Save top_label list under the new_name key
            self.top_label_dict[new_name] = [int(num) if num.strip().isdigit() else num.strip() for num in self.top_marker_input.toPlainText().strip("[]").split(",")]

            try:
                # Save both the marker values and top label (under new_name) to the config file
                with open("Imaging_assistant_config.txt", "w") as f:
                    config = {
                        "marker_values": self.marker_values_dict,
                        "top_label": self.top_label_dict  # Save top_label_dict as a dictionary with new_name as key
                    }
                    json.dump(config, f)  # Save both marker_values and top_label_dict
            except Exception as e:
                print(f"Error saving config: {e}")

        
        self.load_config()  # Reload the configuration after saving
    
    
    def load_config(self):
        """Load the configuration from the file."""
        try:
            with open("Imaging_assistant_config.txt", "r") as f:
                config = json.load(f)
                
                # Load marker values and top label from the file
                self.marker_values_dict = config.get("marker_values", {})
                
                # Retrieve top_label list from the dictionary using the new_name key
                new_name = self.rename_input.text().strip()  # Assuming `new_name` is defined here; otherwise, set it manually
                self.top_label_dict = config.get("top_label", {})  # Default if not found
                # self.top_label = self.top_label_dict.get(new_name, ["MWM", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "MWM"])  # Default if not found
                
        except FileNotFoundError:
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
    
    def paste_image(self):
        """Handle pasting image from clipboard."""
        self.load_image_from_clipboard()
        self.update_live_view()
    
    def load_image_from_clipboard(self):
        """Load an image from the clipboard into self.image."""
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        # print("Clipboard contains:")
        # if mime_data.hasImage():
        #     print("- An image (raw data)")
        # if mime_data.hasUrls():
        #     print("- URLs:", [url.toLocalFile() for url in mime_data.urls()])
        # if mime_data.hasText():
        #     print("- Text:", mime_data.text())
    
        # Check if the clipboard contains an image
        if mime_data.hasImage():
            image = clipboard.image()  # Get the image directly from the clipboard
            self.image = image  # Store the image in self.image
            self.original_image = self.image.copy()
            self.image_contrasted = self.image.copy()
            self.image_master = self.image.copy()
            self.image_before_padding = None
            self.update_live_view()
    
        # Check if the clipboard contains URLs (file paths)
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()  # Get the first file path
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    # Load the image from the file path
                    self.image = QImage(file_path)
                    self.original_image = self.image.copy()
                    self.image_contrasted = self.image.copy()
                    self.image_master = self.image.copy()
                    self.image_before_padding = None
                    self.update_live_view()
    
                    # Update the window title with the image path
                    self.setWindowTitle(f"IMAGING ASSISTANT V3: {file_path}")



            
        
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

            text_title="IMAGING ASSISTANT V3: "
            text_title+=str(self.image_path)
            self.setWindowTitle(text_title)
    
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
        try:
            self.left_padding_slider.setRange(-int(self.image.width()),int(self.image.width()))
            self.right_padding_slider.setRange(-int(self.image.width()),int(self.image.width()))
        except:
            pass
    
    def apply_config(self, config_data):
        self.left_padding_input.setText(config_data["adding_white_space"]["left"])
        self.right_padding_input.setText(config_data["adding_white_space"]["right"])
        self.top_padding_input.setText(config_data["adding_white_space"]["top"])
        try:
            self.bottom_padding_input.setText(config_data["adding_white_space"]["bottom"])
        except:
            pass
    
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
    
        try:
            self.custom_markers = [
                (marker["x"], marker["y"], marker["text"], QColor(marker["color"]), marker["font"], marker["font_size"])
                for marker in config_data.get("custom_markers", [])
            ]
        except:
            pass
        
        # Apply font settings

        
        #DO NOT KNOW WHY THIS WORKS BUT DIRECT VARIABLE ASSIGNING DOES NOT WORK
        
        font_size_new=self.font_size
        font_rotation_new=self.font_rotation
        
        self.font_combo_box.setCurrentFont(QFont(self.font_family))
        self.font_size_spinner.setValue(font_size_new)
        self.font_rotation_input.setValue(font_rotation_new)

        self.update_live_view()
        
    def get_current_config(self):
        config = {
            "adding_white_space": {
                "left": self.left_padding_input.text(),
                "right": self.right_padding_input.text(),
                "top": self.top_padding_input.text(),
                "bottom": self.bottom_padding_input.text(),
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
                "right": [marker[1] for marker in self.right_markers],
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
    
        try:
            # Add custom markers with font and font size
            config["custom_markers"] = [
                {"x": x, "y": y, "text": text, "color": color.name(), "font": font, "font_size": font_size}
                for x, y, text, color, font, font_size in self.custom_markers
            ]
        except AttributeError:
            # Handle the case where self.custom_markers is not defined or invalid
            config["custom_markers"] = []
    
        return config
    
    def add_band(self, event):
        # Ensure there's an image loaded and marker mode is active
        self.live_view_label.preview_marker_enabled = False
        self.live_view_label.preview_marker_text = ""
        if not self.image or not self.marker_mode:
            return
    
        # Get the cursor position from the event
        pos = event.pos()
        cursor_x, cursor_y = pos.x(), pos.y()
        
        if self.show_grid_checkbox.isChecked():
            grid_size = self.grid_size_input.value()
            cursor_x = round(cursor_x / grid_size) * grid_size
            cursor_y = round(cursor_y / grid_size) * grid_size
    
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
        try:
        # Add the band marker based on the active marker mode
            if self.marker_mode == "left" and self.current_marker_index < len(self.marker_values):
                if len(self.left_markers)!=0:
                    self.left_markers.append((image_y, self.marker_values[len(self.left_markers)]))
                    
                else:
                    self.left_markers.append((image_y, self.marker_values[self.current_marker_index]))
                    self.current_marker_index += 1
            elif self.marker_mode == "right" and self.current_marker_index < len(self.marker_values):
                if len(self.right_markers)!=0:
                    self.right_markers.append((image_y, self.marker_values[len(self.right_markers)]))
                else:
                    self.right_markers.append((image_y, self.marker_values[self.current_marker_index]))
                    self.current_marker_index += 1
            elif self.marker_mode == "top" and self.current_top_label_index < len(self.top_label):
                if len(self.top_markers)!=0:
                    self.top_markers.append((image_x, self.top_label[len(self.top_markers)]))
                else:
                    self.top_markers.append((image_x, self.top_label[self.current_top_label_index]))
                    self.current_top_label_index += 1
        except:
            pass            
        # Update the live view with the new markers
        self.update_live_view()
        

        
    def enable_left_marker_mode(self):
        self.marker_mode = "left"
        self.current_marker_index = 0
        self.live_view_label.mousePressEvent = self.add_band
        self.live_view_label.setCursor(Qt.CrossCursor)

    def enable_right_marker_mode(self):
        self.marker_mode = "right"
        self.current_marker_index = 0
        self.live_view_label.mousePressEvent = self.add_band
        self.live_view_label.setCursor(Qt.CrossCursor)
    
    def enable_top_marker_mode(self):
        self.marker_mode = "top"
        self.current_top_label_index
        self.live_view_label.mousePressEvent = self.add_band
        self.live_view_label.setCursor(Qt.CrossCursor)
        
    def remove_padding(self):
        self.image = self.image_before_padding.copy()  # Revert to the image before padding
        self.image_contrasted = self.image.copy()  # Sync the contrasted image
        self.image_padded = False  # Reset the padding state
        self.update_live_view()
        
    def finalize_image(self):
        # Get the padding values from the text inputs
        try:
            padding_left = int(self.left_padding_input.text())
            padding_right = int(self.right_padding_input.text())
            padding_top = int(self.top_padding_input.text())
            padding_bottom = int(self.bottom_padding_input.text())
        except ValueError:
            # Handle invalid input (non-integer value)
            # print("Please enter valid integers for padding.")
            return
        
        self.left_marker_shift=padding_left+20
        
        self.right_marker_shift = self.image.width()*0.75
        
        self.top_marker_shift_added=(padding_top-30)
        
    
        # Ensure self.image_before_padding is initialized
        if self.image_before_padding is None:
            self.image_before_padding = self.image_contrasted.copy()
    
        # Reset to the original image if padding inputs have changed
        if self.image_padded:
            self.image = self.image_before_padding.copy()
            self.image_padded = False
    
        # Calculate the new dimensions with padding
        new_width = self.image.width() + padding_left + padding_right
        new_height = self.image.height() + padding_top + padding_bottom
    
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
        self.image_contrasted = self.image.copy()
        
        self.top_padding_slider.setRange((-self.image.height()+100), (self.image.height()+100))
        self.left_padding_slider.setRange((-self.image.width()+100), (self.image.width()+100))
        self.right_padding_slider.setRange((-self.image.width()+100), (self.image.width()+100))
    
        # Adjust marker shifts to account for padding
        # self.left_marker_shift += padding_left
        
    
        # Update slider values to match the new shifts
        # self.left_padding_slider.setValue(self.left_marker_shift)
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
        
        # Enable the "Predict Molecular Weight" button if markers are present
        if self.left_markers or self.right_markers:
            self.predict_button.setEnabled(True)
        else:
            self.predict_button.setEnabled(False)
    
        # Adjust slider maximum ranges based on the current image width
        self.left_padding_slider.setRange(-int(self.image.width()), int(self.image.width()))
        self.right_padding_slider.setRange(-int(self.image.width()), int(self.image.width()))
    
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
        
        y_offset_global = text_height / 4
    
        # Draw the left markers (aligned right)
        for y_pos, marker_value in self.left_markers:
            y_pos_cropped = (y_pos - y_start) * (scaled_image.height() / self.image.height())
            if 0 <= y_pos_cropped <= scaled_image.height():
                text = f"{marker_value} ⎯ "  ##CHANGE HERE IF YOU WANT TO REMOVE THE "-"
                text_width = font_metrics.horizontalAdvance(text)  # Get text width
                painter.drawText(
                    int(x_offset + self.left_marker_shift - text_width),
                    int(y_offset + y_pos_cropped + y_offset_global),  # Adjust for proper text placement
                    text,
                )
        
        
        # Draw the right markers (aligned left)
        for y_pos, marker_value in self.right_markers:
            y_pos_cropped = (y_pos - y_start) * (scaled_image.height() / self.image.height())
            if 0 <= y_pos_cropped <= scaled_image.height():
                text = f" ⎯ {marker_value}" ##CHANGE HERE IF YOU WANT TO REMOVE THE "-"
                painter.drawText(
                    int(x_offset + self.right_marker_shift + self.right_marker_shift_added + line_padding),
                    int(y_offset + y_pos_cropped + y_offset_global),  # Adjust for proper text placement
                    text,
                )
                
    
        # Draw the top markers (if needed)
        for x_pos, top_label in self.top_markers:
            x_pos_cropped = (x_pos - x_start) * (scaled_image.width() / self.image.width())
            if 0 <= x_pos_cropped <= scaled_image.width():
                text = f"{top_label}"
                painter.save()
                text_width = font_metrics.horizontalAdvance(text)
                text_height= font_metrics.height()
                label_x = x_offset + x_pos_cropped # - text_width/2
                label_y = y_offset + self.top_marker_shift + self.top_marker_shift_added
                painter.translate(label_x, label_y)
                painter.rotate(self.font_rotation)
                painter.drawText(int(0 - text_width/2),0, f"{top_label}")
                painter.restore()
    
        # Draw guide lines
        if draw_guides and self.show_guides_checkbox.isChecked():
            pen = QPen(Qt.red, 2 * render_scale)
            painter.setPen(pen)
            center_x = canvas.width() // 2
            center_y = canvas.height() // 2
            painter.drawLine(center_x, 0, center_x, canvas.height())  # Vertical line
            painter.drawLine(0, center_y, canvas.width(), center_y)  # Horizontal line
            
        
        
        
        # Draw the protein location marker (*)
        if hasattr(self, "protein_location") and self.run_predict_MW == False:
            x, y = self.protein_location
            text = "⎯⎯"
            text_width = font_metrics.horizontalAdvance(text)
            text_height= font_metrics.height()
            painter.drawText(
                int(x * render_scale - text_width / 2),  #Currently left edge # FOR Center horizontally use int(x * render_scale - text_width / 2)
                int(y * render_scale + text_height / 4),  # Center vertically
                text
            )
        
        if hasattr(self, "custom_markers"):
            # Get default font type and size
            default_font_type = QFont(self.custom_font_type_dropdown.currentText())
            default_font_size = int(self.custom_font_size_spinbox.value())
        

        
            for x, y, text, color, *optional in self.custom_markers:
                # Use provided font type and size if available, otherwise use defaults
                marker_font_type = optional[0] if len(optional) > 0 else default_font_type
                marker_font_size = optional[1] if len(optional) > 1 else default_font_size
                
                # If marker_font_type is not already a QFont, create a QFont instance
                if isinstance(marker_font_type, str):
                    font = QFont(marker_font_type)
                else:
                    font = QFont(marker_font_type)  # Clone the font to avoid modifying the original
                
                # Adjust font size for rendering scale
                font.setPointSize(marker_font_size * render_scale)
        
                # Apply the font to the painter
                painter.setFont(font)
                painter.setPen(color)  # Use the selected custom marker color
        
                # Create font metrics to calculate text width and height
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()
        
                # Draw text, center it horizontally and vertically
                painter.drawText(
                    int(x * render_scale - text_width / 2),  # Center horizontally 
                    int(y * render_scale + text_height / 4),  # Center vertically
                    text
                )

        # Draw the grid (if enabled)
        if self.show_grid_checkbox.isChecked():
            grid_size = self.grid_size_input.value() * render_scale
            pen = QPen(Qt.red)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
    
            # Draw vertical grid lines
            for x in range(0, canvas.width(), grid_size):
                painter.drawLine(x, 0, x, canvas.height())
    
            # Draw horizontal grid lines
            for y in range(0, canvas.height(), grid_size):
                painter.drawLine(0, y, canvas.width(), y)
                
                
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
        self.show_grid_checkbox.setChecked(False)
        self.update_live_view()
    
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
        self.image_before_padding = self.image.copy()
        self.image_contrasted=self.image.copy()
        self.image_before_contrast=self.image.copy()
    
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
        self.show_grid_checkbox.setChecked(False)
        self.update_live_view()
        config = self.get_current_config()
    
        # Align the image first (rotate it)
        self.align_image()
        
    
        # Now apply cropping
        cropped_image = self.crop_image()
    
        if cropped_image:
            self.image = cropped_image
            self.image_before_padding = self.image.copy()
            self.image_contrasted=self.image.copy()
            self.image_before_contrast=self.image.copy()
            self.update_live_view()
    
        # Reapply the saved configuration
        self.apply_config(config)
    
        # Reset sliders
        self.crop_x_start_slider.setValue(0)
        self.crop_x_end_slider.setValue(100)
        self.crop_y_start_slider.setValue(0)
        self.crop_y_end_slider.setValue(100)

        
    def save_image(self):
        if not self.image:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
    
        options = QFileDialog.Options()
        save_path=""
        base_save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)", options=options
        )
        if base_save_path:
            # Check if the base name already contains "_original"
            if "_original" not in os.path.splitext(base_save_path)[0]:
                original_save_path = os.path.splitext(base_save_path)[0] + "_original.png"
                modified_save_path = os.path.splitext(base_save_path)[0] + "_modified.png"
                config_save_path = os.path.splitext(base_save_path)[0] + "_config.txt"
                save_path=original_save_path
                
            else:
                original_save_path = base_save_path
                modified_save_path = os.path.splitext(base_save_path)[0].replace("_original", "") + "_modified.png"
                config_save_path = os.path.splitext(base_save_path)[0].replace("_original", "") + "_config.txt"
                save_path=original_save_path
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
            
            self.show_grid_checkbox.setChecked(False)
            self.update_live_view()
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
            
            text_title="IMAGING ASSISTANT V3: "
            text_title+=str(save_path)
            self.setWindowTitle(text_title)
            

    
    
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
        self.show_grid_checkbox.setChecked(False)
        self.update_live_view()
        # Render the high-resolution canvas without guides for clipboard
        self.render_image_on_canvas(
            high_res_canvas, scaled_image, x_start, y_start, render_scale, draw_guides=False
        )
        
        # Copy the high-resolution image to the clipboard
        clipboard = QApplication.clipboard()
        clipboard.setImage(high_res_canvas)  # Copy the rendered image

        
    def clear_predict_molecular_weight(self):
        self.live_view_label.preview_marker_enabled = False
        self.live_view_label.preview_marker_text = ""
        self.live_view_label.setCursor(Qt.ArrowCursor)
        if hasattr(self, "protein_location"):
            del self.protein_location  # Clear the protein location marker
        self.update_live_view()  # Update the display
        
    def predict_molecular_weight(self):
        self.live_view_label.preview_marker_enabled = False
        self.live_view_label.preview_marker_text = ""
        self.live_view_label.setCursor(Qt.CrossCursor)
        
        # Determine which markers to use (left or right)
        self.run_predict_MW=False
        if self.run_predict_MW!=True:
            markers_not_rounded = self.left_markers if self.left_markers else self.right_markers
            markers = [[round(value, 2) for value in sublist] for sublist in markers_not_rounded]
            if not markers:
                QMessageBox.warning(self, "Error", "No markers available for prediction.")
                return
        
            # Get marker positions and values
            marker_positions = np.array([pos for pos, _ in markers])
            marker_values = np.array([val for _, val in markers])
        
            # Ensure there are at least two markers for linear regression
            if len(marker_positions) < 2:
                QMessageBox.warning(self, "Error", "At least two markers are needed for prediction.")
                return
        
            # Normalize distances
            min_position = np.min(marker_positions)
            max_position = np.max(marker_positions)
            normalized_distances = (marker_positions - min_position) / (max_position - min_position)
            # normalized_distances = (marker_positions) / (min_position)
            # print("MARKERS: ", markers)
            
        
            # Allow the user to select a point for prediction
            QMessageBox.information(self, "Instruction", "Click on the protein location in the preview window.")
            self.live_view_label.mousePressEvent = lambda event: self.get_protein_location(
                event, normalized_distances, marker_values, min_position, max_position
            )

            


        
    def get_protein_location(self, event, normalized_distances, marker_values, min_position, max_position):
        # Get cursor position from the event
        pos = event.pos()
        cursor_x, cursor_y = pos.x(), pos.y()
    
        # Dimensions of the displayed image
        displayed_width = self.live_view_label.width()
        displayed_height = self.live_view_label.height()
    
        # Dimensions of the actual image
        image_width = self.image.width()
        image_height = self.image.height()
    
        # Calculate scaling factors
        scale = min(displayed_width / image_width, displayed_height / image_height)
    
        # Calculate offsets
        x_offset = (displayed_width - image_width * scale) / 2
        y_offset = (displayed_height - image_height * scale) / 2
    
        # Transform cursor position to image space
        protein_position = round((cursor_y - y_offset) / scale,2)
        # protein_positio
        # print("PROTEIN POSITION: ", protein_position)
        
        
        
    
        # Normalize the protein position
        normalized_protein_position = (protein_position - min_position) / (max_position - min_position)
        # normalized_protein_position = (protein_position) / (min_position)
    
        # Perform linear regression on the log-transformed data
        log_marker_values = np.log10(marker_values)
        
        #LOG_FIT
        coefficients = np.polyfit(normalized_distances, log_marker_values, 1)
        #Polynomial Fit 6th order
        # coefficients = np.polyfit(normalized_distances, log_marker_values, 6)  # Fit a quadratic curve
        
        # Predict molecular weight
        predicted_log10_weight = np.polyval(coefficients, normalized_protein_position)
        predicted_weight = 10 ** predicted_log10_weight
    
        # Store the protein location in pixel coordinates
        self.protein_location = (cursor_x, cursor_y)

        
    
        # Update the live view to draw the *
        self.update_live_view()
    
        # Plot the graph with the clicked position
        self.plot_molecular_weight_graph(
            normalized_distances,
            marker_values,
            10 ** np.polyval(coefficients, normalized_distances),
            normalized_protein_position,
            predicted_weight,
            r_squared=1 - (np.sum((log_marker_values - np.polyval(coefficients, normalized_distances)) ** 2) /
                           np.sum((log_marker_values - np.mean(log_marker_values)) ** 2))
        )

        
        
    def plot_molecular_weight_graph(
        self, normalized_distances, marker_values, fitted_values, protein_position, predicted_weight, r_squared
    ):
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PyQt5.QtGui import QPixmap
    
        plt.figure(figsize=(5, 3))
        plt.scatter(normalized_distances, marker_values, color="red", label="Marker Data")
        plt.plot(normalized_distances, fitted_values, color="blue", label=f"Fit (R²={r_squared:.3f})")
        plt.axvline(protein_position, color="green", linestyle="--", label=f"Protein Position\n({predicted_weight:.2f} units)")
        plt.xlabel("Normalized Relative Distance")
        plt.ylabel("Molecular Weight (units)")
        plt.yscale("log")  # Log scale for molecular weight
        plt.legend()
        plt.title("Molecular Weight Prediction")
    
        # Convert the plot to a pixmap
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.read())
        buffer.close()
        plt.close()
    
        # Display the plot and results in a message box
        message_box = QMessageBox(self)
        message_box.setWindowTitle("Prediction Result")
        message_box.setText(
            f"The predicted molecular weight is approximately {predicted_weight:.2f} units.\n"
            f"R-squared value of the fit: {r_squared:.3f}"
        )
        label = QLabel()
        label.setPixmap(pixmap)
        message_box.layout().addWidget(label, 1, 0, 1, message_box.layout().columnCount())
        message_box.exec()
        # self.run_predict_MW=True

  
        
    def reset_image(self):
        # Reset the image to original
        if self.image != None:
            self.image = self.image_master.copy()
            self.image_before_padding = self.image.copy()
            self.image_contrasted = self.image.copy() # Update the contrasted image
            self.image_before_padding = self.image.copy()
        self.left_markers.clear()  # Clear left markers
        self.right_markers.clear()  # Clear right markers
        self.top_markers.clear()
        self.custom_markers.clear()
        self.remove_custom_marker_mode()
        self.clear_predict_molecular_weight()
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
    app = QApplication([])
    app.setStyle("Fusion")
    window = CombinedSDSApp()
    window.show()
    app.exec_()
