# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'g:\root\my_life\sbme_2025\3rd_year\second_semester\computer_vision\tasks\task2\CV_task2\GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(948, 775)
        MainWindow.setStyleSheet("/* Apply to the entire application */\n"
"QWidget {\n"
"    \n"
"    \n"
"    font-family: Arial, sans-serif; /* Set font to Arial or a fallback sans-serif */\n"
"}\n"
"\n"
"/* Apply to buttons */\n"
"QPushButton {\n"
"    background-color: #4A90E2; /* Light blue background */\n"
"    border: none; /* No border */\n"
"    color: white; /* White text color */\n"
"    padding: 10px 20px; /* Padding */\n"
"    border-radius: 5px; /* Rounded corners */\n"
"    font-size: 14px; /* Font size */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #357EBD; /* Darker blue on hover */\n"
"}\n"
"\n"
"/* Apply to labels */\n"
"QLabel {\n"
"    color: #333333; /* Dark text color */\n"
"    font-size: 16px; /* Font size */\n"
"}\n"
"\n"
"/* Apply to group boxes */\n"
"QGroupBox {\n"
"    background-color: #ffffff; /* White background */\n"
"    border: 2px solid #4A90E2; /* Light blue border */\n"
"    border-radius: 5px; /* Rounded corners */\n"
"}\n"
"\n"
"/* Apply to sliders */\n"
"QSlider::groove:horizontal {\n"
"    border: 1px solid #999999; /* Gray border */\n"
"    background: #ffffff; /* White background */\n"
"    height: 10px; /* Groove height */\n"
"    border-radius: 5px; /* Rounded corners */\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: #4A90E2; /* Light blue handle */\n"
"    border: 1px solid #4A90E2; /* Light blue border */\n"
"    width: 20px; /* Handle width */\n"
"    margin: -5px 0; /* Center the handle vertically */\n"
"    border-radius: 10px; /* Rounded corners */\n"
"}\n"
"\n"
"/* Apply to combo boxes */\n"
"QComboBox {\n"
"    background-color: #ffffff; /* White background */\n"
"    border: 1px solid #4A90E2; /* Light blue border */\n"
"    border-radius: 5px; /* Rounded corners */\n"
"    padding: 5px; /* Padding */\n"
"}\n"
"\n"
"/* Apply to graphics views */\n"
"QGraphicsView {\n"
"    background-color: #ffffff; /* White background */\n"
"    border: 1px solid #4A90E2; /* Light blue border */\n"
"    border-radius: 5px; /* Rounded corners */\n"
"}\n"
"\n"
"/* Apply to scroll bars */\n"
"QScrollBar:vertical {\n"
"    border: 1px solid #999999; /* Gray border */\n"
"    background: #ffffff; /* White background */\n"
"    width: 10px; /* Scroll bar width */\n"
"    margin: 0px 0px 0px 0px; /* Margin */\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"    background: #4A90E2; /* Light blue handle */\n"
"    min-height: 20px; /* Handle minimum height */\n"
"    border-radius: 5px; /* Rounded corners */\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"    height: 0px; /* No height for add line */\n"
"    subcontrol-position: bottom; /* Align to bottom */\n"
"    subcontrol-origin: margin; /* Origin from margin */\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"    height: 0px; /* No height for sub line */\n"
"    subcontrol-position: top; /* Align to top */\n"
"    subcontrol-origin: margin; /* Origin from margin */\n"
"}\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMinimumSize(QtCore.QSize(100, 0))
        self.groupBox_2.setMaximumSize(QtCore.QSize(600, 16777215))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton.setObjectName("radioButton")
        self.gridLayout_15.addWidget(self.radioButton, 6, 0, 1, 1)
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_2.setObjectName("radioButton_2")
        self.gridLayout_15.addWidget(self.radioButton_2, 7, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_15.addItem(spacerItem, 9, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_15.addItem(spacerItem1, 22, 0, 1, 1)
        self.gridLayout_13 = QtWidgets.QGridLayout()
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.horizontalSlider_6 = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_6.setMaximum(102)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.gridLayout_13.addWidget(self.horizontalSlider_6, 0, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox_2)
        self.label_16.setObjectName("label_16")
        self.gridLayout_13.addWidget(self.label_16, 0, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_11.setObjectName("label_11")
        self.gridLayout_13.addWidget(self.label_11, 0, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_13.addItem(spacerItem2, 1, 1, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_13, 16, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_15.addItem(spacerItem3, 11, 0, 1, 1)
        self.gridLayout_10 = QtWidgets.QGridLayout()
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_10.addWidget(self.label_7, 0, 0, 1, 1)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_3.setMaximum(100)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.gridLayout_10.addWidget(self.horizontalSlider_3, 0, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_10.addWidget(self.label_13, 0, 2, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_10, 10, 0, 1, 1)
        self.gridLayout_11 = QtWidgets.QGridLayout()
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_11.addWidget(self.label_8, 0, 0, 1, 1)
        self.horizontalSlider_4 = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_4.setMaximum(100)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.gridLayout_11.addWidget(self.horizontalSlider_4, 0, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_11.addWidget(self.label_14, 0, 2, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_11, 12, 0, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.kernal_combo = QtWidgets.QComboBox(self.groupBox_2)
        self.kernal_combo.setMinimumSize(QtCore.QSize(0, 30))
        self.kernal_combo.setObjectName("kernal_combo")
        self.kernal_combo.addItem("")
        self.kernal_combo.addItem("")
        self.kernal_combo.addItem("")
        self.kernal_combo.addItem("")
        self.gridLayout_6.addWidget(self.kernal_combo, 0, 1, 1, 1)
        self.kernal_label = QtWidgets.QLabel(self.groupBox_2)
        self.kernal_label.setMaximumSize(QtCore.QSize(100, 16777215))
        self.kernal_label.setObjectName("kernal_label")
        self.gridLayout_6.addWidget(self.kernal_label, 0, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.sigma_label = QtWidgets.QLabel(self.groupBox_2)
        self.sigma_label.setObjectName("sigma_label")
        self.gridLayout_7.addWidget(self.sigma_label, 0, 0, 1, 1)
        self.sigma_slider = QtWidgets.QSlider(self.groupBox_2)
        self.sigma_slider.setMaximum(100)
        self.sigma_slider.setOrientation(QtCore.Qt.Horizontal)
        self.sigma_slider.setObjectName("sigma_slider")
        self.gridLayout_7.addWidget(self.sigma_slider, 0, 1, 1, 1)
        self.slider_sig_label = QtWidgets.QLabel(self.groupBox_2)
        self.slider_sig_label.setObjectName("slider_sig_label")
        self.gridLayout_7.addWidget(self.slider_sig_label, 0, 2, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_7, 2, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout_8.addItem(spacerItem4, 1, 0, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_8, 4, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem5 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem5, 0, 3, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton.setMaximumSize(QtCore.QSize(16777215, 50))
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_3.addWidget(self.pushButton, 0, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem6, 0, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_3.addWidget(self.pushButton_2, 0, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.gridLayout_12 = QtWidgets.QGridLayout()
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_12.addWidget(self.label_9, 0, 0, 1, 1)
        self.horizontalSlider_5 = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_5.setMaximum(100)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.gridLayout_12.addWidget(self.horizontalSlider_5, 0, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox_2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_12.addWidget(self.label_15, 0, 2, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_12, 14, 0, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout_15.addItem(spacerItem7, 1, 0, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout_15.addItem(spacerItem8, 5, 0, 1, 1)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_9.addWidget(self.label_6, 0, 0, 1, 1)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_2.setMaximum(100)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.gridLayout_9.addWidget(self.horizontalSlider_2, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_9.addWidget(self.label_12, 0, 2, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_9, 8, 0, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_15.addItem(spacerItem9, 15, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_15.addWidget(self.label_4, 18, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout_15.addWidget(self.label_3, 20, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_15.addItem(spacerItem10, 23, 0, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_15.addItem(spacerItem11, 24, 0, 1, 1)
        self.gridLayout_14 = QtWidgets.QGridLayout()
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.colors_combo = QtWidgets.QComboBox(self.groupBox_2)
        self.colors_combo.setMinimumSize(QtCore.QSize(0, 30))
        self.colors_combo.setObjectName("colors_combo")
        self.gridLayout_14.addWidget(self.colors_combo, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_10.setObjectName("label_10")
        self.gridLayout_14.addWidget(self.label_10, 0, 0, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_14, 17, 0, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout_15.addItem(spacerItem12, 3, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.mode_name_comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.mode_name_comboBox.setMinimumSize(QtCore.QSize(0, 50))
        self.mode_name_comboBox.setMaximumSize(QtCore.QSize(16777215, 100))
        self.mode_name_comboBox.setObjectName("mode_name_comboBox")
        self.mode_name_comboBox.addItem("")
        self.mode_name_comboBox.addItem("")
        self.mode_name_comboBox.addItem("")
        self.mode_name_comboBox.addItem("")
        self.mode_name_comboBox.addItem("")
        self.gridLayout_5.addWidget(self.mode_name_comboBox, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_2.setObjectName("label_2")
        self.gridLayout_5.addWidget(self.label_2, 0, 0, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_5, 2, 0, 1, 1)
        self.lineEdit_low = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_low.setObjectName("lineEdit_low")
        self.gridLayout_15.addWidget(self.lineEdit_low, 19, 0, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_15.addItem(spacerItem13, 13, 0, 1, 1)
        self.lineEdit_high = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_high.setObjectName("lineEdit_high")
        self.gridLayout_15.addWidget(self.lineEdit_high, 21, 0, 1, 1)
        self.gridLayout_16.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.input_image = QtWidgets.QGraphicsView(self.groupBox)
        self.input_image.setMinimumSize(QtCore.QSize(0, 0))
        self.input_image.setObjectName("input_image")
        self.gridLayout.addWidget(self.input_image, 0, 0, 1, 1)
        self.output_image = QtWidgets.QGraphicsView(self.groupBox)
        self.output_image.setObjectName("output_image")
        self.gridLayout.addWidget(self.output_image, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_16.addWidget(self.groupBox, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 948, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.mode_name_comboBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButton.setText(_translate("MainWindow", "Square Contour"))
        self.radioButton_2.setText(_translate("MainWindow", "Circle Contour"))
        self.label_16.setText(_translate("MainWindow", "0"))
        self.label_11.setText(_translate("MainWindow", "TextLabel"))
        self.label_7.setText(_translate("MainWindow", "TextLabel"))
        self.label_13.setText(_translate("MainWindow", "0"))
        self.label_8.setText(_translate("MainWindow", "TextLabel"))
        self.label_14.setText(_translate("MainWindow", "0"))
        self.kernal_combo.setItemText(0, _translate("MainWindow", "None"))
        self.kernal_combo.setItemText(1, _translate("MainWindow", "3 X 3 "))
        self.kernal_combo.setItemText(2, _translate("MainWindow", "5 X 5"))
        self.kernal_combo.setItemText(3, _translate("MainWindow", "7 X 7"))
        self.kernal_label.setText(_translate("MainWindow", "Kernal Size"))
        self.sigma_label.setText(_translate("MainWindow", "Sigma"))
        self.slider_sig_label.setText(_translate("MainWindow", "0"))
        self.pushButton.setText(_translate("MainWindow", "Browse"))
        self.pushButton_2.setText(_translate("MainWindow", "Apply"))
        self.label.setText(_translate("MainWindow", "Upload Image"))
        self.label_9.setText(_translate("MainWindow", "TextLabel"))
        self.label_15.setText(_translate("MainWindow", "0"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))
        self.label_12.setText(_translate("MainWindow", "0"))
        self.label_4.setText(_translate("MainWindow", "T_low"))
        self.label_3.setText(_translate("MainWindow", "T_high"))
        self.label_10.setText(_translate("MainWindow", "TextLabel"))
        self.mode_name_comboBox.setItemText(0, _translate("MainWindow", "Canny Edge Detector"))
        self.mode_name_comboBox.setItemText(1, _translate("MainWindow", "Line Detection"))
        self.mode_name_comboBox.setItemText(2, _translate("MainWindow", "Elipse Detection"))
        self.mode_name_comboBox.setItemText(3, _translate("MainWindow", "Circle Detection"))
        self.mode_name_comboBox.setItemText(4, _translate("MainWindow", "Snake"))
        self.label_2.setText(_translate("MainWindow", "Mode"))
