from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtWidgets import QTableWidget, QWidget, QMessageBox
from PyQt5.QtWidgets import QLabel, QFileDialog, QTableWidgetItem, QPushButton, QLineEdit, QFrame 
from PyQt5.QtGui import QFont, QPixmap, QIcon

from os import path, listdir, makedirs
from shutil import move
from serial import Serial, PARITY_NONE, STOPBITS_ONE, EIGHTBITS
from datetime import datetime
from training_tab_regression import predict_image
from xlsxwriter import Workbook
from xlrd import open_workbook

PROCESSED_IMAGE_FOLDER = "CUSTOMIZE_4_USER/PROCESSED_IMAGES/"

INITIAL_MODBUS = 0xFFFF
INITIAL_DF1 = 0x0000

# Table CRC16_modbus
table = ( 
0x0000, 0xC0C1, 0xC181, 0x0140, 0xC301, 0x03C0, 0x0280, 0xC241,
0xC601, 0x06C0, 0x0780, 0xC741, 0x0500, 0xC5C1, 0xC481, 0x0440,
0xCC01, 0x0CC0, 0x0D80, 0xCD41, 0x0F00, 0xCFC1, 0xCE81, 0x0E40,
0x0A00, 0xCAC1, 0xCB81, 0x0B40, 0xC901, 0x09C0, 0x0880, 0xC841,
0xD801, 0x18C0, 0x1980, 0xD941, 0x1B00, 0xDBC1, 0xDA81, 0x1A40,
0x1E00, 0xDEC1, 0xDF81, 0x1F40, 0xDD01, 0x1DC0, 0x1C80, 0xDC41,
0x1400, 0xD4C1, 0xD581, 0x1540, 0xD701, 0x17C0, 0x1680, 0xD641,
0xD201, 0x12C0, 0x1380, 0xD341, 0x1100, 0xD1C1, 0xD081, 0x1040,
0xF001, 0x30C0, 0x3180, 0xF141, 0x3300, 0xF3C1, 0xF281, 0x3240,
0x3600, 0xF6C1, 0xF781, 0x3740, 0xF501, 0x35C0, 0x3480, 0xF441,
0x3C00, 0xFCC1, 0xFD81, 0x3D40, 0xFF01, 0x3FC0, 0x3E80, 0xFE41,
0xFA01, 0x3AC0, 0x3B80, 0xFB41, 0x3900, 0xF9C1, 0xF881, 0x3840,
0x2800, 0xE8C1, 0xE981, 0x2940, 0xEB01, 0x2BC0, 0x2A80, 0xEA41,
0xEE01, 0x2EC0, 0x2F80, 0xEF41, 0x2D00, 0xEDC1, 0xEC81, 0x2C40,
0xE401, 0x24C0, 0x2580, 0xE541, 0x2700, 0xE7C1, 0xE681, 0x2640,
0x2200, 0xE2C1, 0xE381, 0x2340, 0xE101, 0x21C0, 0x2080, 0xE041,
0xA001, 0x60C0, 0x6180, 0xA141, 0x6300, 0xA3C1, 0xA281, 0x6240,
0x6600, 0xA6C1, 0xA781, 0x6740, 0xA501, 0x65C0, 0x6480, 0xA441,
0x6C00, 0xACC1, 0xAD81, 0x6D40, 0xAF01, 0x6FC0, 0x6E80, 0xAE41,
0xAA01, 0x6AC0, 0x6B80, 0xAB41, 0x6900, 0xA9C1, 0xA881, 0x6840,
0x7800, 0xB8C1, 0xB981, 0x7940, 0xBB01, 0x7BC0, 0x7A80, 0xBA41,
0xBE01, 0x7EC0, 0x7F80, 0xBF41, 0x7D00, 0xBDC1, 0xBC81, 0x7C40,
0xB401, 0x74C0, 0x7580, 0xB541, 0x7700, 0xB7C1, 0xB681, 0x7640,
0x7200, 0xB2C1, 0xB381, 0x7340, 0xB101, 0x71C0, 0x7080, 0xB041,
0x5000, 0x90C1, 0x9181, 0x5140, 0x9301, 0x53C0, 0x5280, 0x9241,
0x9601, 0x56C0, 0x5780, 0x9741, 0x5500, 0x95C1, 0x9481, 0x5440,
0x9C01, 0x5CC0, 0x5D80, 0x9D41, 0x5F00, 0x9FC1, 0x9E81, 0x5E40,
0x5A00, 0x9AC1, 0x9B81, 0x5B40, 0x9901, 0x59C0, 0x5880, 0x9841,
0x8801, 0x48C0, 0x4980, 0x8941, 0x4B00, 0x8BC1, 0x8A81, 0x4A40,
0x4E00, 0x8EC1, 0x8F81, 0x4F40, 0x8D01, 0x4DC0, 0x4C80, 0x8C41,
0x4400, 0x84C1, 0x8581, 0x4540, 0x8701, 0x47C0, 0x4680, 0x8641,
0x8201, 0x42C0, 0x4380, 0x8341, 0x4100, 0x81C1, 0x8081, 0x4040 )

def calcByte( ch, crc):
    """Given a new Byte and previous CRC, Calc a new CRC-16"""
    if type(ch) == type("c"):
        by = ord(ch)
    else:
        by = ch
    crc = (crc >> 8) ^ table[(crc ^ by) & 0xFF]
    return (crc & 0xFFFF)

def calcString( st, crc):
    """Given a binary string and starting CRC, Calc a final CRC-16 """
    for ch in st:
        crc = (crc >> 8) ^ table[(crc ^ ord(ch)) & 0xFF]
    return crc


class MainTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        
        self.font18 = parent.font18
        self.font16 = parent.font16
        self.font15 = parent.font15
        self.font14 = parent.font14
        self.font13 = parent.font13
        self.font12 = parent.font12
        self.font11 = parent.font11
        self.font10 = parent.font10
        self.allFont = parent.allFont

        self.result_predict60 = -100
        self.result_predict90 = -100
        self.result_predict120 = -100

        self.left = 0
        self.top = 0
        self.height = parent.height
        self.width = parent.width
        self.ser = SerialPort()
        self.tabMainUI()
        
    def tabMainUI(self):
        # 3 columns 
        self.column_w = int(self.width * 0.32)

        # WIDGETS IN NEW MAIN TAB
        # Create widget mode: AUTO - MANUAL
        self.runningModeWidget = QWidget(self)
        self.runningModeWidget.setGeometry(QRect(self.left, self.top , self.width, 30))
        self.groupRunningMode()
        
        # Create widget: Group 60
        self.groupStage60 = QWidget(self)
        self.groupStage60.setGeometry(QRect(self.left, self.top + 35, self.column_w, self.height))
        self.createGroupStage("60")

        # Create widget: Group 90
        self.groupStage90 = QWidget(self)
        self.groupStage90.setGeometry(QRect(int(0.338*self.width), self.top + 35, self.column_w, self.height))
        self.createGroupStage("90")

        # Create widget: Group 120
        self.groupStage120 = QWidget(self)
        self.groupStage120.setGeometry(QRect(int(0.675*self.width), self.top + 35, self.column_w, self.height))
        self.createGroupStage("120")

    # WIDGET RUNNING MODE
    def runningModeButton(self, mode):
        if self.PREMODE != mode:
            self.startButton60.setChecked(False)
            self.startButton90.setChecked(False)
            self.startButton120.setChecked(False)
        else:
            return
        if mode =="manual":
            print("manual")
            self.autoButton.setChecked(False)
            self.MODE = "manual"
            self.PREMODE = "manual"

        if mode == "auto":
            print("auto")
            self.manualButton.setChecked(False)
            self.MODE = "auto"
            self.PREMODE = "auto"

    def groupRunningMode(self):
        self.manualButton  = QPushButton("MANUAL", self.runningModeWidget)
        self.manualButton.setFont(self.font14)
        self.manualButton.setGeometry(3*self.width//4-100, self.top, 100,30)
        self.manualButton.setStyleSheet("QPushButton {background-color: yellow; font-weight: bold}")
        self.manualButton.setCheckable(True)
        self.manualButton.clicked.connect(lambda: self.runningModeButton("manual"))
        
        self.autoButton = QPushButton("AUTO", self.runningModeWidget)
        self.autoButton.setFont(self.font14)
        self.autoButton.setGeometry(self.width//4, self.top, 100,30)
        self.autoButton.setStyleSheet("QPushButton {background-color: yellow; font-weight: bold}")
        self.autoButton.setCheckable(True)
        self.autoButton.setChecked(True)
        self.autoButton.clicked.connect(lambda: self.runningModeButton("auto"))

        self.MODE = "auto"
        self.PREMODE = "auto"
           
    def createGroupStage(self, groupStage):
        self.createGroupLabel(groupStage)        
        self.createGroupImage(groupStage)
        self.createGroupResult(groupStage)
        self.createGroupInfoImg(groupStage)
        self.startGroup(groupStage)

    # Group label 
    def createGroupLabel(self, groupStage):
        self.heightGroupLabel = 30
        if groupStage=="60": 
            self.nameGroupLabel = QLabel("Giai đoạn 60 phút", self.groupStage60)
        if groupStage=="90":
            self.nameGroupLabel = QLabel("Giai đoạn 90 phút", self.groupStage90)
        if groupStage=="120":
            self.nameGroupLabel = QLabel("Giai đoạn 120 phút", self.groupStage120)

        self.nameGroupLabel.setGeometry(self.column_w//4, self.top, self.column_w//2, self.heightGroupLabel)
        self.nameGroupLabel.setFont(self.allFont)
        self.nameGroupLabel.setAlignment(Qt.AlignCenter)
        self.nameGroupLabel.setStyleSheet("QLabel {background-color: yellow}")

    # Group image 
    def createGroupImage(self, groupStage):
        self.imageFolderPath60 = ""
        self.imageFolderPath90 = ""
        self.imageFolderPath120 = ""
        self.imagePath60 = ""
        self.imagePath90 = ""
        self.imagePath120 = ""

        if groupStage=="60":
            # Load image button
            self.loadImageButton60 = QPushButton("Thư mục ảnh", self.groupStage60)
            self.loadImageButton60.setFont(self.allFont)
            self.loadImageButton60.setGeometry(self.left, self.heightGroupLabel + 5, self.column_w//2-50 ,25)
            self.loadImageButton60.setStyleSheet("background-color: orange")
            self.loadImageButton60.clicked.connect(lambda: self.browseImage(groupStage))
            self.loadImageButton60.setIcon(QIcon("folder.png"))
            # Frame contain image
            self.frameImage60 = QFrame(self.groupStage60)
            self.frameImage60.setGeometry(self.left, self.heightGroupLabel + 35, int(0.6*self.column_w)-10, self.height // 2.5)
            self.frameImage60.setStyleSheet("background-color:grey")
            # Label show image
            self.labelImg60 = QLabel(self.frameImage60)
            self.labelImg60.setGeometry(self.left, self.top, int(0.6*self.column_w)-10, self.height // 2.5)

        if groupStage=="90":
            # Load image button
            self.loadImageButton90 = QPushButton("Thư mục ảnh", self.groupStage90)
            self.loadImageButton90.setFont(self.allFont)
            self.loadImageButton90.setGeometry(self.left, self.heightGroupLabel + 5, self.column_w//2-50 ,25)
            self.loadImageButton90.setStyleSheet("background-color: orange")
            self.loadImageButton90.clicked.connect(lambda: self.browseImage(groupStage))
            self.loadImageButton90.setIcon(QIcon("folder.png"))
            # Frame contain image
            self.frameImage90 = QFrame(self.groupStage90)
            self.frameImage90.setGeometry(self.left, self.heightGroupLabel + 35, int(0.6*self.column_w)-10, self.height // 2.5)
            self.frameImage90.setStyleSheet("background-color:grey")
            # Label show image
            self.labelImg90 = QLabel(self.frameImage90)
            self.labelImg90.setGeometry(self.left, self.top, int(0.6*self.column_w)-10, self.height // 2.5)
        
        if groupStage=="120":
            # Load image button
            self.loadImageButton120 = QPushButton("Thư mục ảnh", self.groupStage120)
            self.loadImageButton120.setFont(self.allFont)
            self.loadImageButton120.setGeometry(self.left, self.heightGroupLabel + 5, self.column_w//2-50 ,25)
            self.loadImageButton120.setStyleSheet("background-color: orange")
            self.loadImageButton120.clicked.connect(lambda: self.browseImage(groupStage))
            self.loadImageButton120.setIcon(QIcon("folder.png"))
            # Frame contain image
            self.frameImage120 = QFrame(self.groupStage120)
            self.frameImage120.setGeometry(self.left, self.heightGroupLabel + 35, int(0.6*self.column_w)-10, self.height // 2.5)
            self.frameImage120.setStyleSheet("background-color:grey")
            # Label show image
            self.labelImg120 = QLabel(self.frameImage120)
            self.labelImg120.setGeometry(self.left, self.top, int(0.6*self.column_w)-10, self.height // 2.5)

    # DONE  
    def browseImage(self, groupStage):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        if self.MODE == "auto":
            if groupStage == "60": 
                self.imageFolderPath60 = QFileDialog.getExistingDirectory(self, "Nhập đường dẫn thư mục ảnh giai đoạn 60 phút")
            if groupStage == "90": 
                self.imageFolderPath90 = QFileDialog.getExistingDirectory(self, "Nhập đường dẫn thư mục ảnh giai đoạn 90 phút")
            if groupStage == "120": 
                self.imageFolderPath120 = QFileDialog.getExistingDirectory(self, "Nhập đường dẫn thư mục ảnh giai đoạn 120 phút")

        if self.MODE == "manual":
            if groupStage == "60":
                self.imagePath60,_ = QFileDialog.getOpenFileName(self, "Nhập đường dẫn ảnh giai đoạn 60 phút", "", "Image Files(*.png *.jpg *.jpeg *.JPG *.PNG *.JPEG)", options=options)
            if groupStage == "90":
                self.imagePath90,_ = QFileDialog.getOpenFileName(self, "Nhập đường dẫn ảnh giai đoạn 90 phút", "", "Image Files(*.png *.jpg *.jpeg *.JPG *.PNG *.JPEG)", options=options)
            if groupStage == "120":
                self.imagePath120,_ = QFileDialog.getOpenFileName(self, "Nhập đường dẫn ảnh giai đoạn 120 phút", "", "Image Files(*.png *.jpg *.jpeg *.JPG *.PNG *.JPEG)", options=options)

    # Group result 
    def createGroupResult(self, groupStage):
        font = QFont("Times New Roman")
        font.setPointSize(30)
        
        if groupStage == "60":
            # Label result
            self.labelResult60 = QLabel("Kết quả phân tích", self.groupStage60)
            self.labelResult60.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 5, int(0.4*self.column_w) ,25)
            self.labelResult60.setStyleSheet("background-color: orange")
            self.labelResult60.setFont(self.allFont)
            self.labelResult60.setAlignment(Qt.AlignCenter)

            # Frame result
            self.frameResult60 = QLineEdit(self.groupStage60)
            self.frameResult60.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 35, int(0.4*self.column_w) ,self.height//5) 
            self.frameResult60.setText("99.99%")
            self.frameResult60.setFont(font)
            self.frameResult60.setAlignment(Qt.AlignCenter)
            self.frameResult60.setReadOnly(True)
            self.frameResult60.setStyleSheet("border: 2px solid gray; border-radius: 10px; padding:0 8px;")

            # Table result
            self.showResultTable60 = QTableWidget(self.groupStage60)
            self.showResultTable60.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 35 + self.height//5 + 5, int(0.4*self.column_w) ,self.height//5)
            self.showResultTable60.setRowCount(3)
            self.showResultTable60.setColumnCount(1)
            self.showResultTable60.horizontalHeader().setStretchLastSection(True)
            self.showResultTable60.horizontalHeader().setVisible(False)
            self.showResultTable60.setVerticalHeaderLabels(["T/gian xử lý","Chất lượng", "Kết luận"])
            self.showResultTable60.setEditTriggers(QTableWidget.NoEditTriggers)
            self.showResultTable60.setFont(self.font12)
            for i in range(3):
                self.showResultTable60.setRowHeight(i, self.showResultTable60.size().height()//3)    

        if groupStage == "90":
            # Label result
            self.labelResult90 = QLabel("Kết quả phân tích", self.groupStage90)
            self.labelResult90.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 5, int(0.4*self.column_w) ,25)
            self.labelResult90.setStyleSheet("background-color: orange")
            self.labelResult90.setFont(self.allFont)
            self.labelResult90.setAlignment(Qt.AlignCenter)

            # Frame result
            self.frameResult90 = QLineEdit(self.groupStage90)
            self.frameResult90.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 35, int(0.4*self.column_w) ,self.height//5) 
            self.frameResult90.setText("99.99%")
            self.frameResult90.setFont(font)
            self.frameResult90.setAlignment(Qt.AlignCenter)
            self.frameResult90.setReadOnly(True)
            self.frameResult90.setStyleSheet("border: 2px solid gray; border-radius: 10px; padding:0 8px;")

            # Table result
            self.showResultTable90 = QTableWidget(self.groupStage90)
            self.showResultTable90.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 35 + self.height//5 + 5, int(0.4*self.column_w) ,self.height//5)
            self.showResultTable90.setRowCount(3)
            self.showResultTable90.setColumnCount(1)
            self.showResultTable90.horizontalHeader().setStretchLastSection(True)
            self.showResultTable90.horizontalHeader().setVisible(False)
            self.showResultTable90.setVerticalHeaderLabels(["T/gian xử lý","Chất lượng", "Kết luận"])
            self.showResultTable90.setEditTriggers(QTableWidget.NoEditTriggers)
            self.showResultTable90.setFont(self.font12)
            for i in range(3):
                self.showResultTable90.setRowHeight(i, self.showResultTable90.size().height()//3)
        
        if groupStage == "120":
            # Label result
            self.labelResult120 = QLabel("Kết quả phân tích", self.groupStage120)
            self.labelResult120.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 5, int(0.4*self.column_w) ,25)
            self.labelResult120.setStyleSheet("background-color: orange")
            self.labelResult120.setFont(self.allFont)
            self.labelResult120.setAlignment(Qt.AlignCenter)

            # Frame result
            self.frameResult120 = QLineEdit(self.groupStage120)
            self.frameResult120.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 35, int(0.4*self.column_w) ,self.height//5) 
            self.frameResult120.setText("99.99%")
            self.frameResult120.setFont(font)
            self.frameResult120.setAlignment(Qt.AlignCenter)
            self.frameResult120.setReadOnly(True)
            self.frameResult120.setStyleSheet("border: 2px solid gray; border-radius: 10px; padding:0 8px;")

            # Table result
            self.showResultTable120 = QTableWidget(self.groupStage120)
            self.showResultTable120.setGeometry(int(0.6*self.column_w),  self.heightGroupLabel + 35 + self.height//5 + 5, int(0.4*self.column_w) ,self.height//5)
            self.showResultTable120.setRowCount(3)
            self.showResultTable120.setColumnCount(1)
            self.showResultTable120.horizontalHeader().setStretchLastSection(True)
            self.showResultTable120.horizontalHeader().setVisible(False)
            self.showResultTable120.setVerticalHeaderLabels(["T/gian xử lý","Chất lượng", "Kết luận"])
            self.showResultTable120.setEditTriggers(QTableWidget.NoEditTriggers)
            self.showResultTable120.setFont(self.font12)
            for i in range(3):
                self.showResultTable120.setRowHeight(i, self.showResultTable120.size().height()//3)
        
    # Group infomation image 
    def createGroupInfoImg(self, groupStage):
        if groupStage == "60":
            self.labelInfoImg60 = QLabel("Thông tin ảnh", self.groupStage60)
            self.labelInfoImg60.setGeometry(self.left, int(0.65*self.height), self.column_w, 30)
            self.labelInfoImg60.setAlignment(Qt.AlignCenter)
            self.labelInfoImg60.setFont(self.font14)

            self.tableInfoImage60 = QTableWidget(self.groupStage60)
            self.tableInfoImage60.setGeometry(self.left, int(0.7*self.height), self.column_w, int(0.2*self.height))
            self.tableInfoImage60.setRowCount(3)
            self.tableInfoImage60.setColumnCount(1)
            self.tableInfoImage60.horizontalHeader().setStretchLastSection(True)
            self.tableInfoImage60.horizontalHeader().setVisible(False)
            self.tableInfoImage60.setVerticalHeaderLabels(["Ảnh", "Kích thước", "Trạng thái"])
            self.tableInfoImage60.setEditTriggers(QTableWidget.NoEditTriggers)
            for i in range(3):
                self.tableInfoImage60.setRowHeight(i, self.tableInfoImage60.size().height()//3-1)

        if groupStage == "90":
            self.labelInfoImg90 = QLabel("Thông tin ảnh", self.groupStage90)
            self.labelInfoImg90.setGeometry(self.left, int(0.65*self.height), self.column_w, 30)
            self.labelInfoImg90.setAlignment(Qt.AlignCenter)
            self.labelInfoImg90.setFont(self.font14)

            self.tableInfoImage90 = QTableWidget(self.groupStage90)
            self.tableInfoImage90.setGeometry(self.left, int(0.7*self.height), self.column_w, int(0.2*self.height))
            self.tableInfoImage90.setRowCount(3)
            self.tableInfoImage90.setColumnCount(1)
            self.tableInfoImage90.horizontalHeader().setStretchLastSection(True)
            self.tableInfoImage90.horizontalHeader().setVisible(False)
            self.tableInfoImage90.setVerticalHeaderLabels(["Ảnh", "Kích thước", "Trạng thái"])
            self.tableInfoImage90.setEditTriggers(QTableWidget.NoEditTriggers)
            for i in range(3):
                self.tableInfoImage90.setRowHeight(i, self.tableInfoImage90.size().height()//3-1)

        if groupStage == "120":
            self.labelInfoImg120 = QLabel("Thông tin ảnh", self.groupStage120)
            self.labelInfoImg120.setGeometry(self.left, int(0.65*self.height), self.column_w, 30)
            self.labelInfoImg120.setAlignment(Qt.AlignCenter)
            self.labelInfoImg120.setFont(self.font14)

            self.tableInfoImage120 = QTableWidget(self.groupStage120)
            self.tableInfoImage120.setGeometry(self.left, int(0.7*self.height), self.column_w, int(0.2*self.height))
            self.tableInfoImage120.setRowCount(3)
            self.tableInfoImage120.setColumnCount(1)
            self.tableInfoImage120.horizontalHeader().setStretchLastSection(True)
            self.tableInfoImage120.horizontalHeader().setVisible(False)
            self.tableInfoImage120.setVerticalHeaderLabels(["Ảnh", "Kích thước", "Trạng thái"])
            self.tableInfoImage120.setEditTriggers(QTableWidget.NoEditTriggers)
            for i in range(3):
                self.tableInfoImage120.setRowHeight(i, self.tableInfoImage120.size().height()//3-1)

    # Group start button 
    def startGroup(self, groupStage):
        fontStartButton = QFont("Times New Roman")
        fontStartButton.setPointSize(20)

        if groupStage == "60":
            self.startButton60 = QPushButton("Start", self.groupStage60)
            self.startButton60.setGeometry(self.column_w//4, int(0.55*self.height), self.column_w//2, int(0.1*self.height))
            self.startButton60.setCheckable(True)
            self.startButton60.setFont(fontStartButton)
            self.startButton60.setStyleSheet("font-weight: bold")
            self.startButton60.toggled[bool].connect(self.checkStartFunction60)
        if groupStage == "90":
            self.startButton90 = QPushButton("Start", self.groupStage90)
            self.startButton90.setGeometry(self.column_w//4, int(0.55*self.height), self.column_w//2, int(0.1*self.height))
            self.startButton90.setCheckable(True)
            self.startButton90.setFont(fontStartButton)
            self.startButton90.setStyleSheet("font-weight: bold")
            self.startButton90.toggled[bool].connect(self.checkStartFunction90)
        
        if groupStage == "120":
            self.startButton120 = QPushButton("Start", self.groupStage120)
            self.startButton120.setGeometry(self.column_w//4, int(0.55*self.height), self.column_w//2, int(0.1*self.height))
            self.startButton120.setCheckable(True)
            self.startButton120.setFont(fontStartButton)
            self.startButton120.setStyleSheet("font-weight: bold")
            self.startButton120.toggled[bool].connect(self.checkStartFunction120)

    # START FUNCTION 60
    def checkStartFunction60(self):
        if self.startButton60.isChecked()==True:
            from PLCtab import COMPORTNAME
            self.startButton60.setStyleSheet("color: blue; font-weight: bold")
            self.timerProcess60 = QTimer()
            self.timerSendData60 = QTimer()
            if COMPORTNAME =="":
                QMessageBox.about(self, "Warning", "Chưa chọn cổng COM.")
                self.startButton60.setChecked(False)
                self.startButton60.setStyleSheet("font-weight: bold")
                return
            else:
                self.ser.comportName = COMPORTNAME
                self.ser.Open()

            # Check image path and model
            if self.MODE == "manual":
                if self.imagePath60 =="":
                    
                    sageBox.about(self, "Warning", "Chưa chọn đường dẫn ảnh.")
                    self.startButton60.setChecked(False)
                    return
                self.timerProcess60.stop()

            if self.MODE == "auto":
                if self.imageFolderPath60 == "":
                    QMessageBox.about(self, "Warning", "Chưa chọn đường dẫn thư mục ảnh.")
                    self.startButton60.setChecked(False)
                    return
                
            self.tableInfoImage60.setItem(2,0, QTableWidgetItem("Đang cập nhật ..."))

            # Start access image folder and read image
            self.processImage60()
            #if self.MODE == "auto":
            self.timerProcess60.timeout.connect(self.processImage60)
            self.timerProcess60.start(15000)

            if self.result_predict60 !=-100:
                self.timerSendData60.timeout.connect(self.intervalSendData) 
                self.timerSendData60.start(1000)

        else: # Stop
            self.startButton60.setStyleSheet("color: gray; font-weight: bold")    
            self.tableInfoImage60.setItem(2,0, QTableWidgetItem("Tạm dừng."))
            self.timerProcess60.stop()
            self.timerSendData60.stop()

    def processImage60(self):
        # Clear image and information before read image
        self.labelImg60.clear()
        self.tableInfoImage60.setItem(0,0, QTableWidgetItem(""))
        self.tableInfoImage60.setItem(1,0, QTableWidgetItem(""))
        if self.MODE == "auto":
            if self.imageFolderPath60 == "":
                return
            checkImage = listdir(self.imageFolderPath60)
            if checkImage == []:
                return
            fileImg = checkImage[0]
            if fileImg.endswith(".JPG") or fileImg.endswith(".jpg") or fileImg.endswith(".png") or fileImg.endswith(".PNG") or fileImg.endswith(".jpeg") or fileImg.endswith(".JPEG"):
                path_of_image = path.join(self.imageFolderPath60, fileImg)
                path_of_processed = path.join(path.abspath(PROCESSED_IMAGE_FOLDER), "60")
                makedirs(path_of_processed, exist_ok=True)
            else:
                return
        
        if self.MODE == "manual":
            if self.imagePath60 =="":
                return
            path_of_image = self.imagePath60
            fileImg = self.imagePath60.split("/")[-1]

        self.pixmap60 = QPixmap(path_of_image)
        height_image = self.pixmap60.size().height()
        width_image = self.pixmap60.size().width()
        self.pixmap60 = self.pixmap60.scaled(int(0.6*self.column_w)-10, self.height//2.5)
        self.labelImg60.setPixmap(self.pixmap60)
        # Fill info image
        self.tableInfoImage60.setItem(0,0, QTableWidgetItem(fileImg))
        self.tableInfoImage60.setItem(1,0, QTableWidgetItem(str(height_image)+" x "+str(width_image))) 
        # Predict image
        self.result_predict60, self.processedTime60 = predict_image(path_of_image, "60")
        # print(self.result_predict60)
        self.result_predict_string60 = str(self.result_predict60)
        # Case 60.0 or 9.xx
        if len(self.result_predict_string60) == 4:
            if self.result_predict60 >=10:
                self.result_predict_string60 = self.result_predict_string60 + "0"
            else:
                self.result_predict_string60 = "0" + self.result_predict_string60
        # Case 9.0
        if len(self.result_predict_string60) == 3:
            if self.result_predict60 <10:
                self.result_predict_string60 = "0" + self.result_predict_string60 + "0"
            else:
                self.result_predict_string60 = self.result_predict_string60 + "00"
        # Show result 
        self.frameResult60.setText(self.result_predict_string60 + " %")
        self.showResultTable60.setItem(0,0, QTableWidgetItem(str(self.processedTime60)+"s"))
        self.showResultTable60.setItem(0,1, QTableWidgetItem(self.result_predict_string60+"%"))
        if self.result_predict60 > 85:
            self.showResultTable60.setItem(0,2, QTableWidgetItem("Đạt"))
        else:
            self.showResultTable60.setItem(0,2, QTableWidgetItem("Chưa đạt"))

        try:
            print(path_of_image, path_of_processed)
            move(path_of_image, path_of_processed)
        except:
            print("Error: Image has existed in path")
        self.result_predict_sent60 = self.result_predict_string60[0] + self.result_predict_string60[1] + self.result_predict_string60[3] + self.result_predict_string60[4]
        
        crc16 = calcString( self.result_predict_sent60, INITIAL_MODBUS)
        crc16_h = crc16 >> 8
        crc16_l = crc16 & 0xff
        self.crc16_msg60 = [crc16_l, crc16_h]
        self.Sender(self.result_predict_sent60, self.crc16_msg60)

        print("CRC : ",crc16)
        print("Sent data: ", self.result_predict_sent60, self.crc16_msg60) 
        # Example: 1% => 0100 (48 49 48 48 in ascii) convert to hex: 30 31 30 30
        # Return crc: 74 - byte low; 255 - byte high; convert to hex: 4A, FF

        # Export data
        self.exportData2CSV("60", fileImg, self.processedTime60, self.result_predict60)    
        
    # START FUNCTION 90
    def checkStartFunction90(self):
        if self.startButton90.isChecked()==True:
            self.startButton90.setStyleSheet("color: blue; font-weight: bold")
            self.timerProcess90 = QTimer()
            from PLCtab import COMPORTNAME
            # Check COM port
            if COMPORTNAME =="":
                QMessageBox.about(self, "Warning", "Chưa chọn cổng COM.")
                self.startButton90.setChecked(False)
                return
            else:
                self.ser.comportName = COMPORTNAME
                self.ser.Open()

            # Check image path and model
            if self.MODE == "manual":
                if self.imagePath90 =="":
                    QMessageBox.about(self, "Warning", "Chưa chọn đường dẫn ảnh.")
                    self.startButton90.setChecked(False)
                    return
                self.timerProcess90.stop()

            if self.MODE == "auto":
                if self.imageFolderPath90 == "":
                    QMessageBox.about(self, "Warning", "Chưa chọn đường dẫn thư mục ảnh.")
                    self.startButton90.setChecked(False)
                    return
                
            self.tableInfoImage90.setItem(2,0, QTableWidgetItem("Đang cập nhật ..."))
            # Start access image folder and read image
            self.processImage90()
            #if self.MODE == "auto":
            self.timerProcess90.timeout.connect(self.processImage90)
            self.timerProcess90.start(15000)

        else: # Stop
            self.startButton90.setStyleSheet("color: gray; font-weight: bold")    
            self.tableInfoImage90.setItem(2,0, QTableWidgetItem("Tạm dừng."))
            self.timerProcess90.stop()

    def processImage90(self):
        # Clear image and infomation before read folder image
        self.labelImg90.clear()
        self.tableInfoImage90.setItem(0,0, QTableWidgetItem(""))
        self.tableInfoImage90.setItem(1,0, QTableWidgetItem(""))
        if self.MODE == "auto":
            if self.imageFolderPath90 == "":
                return
            checkImage = listdir(self.imageFolderPath90)
            
            if checkImage == []:
                return
            
            fileImg = checkImage[0]
            if fileImg.endswith(".JPG") or fileImg.endswith(".jpg") or fileImg.endswith(".png") or fileImg.endswith(".PNG") or fileImg.endswith(".jpeg") or fileImg.endswith(".JPEG"):
                path_of_image = path.join(self.imageFolderPath90, fileImg)
                path_of_processed = path.join(path.abspath(PROCESSED_IMAGE_FOLDER), "90")
                makedirs(path_of_processed, exist_ok=True)
            else:
                return

        if self.MODE == "manual":
            if self.imagePath90 =="":
                return
            path_of_image = self.imagePath90
            fileImg = self.imagePath90.split("/")[-1]

        self.pixmap90 = QPixmap(path_of_image)
        height_image = self.pixmap90.size().height()
        width_image = self.pixmap90.size().width()
        self.pixmap90 = self.pixmap90.scaled(int(0.6*self.column_w)-10, self.height//2.5)
        self.labelImg90.setPixmap(self.pixmap90)
        self.tableInfoImage90.setItem(0,0, QTableWidgetItem(fileImg))
        self.tableInfoImage90.setItem(1,0, QTableWidgetItem(str(height_image)+" x "+str(width_image)))
        # Predict image
        self.result_predict90, self.processedTime90 = predict_image(path_of_image, "90")
        self.result_predict_string90 = str(self.result_predict90)
        # Case 60.0 or 9.xx
        if len(self.result_predict_string90) == 4:
            if self.result_predict90 >=10:
                self.result_predict_string90 = self.result_predict_string90 + "0"
            else:
                self.result_predict_string90 = "0" + self.result_predict_string90
        # Case 9.0
        if len(self.result_predict_string90) == 3:
            if self.result_predict90 <10:
                self.result_predict_string90 = "0" + self.result_predict_string90 + "0"
            else:
                self.result_predict_string90 = self.result_predict_string90 + "00"
        # Show result 
        self.frameResult90.setText(self.result_predict_string90 + " %")
        self.showResultTable90.setItem(0,0, QTableWidgetItem(str(self.processedTime90)+"s"))
        self.showResultTable90.setItem(0,1, QTableWidgetItem(self.result_predict_string90+"%"))
        if self.result_predict90 > 90:
            self.showResultTable90.setItem(0,2, QTableWidgetItem("Đạt"))
        else:
            self.showResultTable90.setItem(0,2, QTableWidgetItem("Chưa đạt"))
        try:
            move(path_of_image, path_of_processed)
        except:
            print("Error: Image has existed in path")
        self.result_predict_sent90 = self.result_predict_string90[0] + self.result_predict_string90[1] + self.result_predict_string90[3] + self.result_predict_string90[4]
        # Export data
        self.exportData2CSV("90", fileImg, self.processedTime90, self.result_predict90)    

    # START FUNCTION 120
    def checkStartFunction120(self):
        if self.startButton120.isChecked()==True:
            self.startButton120.setStyleSheet("color: blue; font-weight: bold")
            self.timerProcess120 = QTimer()
            from PLCtab import COMPORTNAME
            # Check COM port
            if COMPORTNAME =="":
                QMessageBox.about(self, "Warning", "Chưa chọn cổng COM.")
                self.startButton120.setChecked(False)
                return
            else:
                self.ser.comportName = COMPORTNAME
                self.ser.Open()    

            # Check image path and model
            if self.MODE == "manual":
                if self.imagePath120 =="":
                    QMessageBox.about(self, "Warning", "Chưa chọn đường dẫn ảnh.")
                    self.startButton120.setChecked(False)
                    return
                self.timerProcess120.stop()

            if self.MODE == "auto":
                if self.imageFolderPath120 == "":
                    QMessageBox.about(self, "Warning", "Chưa chọn đường dẫn thư mục ảnh.")
                    self.startButton120.setChecked(False)
                    return
                          
            self.tableInfoImage120.setItem(2,0, QTableWidgetItem("Đang cập nhật ..."))
            # Start access image folder and read image
            self.processImage120()
            #if self.MODE == "auto":
            self.timerProcess120.timeout.connect(self.processImage120)
            self.timerProcess120.start(15000)     

        else: # Stop
            self.startButton120.setStyleSheet("color: gray; font-weight: bold")    
            self.tableInfoImage120.setItem(2,0, QTableWidgetItem("Tạm dừng."))
            self.timerProcess120.stop()

    def processImage120(self):
        # Clear image and infomation before read folder image
        self.labelImg120.clear()
        self.tableInfoImage120.setItem(0,0, QTableWidgetItem(""))
        self.tableInfoImage120.setItem(1,0, QTableWidgetItem(""))
        if self.MODE == "auto":
            if self.imageFolderPath120 == "":
                return
            checkImage = listdir(self.imageFolderPath120)
            
            if checkImage == []:
                return

            fileImg = checkImage[0]
            if fileImg.endswith(".JPG") or fileImg.endswith(".jpg") or fileImg.endswith(".png") or fileImg.endswith(".PNG") or fileImg.endswith(".jpeg") or fileImg.endswith(".JPEG"):
                path_of_image = path.join(self.imageFolderPath120, fileImg)
                path_of_processed = path.join(path.abspath(PROCESSED_IMAGE_FOLDER), "120")
                makedirs(path_of_processed, exist_ok=True)
            else:
                return

        if self.MODE == "manual":
            if self.imagePath120 =="":
                return
            path_of_image = self.imagePath120
            fileImg = self.imagePath120.split("/")[-1]

        self.pixmap120 = QPixmap(path_of_image)
        height_image = self.pixmap120.size().height()
        width_image = self.pixmap120.size().width()
        self.pixmap120 = self.pixmap120.scaled(int(0.6*self.column_w)-10, self.height//2.5)
        self.labelImg120.setPixmap(self.pixmap120)
        # Fill info image
        self.tableInfoImage120.setItem(0,0, QTableWidgetItem(fileImg))
        self.tableInfoImage120.setItem(1,0, QTableWidgetItem(str(height_image)+" x "+str(width_image))) 
        # Predict image
        self.result_predict120, self.processedTime120 = predict_image(path_of_image, "120")
        self.result_predict_string120 = str(self.result_predict120)
        # Case 60.0 or 9.xx
        if len(self.result_predict_string120) == 4:
            if self.result_predict120 >=10:
                self.result_predict_string120 = self.result_predict_string120 + "0"
            else:
                self.result_predict_string120 = "0" + self.result_predict_string120
        # Case 9.0
        if len(self.result_predict_string120) == 3:
            if self.result_predict120 <10:
                self.result_predict_string120 = "0" + self.result_predict_string120 + "0"
            else:
                self.result_predict_string120 = self.result_predict_string120 + "00"
        # Show result 
        self.frameResult120.setText(self.result_predict_string120 + " %")
        self.showResultTable120.setItem(0,0, QTableWidgetItem(str(self.processedTime120)+"s"))
        self.showResultTable120.setItem(0,1, QTableWidgetItem(self.result_predict_string120+"%"))
        if self.result_predict120 > 90:
            self.showResultTable120.setItem(0,2, QTableWidgetItem("Đạt"))
        else:
            self.showResultTable120.setItem(0,2, QTableWidgetItem("Chưa đạt"))
        try:
            move(path_of_image, path_of_processed)
        except:
            print("Error: Image has existed in path")
        self.result_predict_sent120 = self.result_predict_string120[0] + self.result_predict_string120[1] + self.result_predict_string120[3] + self.result_predict_string120[4]
        # Export data to excel
        self.exportData2CSV("120", fileImg, self.processedTime120, self.result_predict120)           

    def exportData2CSV(self, groupStage, imageName, processedTime, resultPredict):
        header = ["STT", "Tên ảnh", "Thời gian thu thập", "Thời gian xử lý", "Kết quả", "Trạng thái lên men"]
        now = datetime.now()
        dailyCsv = str(now.strftime("%Y_%m_%d"))
        currentTime = str(now.strftime("%H:%M_%d/%m/%Y"))
        folderCSV = path.join("./CUSTOMIZE_4_USER", "RESULT", groupStage)
        makedirs(folderCSV, exist_ok=True)
        nameCSV = path.join(folderCSV, dailyCsv + ".xlsx")

        if int(resultPredict) >90:
            statusFermentation = "Đ"
        else: 
            statusFermentation = "K"
        last_status = "0"

        if path.exists(nameCSV):
            wbRD = open_workbook(nameCSV)
            sheets = wbRD.sheets()
            wb_obj = Workbook(nameCSV)

            for sheet in sheets: # write data from old file
                newSheet = wb_obj.add_worksheet(sheet.name)
                newSheet.set_column(0, 5, 25)
                stt = sheet.nrows 
                for row in range(sheet.nrows):
                    for col in range(sheet.ncols):
                        newSheet.write(row, col, sheet.cell(row, col).value)

                header[0] = str(stt)
                informationRow = [str(stt), imageName, currentTime, str(processedTime)+" s", str(resultPredict)+" %", statusFermentation]
                for col in range(len(header)):
                    newSheet.write(stt, col, informationRow[col])
            try:
                wb_obj.close() # THIS writes    
            except PermissionError:
                print("Can not write excel file while file is opening!!")            
        
        else:
            #print("First")
            wb_obj = Workbook(nameCSV)
            newSheet = wb_obj.add_worksheet()
            newSheet.set_column(0, 5, 25)
            header = ["STT", "Tên ảnh", "Thời gian thu thập", "Thời gian xử lý", "Kết quả", "Trạng thái lên men"]
            for col in range(len(header)):
                newSheet.write(0, col, header[col])
            stt = 1    
            informationRow = [str(stt), imageName, currentTime, str(processedTime)+" s", str(resultPredict)+" %", statusFermentation]
            for col in range(len(header)):
                newSheet.write(stt, col, informationRow[col])

            try:
                wb_obj.close() # THIS writes    
            except PermissionError:
                print("Can not write excel file while file is opening!!") 

    def intervalSendData(self):
        self.Sender(self.result_predict_sent60, self.crc16_msg60)

    def Sender(self, msg, crc_msg):
        msg = str(msg)
        if self.ser.IsOpen():
            self.ser.Send(msg[0])
            self.ser.Send(msg[1])
            self.ser.Send(msg[2])
            self.ser.Send(msg[3])
            self.ser.Send_CRC(crc_msg[0])
            self.ser.Send_CRC(crc_msg[1])
        else:
            self.ser.Open()
            if self.ser.IsOpen():
                self.ser.Send(msg[0])
                self.ser.Send(msg[1])
                self.ser.Send(msg[2])
                self.ser.Send(msg[3])
                self.ser.Send_CRC(crc_msg[0])
                self.ser.Send_CRC(crc_msg[1])
            

class SerialPort:
    def __init__(self):
        self.comportName = ""
        self.isopen = False
        self.baudrate = 9600
        self.parity = PARITY_NONE
        self.stopbits = STOPBITS_ONE
        self.bytesize = EIGHTBITS
        self.timeout = None
        self.serialport = Serial()

    def __del__(self):
        try:
            if self.IsOpen():
                self.serialport.close()
                print("COM port closed")
        except BaseException as e:
            print(e)
            print("Destructor error closing COM port: ")

    def IsOpen(self):
        return self.isopen

    def Open(self):
        if not self.isopen:
            # serialPort = 'comportName', baudrate, bytesize = 8, parity = 'N',
            # stopbits = 1, timeout = None, xonxoff = 0, rtscts = 0)
            self.serialport.port = self.comportName
            self.serialport.baudrate = self.baudrate
            try:
                self.serialport.open()
                self.isopen = True
                print("COM port opened")
            except BaseException as e:
                print(e)
                print("Error opening COM port: ")

    def Close(self):
        if self.isopen:
            try:
                self.serialport.close()
                self.isopen = False
            except BaseException as e:
                print(e)
                print("Close error closing COM port: ")

    def Send(self, message):
        if self.isopen:
            try:
                self.serialport.write(message.encode('utf_8'))
                return True
            except BaseException as e:
                print(e)
                print("Error sending message: ")
        else:
            return False

    def Send_CRC(self, message):
        if self.isopen:
            try:
                self.serialport.write(self.auxility(message))
                return True
            except BaseException as e:
                print(e)
                print("Error sending message: ")
        else:
            return False

    def auxility(self, i):
        return i.to_bytes(1, byteorder='little', signed=False)
# Chuyển số thập phân thành dạng float
# self.serialport.write(bytearray(struct.pack("f", value)))
