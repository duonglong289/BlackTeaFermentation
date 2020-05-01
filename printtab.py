from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
#from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

class PrintTab(QWidget):
    def __init__(self, parent):
        super(PrintTab, self).__init__(parent)
        self.left = 0
        self.top = 0
        self.height = parent.height
        self.width = parent.width
        self.font18 = parent.font18
        self.font14 = parent.font14
        self.PrintTabUI()

    def PrintTabUI(self):
        self.printingWidget = QWidget(self)
        self.printingWidget.setGeometry(self.left, self.top, self.width, self.height)
        
        # GROUP STAGE 60
        # Create a button in the window
        self.folderImageButton60 = QPushButton('Dữ liệu 60 phút', self.printingWidget)
        self.folderImageButton60.setGeometry(self.left + 10, self.height//7, self.width//7, self.height//10)
        self.folderImageButton60.setFont(self.font14)
        self.folderImageButton60.setStyleSheet("background-color: yellow; font-weight: bold")
        self.folderImageButton60.setIcon(QIcon("folder.png"))
        
        self.training60 = QLabel("Chọn file cần in", self.printingWidget)
        self.training60.setGeometry(self.width//7 + 30, self.height//7, self.width//5, self.height//10)
        self.training60.setFont(self.font14)
        self.training60.setStyleSheet("font-weight: bold")

        # GROUP STAGE 90
        # Create a button in the window
        self.folderImageButton90 = QPushButton('Dữ liệu 90 phút', self.printingWidget)
        self.folderImageButton90.setGeometry(self.left + 10, 3*self.height//7, self.width//7, self.height//10)
        self.folderImageButton90.setFont(self.font14)
        self.folderImageButton90.setStyleSheet("background-color: yellow; font-weight: bold")
        self.folderImageButton90.setIcon(QIcon("folder.png"))
        
        self.training90 = QLabel("Chọn file cần in", self.printingWidget)
        self.training90.setGeometry(self.width//7 + 30, 3*self.height//7, self.width//5, self.height//10)
        self.training90.setFont(self.font14)
        self.training90.setStyleSheet("font-weight: bold")

        # GROUP STAGE 120
        # Create a button in the window
        self.folderImageButton120 = QPushButton('Dữ liệu 120 phút', self.printingWidget)
        self.folderImageButton120.setGeometry(self.left + 10, 5*self.height//7, self.width//7, self.height//10)
        self.folderImageButton120.setFont(self.font14)
        self.folderImageButton120.setStyleSheet("background-color: yellow; font-weight: bold")
        self.folderImageButton120.setIcon(QIcon("folder.png"))
        
        self.training120 = QLabel("Chọn file cần in", self.printingWidget)
        self.training120.setGeometry(self.width//7 + 30, 5*self.height//7, self.width//5, self.height//10)
        self.training120.setFont(self.font14)
        self.training120.setStyleSheet("font-weight: bold")

        # PRING BUTTON
        self.printButton = QPushButton('In', self.printingWidget)
        self.printButton.setGeometry(self.width//2, 3*self.height//7, self.width//7, self.height//10)
        self.printButton.setFont(self.font14)
        self.printButton.setStyleSheet("background-color: yellow; font-weight: bold")

    def on_click(self):
        pass
    