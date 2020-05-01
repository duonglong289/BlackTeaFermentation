from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

COMPORTNAME = ""

class PLCTab(QWidget):
    def __init__(self, parent):
        super(PLCTab, self).__init__(parent)
        self.left = 0
        self.top = 0
        self.height = parent.height
        self.width = parent.width
        self.font18 = parent.font18
        self.PLCTabUI()

    def PLCTabUI(self):
        self.plcPanelWidget = QWidget(self)
        self.plcPanelWidget.setGeometry(self.left, self.top, self.width, self.height)

        # Create label
        self.labelCOM = QLabel("Cổng truyền thông", self.plcPanelWidget)
        self.labelCOM.setGeometry(self.left, self.height//3, self.width//5, 100)
        self.labelCOM.setFont(self.font18)
        self.labelCOM.setAlignment(Qt.AlignCenter)
        self.labelCOM.setStyleSheet("background-color: yellow; font-weight: bold")

        # Create textbox
        self.textbox = QLineEdit(self.plcPanelWidget)
        self.textbox.setGeometry(self.width//5 + 20, self.height//3, self.width//5, 100)
        self.textbox.setFont(self.font18)
        self.textbox.setAlignment(Qt.AlignCenter)

        # Create a button in the window
        self.button = QPushButton('Kết nối PLC', self.plcPanelWidget)
        self.button.setGeometry(2*self.width//5 + 40, self.height//3, self.width//5, 100)
        self.button.setFont(self.font18)
        self.button.setStyleSheet("background-color: yellow; font-weight: bold")
    
        # connect button to function on_click
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        comportName = self.textbox.text().upper().strip()
        print("Đã chọn cổng ", comportName)
        global COMPORTNAME
        COMPORTNAME = comportName
        if comportName == "":
            return
        