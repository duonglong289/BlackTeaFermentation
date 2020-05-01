from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QApplication, QWidget, QAction, qApp
from sys import argv
from sys import exit as sys_exit
from maintab import MainTab
from introduction import Introduction
from PLCtab import PLCTab
from training_tab_regression import TrainingTab
from printtab import PrintTab

class MainWindow(QMainWindow):
    def __init__(self, app):
        super(MainWindow, self).__init__()
        screen = app.primaryScreen()
        rect = screen.availableGeometry()
        self.title = " "
        self.left = 0
        self.top = 30
        self.width = rect.width()
        self.height = rect.height() - self.top

        self.initUI()

    def initUI(self):
        # Set font size
        self.font18 = QFont("Times New Roman")
        self.font18.setPointSize(18)
        self.font16 = QFont("Times New Roman")
        self.font16.setPointSize(16)
        self.font15 = QFont("Times New Roman")
        self.font15.setPointSize(15)
        self.font14 = QFont("Times New Roman")
        self.font14.setPointSize(14)
        self.font13 = QFont("Times New Roman")
        self.font13.setPointSize(13)
        self.font12 = QFont("Times New Roman")
        self.font12.setPointSize(12)
        self.font11 = QFont("Times New Roman")
        self.font11.setPointSize(11)
        self.font10 = QFont("Times New Roman")
        self.font10.setPointSize(10)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        centralwidget = QWidget(self)
        self.introductionGroupBox = Introduction(self)

        if self.height > 800:
            self.table_widget = MyTableWidget(self)
        else:
            self.table_widget = MyTableWidget(self)
        self.setCentralWidget(centralwidget)

        exitAct = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('')
        fileMenu.addAction(exitAct)
        
        self.show()


class MyTableWidget(QTabWidget):
    def __init__(self, parent):
        super(QTabWidget, self).__init__(parent)

        self.font18 = parent.font18
        self.font16 = parent.font16
        self.font15 = parent.font15
        self.font14 = parent.font14
        self.font13 = parent.font13
        self.font12 = parent.font12
        self.font11 = parent.font11
        self.font10 = parent.font10

        self.left = parent.left
        self.top = int(parent.height * 0.2)
        self.width = parent.width
        self.height = int(parent.height * 0.8)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.allFont = QFont("Times New Roman")
        self.allFont.setPointSize(14)

        # Create tabs
        self.tabMain = MainTab(self)
        self.tabMain.setFont(self.allFont)

        self.tabPLC = PLCTab(self)

        self.tabTraining = TrainingTab(self)
        self.tabPrinting = PrintTab(self)

        # Add tabs
        self.addTab(self.tabMain, "Main")
        self.addTab(self.tabPLC, "PLC")
        self.addTab(self.tabTraining, "Training")
        self.addTab(self.tabPrinting, "In ấn")


if __name__ == '__main__':
    app = QApplication(argv)
    window = MainWindow(app)
    sys_exit(app.exec_())
