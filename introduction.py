from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel

class Introduction(QGroupBox):
    def __init__(self, parent):
        super(Introduction, self).__init__(parent)

        self.font18 = parent.font18
        self.font16 = parent.font16
        self.font15 = parent.font15
        self.font14 = parent.font14
        self.font13 = parent.font13
        self.font12 = parent.font12
        self.font11 = parent.font11
        self.font10 = parent.font10

        self.left = parent.left
        self.top = parent.top
        self.width = parent.width
        self.height = int(parent.height * 0.22)

        introductionGB_l = 0
        introductionGB_t = -10
        introductionGB_w = self.width
        introductionGB_h = self.height

        self.setGeometry(
            introductionGB_l,
            introductionGB_t,
            introductionGB_w,
            introductionGB_h)
        self.setStyleSheet("QGroupBox{padding-top:0px; margin-top:-20px}")

        # LOGO INTRODUCTION
        logoIntro = QWidget(self)
        logoIntro.setGeometry(
            introductionGB_l,
            introductionGB_t,
            introductionGB_h,
            introductionGB_h)
        h_22 = int(introductionGB_h * 0.22)
        h_06 = int(introductionGB_h * 0.6)
        logo = QLabel(logoIntro)
        logo.move(h_22, 50)
        pixmap = QPixmap("logotron.png")
        pixmap = pixmap.scaled(h_06, h_06)
        logo.setPixmap(pixmap)

        d = introductionGB_w - introductionGB_h
        b = introductionGB_w * 0.4 + introductionGB_h * 0.6
        
        # LEFT INTRODUCTION
        leftIntro = QLabel(self)
        leftIntro.setGeometry(
            introductionGB_h, introductionGB_t, int(
                d * 0.4), introductionGB_h)
        leftIntro.setText(
            "BỘ NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN\nVIỆN KHOA HỌC KĨ THUẬT NÔNG LÂM NGHIỆP\nMIỀN NÚI PHÍA BẮC")
        leftIntro.setAlignment(Qt.AlignCenter)

        # RIGHT INTRODUCTION
        # Create widget
        widgetIntroRight = QWidget(self)
        widgetIntroRight.setGeometry(
            int(b), introductionGB_t, int(
                d * 0.6), introductionGB_h)

        # Create layout in widget
        introRightLayout = QVBoxLayout(widgetIntroRight)
        introRightLayout.setAlignment(Qt.AlignCenter)

        # Create label in layout
        rightIntro1 = QLabel("DỰ ÁN SẢN XUẤT THỬ NGHIỆM")
        rightIntro1.setAlignment(Qt.AlignCenter)

        rightIntro2 = QLabel(
            "Hoàn thiện quy trình công nghệ kiểm soát tính sinh học của hệ Enzym trong quá trình lên men chè đen")
        rightIntro2.setAlignment(Qt.AlignCenter)

        rightIntro3 = QLabel(
            "và thử nghiệm ứng dụng kiểm soát tự động quá trình lên men trên quy mô sản xuất bán công nghiệp.")
        rightIntro3.setAlignment(Qt.AlignCenter)

        managerIntro = QLabel("Chủ nhiệm dự án: PHẠM THANH BÌNH")
        managerIntro.setAlignment(Qt.AlignCenter)
        managerIntro.setStyleSheet("QLabel {text-decoration: underline; }")

        mainIntroWidget_l = introductionGB_l
        mainIntroWidget_t = introductionGB_h * 4 // 5
        mainIntroWidget_h = 40 if self.height < 800 else 50
        mainIntroWidget_w = introductionGB_w

        mainIntroWidget = QWidget(self)
        mainIntroWidget.setGeometry(
            mainIntroWidget_l,
            mainIntroWidget_t,
            mainIntroWidget_w,
            mainIntroWidget_h)

        mainIntroLayout = QVBoxLayout(mainIntroWidget)
        mainIntroLabel = QLabel()
        mainIntroLabel.setText(
            "HỆ THỐNG THU THẬP VÀ PHÂN TÍCH QUÁ TRÌNH LÊN MEN CHÈ ĐEN")
        mainIntroLabel.setAlignment(Qt.AlignCenter)

        mainIntroLabel.setStyleSheet("QLabel {color: red;}")
        mainIntroLayout.addWidget(mainIntroLabel)

        mainIntroWidget.setLayout(mainIntroLayout)

        introRightLayout.addWidget(rightIntro1)
        introRightLayout.addWidget(rightIntro2)
        introRightLayout.addWidget(rightIntro3)
        introRightLayout.addWidget(managerIntro)

        if parent.height > 800:
            leftIntro.setFont(self.font18)

            rightIntro1.setFont(self.font16)
            rightIntro2.setFont(self.font15)
            rightIntro3.setFont(self.font15)
            managerIntro.setFont(self.font16)

            mainIntroLabel.setFont(self.font16)

        else:
            leftIntro.setFont(self.font14)

            rightIntro1.setFont(self.font13)
            rightIntro2.setFont(self.font12)
            rightIntro3.setFont(self.font12)
            managerIntro.setFont(self.font13)

            mainIntroLabel.setFont(self.font14)
