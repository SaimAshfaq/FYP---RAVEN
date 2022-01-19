# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'viewVideo.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(735, 572)
        MainWindow.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0.0191, y1:0.101636, x2:0.991379, y2:0.977, stop:0 black, stop:1 gray);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(290, 30, 161, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(22)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: white;\n"
"background: transparent")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.SearchEdit2 = QtWidgets.QLineEdit(self.centralwidget)
        self.SearchEdit2.setGeometry(QtCore.QRect(290, 100, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(False)
        self.SearchEdit2.setFont(font)
        self.SearchEdit2.setStyleSheet("background-color: white;\n"
"border-radius: 5px;padding:0px 10px")
        self.SearchEdit2.setObjectName("SearchEdit2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(120, 100, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: white;\n"
"background: transparent")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.videoArea = QtWidgets.QLabel(self.centralwidget)
        self.videoArea.setGeometry(QtCore.QRect(120, 210, 541, 341))
        self.videoArea.setStyleSheet("background: gray\n"
"")
        self.videoArea.setText("")
        self.videoArea.setObjectName("videoArea")
        self.search_error = QtWidgets.QLabel(self.centralwidget)
        self.search_error.setGeometry(QtCore.QRect(200, 140, 381, 31))
        self.search_error.setStyleSheet("background-color: transparent;\n"
"color: red;\n"
"font-size: 16px")
        self.search_error.setText("")
        self.search_error.setObjectName("search_error")
        self.switch_page = QtWidgets.QPushButton(self.centralwidget)
        self.switch_page.setGeometry(QtCore.QRect(20, 20, 41, 41))
        self.switch_page.setStyleSheet("background: transparent")
        self.switch_page.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("previous.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.switch_page.setIcon(icon)
        self.switch_page.setIconSize(QtCore.QSize(32, 20))
        self.switch_page.setObjectName("switch_page")
        self.rep_btn = QtWidgets.QPushButton(self.centralwidget)
        self.rep_btn.setGeometry(QtCore.QRect(500, 100, 41, 31))
        self.rep_btn.setStyleSheet("border:none;background:white")
        self.rep_btn.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rep_btn.setIcon(icon1)
        self.rep_btn.setIconSize(QtCore.QSize(24, 24))
        self.rep_btn.setAutoDefault(False)
        self.rep_btn.setDefault(False)
        self.rep_btn.setFlat(False)
        self.rep_btn.setObjectName("rep_btn")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "View Video"))
        self.SearchEdit2.setPlaceholderText(_translate("MainWindow", "  Enter ID"))
        self.label_3.setText(_translate("MainWindow", "Search video by Id:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

