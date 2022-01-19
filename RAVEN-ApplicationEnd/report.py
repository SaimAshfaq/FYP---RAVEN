# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'report.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(741, 600)
        MainWindow.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0.0191, y1:0.101636, x2:0.991379, y2:0.977, stop:0 black, stop:1 gray);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(320, 20, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setStyleSheet("color: white;\n"
"background: transparent")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(290, 210, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: white;\n"
"background: transparent")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(170, 80, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: white;\n"
"background: transparent")
        self.label_3.setObjectName("label_3")
        self.SearchEdit2 = QtWidgets.QLineEdit(self.centralwidget)
        self.SearchEdit2.setGeometry(QtCore.QRect(320, 80, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(False)
        self.SearchEdit2.setFont(font)
        self.SearchEdit2.setStyleSheet("background-color: white;\n"
"border-radius: 5px;padding:0px 10px")
        self.SearchEdit2.setObjectName("SearchEdit2")
        self.search_button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.search_button_2.setGeometry(QtCore.QRect(530, 80, 31, 31))
        self.search_button_2.setStyleSheet("border:none;background:white")
        self.search_button_2.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.search_button_2.setIcon(icon)
        self.search_button_2.setIconSize(QtCore.QSize(24, 24))
        self.search_button_2.setAutoDefault(False)
        self.search_button_2.setDefault(False)
        self.search_button_2.setFlat(False)
        self.search_button_2.setObjectName("search_button_2")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(140, 260, 481, 321))
        self.tableWidget.setAutoFillBackground(False)
        self.tableWidget.setStyleSheet("background: lightgray\n"
"\n"
"")
        self.tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget.setAlternatingRowColors(False)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setRowCount(0)
        self.rep_btn = QtWidgets.QPushButton(self.centralwidget)
        self.rep_btn.setGeometry(QtCore.QRect(340, 120, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(-1)
        self.rep_btn.setFont(font)
        self.rep_btn.setStyleSheet("QPushButton#rep_btn{\n"
"   background-color:rgba(0,0,0,0.5);\n"
"    border-radius: 6px;\n"
"    color: white;\n"
"    font-size: 18px \n"
"}\n"
"\n"
"QPushButton#rep_btn:hover{\n"
"    background-color: qlineargradient(spread:pad, x1:0.0191, y1:0.101636, x2:0.991379, y2:0.977, stop:0 black, stop:1 gray);\n"
"    color: \'white\';\n"
"}")
        self.rep_btn.setObjectName("rep_btn")
        self.error_2 = QtWidgets.QLabel(self.centralwidget)
        self.error_2.setGeometry(QtCore.QRect(220, 179, 351, 31))
        self.error_2.setStyleSheet("color: red;\n"
"font-weight: 600;\n"
"background: transparent;\n"
"font-size: 14px;")
        self.error_2.setText("")
        self.error_2.setObjectName("error_2")
        self.switch_page = QtWidgets.QPushButton(self.centralwidget)
        self.switch_page.setGeometry(QtCore.QRect(10, 10, 41, 41))
        self.switch_page.setStyleSheet("background: transparent")
        self.switch_page.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("previous.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.switch_page.setIcon(icon1)
        self.switch_page.setIconSize(QtCore.QSize(32, 20))
        self.switch_page.setObjectName("switch_page")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "REPORT"))
        self.label_2.setText(_translate("MainWindow", "APPEARANCE DETAILS"))
        self.label_3.setText(_translate("MainWindow", "View Report by Id:"))
        self.SearchEdit2.setPlaceholderText(_translate("MainWindow", "  Enter ID"))
        self.rep_btn.setText(_translate("MainWindow", "View Report of all ID\'s"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

