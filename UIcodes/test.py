import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import cv2
img = cv2.imread("lena.tiff")
class App(QWidget):

    def __init__(self,m,n):
        super().__init__()
        self.title = 'PyQt5 button - pythonspot.com'
        self.left = 900
        self.top = 450
        self.width = 512
        self.height = 512
        self.num_person = m
        self.num_picture = n
        self.buttonList = []
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.button = QPushButton('', self)
        self.button.setToolTip('This is an example button')
        self.button.setStyleSheet("background-image : url(lena.tiff);")
        row,col,_ = img.shape
        self.button.resize(row,col)

        # self.button.move(100,70)
        self.button.clicked.connect(self.on_click)

        # grid = QGridLayout(self)
        # for i in range(self.num_person):
        #     for j in range(self.num_picture):
        #         grid.addWidget(QPushButton(""),i,j)
                # button.clicked.connect(self.on_click)
        
        self.show()

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click on: ')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    m,n = 3,3
    ex = App(m,n)
    sys.exit(app.exec_())