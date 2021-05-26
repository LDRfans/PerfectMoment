import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import cv2
import numpy as np
img = cv2.imread("lena.tiff")
class DataSet:
    def __init__(self, m, n):
        self.selected_matrix = np.zeros((m, n))
    def ClearForReselection(self, coord_x, coord_y):
        for i in range(coord_x):
            if self.selected_matrix[i,coord_y] != 0:
                self.selected_matrix[i,coord_y] = 0
    def SetSelected(self, coord_x, coord_y):
        self.ClearForReselection(coord_x, coord_y)
        self.selected_matrix[coord_x, coord_y] = 1
class SelectBoard(QMainWindow):
    def __init__(self, m, n):
        # super(SelectBoard, self).__init__()
        super().__init__()
        self.num_people = m
        self.num_picture = n
        self.selectButtonDict = {}

        self.setWindowTitle('Image Selection')
        self.setFixedSize(512,512)
        self.generalLayout = QVBoxLayout()

        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        
        self.SetDisplayWindow()
        self.SetSelectionWindow(self.num_people, self.num_picture)

    def SetDisplayWindow(self):
        self.display = QLineEdit()
        self.display.setFixedHeight(35)
        self.display.setAlignment(Qt.AlignRight)
        self.display.setReadOnly(True)
        self.generalLayout.addWidget(self.display)

    def SetSelectionWindow(self, m, n):
        self.selectGrid = QGridLayout()
        self.selection_data = DataSet(m, n)
        for i in range(m):
            for j in range(n):
                button = QPushButton("")
                self.selectButtonDict[button] = (i,j)
                button.clicked.connect(self.CheckClicked)
                self.selectGrid.addWidget(button, i, j)
                
        self.generalLayout.addLayout(self.selectGrid)
        print(self.selectButtonDict)

    def CheckClicked(self):
        print("Clicked button " + str(self.selectButtonDict[self.sender()]))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    m,n = 3,3
    view = SelectBoard(m,n)
    view.show()
    
    sys.exit(app.exec_())