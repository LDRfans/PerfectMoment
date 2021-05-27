import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox
from PyQt5.QtGui import QIcon, QImage, QPixmap
import cv2
import numpy as np
class DataSet:
    def __init__(self, m, n):
        # TODO: Get the image list of the face detection, then input parameter m,n, self.num_people and self.num_picture can be replaced
        self.face_set = np.asarray([[cv2.imread("lena.tiff"),cv2.imread("lena.tiff"),cv2.imread("lena.tiff")],[cv2.imread("lena.tiff"),cv2.imread("lena.tiff"),cv2.imread("lena.tiff")],[cv2.imread("lena.tiff"),cv2.imread("lena.tiff"),cv2.imread("lena.tiff")]])
        self.image_pack = [cv2.imread("lena.tiff"), cv2.imread("q2.jpeg"), cv2.imread("q1_1.jpeg")]
        self.num_people = m
        self.num_picture = n
        self.selected_matrix = np.zeros((m, n))
        # TODO: This is the result output image data
        self.display_image = None
        self.display_image_data = None
    def ClearForReselection(self, coord_x, coord_y):
        # for i in range(coord_x):
            # if self.selected_matrix[i,coord_y] != 0:
        self.selected_matrix[:,coord_y] = 0
    def SetSelected(self, coord_x, coord_y):
        self.ClearForReselection(coord_x, coord_y)
        self.selected_matrix[coord_x, coord_y] = 1
    def ClearAll(self):
        for i in range(self.num_people):
            for j in range(self.num_picture):
                self.selected_matrix[i,j] = 0
    def GetSelectedBackgroundImage(self):
        return self.display_image_data
    def GetSelectedFaces(self):
        return self.selected_matrix
class SelectBoard(QMainWindow):
    def __init__(self, m, n):
        # super(SelectBoard, self).__init__()
        super().__init__()
        self.selection_data = DataSet(m, n)
        self.selectButtonDict = {}

        self.setWindowTitle('Image Selection')
        self.setFixedSize(1024,512)
        self.generalLayout = QHBoxLayout()
        self.subLayout = QVBoxLayout()

        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        
        self.SetDisplayWindow()
        self.SetDisplayImageSelection()
        self.SetSelectionWindow(m, n)

    def SetDisplayWindow(self):
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignLeft)
        self.ShowResultImage(self.selection_data.image_pack[0])
        self.generalLayout.addWidget(self.display)

    def ShowResultImage(self, image):
        self.selection_data.display_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.display.setPixmap(QPixmap.fromImage(self.selection_data.display_image))
        self.display.resize(image.shape[0], image.shape[1])

    def SetDisplayImageSelection(self):
        self.combo = QComboBox(self)
        for i in range(len(self.selection_data.image_pack)):
            self.combo.addItem(str(i))
        self.combo.currentIndexChanged.connect(self.ImageSelectionUpdateDisplay)
        self.subLayout.addWidget(self.combo)
    def ImageSelectionUpdateDisplay(self):
        # print(type(self.combo.currentText()))
        self.selection_data.display_image_data = self.selection_data.image_pack[int(self.combo.currentText())]
        self.ShowResultImage(self.selection_data.display_image_data)
        
    def SetSelectionWindow(self, m, n):
        self.selectGrid = QGridLayout()
        for i in range(m):
            for j in range(n):
                image = self.selection_data.face_set[i,j]
                q_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
                button = QPushButton("")
                button.setIcon(QIcon(QPixmap.fromImage(q_img)))
                button.setAccessibleName(str((i,j)))
                self.selectButtonDict[button] = (i,j)
                button.clicked.connect(self.CheckClicked)
                self.selectGrid.addWidget(button, i, j)
                
        self.subLayout.addLayout(self.selectGrid)
        # for i in self.selectButtonDict.keys():
        #     print(i.accessibleName())
        self.generalLayout.addLayout(self.subLayout)

    def CheckClicked(self):
        # print("Clicked button " + str(self.selectButtonDict[self.sender()]))
        print("Clicked button " + self.sender().accessibleName())
        coord_x, coord_y = self.selectButtonDict[self.sender()]
        self.selection_data.SetSelected(coord_x, coord_y)
        print(self.selection_data.selected_matrix)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    m,n = 3,3
    view = SelectBoard(m,n)
    view.show()
    
    sys.exit(app.exec_())