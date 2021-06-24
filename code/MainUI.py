import os
import sys
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, \
    QVBoxLayout, QComboBox, QSizePolicy
from PyQt5.QtGui import QIcon, QImage, QPixmap
import cv2
import numpy as np
from Fileop import read_img
from Maskgen import generate_mask, generate_pyramid_mask, generate_all_mask
# from Pyramid import pyramid_blend
from Pyramid_z import pyramid_blend
from Homography import face_to_base
from Extract import extract

RESIZE_RATIO = 0.1
BLEND_BIAS = 0


def imageResize(img, ratio):
    row = img.shape[0]
    col = img.shape[1]
    ret = cv2.resize(img, (int(col * ratio), int(row * ratio)))
    return ret


class DataSet:
    def __init__(self, img_list):
        # TODO: Get the image list of the face detection, NEED FACE DETECTION function to replace the line below
        # Detect the face by applying the mask to the image
        print("Extract")
        self.img_info_list = [extract(img, img_list[0]) for img in img_list]    # select img_0 as img_ref
        self.img_shape_list = [img.shape for img in img_list]
        # self.initial_masks = generate_all_mask(self.img_info_list, self.img_shape_list)
        self.image_pack = img_list

        # Give the faces to UI
        print("Face Set")
        # self.face_set = np.asarray([[cv2.imread("lena.tiff"),cv2.imread("lena.tiff"),cv2.imread("lena.tiff")],[cv2.imread("lena.tiff"),cv2.imread("lena.tiff"),cv2.imread("lena.tiff")]])
        self.face_set = np.asarray(self.generateFaceSet())

        # Each row is a picture, each column is a person, a person can only have one face in all pictures, thus only on 1 in each column
        print("Other")
        self.num_picture = len(self.image_pack)
        self.num_people = self.face_set.shape[1]

        # IMPORTANT TODO: For roubustness, we may consider people more than image is invalid, maybe will need triming in face detection
        # if self.num_people > self.num_picture:
        #     print("There are more people in images provided, will cut out", str(self.num_people - self.num_picture), "people. ")
        #     self.face_set = self.face_set[:, :self.num_picture, :, :, :]
        #     self.num_people = self.num_picture

        self.selected_matrix = np.zeros((self.num_picture, self.num_people))
        self.display_image = None
        self.display_image_data = self.image_pack[0]
        # cv2.imshow('d',self.display_image_data)
        # cv2.waitKey()

    ''' Helper function for SetSelected(), it will be called each time checking one button to make sure
        there is only one button in a column is selected (only one face for a person)'''

    def ClearForReselection(self, coord_y):
        # for i in range(coord_x):
        # if self.selected_matrix[i,coord_y] != 0:
        self.selected_matrix[:, coord_y] = 0

    ''' Helper function for class SelectBoard's CheckClicked(), it will set corresponding clicked button coordinate's
        value to 1'''

    def SetSelected(self, coord_x, coord_y):
        self.ClearForReselection(coord_y)
        self.selected_matrix[coord_x, coord_y] = 1

    ''' Function that clear all the values in the selected_matrix'''

    def ClearAll(self):
        for i in range(self.num_picture):
            for j in range(self.num_people):
                self.selected_matrix[i, j] = 0

    ''' Getter function in order to get the display_image_data (type: row*col*3 np.ndarray, cv2 type),
        which is the user selected background, it can be used for later blending data,
        self.display_image_data is also the place to save the blending result'''

    def GetDisplayImageData(self):
        return self.display_image_data

    ''' Getter function in order to get the selected_matrix (type: num_picture * num_people np.ndarray, 0, 1 matrix),
        which is the user selected faces, it can be used for later blending data'''

    def GetSelectedFaces(self):
        return self.selected_matrix

    ''' Setter function to set the display_image_data, used for result displaying and saving '''

    def SetDisplayImageData(self, img):
        self.display_image_data = img

    # def FaceSetRegulation(self):
    #     m,n = self.face_set.shape[:2]
    #     for i in range(m):
    #         for j in range(n):
    #             image = self.face_set[i,j]
    #             row, col = image.shape[:2]
    #             if row > col:
    #                 denominator = col
    #                 diff = (row - col) // 2
    #                 image = image[diff:row-diff,:,:]
    #             else:
    #                 denominator = row
    #                 diff = (col - row) // 2
    #                 image = image[:,diff:col-diff,:]
    #             ratio = 100 / denominator
    #             self.face_set[i,j] = ImageResize(image, ratio)

    def generateFaceSet(self):
        '''
        Generate the face sets from the masks
        :return: The face sets for the GUI
        '''
        faces = []

        # Get faces with black
        # for i in range(0, len(self.initial_masks)):
        #     picture_masks = self.initial_masks[i]
        #     faces_in_picture = []
        #     for mask in picture_masks:
        #         mask_int = mask.copy()
        #         mask_int = mask_int.astype(np.uint8)
        #         rows, cols, c = mask.shape
        #         face = np.zeros(mask.shape, dtype=np.uint8)
        #         face = cv2.add(face,self.image_pack[i],mask=mask_int)
        #         for x in range(0, rows):
        #             for y in range(0, cols):
        #                 if mask[x, y, 0] != 0:
        #                     face[x, y, :] += self.image_pack[i][x, y, :]
        #         faces_in_picture.append(face)
        #     faces.append(faces_in_picture)
        #
        # # Cut the black part
        # for i in range(0, len(faces)):
        #     picture = faces[i]
        #     faces_in_picture = []
        #     for j in range(0, len(picture)):
        #         person = picture[j]
        #         head_bounding_box = self.img_info_list[i][j][0]
        #         x, y, h, w = head_bounding_box
        #         person_cut = person[y:y + h, x:x + w, :]
        #         person_cut = cv2.resize(person_cut, (300, 300))
        #         faces_in_picture.append(person_cut)
        #     faces_cut.append(faces_in_picture)

        #

        for i in range(0, len(self.img_info_list)):
            picture_info = self.img_info_list[i]
            faces_in_picture = []
            for person in picture_info:
                x, y, w, h = person[0]
                face = self.image_pack[i][y:y + h, x:x + w, :]
                face = cv2.resize(face, (100, 100))
                faces_in_picture.append(face)
            faces.append(faces_in_picture)

        return faces


class SelectBoard(QMainWindow):
    def __init__(self, img_pack):
        # super(SelectBoard, self).__init__()
        super().__init__()
        self.selection_data = DataSet(img_pack)
        self.selectButtonDict = {}

        self.height = 0
        self.width = 0
        for img in img_pack:
            row, col, _ = img.shape
            if row > self.height:
                self.height = row // 2
            if col > self.width:
                self.width = col // 2 + 50
        default_size = self.selection_data.num_picture * 100 + 50
        if self.height < default_size:
            self.height = default_size
        self.width = self.width + self.selection_data.num_people * 100
        # ------Set out the structure of the UI window------#
        self.setWindowTitle('Image Selection')
        self.setFixedSize(self.width, self.height)
        self.generalLayout = QHBoxLayout()
        self.subLayout = QVBoxLayout()
        self.optionsLayout = QHBoxLayout()

        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)

        self.SetDisplayWindow()
        self.SetDisplayImageSelection()
        self.SetSelectionWindow(self.selection_data.num_picture, self.selection_data.num_people)
        self.SetOptionButtons()

        self.subLayout.addLayout(self.optionsLayout)
        self.generalLayout.addLayout(self.subLayout)
        # self.generalLayout.addStretch()
        self.subLayout.addStretch()

    # ---Functions that set up the left result and background display window---#
    def SetDisplayWindow(self):
        self.display = QLabel()
        self.display.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.display.setScaledContents(True)
        self.display.setAlignment(Qt.AlignLeft)
        self.display.setScaledContents(True)
        self.ShowResultImage(self.selection_data.image_pack[0])
        self.generalLayout.addWidget(self.display)

    ''' Display the parameter IMAGE(type: numpy.ndarray) on the left window, convert function'''

    def ShowResultImage(self, image):

        # Resize the image for better display
        image_small = image.copy()
        shape = image_small.shape
        image_small = cv2.resize(image_small, (shape[1] // 2, shape[0] // 2))
        # image_small = image

        height, width, channel = image_small.shape
        bytes_perLine = channel * width

        # self.display.resize(image.shape[0], image.shape[1])
        self.selection_data.display_image = QImage(image_small.data, width, height, bytes_perLine,
                                                   QImage.Format_RGB888).rgbSwapped()
        self.display.setPixmap(QPixmap.fromImage(self.selection_data.display_image))

    # ---Functions that set up the right drop down menu for background selection---#
    def SetDisplayImageSelection(self):
        self.combo = QComboBox(self)
        for i in range(len(self.selection_data.image_pack)):
            self.combo.addItem(str(i))
        self.combo.currentIndexChanged.connect(self.ImageSelectionUpdateDisplay)
        self.subLayout.addWidget(self.combo)

    ''' Helper function for SetDisplayImageSelection() 
        Initially add all the input images (original images) in self.selection_data.image_pack (type: numpy.ndarray, m pictures) for background selection,
        will save the data result (type: numpy.ndarray, 1 picture) in self.selection_data.display_image_data'''

    def ImageSelectionUpdateDisplay(self):
        # print(type(self.combo.currentText()))
        # self.selection_data.display_image_data = self.selection_data.image_pack[int(self.combo.currentText())]
        self.selection_data.SetDisplayImageData(self.selection_data.image_pack[int(self.combo.currentText())])
        self.ShowResultImage(self.selection_data.display_image_data)

    def ClearSelectedBackground(self):
        # self.selection_data.display_image_data = self.selection_data.image_pack[0]
        self.selection_data.SetDisplayImageData(self.selection_data.image_pack[0])
        self.combo.setCurrentText('0')
        self.ShowResultImage(self.selection_data.display_image_data)

    # ---Functions that set up the right selection pad for faces selections---#
    def SetSelectionWindow(self, m, n):
        self.selectGrid = QGridLayout()
        for i in range(m):
            for j in range(n):
                image = self.selection_data.face_set[i, j]
                q_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
                button = QPushButton("")
                # button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                button.setCheckable(True)
                button.setIcon(QIcon(QPixmap.fromImage(q_img)))
                button.setIconSize(QSize(100, 100))
                # button.setIconSize(QSize(button.size().width(), button.size().height()))
                button.setAccessibleName(str((i, j)))
                self.selectButtonDict[button] = (i, j)
                button.clicked.connect(self.CheckClicked)
                self.selectGrid.addWidget(button, i, j)

        self.subLayout.addLayout(self.selectGrid)
        # for i in self.selectButtonDict.keys():
        #     print(i.accessibleName())

    ''' Callback function for button.clicked.connect() (internal function),
        it will give out the clicked button's coordinates and pass it to self.selection_data.selected_matrix for record,
        for functions in self.selection_data, see class DataSet for details'''

    def CheckClicked(self):
        # print("Clicked button " + str(self.selectButtonDict[self.sender()]))
        print("Clicked button " + self.sender().accessibleName())
        coord_x, coord_y = self.selectButtonDict[self.sender()]
        self.MarkClickedEachRow(coord_x, coord_y)
        self.selection_data.SetSelected(coord_x, coord_y)
        print(self.selection_data.selected_matrix)

    ''' Visual function that mark selected button red and pop up inappropriate buttons automatically'''

    def MarkClickedEachRow(self, coord_x, coord_y):
        for btn in self.selectButtonDict.keys():
            if btn.isChecked():
                btn.setStyleSheet("background-color : red")
            x, y = self.selectButtonDict[btn]
            if y == coord_y and x != coord_x:
                btn.setChecked(False)
                btn.setStyleSheet("background-color : none")

    ''' Visual function that clear out all selected buttons'''

    def ClearAllClickedMarks(self):
        for btn in self.selectButtonDict.keys():
            btn.setChecked(False)
            btn.setStyleSheet("background-color : none")

    # ---Functions that set up the right options buttons group---#
    def SetOptionButtons(self):
        self.btn_ok = QPushButton("OK")
        self.btn_reset = QPushButton("Reset")
        self.btn_save = QPushButton("Save")
        self.optionsLayout.addWidget(self.btn_ok)
        self.optionsLayout.addWidget(self.btn_reset)
        self.optionsLayout.addWidget(self.btn_save)

        self.btn_reset.clicked.connect(self.Reset)
        self.btn_save.clicked.connect(self.Save)
        self.btn_ok.clicked.connect(self.Confirm)

    ''' Callback function for self.btn_reset, it will reset self.selected_data.selected_matrix to all 0,
        make it right for visual effects'''

    def Reset(self):
        self.selection_data.ClearAll()
        self.ClearAllClickedMarks()
        self.ClearSelectedBackground()
        print("Reset done: ")
        print(self.selection_data.selected_matrix)

    def Save(self):
        print('Saving...')
        cv2.imwrite("../result/Result.jpg", self.result_img)
        print('Done!')

    def Confirm(self):
        face_matrix = self.selection_data.GetSelectedFaces()
        background = self.selection_data.GetDisplayImageData()

        img_base_index = int(self.combo.currentText())
        selected_list = self.ConvertSelectList(face_matrix)
        subject_num = self.selection_data.img_info_list[img_base_index].shape[0]
        img_base = background

        print('Blending...')

        for i in range(subject_num):
            # Skip the same one
            if selected_list[i] == img_base_index:
                continue

            subject_info_list = self.selection_data.img_info_list[selected_list[i]]

            # Homography
            # mask_head, mask_body = cv2.imread("../imgs/homo_test_1/mask_head2.png")//255, cv2.imread("../imgs/homo_test_1/mask_body2.png")//255
            mask_head, mask_body = generate_mask(subject_info_list[i], img_base.shape)

            head_aligned, pt1, pt2 = face_to_base(img_base, img_list[selected_list[i]], mask_body, mask_head)
            # Blending
            head_aligned = head_aligned.astype(np.uint8)

            head_full = 255 * np.ones((img_base.shape), dtype=np.uint8)
            y1, x1 = pt1
            y2, x2 = pt2
            head_full[y1:y2, x1:x2, :] = head_aligned

            mask = generate_pyramid_mask(pt1, pt2, img_base.shape)
            # mask = np.array(mask, np.uint8)

            # Resize for better performance
            y_ref, x_ref, _ = img_base.shape
            y11 = y1 - BLEND_BIAS if y1 - BLEND_BIAS >= 0 else 0
            y21 = y2 + BLEND_BIAS if y2 + BLEND_BIAS < y_ref else y_ref
            x11 = x1 - BLEND_BIAS if x1 - BLEND_BIAS >= 0 else 0
            x21 = x2 + BLEND_BIAS if x2 + BLEND_BIAS < x_ref else x_ref
            head_full_cut = head_full[y11:y21, x11:x21, :]
            img_base_cut = img_base[y11:y21, x11:x21, :]
            mask_cut = mask[y11:y21, x11:x21, :]
            blended_img = pyramid_blend(head_full_cut, img_base_cut, mask_cut)
            img_base[y11:y21, x11:x21, :] = blended_img
            #print(img_base.shape)

            #img_base = blended_img
        print('Done!')
        self.result_img = img_base
        self.ShowResultImage(img_base)

    def ConvertSelectList(self, face_matrix):
        '''
        Convert the GUI face matrix to the 1-d select list
        :param face_matrix:
        :return:
        '''
        selected_list = []
        rows, cols = face_matrix.shape
        for j in range(0, cols):
            for i in range(0, rows):
                # print(face_matrix[i][j])
                if (face_matrix[i][j] == 1):
                    selected_list.append(i)
        return selected_list


if __name__ == '__main__':
    # Load the image
    dir_path = "../material/test5"
    paths = sorted(os.listdir(dir_path))
    img_list = []
    for path in paths:
        if path.split('.')[-1] in ["jpg", "JPG"]:
            img_list.append(cv2.imread(os.path.join(dir_path, path)))

    # paths = ['../imgs/test6/1.jpg', '../imgs/test6/2.jpg']
    # paths = ['../imgs/homo_test_5/1.jpg', '../imgs/homo_test_5/2.jpg']
    # img_list = read_img(paths)
    # img_list = [cv2.imread(path) for path in paths]
    # img_list = [imageResize(img, 1080 / img.shape[0]) for img in img_list]

    # img_list = [imageResize(img, RESIZE_RATIO) for img in img_list_0]

    app = QApplication(sys.argv)
    view = SelectBoard(img_list)

    # view.setFixedSize(1024, 512)
    view.show()

    sys.exit(app.exec_())
