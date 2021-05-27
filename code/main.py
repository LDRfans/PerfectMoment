from Pyramid import pyramid_blend
from Homography import face_to_base
from Extract import extract
import cv2
import numpy as np
from Fileop import read_img

if __name__ == '__main__':
    paths = ['../imgs/homo_test_1/photo1.jpg', '../imgs/homo_test_1/photo2.jpg']
    img_list = read_img(paths)

    img_info_list = [extract(img) for img in img_list]

    # TODO: Give the data to UI

    # Simulate the UI
    subject_num = len(img_info_list[0])
    img_base_index = 0



