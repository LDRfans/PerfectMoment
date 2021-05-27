from Pyramid import pyramid_blend
from Homography import face_to_base
from Extract import extract
import cv2
import numpy as np
from Fileop import read_img
from Maskgen import generate_mask

if __name__ == '__main__':
    paths = ['../imgs/homo_test_1/photo1.jpg', '../imgs/homo_test_1/photo2.jpg']
    img_list = read_img(paths)

    img_info_list = [extract(img) for img in img_list]

    # TODO: Give the data to UI

    # Simulate the UI
    img_base_index = 0
    img_base = img_list[img_base_index]
    subject_num = img_info_list[img_base_index].shape[0]
    selected_list = [1,1]

    # Continue
    for i in range(subject_num):
        subject_info_list = img_info_list[selected_list[i]]
        # Homography
        mask_face, mask_body = generate_mask(subject_info_list[i], img_base.shape)
        # cv2.imshow('1',mask_body)
        # break
        face_aligned, pt1, pt2 = face_to_base(img_base, img_list[selected_list[i]], mask_body, mask_face)
        # Blending



