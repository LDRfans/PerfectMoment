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
        mask_head, mask_body = cv2.imread("../imgs/homo_test_1/mask_head2.png")//255, cv2.imread("../imgs/homo_test_1/mask_body2.png")//255
        # mask_face, mask_body = generate_mask(subject_info_list[i], img_base.shape)
        # cv2.imshow('1',mask_body)
        # cv2.imshow('2',img_list[0])
        # cv2.imshow('3',np.array(mask_body * 255//2+img_list[0]//2,dtype=np.uint8))
        # cv2.waitKey(0)
        # break
        head_aligned, pt1, pt2 = face_to_base(img_base, img_list[selected_list[i]], mask_body, mask_head)
        # Blending
        cv2.imshow("head", head_aligned)
        cv2.waitKey(0)
        break



