from Pyramid import pyramid_blend
from Homography import face_to_base
from Extract import extract
import cv2
import numpy as np
from Fileop import read_img
from Maskgen import generate_mask,generate_pyramid_mask
import logging

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

if __name__ == '__main__':
    paths = ['../imgs/homo_test_1/photo1.jpg', '../imgs/homo_test_1/photo2.jpg']
    img_list = read_img(paths)

    img_info_list = [extract(img,[0.5,0.8,0.6,0.25]) for img in img_list]

    # TODO: Give the data to UI
    log.info('Starting the Selection GUI')

    # Simulate the UI
    img_base_index = 0
    img_base = img_list[img_base_index]
    subject_num = img_info_list[img_base_index].shape[0]
    selected_list = [1, 1]

    # selected_list = []
    # selected_list = SelectUI(img_list, selected_list)
    # print(f"selected_list: {selected_list}")
    # exit()

    log.info('GUI Selection finished')
    log.info('GUI Selection finished')
    log.debug('Your selected base image is ' + str(img_base_index))
    log.debug('Your selected perfect moment is ' + str(selected_list))



    # Continue
    for i in range(subject_num):
        # Skip the same one
        if(selected_list[i] == img_base_index):
            log.debug('Skip the same image. i = ' + str(i))
            continue

        subject_info_list = img_info_list[selected_list[i]]
        # Homography
        # mask_head, mask_body = cv2.imread("../imgs/homo_test_1/mask_head2.png")//255, cv2.imread("../imgs/homo_test_1/mask_body2.png")//255
        mask_head, mask_body = generate_mask(subject_info_list[i], img_base.shape)
        # print(mask_head.shape)
        # print(img_base.shape)
        # cv2.imshow('1',mask_head)
        # cv2.imshow('2',img_list[0])
        # cv2.imshow('3',np.array(mask_head * 255//2+img_list[0]//2,dtype=np.uint8))
        # cv2.waitKey(0)
        head_aligned, pt1, pt2 = face_to_base(img_base, img_list[selected_list[i]], mask_body, mask_head)
        #Blending
        head_aligned = head_aligned.astype(np.uint8)
        # cv2.imshow("head", head_aligned)
        # cv2.waitKey(0)

        head_full = np.zeros((img_base.shape),dtype=np.uint8)
        y1, x1 = pt1
        y2, x2 = pt2
        head_full[y1:y2, x1:x2, :] += head_aligned
        # cv2.imshow('head_full',head_full)
        # cv2.waitKey()

        mask = generate_pyramid_mask(pt1, pt2, img_base.shape)
        print(mask.shape)
        # cv2.imshow('3', np.array(mask * 255 // 2 + img_list[0] // 2, dtype=np.uint8))
        # cv2.waitKey(0)

        blended_img = pyramid_blend(head_full,img_base,mask)
        # cv2.imshow('1',blended_img)
        # cv2.waitKey()
        # break
        img_base = blended_img

    cv2.imshow('1', blended_img)
    cv2.waitKey()



