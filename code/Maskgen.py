import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


def generate_mask(subject_info, base_size):
    '''
    Gernerate the two masks
    :param subject_info:
    :param base_size:
    :return: mask head and mask body
    '''
    head_info, body_info = subject_info
    x, y, _ = base_size
    mask_head = np.zeros((x, y, 3))
    mask_body = np.zeros((x, y, 3))

    head_x, head_y, head_w, head_h = head_info
    body_x, body_y, body_w, body_h = body_info
    # Head
    for i in range(head_y, head_y + head_h):
        for j in range(head_x, head_x + head_w):
            try:
                mask_head[i][j][:] = 1
            except:
                pass

    # Body
    for i in range(body_y, body_y + body_h):
        for j in range(body_x, body_x + body_w):
            try:
                mask_body[i][j][:] = 1
            except:
                pass

    return mask_head, mask_body


def generate_pyramid_mask(pt1, pt2, img_shape):
    # First generate a template
    mask_template = np.zeros((150, 150, 3), dtype=float)
    mask_template[25:125, 25:125, :] = 1
    mask_template = gaussian_filter(mask_template, sigma=7)
    mask_template = mask_template[15:135, 15:135, :]

    # mask_template = np.zeros((150, 150, 3), dtype=float)
    # mask_template[15:135, 15:135, :] = 1
    # mask_template = gaussian_filter(mask_template, sigma=7)
    # mask_template = mask_template[15:135, 15:135, :]

    # Normalize to [0,1]
    mask_template -= mask_template.min()
    mask_template *= (1 / mask_template.max())
    
    x, y, c = img_shape
    mask = np.zeros((x, y, c), dtype=float)

    # Set the mask to the raw 0/1
    y1, x1 = pt1
    y2, x2 = pt2
    # bias = 22
    # mask[y1+bias:y2-bias, x1+bias:x2-bias, :] = 1
    # mask[y1:y2, x1:x2, :] = 1
    mask_resized = cv2.resize(mask_template, (x2 - x1, y2 - y1))
    mask[y1:y2, x1:x2, :] += mask_resized

    # mask2 = gaussian_filter(mask, sigma=10)
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask2',mask2)
    # cv2.imshow('weighted',mask * 0.2+mask2)
    # cv2.waitKey()

    return mask


def generate_all_mask(img_info_list, img_shape_list):
    '''
    Generate all face masks using the info list
    :param img_shape_list: The shape of images
    :param img_info_list: The image info list
    :return: a list of face masks
    '''
    mask_list = []
    for i in range(0,len(img_info_list)):
        picture = img_info_list[i]
        picture_mask_list = []
        for person in picture:
            mask = generate_mask(person, img_shape_list[i])
            face_mask, _ = mask
            picture_mask_list.append(face_mask)
        mask_list.append(picture_mask_list)
    return mask_list







if __name__ == '__main__':
    generate_pyramid_mask((200, 200), (300, 300), (500, 500, 3))
