import numpy as np
from scipy.ndimage import gaussian_filter


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

    head_y, head_x, head_h, head_w = head_info
    body_y, body_x, body_h, body_w = body_info
    # Head
    for i in range(head_y, head_y + head_h):
        for j in range(head_x, head_x + head_w):
            mask_head[i][j][:] = 1

    # Body
    for i in range(body_y, body_y + body_h):
        for j in range(body_x, body_x + body_w):
            mask_body[i][j][:] = 1

    return mask_head, mask_body


def generate_pyramid_mask(pt1, pt2, img_shape):
    x, y, _ = img_shape
    mask = np.zeros((x,y),dtype=np.float)


    return mask

if __name__ == '__main__':
    pass
