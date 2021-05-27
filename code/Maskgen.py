import numpy as np


def generate_mask(subject_info, base_size):
    '''
    Gernerate the two masks
    :param subject_info:
    :param base_size:
    :return: mask head and mask body
    '''
    head_info, body_info = subject_info
    x, y, _ = base_size
    mask_head = np.zeros((x, y), dtype=np.bool)
    mask_body = np.zeros((x, y), dtype=np.bool)

    head_x, head_y, head_w, head_h = head_info
    body_x, body_y, body_w, body_h = body_info
    # Head
    for i in range(head_y, head_y + head_h):
        for j in range(head_x, head_x + head_w):
            mask_head[i][j] = True

    # Body
    for i in range(body_y, body_y + body_h):
        for j in range(body_x, body_x + body_w):
            mask_body[i][j] = True

    return mask_head, mask_body




