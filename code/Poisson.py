import os
from os import path
import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def laplacian_matrix(img_width, img_height):
    '''
    Generate the laplacian matrix
    :param img_width:
    :param img_height:
    :return: The matrix
    '''
    # build D (kernel)
    kernel = scipy.sparse.lil_matrix((img_width, img_width))
    kernel.setdiag(4)
    kernel.setdiag(values=-1, k=-1)
    kernel.setdiag(values=-1, k=1)
    # define Laplacian matrix with D
    mat = scipy.sparse.block_diag([kernel] * img_height).tolil()
    # set -I in the matrix
    mat.setdiag(-1, 1 * img_width)
    mat.setdiag(-1, -1 * img_width)
    return mat

def poisson_blend(front_img, background, mask):
    '''
    The main function to do the poisson blending
    :param background: The background
    :param front_img: The front image
    :param mask: The mask for poisson blending, with 0/1 data
    :return: The blended image
    '''
    y_max, x_max = background.shape[:-1]
    y_min, x_min = 0, 0
    width = x_max - x_min
    height = y_max - y_min

    # Construct Matrix A #
    # generate matrix A
    mat_A = laplacian_matrix(width, height)
    laplacian = mat_A.tocsc()
    # convert mask to binary
    mask[mask != 0] = 1
    # set outside mask part to identity
    for w in range(1, width - 1):
        for h in range(1, height - 1):
            if mask[h, w] == 0:
                offset = w + h * width
                # D
                mat_A[offset, offset] = 1
                mat_A[offset, offset + 1] = 0
                mat_A[offset, offset - 1] = 0
                # I
                mat_A[offset, offset + width] = 0
                mat_A[offset, offset - width] = 0
    # print((mat_A.shape))
    mat_A = mat_A.tocsc()

    # poisson blending #

    # flatten
    mask_flat = mask.flatten()
    # 3 channels
    for channel in range(front_img.shape[2]):
        front_img_flat = front_img[0:height, 0:width, channel].flatten()
        background_flat = background[0:height, 0:width, channel].flatten()
        # inside the mask
        mat_b = laplacian.dot(front_img_flat)
        # outside the mask
        mat_b[mask_flat == 0] = background_flat[mask_flat == 0]

        # x = mat_b
        x = spsolve(mat_A, mat_b)
        x = x.reshape((height, width))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        background[0:height, 0:width, channel] = x

    # if not os.path.exists("output"):
    #     os.mkdir("output")
    # plt.imshow(background[:, :, ::-1])
    # plt.savefig('./img_'+pic_name+'.jpg')
    # plt.show()

    return background




if __name__ == '__main__':
    # Original code
    # scr_dir = 'data'
    # front_img = cv2.imread(path.join(scr_dir, "front.png"))
    # background = cv2.imread(path.join(scr_dir, "back.JPG"))
    # mask = cv2.imread(path.join(scr_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
    # pic_name = 'poisson'
    #
    #
    #
    # y_max, x_max = background.shape[:-1]
    # y_min, x_min = 0, 0
    # width = x_max - x_min
    # height = y_max - y_min
    #
    # # Construct Matrix A #
    # # generate matrix A
    # mat_A = laplacian_matrix(width, height)
    # laplacian = mat_A.tocsc()
    # # convert mask to binary
    # mask[mask != 0] = 1
    # # set outside mask part to identity
    # for w in range(1, width - 1):
    #     for h in range(1, height - 1):
    #         if mask[h, w] == 0:
    #             offset = w + h * width
    #             # D
    #             mat_A[offset, offset] = 1
    #             mat_A[offset, offset + 1] = 0
    #             mat_A[offset, offset - 1] = 0
    #             # I
    #             mat_A[offset, offset + width] = 0
    #             mat_A[offset, offset - width] = 0
    # # print((mat_A.shape))
    # mat_A = mat_A.tocsc()
    #
    # # poisson blending #
    # from scipy.sparse.linalg import spsolve
    #
    # # flatten
    # mask_flat = mask.flatten()
    # # 3 channels
    # for channel in range(front_img.shape[2]):
    #     front_img_flat = front_img[0:height, 0:width, channel].flatten()
    #     background_flat = background[0:height, 0:width, channel].flatten()
    #     # inside the mask
    #     mat_b = laplacian.dot(front_img_flat)
    #     # outside the mask
    #     mat_b[mask_flat == 0] = background_flat[mask_flat == 0]
    #
    #     # x = mat_b
    #     x = spsolve(mat_A, mat_b)
    #     x = x.reshape((height, width))
    #     x[x > 255] = 255
    #     x[x < 0] = 0
    #     x = x.astype('uint8')
    #
    #     background[0:height, 0:width, channel] = x
    #
    # # if not os.path.exists("output"):
    # #     os.mkdir("output")
    # # plt.imshow(background[:, :, ::-1])
    # # plt.savefig('./img_'+pic_name+'.jpg')
    # # plt.show()
    #
    # cv2.imwrite('./img_' + pic_name + '.jpg', background)
    pass
