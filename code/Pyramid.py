import math

import cv2
import numpy as np
import Extract


pts_1 = []
pts_2 = []


def UpSample(img):
    rows, cols, channels = img.shape
    up_img = np.zeros((rows * 2,cols * 2,channels),dtype=np.float)
    up_img[::2,::2,:] = img
    return up_img

def GaussianPyramid(img,iter_num):
    gaussian_filter = np.array([[1, 4, 6, 4, 1]]) * (1 / 16)
    gaussian_filter = np.dot(np.transpose(gaussian_filter),gaussian_filter)

    pyramid = [None for i in range(0,iter_num)]
    pyramid[0] = img

    for i in range(1,iter_num):
        previous_img = pyramid[i-1]
        temp = cv2.filter2D(previous_img,-1,gaussian_filter,borderType=cv2.BORDER_REPLICATE)
        rows, cols, _ = temp.shape
        temp = temp[0:rows:2,0:cols:2,:]
        pyramid[i] = temp

    return pyramid

def LaplacianPyramid(img,iter_num):
    filter = np.array([[1, 4, 6, 4, 1]]) * (1 / 16)
    filter = np.dot(np.transpose(filter), filter)
    g_filter = filter * 4

    pyramid = [None for i in range(0, iter_num)]
    handle_img = img

    for i in range(0,iter_num -1):
        temp = cv2.filter2D(handle_img, -1, filter, borderType=cv2.BORDER_REPLICATE)
        rows, cols, _ = temp.shape
        temp = temp[0:rows:2, 0:cols:2, :]
        original_img = handle_img
        handle_img = temp

        temp = UpSample(handle_img)
        temp = temp[0:rows,0:cols,:]
        temp = cv2.filter2D(temp, -1, g_filter, borderType=cv2.BORDER_REPLICATE)
        e_handle_img = temp
        pyramid[i] = original_img - e_handle_img

    pyramid[iter_num-1] = handle_img
    return pyramid

def Reconstruct(l_pyramid):
    filter = np.array([[1, 4, 6, 4, 1]]) * (1 / 16)
    filter = np.dot(np.transpose(filter), filter)
    g_filter = filter * 4

    l_pyramid_copy = l_pyramid.copy()
    iter_num = len(l_pyramid_copy)

    for i in range(iter_num-1,0,-1):
        temp = l_pyramid_copy[i]
        temp = UpSample(temp)
        rows,cols,_ = l_pyramid_copy[i-1].shape
        temp = temp[0:rows,0:cols,:]
        temp = cv2.filter2D(temp, -1, g_filter, borderType=cv2.BORDER_REPLICATE)
        l_pyramid_copy[i-1] = l_pyramid_copy[i-1] + temp

    return l_pyramid_copy[0]


def pyramid_blend(img1, img2, mask):
    '''
    The main function for pyramid blending
    :param img1: Blending image 1
    :param img2: Blending image 2
    :param mask: The mask of blending, with 0/1 data
    :return: blende img, as type uint8
    '''
    img1 = img1.astype(np.float)
    img2 = img2.astype(np.float)
    mask_pyramid = GaussianPyramid(mask, 5)

    left_pyramid = LaplacianPyramid(img1, 5)
    right_pyramid = LaplacianPyramid(img2, 5)

    blend_pyramid = []
    for i in range(0, 5):
        blend_pyramid.append(left_pyramid[i] * mask_pyramid[i] + right_pyramid[i] * (1 - mask_pyramid[i]))

    blend_image = Reconstruct(blend_pyramid)
    blend_image = blend_image.astype(np.uint8)

    return blend_image






if __name__ == '__main__':

    img1 = cv2.imread('1.png')
    img2 = cv2.imread('2.png')

    # cv2.imshow('img1',img1)
    # cv2.setMouseCallback("img1", mouse1)
    # cv2.waitKey(0)
    # cv2.imshow('img2', img2)
    # cv2.setMouseCallback("img2", mouse2)
    # cv2.waitKey(0)
    pts_1 = Extract.extract(img1)
    pts_2 = Extract.extract(img2)

    pts_1 = pts_1[31:60]
    pts_2 = pts_2[31:60]



    pts_1 = np.float32(pts_1)
    pts_2 = np.float32(pts_2)
    M,_ = cv2.findHomography(pts_2, pts_1)
    res = cv2.warpPerspective(img2, M, (450,590))
    #M = cv2.getAffineTransform(pts_2, pts_1)
    #res = cv2.warpAffine(img2, M, (450,590))
    cv2.imwrite('res.png',res)
    res = cv2.addWeighted(res,0.5,img1,0.5,0)
    cv2.imshow('res',res)
    cv2.imwrite("img-reg.jpg", res)
    # cv2.waitKey(0)

    img2 = cv2.imread('res.png')
    rows, cols, channels = img1.shape
    img1 = img1.astype(np.float)
    img2 = img2.astype(np.float)


    # mask = np.zeros((rows,cols,channels),dtype=np.float)
    # mask[:,0:math.floor(cols/2),:] = np.ones((rows,math.floor(cols/2),channels),dtype=np.float)
    mask = cv2.imread('mask.png')
    mask = mask.astype(np.float)
    mask = mask / 255
    mask_pyramid = GaussianPyramid(mask,5)

    left_pyramid = LaplacianPyramid(img1,5)
    right_pyramid = LaplacianPyramid(img2, 5)

    blend_pyramid = []
    for i in range(0,5):
        blend_pyramid.append(left_pyramid[i] * mask_pyramid[i] + right_pyramid[i] * (1 - mask_pyramid[i]))

    blend_image = Reconstruct(blend_pyramid)
    blend_image = blend_image.astype(np.uint8)
    cv2.imshow('img',blend_image)
    cv2.imwrite("img-pyramid.jpg", blend_image)
    cv2.waitKey()




