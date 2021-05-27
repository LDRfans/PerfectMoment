import cv2


def read_img(path_list:list):
    '''
    Read the image with the image numbers
    :param img_num: The number of images
    :return: A list of image with cv2
    '''
    img_list = [cv2.imread(path) for path in path_list]
    return img_list