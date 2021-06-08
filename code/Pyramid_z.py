import cv2
import numpy as np
from scipy.ndimage import filters

LEVEL = 8


def generate_mask(h, w, length=160):
    mask = np.zeros((h, w))
    mask[:, :w // 2] = 1
    for i in range(length):
        mask[:, w // 2 - length // 2 + i] = 1 - 1 / (length) * i
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    return mask


def gaussian_pyramid(img, levels):
    gas_pyr = []
    img_next_level = img

    # down sample each level
    for i in range(levels):
        gas_pyr.append(img_next_level)
        img_next_level = filters.gaussian_filter(img_next_level, 1)
        img_next_level = img_next_level[::2, ::2]

    return gas_pyr


def laplacian_pyramid(img, levels):
    lap_pyr = []
    gas_pyr = gaussian_pyramid(img, levels)

    for i in range(levels - 1):
        # upsampling
        img = gas_pyr[i + 1]
        img_up = np.zeros((2 * img.shape[0], 2 * img.shape[1]))
        img_up[::2, ::2] = img
        img_up = filters.gaussian_filter(img_up, 1)
        # get laplacian
        lap = gas_pyr[i] - img_up
        lap_pyr.append(lap)

    lap_pyr.append(gas_pyr[-1])
    return lap_pyr


def pyramid_blending(img_1, img_2, mask, levels=8):
    # masks
    mask_1 = mask
    mask_2 = np.ones(mask.shape) - mask
    # gaussians
    gas_pyr_1 = gaussian_pyramid(mask_1, levels)
    gas_pyr_2 = gaussian_pyramid(mask_2, levels)
    # laplacians
    lap_pyr_1 = laplacian_pyramid(img_1, levels)
    lap_pyr_2 = laplacian_pyramid(img_2, levels)

    gas_pyr_1 = np.array(gas_pyr_1, dtype=object)
    lap_pyr_1 = np.array(lap_pyr_1, dtype=object)
    gas_pyr_2 = np.array(gas_pyr_2, dtype=object)
    lap_pyr_2 = np.array(lap_pyr_2, dtype=object)
    L = np.multiply(gas_pyr_1, lap_pyr_1) + np.multiply(gas_pyr_2, lap_pyr_2)

    img = L[-1]

    for i in range(len(L) - 2, -1, -1):
        # upsampling
        img_up = np.zeros((2 * img.shape[0], 2 * img.shape[1]))
        img_up[::2, ::2] = img
        img_up = filters.gaussian_filter(img_up, 1)
        img = L[i] + img_up

    img = np.clip(img, 0, 255)

    return np.array(img, np.uint8)


def pyramid_blend(img_2, img_1, mask):
    # Resize to fit the kernel
    H, W = img_2.shape[0], img_2.shape[1]
    h, w = 512, 512
    img_1 = cv2.resize(img_1, (h, w))
    img_2 = cv2.resize(img_2, (h, w))
    mask = cv2.resize(mask, (h, w))
    mask = cv2.split(mask)[0]

    # Processing channel by channel
    b1, g1, r1 = cv2.split(img_1)
    b2, g2, r2 = cv2.split(img_2)

    blend1 = pyramid_blending(b2, b1, mask, LEVEL)
    blend2 = pyramid_blending(g2, g1, mask, LEVEL)
    blend3 = pyramid_blending(r2, r1, mask, LEVEL)

    img_blended = cv2.merge([blend1, blend2, blend3])

    img_blended = cv2.resize(img_blended, (W, H))

    return img_blended


if __name__ == "__main__":
    img_1 = cv2.imread("./girl.jpeg")
    img_2 = cv2.imread("./man.jpg")

    # Resize to fit the kernel
    h, w = 512, 512
    img_1 = cv2.resize(img_1, (h, w))
    img_2 = cv2.resize(img_2, (h, w))
    mask = generate_mask(h, w)

    # Processing channel by channel
    b1, g1, r1 = cv2.split(img_1)
    b2, g2, r2 = cv2.split(img_2)

    blend1 = pyramid_blending(b2, b1, mask, LEVEL)
    blend2 = pyramid_blending(g2, g1, mask, LEVEL)
    blend3 = pyramid_blending(r2, r1, mask, LEVEL)

    img_blended = cv2.merge([blend1, blend2, blend3])

    img_blended = cv2.resize(img_blended, (386, 503))
    cv2.imshow("test", img_blended)
    cv2.waitKey(0)
    # cv2.imwrite("./img_blended.jpg", img_blended)
