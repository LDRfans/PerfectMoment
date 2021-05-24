"""
Functions:
- homographySolver(): solver H
- findHomography(): find an H that good enough
- stitch_two(): Stitching two images
"""
import random

import cv2
import numpy as np

HOMOGRAPHY_SOLVER = False

SIFT_DISTANCE_RATIO = 0.75
INLIER_RATIO = 0.6
RANSAC = 100

IMG1_PATH = "../imgs/homo_test_1/photo1.jpg"
IMG2_PATH = "../imgs/homo_test_1/photo2.jpg"


def homographySolver(img2_keys, img1_keys, K):
    # print("Solving...")
    # Randomly choose K points
    point_index_list = [random.randint(0, len(img1_keys) - 1) for _ in range(K)]
    # while len(set(point_index_list)) != K:
    #     point_index_list = [random.randint(0, len(img1_keys)-1) for _ in range(K)]
    x_img1 = [img1_keys[i][0] for i in point_index_list]
    y_img1 = [img1_keys[i][1] for i in point_index_list]
    x_img2 = [img2_keys[i][0] for i in point_index_list]
    y_img2 = [img2_keys[i][1] for i in point_index_list]
    X = np.array([(x_img1[i], y_img1[i], 1) for i in range(K)])
    A = []
    for i in range(K):
        row_1 = np.array([np.zeros(3).T, X[i].T, -y_img2[i] * X[i].T]).flatten()
        row_2 = np.array([X[i].T, np.zeros(3).T, -x_img2[i] * X[i].T]).flatten()
        A.append(row_1)
        A.append(row_2)
    A.append(np.append(np.zeros(8), 1).T)
    # A.append(np.ones(9).T)
    A = np.mat(A)
    b = np.append(np.zeros(2 * K), 1).reshape(2 * K + 1, 1)
    # H = np.linalg.solve(A, b)
    H = np.dot(np.dot(A.T, A).I, np.dot(A.T, b))
    H = H / H[8]  # normalize
    # print("img1:", x_img1, y_img1)
    # print("img2:", x_img2, y_img2)
    # print("X:", X)
    # print("A:", A.shape, A)
    # print("b:", b)
    # print("H:", H)
    return H


def findHomography(img1_keys, img2_keys):
    for round in range(RANSAC):
        print("RANSAC:", round)
        # RANSAC
        # Random select
        H = homographySolver(img1_keys, img2_keys, K=4).reshape(3, 3)  # img_1 <- img_2
        # H, status = cv2.findHomography(img2_keys, img1_keys, cv2.RANSAC, 5.0)
        # print("H:", H)
        # exit()
        homo_img1_keys = np.array([np.append(np.array(img1_keys[i]), 1) for i in range(len(img1_keys))])
        homo_img2_keys = np.array([np.append(np.array(img2_keys[i]), 1) for i in range(len(img2_keys))])
        # print(new_img2_keys)
        # Find inliers
        inlier_list = []
        for i in range(len(img1_keys)):
            homo_img2_keys[i] = np.dot(H, homo_img2_keys[i])
            homo_img2_keys[i] = homo_img2_keys[i] / homo_img2_keys[i, 2]  # normalize
            dist = np.linalg.norm(homo_img1_keys[i] - homo_img2_keys[i])
            # print(homo_img1_keys[i], homo_img2_keys[i], dist)
            if dist < INLIER_DIST:
                inlier_list.append(i)
        # print(inlier_list)
        # If the transform is good enough, refine it using all inliers
        if len(inlier_list) > len(img1_keys) * INLIER_RATIO:
            # print(len(inlier_list), "/", len(img1_keys), "Good!")
            img1_inliers = [img1_keys[i] for i in inlier_list]
            img2_inliers = [img2_keys[i] for i in inlier_list]
            # print(img1_keys)
            # print(img1_inliers)
            # Recompute
            # H_new = solver(img1_inliers, img2_inliers, K=len(img1_inliers)).reshape(3, 3)
            # print("H_new:", H_new)
            H_new, _ = cv2.findHomography(np.array(img2_inliers), np.array(img1_inliers), cv2.RANSAC, 5.0)
            # print("H_std:", H_std)
            return H_new
        # else:
        #     print(len(inlier_list), "/", len(img1_keys), len(inlier_list)/len(img1_keys))
    print("RANSAC fail")
    return 0

def stitch_two(img1, img2):

    # Compute SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # Define matecher
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)

    # Select good matches
    good_points = []
    for id_1, match_1, match_2 in raw_matches:
        match_1_dist = np.linalg.norm(descriptors_1[id_1] - descriptors_2[match_1])
        match_2_dist = np.linalg.norm(descriptors_1[id_1] - descriptors_2[match_2])
        if match_1_dist < SIFT_DISTANCE_RATIO * match_2_dist:
            good_points.append((id_1, match_1))

    # Calculate homograph H with RANSAC
    # Here use cv2.findHomography
    if len(good_points) >= 4:
        img1_keys = np.float32([keypoints_1[i].pt for (i, _) in good_points])
        img2_keys = np.float32([keypoints_2[i].pt for (_, i) in good_points])
        # print(len(img1_keys), len(img2_keys))
        if HOMOGRAPHY_SOLVER:
            H = findHomography(img1_keys, img2_keys)  # img_1 <- img_2
        else:
            H, status = cv2.findHomography(img2_keys, img1_keys, cv2.RANSAC, 5.0)
        # print(H)
        # exit()
        # H = findHomography(img1_keys, img2_keys)    # img_1 <- img_2
    else:
        print("findHomography fail...")
    # Stitch the images together. You can use cv2.warpPerspective() in this step.
    height_panorama = img1.shape[0]
    width_panorama = img1.shape[1] + img2.shape[1]
    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    # create masks
    offset = SMOOTHING_WINDOW_SIZE // 2
    barrier = img1.shape[1] - offset
    # L
    mask1 = np.zeros((height_panorama, width_panorama))
    mask1[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
    mask1[:, :barrier - offset] = 1
    mask1 = cv2.merge([mask1, mask1, mask1])
    # cv2.imwrite('./output/mask1.jpg', mask1*255)
    # R
    mask2 = np.zeros((height_panorama, width_panorama))
    mask2[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
    mask2[:, barrier + offset:] = 1
    mask2 = cv2.merge([mask2, mask2, mask2])
    # cv2.imwrite('./output/mask2.jpg', mask2*255)
    # exit()
    # raw stitch
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    panorama1 = panorama1.astype(np.uint8)
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama)) * mask2
    panorama2 = panorama2.astype(np.uint8)
    raw_stitch = panorama1 + panorama2
    # tailoring
    rows, cols = np.where(raw_stitch[:, :, 0] != 0)
    tailored_stitch = raw_stitch[min(rows):max(rows) + 1, min(cols):max(cols) + 1, :]


if __name__ == "__main__":
    # Read images
    img_1 = cv2.imread(IMG1_PATH)
    img_2 = cv2.imread(IMG2_PATH)
    # Resize
    img_1 = cv2.resize(img_1, (256, 256))
    img_2 = cv2.resize(img_2, (256, 256))

    SMOOTHING_WINDOW_SIZE = img_1.shape[1] // 6
    INLIER_DIST = img_1.shape[1] // 6

    sti_0_1 = stitch_two(img_1, img_2)

    cv2.imshow("test", sti_0_1)
    cv2.waitKey(0)
