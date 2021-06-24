"""
Functions:
- homographySolver(): solver H
- findHomography(): find an H that good enough
- stitch_two(): Stitching two images
"""
import random

import cv2
import numpy as np

HOMOGRAPHY_SOLVER = 0

RESIZE = 256
INLIER_DIST = 10
MATCH_DISTANCE_RATIO = 0.6      # good points
INLIER_RATIO = 0.7
RANSAC = 1000

IMG1_PATH = "../imgs/homo_test_1/photo1.jpg"
IMG2_PATH = "../imgs/homo_test_1/photo2.jpg"
MASK1_PATH = "../imgs/homo_test_1/mask_body2.png"
# MASK1_PATH = "../imgs/homo_test_1/mask_person2.png"
MASK2_PATH = "../imgs/homo_test_1/mask_head2.png"


def homographySolver(img2_keys, img1_keys, K):
    # Randomly choose K points
    point_index_list = [random.randint(0, len(img1_keys) - 1) for _ in range(K)]
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
    A = np.mat(A)
    b = np.append(np.zeros(2 * K), 1).reshape(2 * K + 1, 1)
    H = np.dot(np.dot(A.T, A).I, np.dot(A.T, b))
    H = H / H[8]  # normalize
    return H


def findHomography(img1_keys, img2_keys, GOOD_POINTS=4):
    # RANSAC
    for round in range(RANSAC):
        print("RANSAC:", round)
        H = homographySolver(img1_keys, img2_keys, K=GOOD_POINTS).reshape(3, 3)  # img_1 <- img_2

        # Find inliers
        homo_img1_keys = np.array([np.append(np.array(img1_keys[i]), 1) for i in range(len(img1_keys))])
        homo_img2_keys = np.array([np.append(np.array(img2_keys[i]), 1) for i in range(len(img2_keys))])
        inlier_list = []
        for i in range(len(img1_keys)):
            homo_img2_keys[i] = np.dot(H, homo_img2_keys[i])
            homo_img2_keys[i] = homo_img2_keys[i] / homo_img2_keys[i, 2]  # normalize
            dist = np.linalg.norm(homo_img1_keys[i] - homo_img2_keys[i])
            if dist < INLIER_DIST:
                inlier_list.append(i)

        # If the transform is good enough, refine it using all inliers
        if len(inlier_list) > len(img1_keys) * INLIER_RATIO:
            # print(len(inlier_list), "/", len(img1_keys), "Good!")
            img1_inliers = [img1_keys[i] for i in inlier_list]
            img2_inliers = [img2_keys[i] for i in inlier_list]
            # Recompute
            H_new = homographySolver(img1_inliers, img2_inliers, K=len(img1_inliers)).reshape(3, 3)
            return H_new

    # If fails
    print("findHomography fail")
    exit()


def matcher(des_1, des_2):
    # Find nearest

    # Fix des_1, find in des_2:
    nearest_list_1_2 = []
    for i in range(len(des_1)):
        dist = float('inf')
        des_2_index_1 = 0
        des_2_index_2 = 0
        for j in range(len(des_2)):
            dist_new = np.linalg.norm(des_1[i] - des_2[j])
            if dist_new < dist:
                des_2_index_2 = des_2_index_1
                des_2_index_1 = j
                dist = dist_new
        nearest_list_1_2.append((i, des_2_index_1, des_2_index_2))

    # Fix des_2, find in des_1:
    nearest_list_2_1 = []
    for i in range(len(des_2)):
        dist = float('inf')
        des_1_index_1 = 0
        des_1_index_2 = 0
        for j in range(len(des_1)):
            dist_new = np.linalg.norm(des_2[i] - des_1[j])
            if dist_new < dist:
                des_1_index_2 = des_1_index_1
                des_1_index_1 = j
                dist = dist_new
        nearest_list_2_1.append((i, des_1_index_1, des_1_index_2))

    # If is nearest mutually, match them
    matches = []
    for id_1, id_2, _ in nearest_list_1_2:
        if nearest_list_2_1[id_2][1] == id_1:
            matches.append((id_1, id_2, _))

    return matches


def face_to_base(img_base, img_target, mask_body, mask_face):
    img1 = img_base
    img2 = img_target * mask_body

    #Convert the image
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    # Compute SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # Define matecher
    raw_matches = matcher(descriptors_1, descriptors_2)

    # DRAW
    # matcher = cv2.BFMatcher()
    # raw_matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
    # good_points = []
    # good_matches = []
    # for m1, m2 in raw_matches:
    #     if m1.distance < MATCH_DISTANCE_RATIO * m2.distance:
    #         good_points.append((m1.trainIdx, m1.queryIdx))
    #         good_matches.append([m1])
    # img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good_matches, None, flags=2)
    #
    # cv2.imwrite("sift_align.jpg", img3)
    # exit()

    # Select good matches
    good_points = []
    for id_1, match_1, match_2 in raw_matches:
        match_1_dist = np.linalg.norm(descriptors_1[id_1] - descriptors_2[match_1])
        match_2_dist = np.linalg.norm(descriptors_1[id_1] - descriptors_2[match_2])
        if match_1_dist < MATCH_DISTANCE_RATIO * match_2_dist:
            good_points.append((id_1, match_1))

    # Calculate homograph H with RANSAC
    GOOD_POINTS = 10
    if len(good_points) >= GOOD_POINTS:
        GOOD_POINTS = len(good_points)
        img1_keys = np.float32([keypoints_1[i].pt for (i, _) in good_points])
        img2_keys = np.float32([keypoints_2[i].pt for (_, i) in good_points])
        if HOMOGRAPHY_SOLVER:
            H = findHomography(img1_keys, img2_keys, GOOD_POINTS)  # img_1 <- img_2
        else:
            H, status = cv2.findHomography(img2_keys, img1_keys, cv2.RANSAC, 5.0)
    else:
        print("findHomography fail: Good points not enough")

    # Stitch the images together.
    height_panorama = img1.shape[0]
    width_panorama = img1.shape[1]
    panorama = np.zeros((height_panorama, width_panorama, 3))

    # Stitch
    img2 = img_target * mask_face
    panorama[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))

    x1, y1, x2, y2 = img1.shape[1], img1.shape[0], 0, 0
    for row in range(panorama2.shape[0]):
        for col in range(panorama2.shape[1]):
            if panorama2[row, col, 0] or panorama2[row, col, 1] or panorama2[row, col, 2]:
                if x1 > col:  x1 = col
                if y1 > row:  y1 = row
                if x2 < col:  x2 = col
                if y2 < row:  y2 = row

    padding = 15
    x1, y1, x2, y2 = x1 + padding, y1 + padding, x2 - padding, y2 - padding
    face_aligned = panorama2[y1:y2, x1:x2, :]
    panorama[y1:y2, x1:x2, :] = face_aligned

    panorama = panorama.astype(np.uint8)

    # cv2.imshow('1',panorama)
    # cv2.waitKey()


    return face_aligned, [y1, x1], [y2, x2]
    # return panorama


if __name__ == "__main__":
    # Read images
    img_1 = cv2.imread(IMG1_PATH)
    img_2 = cv2.imread(IMG2_PATH)
    mask_1 = cv2.imread(MASK1_PATH) // 255
    mask_2 = cv2.imread(MASK2_PATH) // 255
    # Resize
    # img_1 = cv2.resize(img_1, (RESIZE, RESIZE))
    # img_2 = cv2.resize(img_2, (RESIZE, RESIZE))
    # mask_1 = cv2.resize(mask_1, (RESIZE, RESIZE))
    # mask_2 = cv2.resize(mask_2, (RESIZE, RESIZE))

    sti_0_1 = face_to_base(img_1, img_2, mask_1, mask_2)

    cv2.imshow("test", sti_0_1)
    cv2.waitKey(0)
    # cv2.imwrite("test.jpg", sti_0_1)
