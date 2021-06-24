import cv2

# img1 = cv2.imread()
# img2 = img_target * mask_body
#
# # Convert the image
# img1 = img1.astype(np.uint8)
# img2 = img2.astype(np.uint8)
#
# # Compute SIFT descriptors
# sift = cv2.xfeatures2d.SIFT_create()
# keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
#
# # DRAW
# matcher = cv2.BFMatcher()
# raw_matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
# good_points = []
# good_matches = []
# for m1, m2 in raw_matches:
#     if m1.distance < MATCH_DISTANCE_RATIO * m2.distance:
#         good_points.append((m1.trainIdx, m1.queryIdx))
#         good_matches.append([m1])
# img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good_matches, None, flags=2)
