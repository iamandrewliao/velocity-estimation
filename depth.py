'''
https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html
https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system
https://amroamroamro.github.io/mexopencv/matlab/cv.stereoRectifyUncalibrated.html
https://amroamroamro.github.io/mexopencv/matlab/cv.findFundamentalMat.html
'''

import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
img1 = cv.imread('left-fullcar.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('right-fullcar.jpg', cv.IMREAD_GRAYSCALE)  # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.4*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.imshow(img3,),plt.show()

#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.5
good_matches = []
for m,n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
# #-- Draw matches
# img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
# show_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# #-- Show detected matches
# plt.imshow(show_matches,),plt.show()
# cv.waitKey()
# p1 = set()
# p2 = set()
p1 = []
p2 = []
# [x1, y1] = kp1[good_matches[0].queryIdx].pt

for good_match in good_matches:
    [x1, y1] = kp1[good_match.queryIdx].pt
    p1.append([x1, y1])
    [x2, y2] = kp2[good_match.trainIdx].pt
    p2.append([x2, y2])

p1 = np.array(p1)
p2 = np.array(p2)
# print(f"p1: {p1}\nlen: {len(p1)}")
# print(f"p2: {p2}\nlen: {len(p2)}")

[F, inliers] = cv.findFundamentalMat(p1, p2)

p1 = p1[inliers.ravel() == 1]
p2 = p2[inliers.ravel() == 1]

h1, w1 = img1.shape
h2, w2 = img2.shape

_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(p1), np.float32(p2), F, imgSize=[w1, h1])
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
# cv.imshow("img1_rectified", img1_rectified)
# cv.imshow("img2_rectified", img2_rectified)
# cv.imwrite("img1_rectified.png", img1_rectified)
# cv.imwrite("img2_rectified.png", img2_rectified)
# # cv.waitKey()
# stereo = cv.StereoBM_create(numDisparities=16, blockSize=5)
stereo = cv.StereoSGBM.create(minDisparity=-30, numDisparities=112, blockSize=7)
disparity = stereo.compute(img1_rectified, img2_rectified)
# getting the original colormap using cm.get_cmap() function
orig_map = matplotlib.colormaps.get_cmap('hot')
# reversing the original colormap using reversed() function
reversed_map = orig_map.reversed()
plt.imshow(disparity, reversed_map)
plt.clim(-30, 30)
plt.colorbar()
plt.show()
