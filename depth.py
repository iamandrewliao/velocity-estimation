'''
https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html
https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system
https://amroamroamro.github.io/mexopencv/matlab/cv.stereoRectifyUncalibrated.html
https://amroamroamro.github.io/mexopencv/matlab/cv.findFundamentalMat.html
'''

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

video_path1 = "vids/left_fullcar.mp4"
video_path2 = "vids/right_fullcar.mp4"

cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

# # Check if frame counts are the same
# print(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
# print(int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

window = 10
disparities = np.array([])
frame_count = 0
# Initiate SIFT detector
sift = cv2.SIFT_create()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

while cap1.isOpened() or cap2.isOpened():
    okay1, img1 = cap1.read()
    okay2, img2 = cap2.read()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (1280, 720), interpolation=cv2.INTER_AREA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (1280, 720), interpolation=cv2.INTER_AREA)
    # cv2.namedWindow("frames", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("frames", 1600, 500)
    # cv2.imshow("frames", np.concatenate((img1, img2), axis=1))
    if okay1 and okay2:
        # cv2.namedWindow("frames", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("frames", 1600, 500)
        # cv2.imshow("frames", np.concatenate((frame1, frame2), axis=1))
        if frame_count % window == 0:
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            matches = flann.knnMatch(des1, des2, k=2)
            # Filter matches using the Lowe's ratio test
            ratio_thresh = 0.5
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            p1 = []
            p2 = []
            for good_match in good_matches:
                [x1, y1] = kp1[good_match.queryIdx].pt
                p1.append([x1, y1])
                [x2, y2] = kp2[good_match.trainIdx].pt
                p2.append([x2, y2])
            p1 = np.array(p1)
            p2 = np.array(p2)
            [F, inliers] = cv2.findFundamentalMat(p1, p2)

            p1 = p1[inliers.ravel() == 1]
            p2 = p2[inliers.ravel() == 1]

            h1, w1 = img1.shape
            h2, w2 = img2.shape

            _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(p1), np.float32(p2), F, imgSize=[w1, h1])
            img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
            img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
            # cv2.namedWindow("frames", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("frames", 1600, 500)
            # cv2.imshow("frames", np.concatenate((img1_rectified, img2_rectified), axis=1))
            stereo = cv2.StereoSGBM.create(minDisparity=-20, numDisparities=112, blockSize=19)
            # Updating the parameters based on the trackbar positions
            # numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
            # blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
            # minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')
            # Setting the updated parameters before computing disparity map
            # stereo.setNumDisparities(numDisparities)
            # stereo.setBlockSize(blockSize)
            # stereo.setMinDisparity(minDisparity)
            disparity = stereo.compute(img1_rectified, img2_rectified)
            # Converting to float32
            disparity = disparity.astype(np.float32)
            # Scaling down the disparity values
            disparity = (disparity / 16.0)
            print(frame_count)
            # # Displaying the disparity map
            # cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("disp", 1280, 720)
            # cv2.imshow("disp", disparity)
            orig_map = matplotlib.colormaps.get_cmap('hot')
            # reversing the original colormap using reversed() function
            reversed_map = orig_map.reversed()
            plt.imshow(disparity, reversed_map)
            # plt.clim(-30, 30)
            plt.colorbar()
            plt.show()
            cur_disp = input("Enter disparity: ")
            disparities = np.append(disparities, cur_disp)

    frame_count += 1

    if not okay1 or not okay2:
        print('Cant read the video, Exit!')
        break

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # # Wait for a key press
    # key = cv2.waitKey(0) & 0xFF
    # # if pressed key is "Space", continue to the next frame
    # if key == ord(' '):
    #     continue
    # # if pressed key is "q", quit
    # elif key == ord('q'):
    #     break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

disparities.tofile('disparities.csv', sep=',')
