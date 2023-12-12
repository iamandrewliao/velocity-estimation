# https://docs.opencv.org/4.2.0/d3/d14/tutorial_ximgproc_disparity_filtering.html

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
# from PIL import Image
# import torch
# from torchvision import transforms
# import yaml
from ultralytics import YOLO


# import yolov7

def get_keypoints_and_descriptors(left, right):  # FOR UNCALIBRATED STEREO
    """Use ORB detector and FLANN matcher to get keypoints, descritpors,
    and corresponding matches that will be good for computing
    homography.
    """
    # orb = cv.ORB_create()
    # kp1, des1 = orb.detectAndCompute(left, None)
    # kp2, des2 = orb.detectAndCompute(right, None)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left, None)
    kp2, des2 = sift.detectAndCompute(right, None)

    ############## Using FLANN matcher ##############
    # Each keypoint of the first image is matched with a number of
    # keypoints from the second image. k=2 means keep the 2 best matches
    # for each keypoint (best matches = the ones with the smallest
    # distance measurement).
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    return kp1, des1, kp2, des2, flann_match_pairs


def lowes_ratio_test(matches, ratio_threshold=0.6):  # FOR UNCALIBRATED STEREO
    """Filter matches using the Lowe's ratio test.

    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    filtered_matches = []
    print(len(matches))
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    print(len(filtered_matches))
    return filtered_matches


def draw_matches(left, right, kp1, des1, kp2, des2, flann_match_pairs):  # FOR UNCALIBRATED STEREO
    """Draw the first 8 matches between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv.drawMatches(
        left,
        kp1,
        right,
        kp2,
        flann_match_pairs[:8],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def compute_fundamental_matrix(matches, kp1, kp2, method=cv.FM_RANSAC):  # FOR UNCALIBRATED STEREO
    """Use the set of good matches to estimate the Fundamental Matrix.

    See  https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    for more info.
    """
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    for m in matches[:8]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        # You can play with the Threshold and confidence values here
        # until you get something that gives you reasonable results. I
        # used the defaults
        fundamental_matrix, inliers = cv.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            # ransacReprojThreshold=3,
            # confidence=0.99,
        )
    return fundamental_matrix, inliers, pts1, pts2


# device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# def load_model():
#    model = torch.load('yolov7/yolov7-mask.pt')
#    model.eval()
#
#    if torch.cuda.is_available():
#       model.half().to(device)


# Values for

# baseline = 508 #this is equivalent to 20 inches in mm
# left_focal_length=17 #mm
# left_pixel_size=0.0042 #mm, 4.2um
# right_focal_length=18 #mm
# right_pixel_size=0.00389 #mm, 3.89um
# camera_left_K = np.array([[left_focal_length/left_pixel_size,0,955],
#                           [0,left_focal_length/left_pixel_size,540],
#                           [0,0,1]])
# camera_right_K = np.array([[right_focal_length/right_pixel_size,0,955],
#                           [0,right_focal_length/right_pixel_size,540],
#                           [0,0,1]])
lmtx = np.array(
    "1950.38453398153	0	960.060648034798 0	1946.91012104234	533.551319346158 0	0	1".split(),
    dtype=np.float64).reshape(3, 3)
rmtx = np.array(
    "2083.73319460427	0	971.580309710173 0	2081.65248106588	540.602552427858 0	0	1".split(),
    dtype=np.float64).reshape(3, 3)
ldist = np.array("-0.267445229737809	3.12714057241085	0 0 -15.9452175285924".split(), dtype=np.float64).reshape(1,
                                                                                                                      5)
rdist = np.array("-0.0335279726196925	-0.197931559581053	0 0 0.314539234991250".split(),
                 dtype=np.float64).reshape(1, 5)
R = np.array(
    "0.999543512313102	-0.0301402160652694	-0.00208191458214724 0.0299484490676848	0.997550465683488	-0.0632151786606444 0.00398213400437069	0.0631239715971340	0.997997748904563".split(),
    dtype=np.float64).reshape(3, 3)
# T=np.array("-587.809867688920	-66.3714985637734	244.964055509695".split(),dtype=np.float64).reshape(3,1)
T = np.array("-587.809867688920	-1.19578492505585	-6.43761287625161".split(), dtype=np.float64).reshape(3, 1)

model = YOLO('yolov8n-seg.pt')

vid_left = cv.VideoCapture("vids/Dec10/pass2/Dec10_left2.mp4")
vid_right = cv.VideoCapture("vids/Dec10/pass2/Dec10_right2.mp4")

if (vid_left.isOpened() == False or vid_right.isOpened() == False):
    print("Error opening video stream or file")
fps = vid_left.get(cv.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
frame_count = int(vid_left.get(cv.CAP_PROP_FRAME_COUNT))
width = vid_left.get(cv.CAP_PROP_FRAME_WIDTH)
height = vid_left.get(cv.CAP_PROP_FRAME_HEIGHT)
print(width, type(width), height, type(height))
R1, R2, Pn1, Pn2, _, _, _ = cv.stereoRectify(lmtx, ldist, rmtx, rdist, (1920, 1080), R, T, alpha=0)

duration = frame_count / fps  # necessary to calculate speed
count = 0  # works in conjunction with "window" below
# window is the amount of data points we look at to get the current speed
window = 5

disparities = np.array([])

block_size = 3
min_disp = -30
max_disp = -30 + 144  # num_disp must be divisible by 16
num_disp = max_disp - min_disp
# left_matcher = cv.StereoSGBM.create(numDisparities=32,blockSize=5,preFilterCap=7,disp12MaxDiff=4)
left_matcher = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=5,
    speckleWindowSize=150,
    speckleRange=1,
    disp12MaxDiff=0,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    mode=cv.STEREO_SGBM_MODE_SGBM
)
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
wls_filter.setLambda(30000)
wls_filter.setSigmaColor(3.1)
# Read until video is completed
positions = np.array(pd.read_csv('positions.csv', delimiter=','))
print(positions, "positions")
yPositions = positions[:, 2]
xPositions = positions[:, 1]

filtered_Disp_Buffer = np.zeros((2, 1080, 1920))
XVelBuffer = []
YVelBuffer = []
while (vid_left.isOpened() and vid_right.isOpened()):
    # Capture frame-by-frame
    ret_left, left = vid_left.read()
    ret_right, right = vid_right.read()
    if ret_left == True and ret_right == True:
        h, w, d = left.shape

        mapL1, mapL2 = cv.initUndistortRectifyMap(lmtx, ldist, R1, Pn1, (left.shape[1], left.shape[0]), cv.CV_32FC1)
        mapR1, mapR2 = cv.initUndistortRectifyMap(rmtx, rdist, R2, Pn2, (right.shape[1], right.shape[0]), cv.CV_32FC1)

        left_rect = cv.remap(left, mapL1, mapL2, cv.INTER_LINEAR)
        right_rect = cv.remap(right, mapR1, mapR2, cv.INTER_LINEAR)

        # mask for first car
        # mask_polygon = result.masks[0].xy[0]  # mask coordinates
        # mask_img = Image.fromarray(mask, "I")
        # mask_img.show()

        scale = 0.5
        left_for_matcher = cv.resize(left_rect, None, scale, scale, cv.INTER_LINEAR)
        right_for_matcher = cv.resize(right_rect, None, scale, scale, cv.INTER_LINEAR)

        left_for_matcher = cv.cvtColor(left_for_matcher, cv.COLOR_BGR2GRAY)
        right_for_matcher = cv.cvtColor(right_for_matcher, cv.COLOR_BGR2GRAY)
        #   right=cv.warpAffine(right,
        #                       np.array([[1,0,0],
        #                                 [0,1,-26]],dtype=np.float32),
        #                                 (right.shape[1],right.shape[0]))
        # left=cv.fastNlMeansDenoising(left)
        # right=cv.fastNlMeansDenoising(right)
        # left=cv.GaussianBlur(left,[15,15],3)
        # right=cv.GaussianBlur(right,[15,15],3)
        # left=cv.GaussianBlur(left,[9,9],2)
        # right=cv.GaussianBlur(right,[9,9],2)
        # erosion_shape=cv.MORPH_RECT
        # erosion_size=3
        # element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
        #                                     (erosion_size, erosion_size))
        #
        # dilate_shape=cv.MORPH_CROSS
        # dilate_size=3
        # element = cv.getStructuringElement(dilate_shape, (2 * dilate_size + 1, 2 * dilate_size + 1),
        #                                     (dilate_size, dilate_size))
        # left=cv.erode(left,element)
        # right=cv.erode(right,element)
        # left=cv.dilate(left,element)
        # right=cv.dilate(right,element)
        left_for_matcher = cv.equalizeHist(left_for_matcher)
        right_for_matcher = cv.equalizeHist(right_for_matcher)

        # sift = cv.SIFT_create()
        # orb = cv.ORB_create()

        # frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        # left = frame[:,0:int(w/2)]
        # right = frame[:,int(w/2):]

        # Display the resulting frame
        # cv.imshow('Left Frame',left)
        # cv.imshow('Right Frame',right)
        cv.namedWindow("frames", cv.WINDOW_NORMAL)
        cv.resizeWindow("frames", 1600, 500)
        cv.imshow("frames", np.concatenate((left, right), axis=1))
        # edge_right = cv.Canny(image=right, threshold1=100, threshold2=200) # Canny Edge Detection
        # edge_left = cv.Canny(image=left, threshold1=100, threshold2=200)
        # # Display Canny Edge Detection Image
        # cv.imshow('Canny Edge Detection', edge_right)

        if count % window == 0:
            ############## Find good keypoints to use ##############
            # kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(left, right)
            # good_matches = lowes_ratio_test(flann_match_pairs, 0.8)
            # # draw_matches(left, right, kp1, des1, kp2, des2, good_matches)

            # ############## Compute Fundamental Matrix ##############
            # F, I, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)

            # ############## Stereo rectify uncalibrated ##############
            # h1, w1 = left.shape
            # h2, w2 = right.shape
            # thresh = 0
            # _, H1, H2 = cv.stereoRectifyUncalibrated(
            #     np.float32(points1), np.float32(points2), F, imgSize=(w1, h1), threshold=thresh,
            # )

            ############## Undistort (Rectify) ##############

            # mapL1, mapL2 = cv.initUndistortRectifyMap(lmtx, ldist, R1, Pn1, (left.shape[1], left.shape[0]), cv.CV_32FC1)
            # mapR1, mapR2 = cv.initUndistortRectifyMap(rmtx, rdist, R2, Pn2, (right.shape[1], right.shape[0]), cv.CV_32FC1)

            # left_rect = cv.remap(left, mapL1, mapL2, cv.INTER_LINEAR)
            # right_rect = cv.remap(right, mapR1, mapR2, cv.INTER_LINEAR)
            # left_undistorted = cv.warpPerspective(left, H1, (w1, h1))
            # right_undistorted = cv.warpPerspective(right, H2, (w2, h2))
            # cv.imwrite("undistorted_L.png", left_rect)
            # cv.imwrite("undistorted_R.png", right_rect)

            ############## Calculate Disparity (Depth Map) ##############
            results = model.predict(left_rect, imgsz=1920, max_det=2, conf=0.3)
            result = (results[0])

            mask = result.masks[0].data[0].numpy()

            # Using StereoBM
            left_disp = left_matcher.compute(left_for_matcher, right_for_matcher)
            left_disp = left_disp.astype(np.float32)
            left_disp /= 16.0
            right_disp = right_matcher.compute(right_for_matcher, left_for_matcher)
            right_disp = right_disp.astype(np.float32)
            right_disp /= 16.0
            filtered_disp = wls_filter.filter(left_disp, left, disparity_map_right=right_disp)
            average_disp = np.average(np.array([filtered_disp[mask[:-8][:].astype(np.bool_)]]))
            disparities = np.append(disparities, average_disp)  # add current disparity of car to disparities array

            filtered_Disp_Buffer = np.concatenate([np.expand_dims(filtered_disp, 0), filtered_Disp_Buffer])[:-1]
            print(filtered_Disp_Buffer.shape)
            # plt.imshow(left_disp)
            # plt.show()
            # plt.imshow(filtered_disp)
            # plt.show()
            # baseline = 508 #this is equivalent to 20 inches in mm
            # left_focal_length=17 #mm
            # left_pixel_size=0.0042 #mm, 4.2um
            # right_focal_length=18 #mm
            # right_pixel_size=0.00389 #mm, 3.89um
            # camera_left_K = np.array([[left_focal_length/left_pixel_size,0,955],
            #                           [0,left_focal_length/left_pixel_size,540],
            #                           [0,0,1]])

            orig_map = matplotlib.colormaps.get_cmap('hot')
            reversed_map = orig_map.reversed()

            plt.imshow(filtered_disp, reversed_map)
            plt.colorbar()
            plt.show()
            # Press Q on keyboard to  exit
            # if cv.waitKey(25) & 0xFF == ord('q'):
            #   break
            # camera_left_K = np.array([[left_focal_length/left_pixel_size,0,955],
            #                           [0,left_focal_length/left_pixel_size,540],
            #                           [0,0,1]])

            # calculate velocity
            # left_focal_length=17 #mm
            # left_pixel_size=0.0042 #mm, 4.2um
            # right_focal_length=18 #mm
            # right_pixel_size=0.00389 #mm, 3.89um
            print(filtered_disp.shape)
            if count > 0:
                y1 = yPositions[int(count / window)]
                y0 = yPositions[int(count / window) - 1]
                x1 = xPositions[int(count / window)]
                x0 = xPositions[int(count / window) - 1]

                X0 = .508 * (x0 - 955) / filtered_Disp_Buffer[1, int(y0), int(x0)]
                X1 = .508 * (x1 - 955) / filtered_Disp_Buffer[0, int(y1), int(x1)]
                Y1 = .508 * (y1 - 540) / filtered_Disp_Buffer[0, int(y1), int(x1)]
                Y0 = .508 * (y0 - 540) / filtered_Disp_Buffer[1, int(y0), int(y0)]
                Z0 = (.017 * .508) / (filtered_Disp_Buffer[1, int(y0), int(x0)] * 4.2e-6)
                Z1 = (.017 * .508) / (filtered_Disp_Buffer[0, int(y1), int(x1)] * 4.2e-6)

                time_elapsed = window / fps

                Y_Vel = (Y1 - Y0) / time_elapsed
                X_Vel = (X1 - X0) / time_elapsed
                Z_Vel = (Z1 - Z0) / time_elapsed
                XVelBuffer.append(X_Vel)
                YVelBuffer.append(Y_Vel)
                print(X_Vel, "X_Velocity")
                print(Y_Vel, "Y_Velocity")
                print(Z_Vel, "Z_Velocity")
                print("disparity -1", filtered_Disp_Buffer[1, int(y0), int(x0)])
                print("disparity current", filtered_Disp_Buffer[0, int(y1), int(x1)])
                print("current depth", Z1)
                print("previous depth", Z0)
            # Break the loop
        count += 1
    else:
        break
print(np.mean(XVelBuffer) * 2.23693, "Average XVEL")  # vel in m/s
print(np.mean(YVelBuffer) * 2.23693, "Average YVEL")  # vel in m/s
# When everything done, release the video capture object
vid_left.release()
vid_right.release()

# Closes all the frames
cv.destroyAllWindows()

disparities.tofile('disparities.csv', sep=',')
