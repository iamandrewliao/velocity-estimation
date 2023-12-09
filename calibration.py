#https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system

import cv2 as cv
import glob
import numpy as np

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synced frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
    # print(c1_images_names,c2_images_names)
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    rows = 4 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 0.025 #change this to the real world square size. Or not.
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows,columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows,columns), None)
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (4,8), corners1, c_ret1)
            cv.imshow('img', frame1)
 
            cv.drawChessboardCorners(frame2, (4,8), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    return R, T,E,F
 


images_folder = "D2/*"
image_names = sorted(glob.glob(images_folder))
left_images=[]
right_images = []
for imname in image_names:
    # print(imname)
    im=cv.imread(imname,1)
    cv.imshow('initial',im)
    if "left" in imname:
        left_images.append(im)
    if "right" in imname:
        right_images.append(im)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
rows=4
columns=8
world_scaling=1

objp = np.zeros((rows*columns,3),np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
objp = world_scaling*objp

width = left_images[0].shape[1]
height = left_images[0].shape[0]

left_imgpoints=[]
right_imgpoints=[]
left_objpoints=[]
right_objpoints=[]

for i in range(len(left_images)):
    gray_left=cv.cvtColor(left_images[i],cv.COLOR_BGR2GRAY)
    gray_right=cv.cvtColor(right_images[i],cv.COLOR_BGR2GRAY)
    
    left_ret, left_corners = cv.findChessboardCorners(gray_left, (rows,columns), None)
    right_ret, right_corners = cv.findChessboardCorners(gray_right, (rows,columns), None)
    # print(left_ret,right_ret)
    if left_ret == True & right_ret == True:
        conv_size = (11,11)

        left_corners = cv.cornerSubPix(gray_left,left_corners,conv_size, (-1,-1), criteria)
        cv.drawChessboardCorners(left_images[i],(rows,columns),left_corners,left_ret)
        cv.imshow('left corners',left_images[i])
        cv.waitKey(0)
        left_objpoints.append(objp)
        left_imgpoints.append(left_corners)

        conv_size = (11,11)

        right_corners = cv.cornerSubPix(gray_right,right_corners,conv_size, (-1,-1), criteria)
        cv.drawChessboardCorners(right_images[i],(rows,columns),right_corners,right_ret)
        cv.imshow('right corners',right_images[i])
        cv.waitKey(0)
        right_objpoints.append(objp)
        right_imgpoints.append(right_corners)

lret, lmtx, ldist, lrvecs, ltvecs = cv.calibrateCamera(left_objpoints,
                                                       left_imgpoints, 
                                                       (width,height),
                                                       None,
                                                       None)
rret, rmtx, rdist, rrvecs, rtvecs = cv.calibrateCamera(right_objpoints,
                                                       right_imgpoints, 
                                                       (width,height),
                                                       None,
                                                       None)
# print(lret)
# print(lmtx)
# print(ldist)
# print(lrvecs)
# print(ltvecs)
# print(rret)
# print(rmtx)
# print(rdist)
# print(rrvecs)
# print(rtvecs)


R,T,E,F = stereo_calibrate(lmtx, ldist, rmtx, rdist, 'synced/*')
# print(R,T)
# np.savez('camera_parameters',lmtx=lmtx,ldist=ldist,rmtx=rmtx,rdist=rdist,R=R,T=T)


R1, R2, Pn1, Pn2, _,_,_ = cv.stereoRectify(lmtx,ldist,rmtx,rdist,(width,height),R,T,alpha=0)
rectify1=R1.dot(np.linalg.inv(lmtx))
rectify2=R2.dot(np.linalg.inv(rmtx))

left = cv.imread('left.png')
right=cv.imread('right.png')

mapL1, mapL2 = cv.initUndistortRectifyMap(lmtx, ldist, rectify1, Pn1, (left.shape[1], left.shape[0]), cv.CV_32FC1)
mapR1, mapR2 = cv.initUndistortRectifyMap(rmtx, rdist, rectify1, Pn2, (right.shape[1], right.shape[0]), cv.CV_32FC1)

left_rect = cv.remap(left, mapL1, mapL2, cv.INTER_LINEAR)
right_rect = cv.remap(right, mapR1, mapR2, cv.INTER_LINEAR)

cv.imshow('left rectified',left_rect)
cv.imshow('right rectified', right_rect)
cv.waitKey(0)
cv.destroyAllWindows()