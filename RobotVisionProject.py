import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Values for 


baseline = 20 #this is in inches currently, can change as needed
left_focal_length=17 #mm
left_pixel_size=0.0042 #mm, 4.2um
right_focal_length=18 #mm
right_pixel_size=0.00389 #mm, 3.89um
camera_left_K = np.array([[left_focal_length/left_pixel_size,0,955],
                          [0,left_focal_length/left_pixel_size,540],
                          [0,0,1]])
camera_right_K = np.array([[right_focal_length/right_pixel_size,0,955],
                          [0,right_focal_length/right_pixel_size,540],
                          [0,0,1]])


vid_left = cv.VideoCapture("vids/left.mp4")
vid_right = cv.VideoCapture("vids/right.mp4")

if (vid_left.isOpened()== False or vid_right.isOpened()==False): 
  print("Error opening video stream or file")
 
# fps = vid_left.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
# frame_count = int(vid_left.get(cv.CAP_PROP_FRAME_COUNT))
# duration = frame_count/fps
# print(duration)
count=0
# Read until video is completed
while(vid_left.isOpened() and vid_right.isOpened()):
  # Capture frame-by-frame
  count+=1
  ret_left, frame_left = vid_left.read()
  ret_right, frame_right = vid_right.read()
  h,w,d=frame_left.shape
  print(h,w,d)
  print(frame_right.shape)
  left=cv.cvtColor(frame_left,cv.COLOR_BGR2GRAY)
  right=cv.cvtColor(frame_right,cv.COLOR_BGR2GRAY)
  # frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
  
  # left = frame[:,0:int(w/2)]
  # right = frame[:,int(w/2):]

  if count==320:
    cv.imwrite("left.png",left)
    cv.imwrite("right.png",right)

  if ret_left == True and ret_right == True:
 
    # Display the resulting frame
    cv.imshow('Left Frame',left)
    cv.imshow('Right Frame',right)
    stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left,right)
    plt.imshow(disparity,'gray')
    plt.show()
    
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
for vid in [vid_left, vid_right]:
  vid.release()
 
# Closes all the frames
cv.destroyAllWindows()

