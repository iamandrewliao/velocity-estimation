# velocity-estimation  
### True velocity estimation using YOLOv8 object tracking + segmentation and OpenCV stereo vision techniques  
https://docs.google.com/presentation/d/1F0fz9mAL1xf_owllBixQae-fdZ8KYEeUk3C7Bm1cz2k/edit?usp=sharing  
  
The main scripts are:
1) uncalibrated_stereo.py, which uses uncalibrated stereo vision techniques + object segmentation to obtain disparity
2) calibrated_stereo.py, which uses calibrated stereo vision techniques + object segmentation to obtain disparity; the code to calculate true velocity is in this script
3) x_axis.py, which uses object tracking to obtain x-axis positions  

final project for EE5271: Robot Vision @ UMN Fall '23 taught by Changhyun Choi  
