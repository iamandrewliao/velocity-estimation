from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# imgL = cv2.imread('left-fullcar.jpg', cv2.IMREAD_GRAYSCALE)
# imgR = cv2.imread('right-fullcar.jpg', cv2.IMREAD_GRAYSCALE)
# # cv2.imshow("L", imgL)
# # cv2.waitKey(0)
# stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity,'gray')
# plt.show()

# img1 = cv2.imread('left-fullcar.jpg')
# img2 = cv2.imread('right-fullcar.jpg')

model = YOLO('yolov8n.pt')

video_path = "vids/right_fullcar.mp4"

cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
# we will use track_idx to access position data to calculate speed later
track_idx = 0
# window is the amount of data points we look at to get the current speed
window = 10
# fps is necessary for calculating speed
fps = 60

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()  # (x,y) = center point
        track_ids = results[0].boxes.id.int().cpu().tolist()  # id = track IDs of the boxes (if available)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x)-float(w/2), float(y)))  # x, y is the center point
            # print(track_history[1])
            # if len(track) > 30:  # retain 90 tracks for 90 frames
            #     track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(229, 255, 0), thickness=5)
        # print(track_history[1])

        # Speed calculation (rolling avg; position change over time window)
        if len(track_history[1])-track_idx >= window:  # e.g. if x amt of data points are available in the track_history
            # track_idx is used to move through track_history to get the last x data points to get "current" speed
            # print(f"track_idx: {track_idx}")
            # print(f"len(track_history[1])-track_idx: {len(track_history[1])-track_idx}")
            # print("speed calculation possible")
            x_init = track_history[1][track_idx][0]
            x_final = track_history[1][-1][0]
            cur_speed = (x_init - x_final) / ((1 / fps) * window)  # window is essentially the # of frames considered
            cv2.putText(annotated_frame, f"Speed (pixels/sec): {cur_speed}", (50, 50), 2, 1, (229, 255, 0), 2)
            track_idx += 1

        # Display the annotated frame
        cv2.imshow("Annotated Frame", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
#
# print(track_history[1])  # output the track coordinates for car id=1
# # the track coordinates define the x-axis movement aka side-to-side
# # this isn't perfectly perpendicular to the cameras though, so depth information will be needed
# x_init = track_history[1][0][0]
# x_final = track_history[1][-1][0]
# fps = 60
# frames = len(track_history[1]) # should match number of frames in video: print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
# print(f"average x-axis (side-to-side) speed in pixels/second: {(x_init-x_final)/((1/fps)*frames)}")
