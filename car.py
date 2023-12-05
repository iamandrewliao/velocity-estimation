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

model = YOLO('yolov8n.pt')

video_path = "vids/right_fullcar.mp4"

cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
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
            # if len(track) > 30:  # retain 90 tracks for 90 frames
            #     track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(229, 255, 0), thickness=5)

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

print(track_history[1])  # output the track coordinates for car id=1
# the track coordinates define the x-axis movement aka side-to-side
# this isn't perfectly perpendicular to the cameras though, so depth information will be needed
x_init = track_history[1][0][0]
x_final = track_history[1][-1][0]
fps = 60
frames = len(track_history[1]) # should match number of frames in video: print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print(f"x-axis (side-to-side) speed in pixels/second: {(x_init-x_final)/((1/fps)*frames)}")
