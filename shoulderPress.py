# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# def calculate_angle(img, p1, p2, p3, lmList):
#     # Get the landmarks
#     x1, y1 = lmList[p1][1:]
#     x2, y2 = lmList[p2][1:]
#     x3, y3 = lmList[p3][1:]

#     # Calculate the Angle
#     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                          math.atan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360

#     # Draw the lines and circles
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#     cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase the width of the window
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Adjust the height proportionally

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# # Setup Mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(img)

#         if results.pose_landmarks:
#             lmList = []
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])

#             if len(lmList) != 0:
#                 # Calculate angles for both arms
#                 angle_right = calculate_angle(img, 12, 14, 16, lmList)  # Right arm (shoulder, elbow, wrist)
#                 angle_left = calculate_angle(img, 11, 13, 15, lmList)  # Left arm (shoulder, elbow, wrist)

#                 # Interpolation for both arms
#                 per_right = np.interp(angle_right, (210, 310), (0, 100))
#                 bar_right = np.interp(angle_right, (220, 310), (720, 100))

#                 per_left = np.interp(angle_left, (210, 310), (0, 100))
#                 bar_left = np.interp(angle_left, (220, 310), (720, 100))

#                 # Check for shoulder press movement
#                 color_right = (255, 0, 255)
#                 color_left = (255, 0, 255)

#                 if per_right == 100 and per_left == 100:
#                     color_right = (0, 255, 0)
#                     color_left = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per_right == 0 and per_left == 0:
#                     color_right = (0, 255, 0)
#                     color_left = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0

#                 # Draw the bars for both arms
#                 cv2.rectangle(img, (1200, 100), (1250, 720), color_right, 3)
#                 cv2.rectangle(img, (1200, int(bar_right)), (1250, 720), color_right, cv2.FILLED)
#                 cv2.putText(img, f'{int(per_right)} %', (1180, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_right, 2)

#                 cv2.rectangle(img, (100, 100), (150, 720), color_left, 3)
#                 cv2.rectangle(img, (100, int(bar_left)), (150, 720), color_left, cv2.FILLED)
#                 cv2.putText(img, f'{int(per_left)} %', (80, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_left, 2)

#                 # Draw the curl count
#                 cv2.rectangle(img, (10, 600), (160, 720), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(img, str(int(count)), (45, 690), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 10)

#         # Convert back to BGR for displaying in OpenCV
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imshow('Mediapipe Feed', img)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# def calculate_angle(img, p1, p2, p3, lmList):
#     # Get the landmarks
#     x1, y1 = lmList[p1][1:]
#     x2, y2 = lmList[p2][1:]
#     x3, y3 = lmList[p3][1:]

#     # Calculate the Angle
#     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                          math.atan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360

#     # Draw the lines and circles
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#     cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase the width of the window
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Adjust the height proportionally

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# # Setup Mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         frame = cv2.flip(frame, 1)
#         # Recolor image to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(img)
#         # Recolor image back to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         if results.pose_landmarks:
#             lmList = []
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frame.shape  # Use the original frame size
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])

#             if len(lmList) != 0:
#                 # Calculate angles for both arms
#                 angle_right = calculate_angle(frame, 12, 14, 16, lmList)  # Right arm (shoulder, elbow, wrist)
#                 angle_left = calculate_angle(frame, 11, 13, 15, lmList)  # Left arm (shoulder, elbow, wrist)

#                 # Ensure angles are mirrored correctly for left arm
#                 if angle_left > 180:
#                     angle_left = 360 - angle_left

#                 # Interpolation for both arms
#                 per_right = np.interp(angle_right, (210, 310), (0, 100))
#                 bar_right = np.interp(angle_right, (220, 310), (720, 100))

#                 per_left = np.interp(angle_left, (30, 130), (0, 100))  # Adjust the range for left arm
#                 bar_left = np.interp(angle_left, (30, 130), (720, 100))  # Adjust the range for left arm

#                 # Check for shoulder press movement
#                 color_right = (255, 0, 255)
#                 color_left = (255, 0, 255)

#                 if per_right == 100 and per_left == 100:
#                     color_right = (0, 255, 0)
#                     color_left = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per_right == 0 and per_left == 0:
#                     color_right = (0, 255, 0)
#                     color_left = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0

#                 # Draw the bars for both arms
#                 cv2.rectangle(frame, (1200, 100), (1250, 720), color_right, 3)
#                 cv2.rectangle(frame, (1200, int(bar_right)), (1250, 720), color_right, cv2.FILLED)
#                 cv2.putText(frame, f'{int(per_right)} %', (1180, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_right, 2)

#                 cv2.rectangle(frame, (50, 100), (100, 720), color_left, 3)  # Updated position for left bar
#                 cv2.rectangle(frame, (50, int(bar_left)), (100, 720), color_left, cv2.FILLED)
#                 cv2.putText(frame, f'{int(per_left)} %', (20, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_left, 2)

#                 # Draw the curl count
#                 cv2.rectangle(frame, (10, 600), (160, 720), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(frame, str(int(count)), (45, 690), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 10)

#         # Convert back to BGR for displaying in OpenCV
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         cv2.imshow('Mediapipe Feed', frame)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# def calculate_angle(img, p1, p2, p3, lmList):
#     # Get the landmarks
#     x1, y1 = lmList[p1][1:]
#     x2, y2 = lmList[p2][1:]
#     x3, y3 = lmList[p3][1:]

#     # Calculate the Angle
#     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                          math.atan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360

#     # Draw the lines and circles
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#     cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#     # print(angle)
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase the width of the window
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Adjust the height proportionally

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# # Setup Mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         frame = cv2.flip(frame, 1)
#         # Recolor image to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(img)
#         # Recolor image back to RGB
#         # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         color_right = (255, 0, 255)
#         color_left = (255, 0, 255)
#         per_left,per_right,bar_left,bar_right=0,0,0,0

#         if results.pose_landmarks:
#             lmList = []
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frame.shape  # Use the original frame size
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])

#             if len(lmList) != 0:
#                 # Calculate angles for both arms
#                 angle_right = calculate_angle(frame, 12, 14, 16, lmList)  # Right arm (shoulder, elbow, wrist)
#                 angle_left = calculate_angle(frame, 11, 13, 15, lmList)   # Left arm (shoulder, elbow, wrist)

#                 print("Left: ", angle_left)                
#                 print("Right: ", angle_right)                


#                 # Ensure angles are mirrored correctly for left arm
#                 # if angle_left > 180:
#                 #     angle_left = 360 - angle_left

#                 # # Normalize percentages for bars based on the new angle ranges
#                 # per_right = np.interp(angle_right, (-170,180), (0, 100))  # Right arm range: 120° to 78°
#                 # bar_right = np.interp(angle_right, (-170,180), (720, 100))  # Adjust bar height for decreasing angles

#                 # per_left = np.interp(angle_left, (-50,20), (0, 100))  # Left arm range: 230° to 265°
#                 # bar_left = np.interp(angle_left, (-50,20), (720, 100))  # Adjust bar height for decreasing angles
#                 angle_right = int(np.interp(angle_right, (-170,180), (100, 0)))
#                 angle_left = int(np.interp(angle_left, (-50, 20), (100, 0)))

#                 # # Cap percentages at 100% for clean visualization
#                 # per_right = min(max(per_right, 0), 100)
#                 # per_left = min(max(per_left, 0), 100)

#                 # Check for shoulder press movement
#                 color_right = (255, 0, 255)
#                 color_left = (255, 0, 255)

#                 if per_right >= 95 and per_left >= 95:
#                     color_right = (0, 255, 0)
#                     color_left = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per_right <= 10 and per_left <= 10:
#                     color_right = (0, 255, 0)
#                     color_left = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0

#         # # # Normalize percentages for bars based on the new angle ranges
#         # per_right = np.interp(angle_right, (110,95), (0, 100))  # Right arm range: 120° to 78°
#         # bar_right = np.interp(angle_right, (110,95), (720, 100))  # Adjust bar height for decreasing angles

#         # per_left = np.interp(angle_left, (400,200), (0, 100))  # Left arm range: 230° to 265°
#         # bar_left = np.interp(angle_left, (400,200), (720, 100))  # Adjust bar height for decreasing angles

#         # Map angles to percentages for bar movement
#         per_right = np.interp(angle_right, (180, 60), (0, 100))  # Right arm: 180° to 60°
#         bar_right = np.interp(angle_right, (180, 60), (720, 100))  # Adjust bar height: 720 to 100

#         per_left = np.interp(angle_left, (130, 50), (0, 100))  # Left arm: 130° to 50°
#         bar_left = np.interp(angle_left, (130, 50), (720, 100))  # Adjust bar height: 720 to 100


#         # Draw the bars for both arms
#         cv2.rectangle(frame, (1200, 100), (1250, 720), color_right, 3)
#         cv2.rectangle(frame, (1200, int(bar_right)), (1250, 720), color_right, cv2.FILLED)
#         cv2.putText(frame, f'{int(per_right)} %', (1180, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_right, 2)

#         cv2.rectangle(frame, (50, 100), (100, 720), color_left, 3)  # Updated position for left bar
#         cv2.rectangle(frame, (50, int(bar_left)), (100, 720), color_left, cv2.FILLED)
#         cv2.putText(frame, f'{int(per_left)} %', (20, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_left, 2)

#         # Draw the curl count
#         cv2.rectangle(frame, (10, 600), (160, 720), (0, 255, 0), cv2.FILLED)
#         cv2.putText(frame, str(int(count)), (45, 690), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 10)


#         # Convert back to BGR for displaying in OpenCV
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         cv2.imshow('Mediapipe Feed', frame)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(img, p1, p2, p3, lmList):
    # Get the landmarks
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    # Calculate the Angle
    angle = math.degrees(abs(math.atan2(y3 - y2, x3 - x2) -
                         math.atan2(y1 - y2, x1 - x2)))
    if angle > 180:
        angle = 360-angle

    # Draw the lines and circles
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
    cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
    cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
    cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
    cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase the width of the window
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Adjust the height proportionally

# Curl counter variables
counter = 0 
count = 0
dir = 0
stage = None

# Setup Mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Recolor image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        # Recolor image back to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lmList = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape  # Use the original frame size
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Calculate angles for both arms
                angle_right = calculate_angle(frame, 12, 14, 16, lmList)  # Right arm (shoulder, elbow, wrist)
                angle_left = calculate_angle(frame, 11, 13, 15, lmList)  # Left arm (shoulder, elbow, wrist)

                # Ensure angles are mirrored correctly for left arm
                # if angle_left > 180:
                #     angle_left = 360 - angle_left

                # Interpolation for both arms
                per_right = np.interp(angle_right, (70, 125), (100, 0))
                bar_right = np.interp(angle_right, (75, 125), (100, 720))

                per_left = np.interp(angle_left, (75, 125), (100, 0))  # Adjust the range for left arm
                bar_left = np.interp(angle_left, (70, 125), (100, 720))  # Adjust the range for left arm

                # Check for shoulder press movement
                color_right = (255, 0, 255)
                color_left = (255, 0, 255)

                if per_right == 100 and per_left == 100:
                    color_right = (0, 255, 0)
                    color_left = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per_right == 0 and per_left == 0:
                    color_right = (0, 255, 0)
                    color_left = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0

                # Draw the bars for both arms
                cv2.rectangle(frame, (1200, 100), (1250, 720), color_right, 3)
                cv2.rectangle(frame, (1200, int(bar_right)), (1250, 720), color_right, cv2.FILLED)
                cv2.putText(frame, f'{int(per_right)} %', (1180, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_right, 2)

                cv2.rectangle(frame, (50, 100), (100, 720), color_left, 3)  # Updated position for left bar
                cv2.rectangle(frame, (50, int(bar_left)), (100, 720), color_left, cv2.FILLED)
                cv2.putText(frame, f'{int(per_left)} %', (20, 75), cv2.FONT_HERSHEY_PLAIN, 2, color_left, 2)

                # Draw the curl count
                cv2.rectangle(frame, (10, 600), (160, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, str(int(count)), (45, 690), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 10)

        # Convert back to BGR for displaying in OpenCV
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Mediapipe Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()