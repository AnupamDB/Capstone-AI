# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # mp_drawing = mp.solutions.drawing_utils
# # # mp_pose = mp.solutions.pose

# # # cap = cv2.VideoCapture(0)

# # # # Curl counter variables
# # # counter = 0 
# # # stage = None

# # # def calculate_angle(a,b,c):
# # #     a = np.array(a) # First
# # #     b = np.array(b) # Mid
# # #     c = np.array(c) # End
    
# # #     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
# # #     angle = np.abs(radians*180.0/np.pi)
    
# # #     if angle >180.0:
# # #         angle = 360-angle
        
# # #     return angle 

# # # ## Setup mediapipe instance
# # # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# # #     while cap.isOpened():
# # #         ret, frame = cap.read()
        
# # #         # Recolor image to RGB
# # #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # #         image.flags.writeable = False
      
# # #         # Make detection
# # #         results = pose.process(image)
    
# # #         # Recolor back to BGR
# # #         image.flags.writeable = True
# # #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
# # #         # Extract landmarks
# # #         try:
# # #             landmarks = results.pose_landmarks.landmark
            
# # #             # Get coordinates
# # #             shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# # #             elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# # #             wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
# # #             # Calculate angle
# # #             angle = calculate_angle(shoulder, elbow, wrist)
            
# # #             # Visualize angle
# # #             cv2.putText(image, str(angle), 
# # #                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
# # #                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
# # #                                 )
            
# # #             # Curl counter logic
# # #             if angle > 160:
# # #                 stage = "down"
# # #             if angle < 30 and stage =='down':
# # #                 stage="up"
# # #                 counter +=1
# # #                 print(counter)

# # #             cv2.circle(image, tuple(np.multiply(shoulder, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
# # #             cv2.circle(image, tuple(np.multiply(elbow, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
# # #             cv2.circle(image, tuple(np.multiply(wrist, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
# # #             cv2.line(image, tuple(np.multiply(shoulder, [640, 480]).astype(int)), tuple(np.multiply(elbow, [640, 480]).astype(int)), (0, 255, 0), 2)
# # #             cv2.line(image, tuple(np.multiply(elbow, [640, 480]).astype(int)), tuple(np.multiply(wrist, [640, 480]).astype(int)), (0, 255, 0), 2)
                       
# # #         except:
# # #             pass
        
# # #         # Render curl counter
# # #         # Setup status box
# # #         cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
# # #         # Rep data
# # #         cv2.putText(image, 'REPS', (15,12), 
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# # #         cv2.putText(image, str(counter), 
# # #                     (10,60), 
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
# # #         # Stage data
# # #         cv2.putText(image, 'STAGE', (65,12), 
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# # #         cv2.putText(image, stage, 
# # #                     (60,60), 
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
# # #         # Render detections
# # #         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
# # #                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
# # #                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
# # #                                  )               
        
# # #         cv2.imshow('Mediapipe Feed', image)

# # #         if cv2.waitKey(10) & 0xFF == ord('q'):
# # #             break

# # #     cap.release()
# # #     cv2.destroyAllWindows()


# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # def calculate_angle(a, b, c):
# #     a = np.array(a)  # First
# #     b = np.array(b)  # Mid
# #     c = np.array(c)  # End
    
# #     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
# #     angle = np.abs(radians * 180.0 / np.pi)
    
# #     if angle > 180.0:
# #         angle = 360 - angle
        
# #     return angle

# # mp_drawing = mp.solutions.drawing_utils
# # mp_pose = mp.solutions.pose

# # cap = cv2.VideoCapture(0)

# # # Curl counter variables
# # counter = 0 
# # stage = None

# # ## Setup mediapipe instance
# # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# #     while cap.isOpened():
# #         ret, frame = cap.read()
        
# #         # Create a blank image with the same dimensions as the frame
# #         blank_image = np.zeros_like(frame)
        
# #         # Recolor image to RGB
# #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         image.flags.writeable = False
      
# #         # Make detection
# #         results = pose.process(image)
    
# #         # Recolor back to BGR
# #         image.flags.writeable = True
# #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
# #         # Extract landmarks
# #         try:
# #             landmarks = results.pose_landmarks.landmark
            
# #             # Get coordinates
# #             shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
# #                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# #             elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
# #                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# #             wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
# #                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
# #             # Calculate angle
# #             angle = calculate_angle(shoulder, elbow, wrist)
            
# #             # Curl counter logic
# #             if angle > 160:
# #                 stage = "down"
# #             if angle < 30 and stage == 'down':
# #                 stage = "up"
# #                 counter += 1
# #                 print(counter)
                
# #             # Draw landmarks
# #             cv2.circle(blank_image, tuple(np.multiply(shoulder, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
# #             cv2.circle(blank_image, tuple(np.multiply(elbow, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
# #             cv2.circle(blank_image, tuple(np.multiply(wrist, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
# #             cv2.line(blank_image, tuple(np.multiply(shoulder, [640, 480]).astype(int)), tuple(np.multiply(elbow, [640, 480]).astype(int)), (0, 255, 0), 2)
# #             cv2.line(blank_image, tuple(np.multiply(elbow, [640, 480]).astype(int)), tuple(np.multiply(wrist, [640, 480]).astype(int)), (0, 255, 0), 2)
                       
# #         except:
# #             pass
        
# #         # Render curl counter on the blank image
# #         # Setup status box
# #         cv2.rectangle(blank_image, (0, 0), (225, 73), (245, 117, 16), -1)
        
# #         # Rep data
# #         cv2.putText(blank_image, 'REPS', (15, 12), 
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
# #         cv2.putText(blank_image, str(counter), 
# #                     (10, 60), 
# #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
# #         # Stage data
# #         cv2.putText(blank_image, 'STAGE', (65, 12), 
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
# #         cv2.putText(blank_image, stage, 
# #                     (60, 60), 
# #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
# #         cv2.imshow('Mediapipe Feed', blank_image)

# #         if cv2.waitKey(10) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# def calculate_angle(img, p1, p2, p3, lmList):
#     # a = np.array(a)  # First
#     # b = np.array(b)  # Mid
#     # c = np.array(c)  # End
#     # Get the landmarks
#     x1, y1 = lmList[p1][1:]
#     x2, y2 = lmList[p2][1:]
#     x3, y3 = lmList[p3][1:]

#     # Calculate the Angle
#     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                             math.atan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360

#         # print(angle)

#         # Draw
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#     cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#     return angle
    
#     # radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     # angle = np.abs(radians * 180.0 / np.pi)
    
#     # if angle > 180.0:
#     #     angle = 360 - angle
        
#     # return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(img)
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(img, results.pose_landmarks,
#                                            mp_pose.POSE_CONNECTIONS)
        
#         lmList = []
#         if results.pose_landmarks:
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 # print(id, lm)
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])
#                 cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

#         # Extract landmarks
#         # try:
#         #     landmarks = results.pose_landmarks.landmark
            
#             # # Get coordinates
#             # shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#             #             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             # elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#             #          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             # wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#             #          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#             if len(lmList) != 0:
#                 # Calculate angle
#                 angle = calculate_angle(img, 12, 14, 16, lmList)
                
#                 # # Curl counter logic
#                 # if angle > 160:
#                 #     stage = "down"
#                 # if angle < 30 and stage == 'down':
#                 #     stage = "up"
#                 #     counter += 1
#                 #     print(counter)

#                 per = np.interp(angle, (210, 310), (0, 100))
#                 bar = np.interp(angle, (220, 310), (650, 100))
#                 # print(angle, per)

#                 # Check for the dumbbell curls
#                 color = (255, 0, 255)
#                 if per == 100:
#                     color = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per == 0:
#                     color = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0
#                 print(count)
                
#             # Draw landmarks and connections
#         #     cv2.circle(frame, tuple(np.multiply(shoulder, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
#         #     cv2.circle(frame, tuple(np.multiply(elbow, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
#         #     cv2.circle(frame, tuple(np.multiply(wrist, [640, 480]).astype(int)), 10, (0, 255, 0), -1)
#         #     cv2.line(frame, tuple(np.multiply(shoulder, [640, 480]).astype(int)), tuple(np.multiply(elbow, [640, 480]).astype(int)), (0, 255, 0), 2)
#         #     cv2.line(frame, tuple(np.multiply(elbow, [640, 480]).astype(int)), tuple(np.multiply(wrist, [640, 480]).astype(int)), (0, 255, 0), 2)
                       
#         # except:
#         #     pass
        
#         # # Render curl counter on the frame
#         # # Setup status box
#         # cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)
        
#         # # Rep data
#         # cv2.putText(frame, 'REPS', (15, 12), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         # cv2.putText(frame, str(counter), 
#         #             (10, 60), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
#         # # Stage data
#         # cv2.putText(frame, 'STAGE', (65, 12), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         # cv2.putText(frame, stage, 
#         #             (60, 60), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#         # Draw Bar
#         cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
#         cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
#         cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
#                     color, 4)

#         # Draw Curl Count
#         cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
#                     (255, 0, 0), 25)
        
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
#                             math.atan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360

#     # Draw
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#     cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# ## Setup mediapipe instance
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
#                 cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

#             if len(lmList) != 0:
#                 # Calculate angle
#                 angle = calculate_angle(img, 12, 14, 16, lmList)

#                 per = np.interp(angle, (210, 310), (0, 100))
#                 bar = np.interp(angle, (220, 310), (650, 100))

#                 # Check for the dumbbell curls
#                 color = (255, 0, 255)
#                 if per == 100:
#                     color = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per == 0:
#                     color = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0
#                 print(count)

#                 # Draw Bar
#                 cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
#                 cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
#                 cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

#                 # Draw Curl Count
#                 cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

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
#                             math.atan2(y1 - y2, x1 - x2))
#     if angle < 0:
#         angle += 360

#     # Draw the lines only, without the circles
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# ## Setup mediapipe instance
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
#                 # Calculate angle
#                 angle = calculate_angle(img, 12, 14, 16, lmList)

#                 per = np.interp(angle, (210, 310), (0, 100))
#                 bar = np.interp(angle, (220, 310), (650, 100))

#                 # Check for the dumbbell curls
#                 color = (255, 0, 255)
#                 if per == 100:
#                     color = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per == 0:
#                     color = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0
#                 print(count)

#                 # Draw Bar
#                 cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
#                 cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
#                 cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

#                 # Draw Curl Count
#                 cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

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

#     # Draw
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

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# # Setup mediapipe instance
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
#                 # Calculate angle
#                 angle = calculate_angle(img, 12, 14, 16, lmList)

#                 per = np.interp(angle, (210, 310), (0, 100))
#                 bar = np.interp(angle, (220, 310), (650, 100))

#                 # Check for the dumbbell curls
#                 color = (255, 0, 255)
#                 if per == 100:
#                     color = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per == 0:
#                     color = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0
#                 print(count)

#                 # Draw Bar
#                 cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
#                 cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
#                 cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

#                 # Draw Curl Count
#                 cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

#                 # Draw the specific landmarks
#                 for i in [12, 14, 16]:  # Only draw these landmarks (shoulder, elbow, wrist)
#                     cx, cy = lmList[i][1:]
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

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

#     # Draw the lines and points for the specific landmarks
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
    
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# # Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(img)

#         if results.pose_landmarks:
#             lmList = []
#             h, w, c = img.shape
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])

#             if len(lmList) != 0:
#                 # Calculate angle
#                 angle = calculate_angle(img, 12, 14, 16, lmList)

#                 per = np.interp(angle, (210, 310), (0, 100))
#                 bar = np.interp(angle, (220, 310), (650, 100))

#                 # Check for the dumbbell curls
#                 color = (255, 0, 255)
#                 if per == 100:
#                     color = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per == 0:
#                     color = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0
#                 print(count)

#                 # Draw Bar on the right side
#                 cv2.rectangle(img, (w-150, 100), (w-75, 650), color, 3)
#                 cv2.rectangle(img, (w-150, int(bar)), (w-75, 650), color, cv2.FILLED)
#                 cv2.putText(img, f'{int(per)} %', (w-150, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

#                 # Draw Curl Count on the bottom left
#                 cv2.rectangle(img, (0, h-270), (250, h), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(img, str(int(count)), (45, h-50), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

#                 # Draw the specific landmarks and lines
#                 for i in [12, 14, 16]:  # Only draw these landmarks (shoulder, elbow, wrist)
#                     cx, cy = lmList[i][1:]
#                     cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
#                     cv2.circle(img, (cx, cy), 15, (0, 0, 255), 2)

#                 # Draw connecting lines between the points (shoulder to elbow, elbow to wrist)
#                 cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[14][1], lmList[14][2]), (255, 255, 255), 3)
#                 cv2.line(img, (lmList[14][1], lmList[14][2]), (lmList[16][1], lmList[16][2]), (255, 255, 255), 3)

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

#     # Draw the lines and points for the specific landmarks
#     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#     cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#     cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#     cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
    
#     return angle

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # Full-screen window setup
# cv2.namedWindow("Mediapipe Feed", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Mediapipe Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# cap = cv2.VideoCapture(0)

# # Curl counter variables
# counter = 0 
# count = 0
# dir = 0
# stage = None

# # Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(img)

#         if results.pose_landmarks:
#             lmList = []
#             h, w, c = img.shape
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])

#             if len(lmList) != 0:
#                 # Calculate angle
#                 angle = calculate_angle(img, 12, 14, 16, lmList)

#                 per = np.interp(angle, (210, 310), (0, 100))
#                 bar = np.interp(angle, (220, 310), (650, 100))

#                 # Check for the dumbbell curls
#                 color = (255, 0, 255)
#                 if per == 100:
#                     color = (0, 255, 0)
#                     if dir == 0:
#                         count += 0.5
#                         dir = 1
#                 if per == 0:
#                     color = (0, 255, 0)
#                     if dir == 1:
#                         count += 0.5
#                         dir = 0
#                 print(count)

#                 # Draw Bar on the right side
#                 cv2.rectangle(img, (w-150, 100), (w-75, 650), color, 3)
#                 cv2.rectangle(img, (w-150, int(bar)), (w-75, 650), color, cv2.FILLED)
#                 cv2.putText(img, f'{int(per)} %', (w-150, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

#                 # Draw Curl Count on the bottom left
#                 cv2.rectangle(img, (0, h-270), (250, h), (0, 255, 0), cv2.FILLED)
#                 cv2.putText(img, str(int(count)), (45, h-50), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

#                 # Draw the specific landmarks and lines
#                 for i in [12, 14, 16]:  # Only draw these landmarks (shoulder, elbow, wrist)
#                     cx, cy = lmList[i][1:]
#                     cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
#                     cv2.circle(img, (cx, cy), 15, (0, 0, 255), 2)

#                 # Draw connecting lines between the points (shoulder to elbow, elbow to wrist)
#                 cv2.line(img, (lmList[12][1], lmList[12][2]), (lmList[14][1], lmList[14][2]), (255, 255, 255), 3)
#                 cv2.line(img, (lmList[14][1], lmList[14][2]), (lmList[16][1], lmList[16][2]), (255, 255, 255), 3)

#         # Convert back to BGR for displaying in OpenCV
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imshow('Mediapipe Feed', img)

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
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                            math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

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
        
        # Recolor image to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            lmList = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Calculate angle
                angle = calculate_angle(img, 12, 14, 16, lmList)

                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (720, 100))

                # Check for dumbbell curls
                color = (255, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0

                # Draw the bar
                cv2.rectangle(img, (1200, 100), (1250, 720), color, 3)
                cv2.rectangle(img, (1200, int(bar)), (1250, 720), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (1180, 75), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                # Draw the curl count
                cv2.rectangle(img, (10, 600), (160, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(count)), (45, 690), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 10)

        # Convert back to BGR for displaying in OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Mediapipe Feed', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
