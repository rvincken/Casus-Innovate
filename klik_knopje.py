import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

button_x, button_y, button_w, button_h = 100, 100, 150, 50

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose, \
     mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 255, 0), -1)
        cv2.putText(frame, "Open Venster", (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 8:
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

                        if button_x < cx < button_x + button_w and button_y < cy < button_y + button_h:
                            print("Knop ingedrukt! Venster openen...")
                            os.system("open -a Safari")

        cv2.imshow('Hand Tracking met Knop Interactie', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()