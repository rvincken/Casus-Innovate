import cv2
import mediapipe as mp
import os
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

button_x, button_y, button_w, button_h = 100, 100, 150, 50

POSE_COLOR = (0, 255, 0)
HAND_COLOR = (255, 0, 0)
FACE_COLOR = (0, 0, 255)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = holistic.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 255, 0), -1)
        cv2.putText(frame, "Open Venster", (button_x + 10, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=POSE_COLOR, thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        if result.face_landmarks:
            mp_drawing.draw_landmarks(frame, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=FACE_COLOR, thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if button_x < cx < button_x + button_w and button_y < cy < button_y + button_h:
                        print("Knop ingedrukt! Venster openen...")
                        os.system("start chrome.exe")
                        time.sleep(1)
                    

        cv2.imshow('Pose & Handtracking met Knop Interactie', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
