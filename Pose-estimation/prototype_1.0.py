#
#installeer eerst Mediapipe en OpenCV
#pip install mediapipe & pip install opencv-python
#
import cv2
import mediapipe as mp

# Mediapipe initialiseren
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Camera openen
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# MediaPipe kleuren voor betere weergave
POSE_COLOR = (0, 255, 0)    # Groen voor skelet
HAND_COLOR = (255, 0, 0)    # Blauw voor handen
FACE_COLOR = (0, 0, 255)    # Rood voor gezicht


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #check of de camera beschikbaar is
    while cap.isOpened():
        #
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converteer frame naar RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose, hand en gezicht detecteren
        result = holistic.process(frame_rgb)
        
        # Teken skelet (33 punten)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=POSE_COLOR, thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
        
        # Teken handen (21 punten per hand)
        #linker hand
        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
        #rechterhand
        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        #print(result.right_hand_landmarks)
        #print(result.left_hand_landmarks)
        # Teken gezichtslandmarks (468 punten)
        if result.face_landmarks:
            mp_drawing.draw_landmarks(frame, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=FACE_COLOR, thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

        # Toon de output
        cv2.imshow('Advanced Pose, Hand & Face Tracking', frame)
        
        #als q wordt gedrukt stop het programma
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
