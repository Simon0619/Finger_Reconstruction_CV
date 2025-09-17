import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Finger tip landmark indices (5 fingertips among 21 landmarks)
finger_tip_ids = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]

# Open video file
video_path = 'pre_training.mp4'
cap = cv2.VideoCapture(video_path)

frame_num = 0
all_data = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    frame_features = [frame_num]

    if results.multi_hand_landmarks:
        hands_detected = results.multi_hand_landmarks[:2]  # Only take up to 2 hands

        for hand_landmarks in hands_detected:
            for fid in finger_tip_ids:
                landmark = hand_landmarks.landmark[fid]
                frame_features.extend([landmark.x, landmark.y, landmark.z])

        # Pad with zeros if less than 2 hands are detected
        missing_hands = 2 - len(hands_detected)
        for _ in range(missing_hands * len(finger_tip_ids)):
            frame_features.extend([0.0, 0.0, 0.0])
    else:
        # No hands detected, fill with 10 fingers × 3D = 30 zeros
        frame_features.extend([0.0] * 30)

    all_data.append(frame_features)
    frame_num += 1

cap.release()
hands.close()

# Save to CSV
columns = ['frame']
for hand in ['L', 'R']:
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        columns.extend([f'{hand}_{finger}_x', f'{hand}_{finger}_y', f'{hand}_{finger}_z'])

df = pd.DataFrame(all_data, columns=columns)
df.to_csv('finger_features.csv', index=False)

print(f"Feature extraction completed. Saved to 'finger_features.csv', total {len(df)} frames.")
