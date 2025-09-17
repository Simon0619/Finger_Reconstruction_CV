import cv2
import pandas as pd
import json
from joblib import load

# === Configurations ===
video_path = 'pre_training.mp4'
features_path = 'finger_features.csv'
output_path = 'output_annotated_with_keys.mp4'

# === Load feature data ===
features_df = pd.read_csv(features_path).set_index('frame')

# === Load trained press detection model ===
model = load("press_classifier.pkl")
X = features_df.drop(columns=["frame"], errors='ignore')
y_pred = model.predict(X)

# === Open video file ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
frame_num = 0

# === Define white keys (used for mapping) ===
white_keys = [
    'F2', 'G2', 'A2', 'B2',
    'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
    'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5',
    'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6'
]

WHITE_KEY_COUNT = len(white_keys)
white_key_width = w / WHITE_KEY_COUNT
white_key_y_threshold = int(h * 0.95)  # Bottom limit for white key region

# === Mapping: finger names to landmark indices ===
finger_joint_map = {
    'L_thumb': 4, 'L_index': 8, 'L_middle': 12, 'L_ring': 16, 'L_pinky': 20,
    'R_thumb': 4, 'R_index': 8, 'R_middle': 12, 'R_ring': 16, 'R_pinky': 20
}

# === Detect which key each finger is pressing ===
def detect_pressed_keys(row):
    pressed = {}
    for finger_name, joint_idx in finger_joint_map.items():
        x_norm = row.get(f'{finger_name}_x')
        y_norm = row.get(f'{finger_name}_y')
        if x_norm is None or y_norm is None:
            continue

        x = int(x_norm * w)
        y = int(y_norm * h)

        if y < white_key_y_threshold:
            key_index = int(x / white_key_width)
            if 0 <= key_index < WHITE_KEY_COUNT:
                key = white_keys[key_index]
                pressed[finger_name] = (x, y, key)
    return pressed

print("🎥 Annotating video with model-based key press predictions...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    press_text = "NO"
    pressed = {}

    # Check if current frame is predicted as keypress
    if frame_num in features_df.index and y_pred[frame_num] == 1:
        row = features_df.loc[frame_num]
        pressed = detect_pressed_keys(row)
        press_text = "YES: " + ", ".join([f"{f} {k[2]}" for f, k in pressed.items()])

        # Draw red dot on fingertip and green key label
        for finger_name, (x, y, key) in pressed.items():
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(frame, key, (x + 4, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame number and press status
    cv2.putText(frame, f'Frame: {frame_num}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f'Press: {press_text}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if y_pred[frame_num] == 1 else (0, 0, 255), 2)

    out.write(frame)
    frame_num += 1

cap.release()
out.release()
print(f"\n Annotated video saved as: {output_path}")
