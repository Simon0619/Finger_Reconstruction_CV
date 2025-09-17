import pandas as pd
import joblib
import json
import cv2

# === Step 1: Load classifier model and features ===
model = joblib.load('press_classifier.pkl')
df = pd.read_csv('finger_features.csv')

# === Step 2: Load keyboard key mapping (bounding boxes) ===
with open('key_mapping.json', 'r') as f:
    key_map = json.load(f)

# === Step 3: Get video resolution from the first frame (or hardcode) ===
cap = cv2.VideoCapture('pre_training.mp4')
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
cap.release()

# === Step 4: Predict press/non-press label for each frame ===
X = df.drop(columns=['frame'])
y_pred = model.predict(X)

# === Step 5: Check which keys are pressed in each pressed frame ===
def detect_pressed_keys(row):
    pressed_keys = []
    margin = 10  # Add a small tolerance margin

    for key_name, box in key_map.items():
        x_min = box['x_min'] - margin
        x_max = box['x_max'] + margin
        y_min = box['y_min'] - margin
        y_max = box['y_max'] + margin

        for finger in ['L_thumb', 'L_index', 'L_middle', 'L_ring', 'L_pinky',
                       'R_thumb', 'R_index', 'R_middle', 'R_ring', 'R_pinky']:
            x_norm = row.get(f'{finger}_x', 0)
            y_norm = row.get(f'{finger}_y', 0)

            x = int(x_norm * frame_width)
            y = int(y_norm * frame_height)

            if x_min <= x <= x_max and y_min <= y <= y_max:
                pressed_keys.append(key_name)
                break  # One finger is enough to trigger a key

    return pressed_keys

# === Step 6: Collect press events for pressed frames only ===
events = []

for i in range(len(df)):
    if y_pred[i] == 1:
        row = df.iloc[i]
        keys = detect_pressed_keys(row)
        events.append({
            'frame': int(row['frame']),
            'keys': keys
        })

# === Step 7: Save to JSON ===
with open('press_events.json', 'w') as f:
    json.dump(events, f, indent=2)

print(f" Detected {len(events)} key press events. Saved to 'press_events.json'")
