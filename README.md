# Project：     F12 Visual Reconstruction of played music
# Author：      Deng Jianwei
# Student ID：  245188/10973015


## Introduction & Goal

This project detects piano key presses from video using hand joint coordinates and machine learning(Random Forest Classification).
To identify which keys are pressed and at "which frame" by analyzing hand motion from a "pre-training" video.


## Pipeline Description

1. "Feature Extraction"    (`extract_hand_features.py`)
   - Extracts MediaPipe 3D fingertip coordinates from video
   - Output: `finger_features.csv`

2. "Manual Labeling"         (`manual_label_tool.py`)
   - Manually label whether a key is pressed at each frame (binary)
   - Output: `labels.csv`

3. "Train Classifier"      (`train_press_classifier.py`)
   - Trains a Random Forest model to classify key-press frames
   - Output: `press_classifier.pkl`

4. "Predict Pressed Keys"  (`predict_pressed_keys.py`)
   - Uses the classifier + fingertip positions to determine which key was pressed
   - Output: `press_events.json`

5. "Visualization"         (`visualize_dynamic_keys.py`)
   - Overlays red circles and predicted key names on video
   - Output: `output_annotated_with_keys.mp4`


## File Descriptions
```
project/
├── extract_hand_features.py             # Extract fingertip positions using MediaPipe
├── manual_label_tool.py                 # Manually label frames as keypress or not
├── train_press_classifier.py            # Train classifier based on finger features
├── predict_pressed_keys.py              # Predict keypress events from features
├── visualize_dynamic_keys.py            # Generate annotated output video
├── pre_training.mp4                     # Input video
├── finger_features.csv                  # Extracted features per frame
├── labels.csv                           # Frame-wise labels (keypress or not)
├── press_classifier.pkl                 # Saved trained model
├── key_mapping.json                     # Piano key region mapping
├── press_events.json                    # Output detected keypress events
├── output_annotated_with_keys.mp4       # Final annotated video
└── keyboard.jpg                         # Used for manual key mapping
```
---

## How to Run（）

### Step 1: Create Virtual Environment (Recommended)
```bash
python -m venv venv310
source venv310/bin/activate       # On Windows: .\venv310\Scripts\Activate   
```

### Step 2: Install Required Libraries
```bash
pip install opencv-python mediapipe pandas numpy scikit-learn joblib
```

---

### Step 3: Feature Extraction
Run the script to extract fingertip 3D coordinates from video:
```bash
python extract_hand_features.py
```

---

### Step 4: Manually Label Pressed Frames
Label each frame as press (1) or non-press (0) by pressing space or other keys:
```bash
python manual_label_tool.py
```

---

### Step 5: Train Classifier
Train a Random Forest model to distinguish pressed vs. non-pressed frames:
```bash
python train_press_classifier.py
```
---

### Step 6: Predict Pressed Keys
Combine the classifier and geometric rules to determine which keys were pressed:
```bash
python predict_pressed_keys.py 
```
This creates:
- `press_events.json`: predicted press events
---

### Step 7: Visualize Output
Overlay predictions on the video:
```bash
python visualize_dynamic_keys.py
```
Result saved to: `output_annotated_with_keys.mp4`

---

## Output Files

- `finger_features.csv`: Coordinates from MediaPipe
- `labels.csv`: Manual labels
- `press_classifier.pkl`: Saved model
- `press_events.json`: Final output events
- `output_annotated_with_keys.mp4`: Annotated video
---

## Notes

- This system uses fingertip positions to detect white key presses only.
- Classifier used: Random Forest with balanced class weights.
