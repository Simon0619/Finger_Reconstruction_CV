import cv2
import csv

# Open video
video_path = 'pre_training.mp4'
cap = cv2.VideoCapture(video_path)

labels = []
frame_num = 0

print(" Press [Space] to label the frame as key pressed (1)")
print(" Press any other key to label as not pressed (0)")
print(" Press [q] to quit labeling")

#Continue Reading
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Show frame number on video
    display_frame = frame.copy()
    cv2.putText(display_frame, f'Frame {frame_num}', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Frame Labeling", display_frame)

    key = cv2.waitKey(0)
    if key == ord('q'):
        print("Labeling stopped early.")
        break
    elif key == ord(' '):  # Space key
        labels.append(1)
        print(f"[✓] Frame {frame_num} → Pressed")
    else:
        labels.append(0)
        print(f"[ ] Frame {frame_num} → Not pressed")
    frame_num += 1

cap.release()
cv2.destroyAllWindows()

# Save labels to CSV
with open('labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'label'])
    for i, label in enumerate(labels):
        writer.writerow([i, label])

print(f"\n Total {len(labels)} labels saved to 'labels.csv'")
