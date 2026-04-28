import cv2
from ultralytics import YOLO

# Load YOLO model (person detection only)
model = YOLO("yolov8n.pt")

# Load video
video_path = "people(2).mp4"
cap = cv2.VideoCapture(video_path)

# Frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional resize (recommended for stability)
RESIZE = True
target_width, target_height = 1280, 720

# Middle reference line (not strict, just visual aid)
line_x = int(width * 0.75)

# Counters
in_count = 0
out_count = 0

# Tracking memory
track_history = {}
counted_ids = {}  # id → "in" or "out"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    if RESIZE:
        frame = cv2.resize(frame, (target_width, target_height))
        h, w = frame.shape[:2]
        line_x = w // 2

    # Run YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],  # person only
        conf=0.4,
        iou=0.7,
        verbose=False
    )

    if results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0])

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Initialize history
            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append(cx)

            # Keep only last 2 positions
            if len(track_history[track_id]) > 2:
                track_history[track_id].pop(0)

            # Direction detection
            if len(track_history[track_id]) == 2:
                prev_x = track_history[track_id][0]
                curr_x = track_history[track_id][1]

                # LEFT → RIGHT = IN
                if prev_x < line_x and curr_x >= line_x:
                    if track_id not in counted_ids:
                        counted_ids[track_id] = "in"
                        in_count += 1

                # RIGHT → LEFT = OUT
                elif prev_x > line_x and curr_x <= line_x:
                    if track_id not in counted_ids:
                        counted_ids[track_id] = "out"
                        out_count += 1

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Show ID
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw reference line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)

    # Display counts
    cv2.putText(frame, f"IN: {in_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"OUT: {out_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People IN/OUT Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final IN:", in_count)
print("Final OUT:", out_count)