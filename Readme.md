# People IN/OUT Counter using YOLOv8 + ByteTrack

A real-time people counting system built using **YOLOv8**, **OpenCV**, and **ByteTrack**.  
This project detects, tracks, and counts people entering or leaving an area based on movement direction across a virtual line.

---

# Features

- Real-time person detection
- Object tracking using ByteTrack
- IN/OUT counting system
- Direction detection
- Unique ID assignment
- Displays tracking IDs
- Supports custom videos
- Easy to customize for different counting zones

---

# Demo

The system:

1. Detects people using YOLOv8
2. Assigns a unique tracking ID
3. Tracks movement direction
4. Counts:
   - LEFT → RIGHT as **IN**
   - RIGHT → LEFT as **OUT**

---

# Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack

---

# Project Structure

```bash
people-counter/
│
├── people(2).mp4
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Anwaarhere27/People-Counter.git
cd people-counter
```

---

## 2. Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install ultralytics opencv-python
```

---

# Download YOLOv8 Model

The project uses YOLOv8 Nano model:

```python
model = YOLO("yolov8n.pt")
```

The model will automatically download on first run.

---

# Run the Project

```bash
python main.py
```

Press:

```bash
q
```

to quit the application.

---

# How Counting Works

A vertical reference line is placed at the center of the frame.

The system tracks the horizontal movement of each detected person:

- LEFT → RIGHT = IN
- RIGHT → LEFT = OUT

Each tracked person is counted only once using a unique tracking ID.

---

# Current Detection Class

Currently detecting:

```python
classes=[0]
```

COCO Class IDs:

| Object | Class ID |
|---------|----------|
| Person | 0 |

---

# Configuration

## Resize Frames

```python
RESIZE = True
target_width, target_height = 1280, 720
```

---

## Change Reference Line Position

```python
line_x = w // 2
```

Example:

```python
line_x = int(w * 0.75)
```

---

## Tracking Settings

```python
tracker="bytetrack.yaml"
```

---

# Example Output

```bash
Final IN: 14
Final OUT: 9
```

---

# Future Improvements

- Multi-zone counting
- Heatmap visualization
- Crowd analytics
- Live webcam support
- Database logging
- Streamlit dashboard
- Direction arrows
- Multi-camera tracking

---

# Requirements

Example `requirements.txt`

```txt
ultralytics
opencv-python
numpy
```

---

# Screenshots

Add screenshots here:

```bash
screenshots/output.png
```

---


# Author

**Anwaar Muhammad**  
AI/ML Engineer | Computer Vision | Generative AI

GitHub: https://github.com/Anwaarhere27
