# Real-time Hand Tracking with Mediapipe

This project demonstrates real-time hand tracking using the Mediapipe library in Python with OpenCV.

## Description

The project includes the following steps:

1. **Initialization**: The OpenCV video capture is set up to capture frames from the webcam. The hand tracking model from Mediapipe is initialized.

2. **Hand Detection and Landmark Tracking**: Frames from the webcam are processed to detect and track hand landmarks using the Mediapipe hand tracking model. Landmarks for each detected hand are extracted and visualized on the frame.

3. **Finger Counting**: Finger counting is performed based on the detected landmarks. The position and orientation of each finger are analyzed to determine the count of extended fingers.

4. **Display**: The finger count is displayed on the frame in real-time.

## Requirements

To run the project, you'll need:

- Python (version 3.9)
- OpenCV 
- Mediapipe 

## Setup

1. Clone the repository:
git clone https://github.com/MohamedHamdy549/Hand-Tracker.git
2. Install the required dependencies:
pip install opencv-python mediapipe

## Usage

1. Run the Python script `hand_tracking.py`.
2. Ensure that your webcam is enabled and properly configured.
3. Perform hand gestures in front of the webcam, and the finger count will be displayed in real-time.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
