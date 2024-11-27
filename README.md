# Real Time Rep Counter with Mediapipe

## Overview

This project implements a real-time rep counter using OpenCV and Mediapipe. The application leverages Mediapipe's Pose estimation to track and analyze arm movements. The program visualizes the current curl count and provides feedback on the curl progress through a dynamic progress bar.

## Features

- **Real-time Pose Estimation:** Utilizes Mediapipe's Pose module to estimate and track key body landmarks.
- **Angle Calculation:** Computes the angle to determine the exercise position.
- **Curl Counting:** Counts the number of reps based on the angle changes.
- **Visual Feedback:** Displays a progress bar and rep count on the video feed.
- **Customizable Window Layout:** Adjustable window size and positioning for optimal user experience.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- NumPy

You can install the required Python packages using pip:

```bash
pip install opencv-python mediapipe numpy
```

## Code Explanation
- **calculate_angle:** Function to compute the angle between any three body landmarks and draw visual aids (lines and circles) on the frame.
- **Main Loop:** Captures video frames, processes them with Mediapipe to estimate pose landmarks, calculates the required angle, updates the rep count, and visualizes the progress and count on the video feed.

## Troubleshooting
**Window Not Closing:** If the 'X' button does not close the window, use the 'q' key to quit the application. Ensure that you have the latest versions of OpenCV and Mediapipe.

## Contributing
Feel free to open issues or submit pull requests to improve the functionality or add features.
