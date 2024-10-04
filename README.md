# VisionAid: Object Detection with Distance Estimation and Voice

This project applies TensorFlow's Object Detection API to live webcam feeds, providing real-time object detection along with distance estimation. The detected objects are announced via speech output using the `pyttsx3` library, making it useful for individuals with visual impairments. The application can be used both as a mobile app or website.

## Features
- **Real-time Object Detection:** Detects objects using a webcam feed.
- **Distance Estimation:** Estimates the distance of detected objects.
- **Voice Feedback:** Uses `pyttsx3` for audible feedback, announcing the class of detected objects.
- **Cross-Platform:** Can run on Android mobile devices or be accessed via a web interface.

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow
- `pyttsx3` for voice output
- `engine.io` for real-time communication

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Rabia-Usman/VisionAid-Object-Detection-with-Distance-Estimation-and-Voice.git
2. Navigate to the project directory:
   `cd VisionAid-Object-Detection-with-Distance-Estimation-and-Voice`
3. Install the required dependencies:
   `pip install -r requirements.txt`
4. Run the object detection script:
   `python webcam_blind_voice.py`

### Usage
1. Ensure your webcam is connected and run the script to start detecting objects.
2. The system will announce detected objects along with their estimated distance.
