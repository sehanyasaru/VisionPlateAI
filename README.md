Overview

This project demonstrates how to use YOLO for object detection and EasyOCR for text recognition to identify and extract license plates from a video. It leverages computer vision libraries such as OpenCV, EasyOCR, and cvzone for processing frames, detecting objects, and recognizing text.

### Requirements

Ensure you have the following dependencies installed:

1. **Python 3.8 or later**
2. **OpenCV (opencv-python)**
3. **EasyOCR (easyocr)**
4. **Ultralytics YOLO (ultralytics)**
5. **cvzone (cvzone)**
6. **Math (standard Python library)**

### Installation

1. Clone the repository or download the script file.
2. Install the required Python libraries using pip:
`pip install opencv-python easyocr ultralytics cvzone`
3. Update the video_path variable in the script to point to the video file you want to process.
**Link for the video that I have been used** :- [https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/](url)

### Code Explanation
**Initialization**
`model = YOLO("best1.pt")
video_path = "C:\\Users\\User\\Desktop\\FInal OCR\\sample.mp4"
cap = cv2.VideoCapture(video_path)
reader = easyocr.Reader(['en'])`

- Load the YOLO model `best1.pt`
- Set the path to the input video.
- Initialize the EasyOCR reader for English text recognition.

**Processing the Video**
`while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))`

- Read video frames in a loop until the video ends.
- Resize each frame to a fixed size (640x480) for uniform processing.

**Object Detection with YOLO**
`  result = model(frame)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)`

- Pass each frame through the YOLO model for object detection.
- Extract bounding box coordinates, confidence scores, and class IDs for detected objects

**License Plate Text Extraction**
` plate_roi = frame[y1:y2, x1:x2]
            extracted_results = reader.readtext(plate_roi)
            
            for result1 in extracted_results:
                text = result1[1]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"License Plate: {text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)`

- Crop the region of interest (ROI) containing the license plate.
- Use EasyOCR to extract text from the cropped image.
- Draw a rectangle around the detected license plate and display the extracted text on the frame.

**Displaying Output**
`    cv2.imshow('YOLO Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break`

- Display the processed frame in a window.
- Exit the loop if the 'q' key is pressed.

**Cleanup**
```
cap.release()
cv2.destroyAllWindows()
```
- Release the video capture object and close all OpenCV windows.

**Notes**

- Ensure the YOLO model file `(best1.pt)` is trained or fine-tuned for license plate detection.
- Update the EasyOCR language list if you want to recognize text in other languages.
- The performance of the text recognition depends on the quality of the video and the clarity of the license plates.

**Limitations**

- Detection accuracy may vary depending on lighting conditions and motion blur in the video.
- OCR may fail to recognize text accurately if the license plate is too small or obscured.
