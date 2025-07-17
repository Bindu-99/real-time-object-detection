import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

print("Starting object detection program...")
print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")

class RealTimeObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully")
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        print(f"Available classes: {len(self.class_names)}")
        
    def process_frame(self, frame):
        print("Processing frame...")
        # Convert frame to RGB (YOLO expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(rgb_frame)[0]
        print(f"Detected {len(results.boxes)} objects")
        
        # Process results
        boxes = results.boxes
        for box in boxes:
            if box.conf.item() >= self.confidence_threshold:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class name
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f'{class_name}: {box.conf.item():.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def main():
    print("Initializing detector...")
    detector = RealTimeObjectDetector()
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        processed_frame = detector.process_frame(frame)
        
        # Display the frame in a window
        cv2.imshow('Object Detection', processed_frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break
    
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()