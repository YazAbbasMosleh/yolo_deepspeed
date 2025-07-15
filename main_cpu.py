

import torch
from ultralytics import YOLO
import sys
import cv2
import numpy as np
from torchvision import transforms
import time
import os

# COCO class names for YOLO (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Force CPU usage
device = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Completely disable CUDA
torch.cuda.is_available = lambda: False  # Override CUDA check

# Check hardware availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {device}")

# Load the YOLO model and move to CPU
try:
    model = YOLO("yolo11n.pt").to(device)
    print("YOLO model loaded successfully on CPU!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img

# Perform inference
image_path = "test.jpg"
try:
    # Load original image
    original_img = preprocess_image(image_path)
    if original_img is None:
        raise ValueError(f"Failed to load image for visualization: {image_path}")

    # Verify devices
    print(f"\nDevice Verification:")
    print(f"Model device: cpu")  # We've forced CPU usage
    print(f"Processing image: {image_path}")

    # Warm-up
    print("\nPerforming warm-up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(original_img, verbose=False)

    # Timed inference
    print("\nPerforming 100 timed inference runs...")
    inference_times = []
    for i in range(100):
        start_time = time.time()
        with torch.no_grad():
            results = model(original_img, verbose=False)
        inference_times.append(time.time() - start_time)
    
    avg_time_ms = sum(inference_times) / len(inference_times) * 1000
    print(f"\nAverage inference time over 100 runs: {avg_time_ms:.2f} ms")

    # Process results from the last run
    for result in results:
        # Draw boxes on the original image
        boxes = result.boxes
        if boxes is not None:
            print(f"\nDetected {len(boxes)} objects:")
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = COCO_CLASSES[cls]
                label = f"{class_name}: {conf:.2f}"
                print(f"- {label} at [{x1}, {y1}, {x2}, {y2}]")
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = "output_testtttttt.jpg"
    cv2.imwrite(output_path, original_img)
    print(f"\nAnnotated image saved to: {output_path}")

except Exception as e:
    print(f"\nError during inference: {e}")
    sys.exit(1)