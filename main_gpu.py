## GPU
import deepspeed
import torch
from ultralytics import YOLO
import sys
import cv2
import numpy as np
from torchvision import transforms
from ultralytics.utils import ops
import time

# COCO class names for YOLO11n (80 classes)
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

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected'}")

# Load the YOLO model
try:
    model = YOLO("yolo11n.pt")
    print("YOLO model loaded successfully!")
    model = model.model  # Access the underlying PyTorch model
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Initialize DeepSpeed
try:
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config_gpu.json"
    )
    print("DeepSpeed initialized successfully!")
except Exception as e:
    print(f"Error initializing DeepSpeed: {e}")
    sys.exit(1)

# Preprocess the image
def preprocess_image(image_path, img_size=640):
    # Load image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Store original dimensions
    orig_h, orig_w = img.shape[:2]
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to model input size (e.g., 640x640 for YOLO11n)
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert to tensor and normalize
    img = transforms.ToTensor()(img)  # Shape: (3, 640, 640)
    img = img.unsqueeze(0)  # Add batch dimension: (1, 3, 640, 640)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        img = img.cuda()
    
    return img, orig_w, orig_h

# Perform inference
image_path = "yamal.jpg"  # Ensure this file exists
try:
    # Preprocess the image once
    img_tensor, orig_w, orig_h = preprocess_image(image_path)
    original_img = cv2.imread(image_path)  # Load original image for visualization
    if original_img is None:
        raise ValueError(f"Failed to load image for visualization: {image_path}")
    
    # Warm-up: Run 10 iterations to stabilize
    print("Performing warm-up (10 iterations)...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(img_tensor)

    # Timed inference: Run 100 iterations
    print("Performing 100 timed inference runs...")
    inference_times = []
    for i in range(100):
        start_time = time.time()
        with torch.no_grad():
            results = model(img_tensor)  
        inference_times.append(time.time() - start_time)
    
    # Calculate average inference time
    avg_time_ms = sum(inference_times) / len(inference_times) * 1000  # Convert to milliseconds
    print(f"-----------------------------------------------------Average inference time over 100 runs: {avg_time_ms:.2f} ms")

    if isinstance(results, tuple):
        pred = results[0]  
    else:
        pred = results
    
    print("Inference completed successfully!")
    print(f"Prediction tensor shape: {pred.shape}")
    
    # Post-process results (apply NMS) for the last run
    results = ops.non_max_suppression(
        pred,
        conf_thres=0.25,  # Confidence threshold
        iou_thres=0.45,   # IoU threshold for NMS
        max_det=300       # Maximum number of detections
    )
    
    # Draw bounding boxes on the original image with scaling
    for det in results:
        if len(det):
            print(f"Detections: {det.shape} (boxes: xyxy, conf, cls)")
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(float, xyxy)  # Convert to float for scaling
                # Scale coordinates from 640x640 to original image size
                scale_x = orig_w / 640
                scale_y = orig_h / 640
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                class_name = COCO_CLASSES[int(cls)]
                label = f"{class_name}: {conf:.2f}"
                print(f"Scaled Box: [{x1}, {y1}, {x2}, {y2}], Confidence: {conf:.2f}, Class: {class_name}")
                # Draw rectangle and label
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("No detections")
    
    # Save the annotated image
    output_path = "output_yamal.jpg"
    cv2.imwrite(output_path, original_img)
    print(f"Annotated image saved to: {output_path}")
    
    # Optional: Display the image (if running in a GUI environment)
    # cv2.imshow("Detections", original_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

except Exception as e:
    print(f"Error during inference: {e}")
    sys.exit(1)
finally:
    # Clean up distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()







