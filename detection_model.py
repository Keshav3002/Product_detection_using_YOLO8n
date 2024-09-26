from ultralytics import YOLO
# from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2



class ProductDetector:
    def __init__(self):
        """Initialize the YOLOv8 model and feature extraction model."""
        self.yolo_model = YOLO("yolov8n.pt")  # Load the pre-trained YOLOv8 model
        
        # Load VGG16/imagenet model for feature extraction
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        output_layer = base_model.output
        output_layer = GlobalAveragePooling2D()(output_layer)
        self.feature_extractor = Model(inputs=base_model.input, outputs=output_layer)
        
    def detect_objects(self, img):
        """Run YOLOv8 detection on the image and return bounding boxes."""
        results = self.yolo_model(img)
        print("here!!!!")
        for result in results:
            # Get all detected boxes
            boxes = result.boxes
            names = result.names  # Dictionary of class names (e.g., {0: 'person', 1: 'bicycle', ...})

            if boxes is not None:
                # Iterate through each detected object in the result
                for box in boxes:
                    class_id = int(box.cls[0])  # Class ID
                    label = names[class_id]  # Get the label from the class ID
                    conf = box.conf[0]  # Confidence score
                    bbox = box.xyxy[0]  # Bounding box coordinates (x1, y1, x2, y2)

                    # Print the label, confidence score, and bounding box
                    # print(f"cls: {class_id}, Label: {label}, Confidence: {conf:.2f}, BBox: {bbox}")
        return results

    def extract_features_from_box(self, img, box):
        """Extract features from a bounding box region using VGG16."""
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        crop_img = img[y1:y2, x1:x2]  # Crop the image using bounding box
        crop_img = cv2.resize(crop_img, (224, 224))  # Resize to match VGG16 input size

        # Preprocess the image
        crop_img = crop_img.astype('float32') / 255.0
        crop_img = np.expand_dims(crop_img, axis=0)

        # Extract features
        features = self.feature_extractor.predict(crop_img)
        return features.flatten()
