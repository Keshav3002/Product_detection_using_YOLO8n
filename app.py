from flask import Flask, request, jsonify, render_template
from detection_model import ProductDetector
from product_grouping import ProductGrouper
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'static/output/'
upload_path = 'static/output/'
# Initialize the product detector
detector = ProductDetector()

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

@app.route('/detect-and-group', methods=['POST'])
def detect_and_group():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_PATH'], filename)
    file.save(filepath)
    
    # Load the image using OpenCV
    img = cv2.imread(filepath)
    
    # Run YOLOv8 detection
    results = detector.detect_objects(img)
    
    # Prepare lists for bounding boxes and feature vectors
    bounding_boxes = []
    feature_vectors = []

    for result in results:
        names = result.names
        for box in result.boxes:
            # Extract box information
            x1, y1, x2, y2 = map(float, box.xyxy[0])  # Bounding box coordinates
            bounding_boxes.append([x1, y1, x2, y2])
            
            # Extract features from the bounding box
            feature_vector = detector.extract_features_from_box(img, [x1, y1, x2, y2])
            feature_vectors.append(feature_vector)

    # Convert feature vectors to numpy array
    feature_vectors = np.array(feature_vectors)
    
    # Determine the number of clusters based on the number of feature vectors
    n_clusters = min(5, len(feature_vectors))  # Here, 5 is the max number of clusters you desire

    clusters = []  # Default empty list for clusters
    if n_clusters > 0:  # Check if there are samples to cluster
        grouper = ProductGrouper(n_clusters=n_clusters)  # Initialize grouper with dynamic clusters
        clusters = grouper.group(feature_vectors)  # Apply K-means clustering

    # Prepare the response with bounding boxes and cluster labels
    detections = []
    for i, box in enumerate(bounding_boxes):
        cls = int(result.boxes[i].cls)  # Class index (e.g., 39)
        label = names[cls]
        # print(cls)
        detections.append({
            'class_name': cls,
            'confidence': float(result.boxes[i].conf),  # Confidence score
            'bbox': box,
            'label': label,
            'cluster': label if len(clusters)>0 else None  # Cluster label
        })
    
    # Save the detection result image with bounding boxes and cluster labels
    result_image_path = os.path.join(app.config['UPLOAD_PATH'], f"result_{filename}")
    # for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
    #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #     cv2.putText(img, f'{names[cls]}' if len(clusters)>0 else 'One Cluster', 
    #                 (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #####################################
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection['bbox'])
        label = detection['label']  # Retrieve label for each bounding box
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        cv2.putText(img, f'{label}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #####################################
    cv2.imwrite(result_image_path, img)

    # Render the results in an HTML template
    return render_template('result.html', detections=detections, result_image_path=result_image_path)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
