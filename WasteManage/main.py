import yolov5
import cv2
import os

# load model
model = yolov5.load('keremberke/yolov5m-garbage')
  
# set model parameters
model.conf = 0.3  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img_path = r'C:\Users\Somashekar\OneDrive\Desktop\CommunityWatch\results\biodegradable5_jpg.rf.c45122faeda6d1739a64b6de2e75ee5a.jpg'

# perform inference
results = model(img_path, size=640)

# inference with test time augmentation
results = model(img_path, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# Load the image using OpenCV
image = cv2.imread(img_path)

# Draw bounding boxes on the image
for box, score, category in zip(boxes, scores, categories):
    box = list(map(int, box))
    class_name = str(int(category))  # Convert category to string for display
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)  # Reduce thickness to 1
    cv2.putText(image, f'{class_name}: {score:.2f}', (box[0], box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Adjust text position and size

# Save the image with bounding boxes
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, os.path.basename(img_path))
cv2.imwrite(save_path, image)

print(f"Saved annotated image to {save_path}")
