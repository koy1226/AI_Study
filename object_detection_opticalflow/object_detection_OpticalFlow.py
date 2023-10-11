import cv2
from tracker import *
import numpy as np
import torch

import sys
sys.path.append(r"Path")
#from models.experimental import attempt_load
from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model = YOLO('yolov8x.pt')  # load yolov5
#metrics = model.val()

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # Convert gray to bgr
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw lines
    cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)
    
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)
        
    return vis

# Create tracker object
tracker = EuclideanDistTracker()

# Initialize webcam
cap = cv2.VideoCapture(0)

ret, previous_frame = cap.read()
if not ret:
    print("Failed to grab frame")

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read Webcam video frame
    ret, frame = cap.read()
    # print(frame.shape)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Compute optical flow
    optical_flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract flow magnitude and angle
    magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    
    # Object detection with YOLOv8
    with torch.no_grad():
        # Comvert NumPy Arr to PyTorch tensor
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # print(input_tensor.shape)
        results = model(input_tensor)
    # print(type(results), results)
    
    first_result = results[0]
    
    # boxes와 확률 정보를 추출
    # boxes = first_result[:, :4].cpu().numpy()  # xyxy format
    boxes = first_result.boxes.cpu().numpy() if first_result.boxes is not None else None
    print(first_result.probs)
    print(first_result)
    # confidences = first_result.probs.cpu().numpy() if first_result.probs is not None else None
    class_names = first_result.names

    # # 신뢰도가 0.5 이상인 결과만 추출
    # if confidences is not None:  # None 확인 추가
    #     confident_indices = confidences > 0.3
    #     confident_boxes = boxes[confident_indices]
    #     confident_labels = class_names[confident_indices]
    # else:
    #     print("No confidences available")
    #     continue

    # Draw bounding box and put label
    if boxes is not None:
        print(boxes)
        print("Shape of boxes:", boxes.shape)
        print("Total number of boxes:", len(boxes))
        for i in range(len(boxes)):
            print("Box {}: {}".format(i, boxes[i].data))
            x1, y1, x2, y2, _, label = boxes[i].data[0] 

            label = int(label)  # label을 정수로 변환
            if class_names[label] in ['person', 'oven', 'bed']:
                continue    
            class_name = class_names[label]  # 해당 label에 대한 class name
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 해당 객체의 옵티컬 플로우 계산
            obj_flow = optical_flow[y:y+h, x:x+w]
            print("Optical flow for object with label", class_names[label], ":", obj_flow)

    cv2.imshow("Frame", frame)
    cv2.imshow('Frame2', draw_flow(gray, optical_flow))

    key = cv2.waitKey(100) & 0xff
    if key == ord('q'):
        break

    # Update previous frame
    previous_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
