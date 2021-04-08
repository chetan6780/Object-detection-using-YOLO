# When it comes to deep learning-based object detection, there are three primary object detectors you’ll encounter:

# 1. R-CNN and their variants, including the original R-CNN, Fast R- CNN, and Faster R-CNN
# 2. Single Shot Detector (SSDs)
# 3. YOLO (You Only Look Once)

# While R-CNNs tend to very accurate, the biggest problem with the R-CNN family of networks is their speed — they were incredibly slow, obtaining only 5 FPS on a GPU. To help increase the speed of deep learning-based object detectors, both Single Shot Detectors (SSDs) and YOLO use a one-stage detector strategy.These algorithms treat object detection as a regression problem, taking a given input image and simultaneously learning bounding box coordinates and corresponding class label probabilities.In general, single-stage detectors tend to be less accurate than two-stage detectors but are significantly faster. YOLO is a great example of a single stage detector.

# I have tried both SSD and YOLO and i got better results with YOLO therefore below is the implimentaion of ***Object detection using YOLO***

# Load Required Packages
import cv2
import numpy as np

# paths for weight and configuration files and video path
weight_path = r'C:\Users\Admin\Desktop\TSF\TASKS\1\object-detection-using-YOLO\yolov3.weights'
cfg_path = r'C:\Users\Admin\Desktop\TSF\TASKS\1\object-detection-using-YOLO\yolov3.cfg'
video_path= r'C:\Users\Admin\Desktop\TSF\TASKS\1\object-detection-using-YOLO\media\sonali_kul.mp4'

# creating the list of names from the coco dataset
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
# print(classes)

net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

# capturing the video
cap = cv2.VideoCapture(video_path)

# setting up the confidence threshold and nms threshold
confidence_threshold = 0.5
nms_threshold = 0.5

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # image is in BGR we need to convert it to RGB and we convert it to blob image
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # here we set the input from blob to network and
    # get output layer names and from these names we can obtain the layer outputs as list
    # layer outputs gets 85 parameters first 4 are (x,y,w,h) and 5 th is confidence and remaining 80 are object names
    net.setInput(blob)  
    output_layers_names = net.getUnconnectedOutLayersNames() 
    layerOutputs = net.forward(output_layers_names)

    # creating variables to store the data
    boxes = []
    confidences = []
    class_ids = []

    # Here we will extract the data from output layer
    for output in layerOutputs:
        for detection in output:
            
            scores = detection[5:]        # store score of 80 classes which starts from 6th index
            class_id = np.argmax(scores)  # we get maximum score location
            confidence = scores[class_id] # get the score of the class id
            
            # if confidence is satisfying we extract the center location and width, height of detected image
            if confidence > confidence_threshold: 
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # get the position of upper corners
                x = int(center_x-w/2)
                y = int(center_y-h/2)

                # collecting all information
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # To remove the multiple bounding boxes which have been detected we use Non-Maximum Suppression method
    # it suppresses weak, overlapping bounding boxes in result giving only string bounding boxes in output
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Defining font and variable color
    font = cv2.FONT_HERSHEY_COMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # loop through each of the object detected and extract the information
    if len(indexes) != 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i]*100, 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            # putText: (img, text, org, fontFace, fontScale, color, thickness)
            cv2.putText(img, label.upper()+" "+confidence, (boxes[i][0], boxes[i][1]),font, .85, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # 27 = esc key
        break

# here we release the captur object
cap.release()
cv2.destroyAllWindows()
