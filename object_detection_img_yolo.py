import cv2
import numpy as np

# paths for weight , configuration file and input image
weight_path = r'C:\Users\Admin\Desktop\TSF\TASKS\1\object-detection-using-YOLO\yolov3.weights'
cfg_path = r'C:\Users\Admin\Desktop\TSF\TASKS\1\object-detection-using-YOLO\yolov3.cfg'
img_path = r'C:\Users\Admin\Desktop\TSF\TASKS\1\object-detection-using-YOLO\media\person.jpg'

net = cv2.dnn.readNet(cfg_path, weight_path)

# creating the list of names from the coco dataset
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
print(classes)

img = cv2.imread(img_path)
height, width, _ = img.shape

# image is in BGR we need to convert it to RGB and we convert it to blob image
blob = cv2.dnn.blobFromImage(
    img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

# here we set the input from blob to network and get output layer names and from these names we can obtain the layer outputs as list layer outputs gets 85 parameters first 4 are (x,y,w,h) and 5 th is confidence and remaining 80 are object names
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
        class_id = np.argmax(scores)  # get the maximum score location
        confidence = scores[class_id] # get the score of the class id
        
        # if confidence is satisfying we extract the center location and width, height of detected image
        if confidence > 0.5:  
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
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Defining font and variable color
font = cv2.FONT_HERSHEY_COMPLEX
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

# loop through each of the object detected and extract the information
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i]*100, 2))
    color = colors[i]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    # putText: (img, text, org, fontFace, fontScale, color, thickness)
    cv2.putText(img, label.upper()+" "+confidence, (boxes[i][0], boxes[i][1]),
                font, .85, (0, 0, 255), 2)

# Here we show the output and then clear all windows
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
