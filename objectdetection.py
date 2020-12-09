import cv2
import numpy as np


#Loading YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]  - 1] for i in net.getUnconnectedOutLayers()]


#process image
img = cv2.imread("image.jpg")
height, width, channels = img.shape
#Object detection blob is to extract feature from the image
blob = cv2.dnn.blobFromImage(img, 0.00392, (640, 480), (0, 0, 0), True, crop=False)

#this blob is passed to yolo
net.setInput(blob)
outs = net.forward(outputlayers)


#showing info
class_ids=[]
confidences = []
objects = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5: #0-1
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x  - w/2)
            y = int(center_y  - h/2)
            #cv2.circle(img , (center_x,center_y), 10, (0,255,0), 2)
            objects.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(objects, confidences, 0.5, 0.4)#to reduce the multiple detection(noices)
print(len(objects))
print(indexes)
for i in range(len(objects)):
    if i in indexes:
        x, y, w, h = objects[i]
        lable = str(classes[class_ids[i]])
        print(lable)


cv2.imshow("image.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
quit()
