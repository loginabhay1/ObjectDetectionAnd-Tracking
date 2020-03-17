import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

classes = None
scale = 0.00392
net = cv2.dnn.readNet('./yolov3_last.weights','./yolov3.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open('./object.names','r') as file:
    classes = [line.strip() for line in file.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def Object_Detect(frame):
    Width = frame.shape[1]
    Height = frame.shape[0]

    blob = cv2.dnn.blobFromImage(frame, scale, (608,608), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network and gather predictions from output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    boxe = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer get the confidence, class id, 
    # bounding box params and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes_1 = (x,y,w,h)
                boxe.append([x,y,w,h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxe, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining after nms and draw bounding box
    # for i in indices:
    #     i = i[0]
    #     box = boxes[i]
    #     x = box[0]
    #     y = box[1]
    #     w = box[2]
    #     h = box[3]

    #     draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    return (boxes_1,indices)



#---------------------------------------------------------#
if __name__ == "__main__":

  cap = cv2.VideoCapture('./../tv_hdmi_obj.mp4')
  i = 0
  while(1):
      ok, frame = cap.read()
      if (i%15 == 0):
          frame = np.asarray(frame)
          boxes,indices = Object_Detect(frame)
          print('Object Detection boxes: ',boxes)
          point = True
      elif point == True:
        #   ok, frame = cap.read()
          tracker = cv2.TrackerKCF_create()
          ok = tracker.init(frame, boxes)
          print('Initial tracker boxes: ', boxes)
          point = False
      else:
          ok, bbox = tracker.update(frame)
          print('Tracking boxes: ', bbox)
          if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            frame = cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
      
      cv2.imshow('Window', frame)
      key = cv2.waitKey(1) & 0xff
      if key == ord("b"):
          break  
      i = i+1
#   track_the_object = Tracking()
#   print(boxes,'\n', indices)
