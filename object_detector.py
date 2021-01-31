import cv2 as cv
import numpy as np
from typing import List

CONF_THRESHOLD = 0.5    # Confidence threshold
NMS_THRESHOLD = 0.4     # Non-maximum suppression threshold
INPUT_WIDTH = 416       # Width of network's input image
INPUT_HEIGHT = 416      # Height of network's input image

class Detection:
    def __init__(self, box, class_id, confidence) -> None:
        self.box = box
        self.class_id = class_id
        self.confidence = confidence
        

class ObjectDetector:
    def __init__(self, weights_path, config_path, classes_path) -> None:

        with open(classes_path, 'rt') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.net = cv.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def draw_labels(self, detections, img) -> None: 
        font = cv.FONT_HERSHEY_PLAIN
        for detection in detections:
            x, y, w, h = detection.box
            label = str(self.classes[detection.class_id])
            color = (0, 0, 255)
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y - 5), font, 1, color, 1)
            
    def process_output(self, frame, outs) -> List[Detection]:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONF_THRESHOLD:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])

        # Apply non maximum suppression to eliminate redundancy
        indices = cv.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        filtered_boxes = [boxes[i[0]] for i in indices]
        filtered_classes = [class_ids[i[0]] for i in indices]
        filtered_confidences = [class_ids[i[0]] for i in indices]

        return [Detection(filtered_boxes[i], filtered_classes[i], filtered_confidences[i]) for i in range(len(filtered_boxes))]

    def getOutputsNames(self):
        layersNames = self.net.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def stream(self, output_file):
        cap = cv.VideoCapture(0)
        vid_writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
        while True:
            ret, frame = cap.read()
            if ret != True:
                break

            blob = cv.dnn.blobFromImage(frame, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

            self.net.setInput(blob)

            raw_output = self.net.forward(self.getOutputsNames())

            detections = self.process_output(frame, raw_output)

            self.draw_labels(detections, frame)

            t, _ = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            cv.imshow("Video", frame)
            vid_writer.write(frame.astype(np.uint8))
            
            if cv.waitKey(1) & 0XFF == ord('q'):
                break
            
        cap.release()