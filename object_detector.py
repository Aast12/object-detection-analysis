import cv2
import numpy as np
from typing import List
import pandas as pd

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

        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def draw_labels(self, detections, img) -> None:
        font = cv2.FONT_HERSHEY_PLAIN
        for detection in detections:
            x, y, w, h = detection.box
            label = str(self.classes[detection.class_id])
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    def build_records(self, detections: List[Detection], frame_number, curr_ms):
        records = []
        for detection in detections:
            records.append({
                'frame_number': frame_number,
                'timestamp': curr_ms,
                'class': self.classes[detection.class_id],
                'confidence': detection.confidence,
                'x': detection.box[0],
                'y': detection.box[1],
                'width': detection.box[2],
                'height': detection.box[3]
            })

        return records

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
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        filtered_boxes = [boxes[i[0]] for i in indices]
        filtered_classes = [class_ids[i[0]] for i in indices]
        filtered_confidences = [confidences[i[0]] for i in indices]

        return [Detection(filtered_boxes[i], filtered_classes[i], filtered_confidences[i]) for i in range(len(filtered_boxes))]

    def get_outputs_names(self):
        layersNames = self.net.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def label_frame(self, frame, detections):
        self.draw_labels(detections, frame)

        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms, frame: %d' % (
            t * 1000.0 / cv2.getTickFrequency())

        cv2.putText(frame, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    def stream_webcam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        # vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        records = []

        frame_number = 0
        while True:
            ret, frame = cap.read()
            curr_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            if ret != True:
                break

            frame_number += 1

            blob = cv2.dnn.blobFromImage(
                frame, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            raw_output = self.net.forward(self.get_outputs_names())
            detections = self.process_output(frame, raw_output)

            records = records + self.build_records(detections, frame_number, curr_ms)

            self.label_frame(frame, detections)

            cv2.imshow("Video", frame)
            # vid_writer.write(frame.astype(np.uint8))

            if cv2.waitKey(1) & 0XFF == ord('q'):
                break

        cap.release()
        # vid_writer.release()

        return pd.DataFrame(records)

    def stream_videofile(self, file_path, process):
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_number = 0

        try: 
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number * (fps // 5))
                
                ret, frame = cap.read()

                curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                curr_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                if not ret:
                    break

                blob = cv2.dnn.blobFromImage(
                    frame, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
                self.net.setInput(blob)

                raw_output = self.net.forward(self.get_outputs_names())

                detections = self.process_output(frame, raw_output)

                process(detections, curr_frame, curr_ms)

                frame_number += 1
        except:
            cap.release()    

        cap.release()