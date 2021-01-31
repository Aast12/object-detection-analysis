from object_detector import ObjectDetector

detector = ObjectDetector('config/yolov3.weights',
                          'config/yolov3.cfg', 'config/coco.names')

detector.stream('yolo_out_py.avi')
