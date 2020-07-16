from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import time
import logging as logger
from model import Model

class FaceDetection(Model):
    '''
    Class for the Face Detection Model.
    '''
    def predict(self, image):
        p_image = self.preprocess_input(image)
        infer_start_time = time.time()
        self.exec_network.start_async(request_id = 0, inputs={self.input_name: p_image})
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            self.total_infer_time = time.time() - infer_start_time
            coords = self.preprocess_output(outputs, image.shape[0], image.shape[1])
            if len(coords) == 0:
                return None, None
            else:
                coords = coords[0]
                cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        return cropped_face, coords

    def preprocess_output(self, outputs, height, width):
        # getting coordinates of the face from the inference
        coords = []
        if len(outputs) > 0 and len(outputs[0]) > 0 and len(outputs[0][0]) > 0:
            for result in outputs[0][0]:
                _, label, conf, x1, y1, x2, y2 = result
                if conf > self.threshold:
                    x_min = int(x1 * width)
                    y_min = int(y1 * height)
                    x_max = int(x2 * width)
                    y_max = int(y2 * height)
                    coords.append([x_min, y_min, x_max, y_max])
        
        return coords
