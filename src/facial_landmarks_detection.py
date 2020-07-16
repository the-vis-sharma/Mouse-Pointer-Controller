from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import time
import logging as logger
from model import Model

class FacialLandmarksDetection(Model):
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def predict(self, image):
        p_image = self.preprocess_input(image)
        infer_start_time = time.time()
        self.exec_network.start_async(request_id = 0, inputs={self.input_name: p_image})
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            self.total_infer_time = time.time() - infer_start_time
            left_eye_coords, right_eye_coords = self.preprocess_output(outputs, image.shape[0], image.shape[1])
            if len(left_eye_coords) == 0 or len(right_eye_coords) == 0:
                return None, None, None, None
            else:
                left_eye = image[left_eye_coords[1]:left_eye_coords[3], left_eye_coords[0]:left_eye_coords[2]]
                right_eye = image[right_eye_coords[1]:right_eye_coords[3], right_eye_coords[0]:right_eye_coords[2]]

        return left_eye, right_eye, left_eye_coords, right_eye_coords

    def preprocess_output(self, outputs, height, width):
        # getting coordinates of the left and right eyes from the inference
        margin = 20
        if len(outputs) > 0:
            result = outputs[0]
            x1 = result[0][0][0] * width
            y1 = result[1][0][0] * height
            x2 = result[2][0][0] * width
            y2 = result[3][0][0] * height
            
            left_eye_coords = [int(x1 - margin), int(y1 - margin), int(x1 + margin), int(y1 + margin)]
            right_eye_coords = [int(x2 - margin), int(y2 - margin), int(x2 + margin), int(y2 + margin)]
        
        return left_eye_coords, right_eye_coords
