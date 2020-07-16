from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import time
import logging as logger
from model import Model

class HeadPoseEstimation(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def predict(self, image):
        p_image = self.preprocess_input(image)
        infer_start_time = time.time()
        self.exec_network.start_async(request_id = 0, inputs={self.input_name: p_image})
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs
            self.total_infer_time = time.time() - infer_start_time
            angles = self.preprocess_output(outputs)
            
        return angles

    def preprocess_output(self, outputs):
        # getting coordinates of the left and right eyes from the inference
        fc_y = outputs['angle_y_fc'][0][0]
        fc_p = outputs['angle_p_fc'][0][0]
        fc_r = outputs['angle_r_fc'][0][0]
            
        return [fc_y, fc_p, fc_r]
