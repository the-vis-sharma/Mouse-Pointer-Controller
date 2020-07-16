from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import math
import time
import logging as logger
from model import Model

class GazeEstimation(Model):
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        self.model_structure = model_name
        self.model_weights = os.path.splitext(model_name)[0]+'.bin'
        self.device = device
        self.threshold = threshold
        self.exec_network = None
        self.total_infer_time = None

        # adding extensions
        if extensions and self.device == "CPU":
            self.plugin.add_extension(self.extensions, self.device)

        try:
            self.plugin = IECore()
            self.model=IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the Gaze Estimation Model. Have you enterred the correct model path?")

        self.input_name=[x for x in self.model.inputs.keys()]
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=[x for x in self.model.outputs.keys()]
        # self.output_shape=self.model.outputs[self.output_name[1]].shape

    def predict(self, left_eye, right_eye, head_pose):
        p_left_eye = self.preprocess_input(left_eye)
        p_right_eye = self.preprocess_input(right_eye)
        inputs = {'head_pose_angles':head_pose, 'left_eye_image':p_left_eye, 'right_eye_image':p_right_eye}
        infer_start_time = time.time()
        self.exec_network.start_async(request_id = 0, inputs=inputs)
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name[0]]
            self.total_infer_time = time.time() - infer_start_time
            x, y, result = self.preprocess_output(outputs, head_pose)

        return [x, y], result

    def preprocess_output(self, outputs, head_pose):
        # getting coordinates of the left and right eyes from the inference
        if len(outputs) > 0:
            result = outputs[0]
            roll = head_pose[2]
            angle = roll * math.pi / 180
            sin = math.sin(angle)
            cos = math.cos(angle)
            x = result[0] * cos + result[1] * sin
            y = -result[0] * sin + result[1] * cos

        return x, y, result
