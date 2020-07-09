from openvino.inference_engine import IENetwork, IECore
import os
import cv2

class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        self.model_structure = model_name
        self.model_weights = os.path.splitext(model_name)[0]+'.bin'
        self.device = device
        self.threshold = threshold
        self.exec_network = None

        # adding extensions
        if extensions and self.device == "CPU":
            self.plugin.add_extension(self.extensions, self.device)

        try:
            self.plugin = IECore()
            self.model=IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the Head Pose Estimation Model. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        # Load the IENetwork into the plugin
        if self.check_model():
            self.exec_network = self.plugin.load_network(self.model, self.device)
        else:
            exit(1)

    def predict(self, image):
        p_image = self.preprocess_input(image)
        self.exec_network.start_async(request_id = 0, inputs={self.input_name: p_image})
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            outputs = self.exec_network.requests[0].outputs
            angles = self.preprocess_output(outputs)
            
        return angles

    def check_model(self):
        # check model for unsupported layers
        keys = self.model.layers.keys()
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [layer for layer in keys if layer not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Found unsupported Layers: {}".format(unsupported_layers))
            print("Check if you have any extention for these layers")
            return False
        else:
            return True

    def preprocess_input(self, image):
        # Pre-process the image as needed
        p_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose(2, 0, 1)
        p_image = p_image.reshape(1, *p_image.shape)

        return p_image

    def preprocess_output(self, outputs):
        # getting coordinates of the left and right eyes from the inference
        fc_y = outputs['angle_y_fc'][0][0]
        fc_p = outputs['angle_p_fc'][0][0]
        fc_r = outputs['angle_r_fc'][0][0]
            
        return [fc_y, fc_p, fc_r]
