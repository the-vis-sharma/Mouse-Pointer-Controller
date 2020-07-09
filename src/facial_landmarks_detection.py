from openvino.inference_engine import IENetwork, IECore
import os
import cv2

class FacialLandmarksDetection:
    '''
    Class for the Facial Landmarks Detection Model.
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
            raise ValueError("Could not Initialise the Facial Landmarks Detection Model. Have you enterred the correct model path?")

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
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            left_eye_coords, right_eye_coords = self.preprocess_output(outputs, image.shape[0], image.shape[1])
            if len(left_eye_coords) == 0 or len(right_eye_coords) == 0:
                return None, None, None, None
            else:
                left_eye = image[left_eye_coords[1]:left_eye_coords[3], left_eye_coords[0]:left_eye_coords[2]]
                right_eye = image[right_eye_coords[1]:right_eye_coords[3], right_eye_coords[0]:right_eye_coords[2]]

        return left_eye, right_eye, left_eye_coords, right_eye_coords

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
