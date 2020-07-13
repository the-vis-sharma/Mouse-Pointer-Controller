from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import time
import logging as logger

class FaceDetection:
    '''
    Class for the Face Detection Model.
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
            raise ValueError("Could not Initialise the Face Detection Model. Have you enterred the correct model path?")

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

    def check_model(self):
        # check model for unsupported layers
        keys = self.model.layers.keys()
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [layer for layer in keys if layer not in supported_layers]
        if len(unsupported_layers) != 0:
            logger.error("Found unsupported Layers: {}".format(unsupported_layers))
            logger.error("Check if you have any extention for these layers")
            return False
        else:
            return True

    def preprocess_input(self, image):
        # Pre-process the image as needed
        try:
            p_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            p_image = p_image.transpose(2, 0, 1)
            p_image = p_image.reshape(1, *p_image.shape)
            return p_image
        except Exception as e:
            logger.error(str(e))

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
