import os

import cv2
import numpy


class ONNX:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CPUExecutionProvider'])

        self.inputs = self.session.get_inputs()[0]
        self.input_size = (256, 192)
        self.mean = numpy.array([0.485, 0.456, 0.406])[numpy.newaxis, numpy.newaxis, :]
        self.std = numpy.array([0.229, 0.224, 0.225])[numpy.newaxis, numpy.newaxis, :]

    def __call__(self, image):
        image = self.resize(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255
        image = image - self.mean
        image = image / self.std
        image = image.transpose((2, 0, 1))
        image = image[numpy.newaxis, ...].astype('float32')

        outputs = self.session.run(output_names=None,
                                   input_feed={self.inputs.name: image})
        return 'Female' if outputs[0][0][22] > 0.5 else 'Male'

    def resize(self, image):
        shape = image.shape
        scale_y = self.input_size[0] / float(shape[0])
        scale_x = self.input_size[1] / float(shape[1])
        image = cv2.resize(image,
                           None,
                           None,
                           fx=scale_x,
                           fy=scale_y,
                           interpolation=cv2.INTER_LINEAR)
        return image
