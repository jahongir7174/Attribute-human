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

        self.size = (256, 192)
        self.inputs = self.session.get_inputs()[0]

        self.mean = numpy.array([[[0.485, 0.456, 0.406]]])
        self.mean = self.mean.astype('float32')

        self.std = numpy.array([[[0.229, 0.224, 0.225]]])
        self.std = self.std.astype('float32')

    def __call__(self, images):
        """
        :param images: list of BGR images
        :return: list of predicted results
        """
        inputs = []
        for image in images:
            image = self.resize(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype('float32') / 255
            image = image - self.mean
            image = image / self.std
            image = image.transpose((2, 0, 1))
            inputs.append(image)
        inputs = numpy.stack(inputs)
        outputs = self.session.run(output_names=None,
                                   input_feed={self.inputs.name: inputs})[0]
        return ['Female' if x[22] > 0.5 else 'Male' for x in outputs]

    def resize(self, image):
        shape = image.shape[:2]

        scale_y = self.size[0] / float(shape[0])
        scale_x = self.size[1] / float(shape[1])

        return cv2.resize(image, dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
