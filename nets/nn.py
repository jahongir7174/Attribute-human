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

        self.threshold = 0.5
        self.hold_threshold = 0.6
        self.glasses_threshold = 0.3

        self.age = ['0~18', '18~60', '60~']
        self.direction = ['Front', 'Side', 'Back']

        self.bag = ['HandBag', 'ShoulderBag', 'Backpack']
        self.upper = ['UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice']
        self.lower = ['LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress']

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
        results = []

        for output in outputs:
            result = []

            # gender
            gender = 'Female' if output[22] > 0.5 else 'Male'
            result.append(gender)

            # age
            age = 'Age: ' + self.age[numpy.argmax(output[19:22])]
            result.append(age)

            # direction
            direction = 'Direction: ' + self.direction[numpy.argmax(output[23:])]
            result.append(direction)

            # glasses
            glasses = 'Glasses: '
            if output[1] > self.glasses_threshold:
                glasses += 'True'
            else:
                glasses += 'False'
            result.append(glasses)

            # hat
            hat = 'Hat: '
            if output[0] > self.threshold:
                hat += 'True'
            else:
                hat += 'False'
            result.append(hat)

            # hold object
            hold_object = 'HoldObjectsInFront: '
            if output[18] > self.hold_threshold:
                hold_object += 'True'
            else:
                hold_object += 'False'
            result.append(hold_object)

            # bag
            bag = self.bag[numpy.argmax(output[15:18])]
            bag_score = output[15 + numpy.argmax(output[15:18])]
            bag_label = 'Bag: ' + bag if bag_score > self.threshold else 'No bag'
            result.append(bag_label)

            # upper
            upper_res = output[4:8]
            upper_label = 'Upper:'
            sleeve = 'LongSleeve' if output[3] > output[2] else 'ShortSleeve'
            upper_label += ' {}'.format(sleeve)
            for i, r in enumerate(upper_res):
                if r > self.threshold:
                    upper_label += ' {}'.format(self.upper[i])
            result.append(upper_label)

            # lower
            lower_res = output[8:14]
            lower_label = 'Lower: '
            has_lower = False
            for i, l in enumerate(lower_res):
                if l > self.threshold:
                    lower_label += ' {}'.format(self.lower[i])
                    has_lower = True
            if not has_lower:
                lower_label += ' {}'.format(self.lower[numpy.argmax(lower_res)])

            result.append(lower_label)

            # shoe
            shoe = 'Boots' if output[14] > self.threshold else 'No boots'
            result.append(shoe)
            results.append(result)
        return results

    def resize(self, image):
        shape = image.shape[:2]

        scale_y = self.size[0] / float(shape[0])
        scale_x = self.size[1] / float(shape[1])

        return cv2.resize(image, dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
