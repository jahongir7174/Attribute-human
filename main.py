import argparse
import warnings

import cv2

from nets import nn

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs='+')
    args = parser.parse_args()

    # Load model
    model = nn.ONNX(onnx_path='./weights/model.onnx')

    inputs = []
    filenames = []
    for filename in args.image_path:
        filenames.append(filename)
    for filename in filenames:
        image = cv2.imread(filename)
        inputs.append(image)

    outputs = model(inputs)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
