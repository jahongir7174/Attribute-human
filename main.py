import argparse
import warnings

import cv2

from nets import nn

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    # Load model
    model = nn.ONNX(onnx_path='./weights/model.onnx')

    image = cv2.imread(args.image_path)
    gender = model(image)
    print(gender)


if __name__ == "__main__":
    main()
