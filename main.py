import warnings

from nets import nn

warnings.filterwarnings("ignore")


def main():
    import cv2

    # Load model
    model = nn.ONNX(onnx_path='./weights/model.onnx')

    image = cv2.imread('./data/2.jpg')
    gender = model(image)
    print(gender)


if __name__ == "__main__":
    main()
