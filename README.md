Human Attribute Recognition

### Installation

```
conda create -n ONNX python=3.8
conda activate ONNX
pip install opencv-python==4.5.5.64
pip install onnxruntime
```

### Run

* Run `python main.py ./demo/demo.jpg`

### Results

<img src="./demo/demo.jpg" width="960" alt="">

```
['Male', 'Age: 18~60', 'Direction: Front', 'Glasses: True', 'Hat: False', 'HoldObjectsInFront: True', 'No bag', 'Upper: LongSleeve', 'Lower:  LongCoat Trousers', 'No boots']
```
#### Reference

* https://github.com/MhLiao/DB
